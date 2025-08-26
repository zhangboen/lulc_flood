from pyogrio import read_dataframe,write_dataframe
from shapely.geometry import Polygon, MultiPolygon
import geopandas as gpd
import os,glob,sys,time,re
import networkx as nx
import pandas as pd
from shapely.geometry import box
import numpy as np
from typing import Dict, List, Tuple
import multiprocessing as mp
from parallel_pandas import ParallelPandas
from tqdm.auto import tqdm
from functools import reduce
from pathlib import Path
from shapely.ops import unary_union
from packaging import version
import argparse

pandas_version = pd.__version__

dir0 = Path('/data/ouce-drift/cenv1021/')

def fill_holes(polygon):
    """Fills holes in a polygon."""
    if polygon.geom_type == 'Polygon':
        exteriors = [Polygon(polygon.exterior)]
        rings = list(polygon.interiors) #List all interior rings
    elif polygon.geom_type == 'MultiPolygon':
        rings = []
        exteriors = []
        for geom in polygon.geoms:
            exteriors.append(Polygon(geom.exterior))
            rings = rings + list(geom.interiors)
    if len(rings)>0: #If there are any rings
        to_fill = [Polygon(ring) for ring in rings] #List the ones to fill
        newgeom = reduce(lambda geom1, geom2: geom1.union(geom2),exteriors+to_fill) #Union the original geometry with all holes
        return newgeom
    else:
        return polygon

def get_upstream(global_id, G):
    nodes = nx.edge_dfs(G, global_id, orientation='reverse')
    if len(list(nodes)) == 0:
        x = [global_id]
    else:
        y = list(list(zip(*(nx.edge_dfs(G, global_id, orientation='reverse'))))[0])
        x = y+[global_id] 
    return x

def get_upstream_func(global_id):
    # get upstream reaches
    upstreams = get_upstream(global_id, G)
    return upstreams

def reach2Graph(reach):
    newcol = reach.downstream_line_ids.str.split(',',expand=True)
    colName = ['down%d'%(a+1) for a in range(newcol.shape[1])]
    reach[colName] = newcol
    reach[colName] = np.where(reach[colName]=='', np.nan, reach[colName])
    # create graph
    G = nx.MultiDiGraph()
    for colName0 in colName:
        path1 = reach[['global_id',colName0]].dropna().astype(int)
        path1 = path1.loc[path1[colName0].isin(reach.global_id.values),:]
        G.add_edges_from(list(zip(path1.global_id, path1[colName0])))
    # add node into graph with attribute area
    nodes = reach.global_id.values
    G.add_nodes_from(nodes)
    return G

def create_fishnet(gdf0, nrow = 10, ncol = 10):
    xmin, ymin, xmax, ymax = gdf0.total_bounds
    # Calculate cell size based on the number of rows and columns
    dx = (xmax - xmin) / ncol
    dy = (ymax - ymin) / nrow
    fishnet = []
    for i in range(nrow):
        for j in range(ncol):
            x = xmin + j * dx
            y = ymin + i * dy
            fishnet.append(box(x, y, x + dx, y + dy))
    gdf_fishnet = gpd.GeoDataFrame(pd.DataFrame({'grid_id':np.arange(len(fishnet))}), geometry = fishnet, crs = 'epsg:8857')
    return gdf_fishnet

def step_dissolve_polygon(gdf0, parallel = True):
    '''dissolve for huge datasets, such as 10,000 records'''
    # Create a grid
    gdf_fishnet = create_fishnet(gdf0, nrow = 10, ncol = 10)
    # Assign polygons to grid cells
    gdf0_join = gpd.sjoin(gdf0, gdf_fishnet, how = 'left')
    gdf0_join = gdf0_join.loc[~gdf0_join.grid_id.isna(),:]
    if version.parse(pandas_version) >= version.parse("2.0.0"):
        gdf0_join = gdf0_join.groupby('global_id', group_keys=False).apply(lambda x:x.iloc[0,:], include_groups=False)
    else:
        gdf0_join = gdf0_join.groupby('global_id').apply(lambda x:x.iloc[0,:])
    if parallel:
        # parallel dissolving
        gdf0_join = gdf0_join.groupby('grid_id').p_apply(lambda x:x.dissolve()).reset_index(drop = True)
    else:
        gdf0_join = pd.concat([
            gdf0_join.loc[gdf0_join.grid_id==x,:].dissolve() for x in gdf0_join.grid_id.unique()
        ])
    gdf0_join['global_id'] = np.arange(gdf0_join.shape[0])
    gdf0_join = gdf0_join[['global_id','geometry']]
    gdf0_join = gdf0_join.set_crs('epsg:8857')

    # further dissolve parallel
    print(f'Dissolving {gdf0_join.shape[0]} geometries......')
    gdf_fishnet = create_fishnet(gdf0_join, nrow = 3, ncol = 3)
    gdf0_join = gpd.sjoin(gdf0_join, gdf_fishnet, how = 'left')
    gdf0_join = gdf0_join.loc[~gdf0_join.grid_id.isna(),:]
    if version.parse(pandas_version) >= version.parse("2.0.0"):
        gdf0_join = gdf0_join.groupby('global_id', group_keys=False).apply(lambda x:x.iloc[0,:], include_groups=False)
    else:
        gdf0_join = gdf0_join.groupby('global_id').apply(lambda x:x.iloc[0,:])
    gdf0_join = gdf0_join.groupby('grid_id').p_apply(lambda x:x.dissolve())

    gdf0_join = gdf0_join.dissolve()
    polygon = gdf0_join.geometry.values[0]
    return polygon

def dissolve(ohdb_id, cfg):
    domain = cfg['domain']
    if cfg['use_dis'] == True:
        outName = dir0 / f'data/GRIT/full_catchment/raw/GRITv06_full_catchment_EPSG8857_{domain}_{ohdb_id}_nearest.gpkg'
    else:
        outName = dir0 / f'data/GRIT/full_catchment/raw/GRITv06_full_catchment_EPSG8857_{domain}_{ohdb_id}.gpkg'
    if os.path.exists(outName):
        return

    tmp0 = tmp.loc[tmp.ohdb_id==ohdb_id,:].reset_index()
    if cfg['use_ref'] == True:
        ohdb_darea = tmp0.ohdb_catchment_area_merit.values[0]
    else:
        ohdb_darea = np.nan

    # a gauge is matched to multiple GRIT reaches typically, so their common basins are dissolved at first, 
    upstream_dict = {k:get_upstream_func(k) for k in tmp0.global_id.values}
    # in case some reaches have thousands of upstream reaches, but some reaches have only several upstream reaches
    # , so we also focus on the common upstream reaches for those reaches have over 1000 upstream reaches
    upstream_dict_1000 = {k:v for k,v in upstream_dict.items() if len(v) >= 1000}

    if len(upstream_dict_1000) > 0: # if some reaches have over 1000 upstream reaches
        upstream_c = list(set.intersection(*map(set, list(upstream_dict_1000.values()))))
    else:                           # if not
        upstream_c = list(set.intersection(*map(set, list(upstream_dict.values()))))
    print('Common upstream reaches have ', len(upstream_c))
    catch0_c = None
    catch_id_c = []
    if len(upstream_c) > 0:
        reach0_c = reach.loc[upstream_c]
        minx,miny,maxx,maxy = reach0_c.total_bounds
        catch0_c = catch.loc[(catch.minx<=maxx)&(catch.maxx>=minx)&(catch.miny<=maxy)&(catch.maxy>=miny),:]
        # spatial join
        reach0_c = reach0_c[['geometry']].reset_index().drop(columns=['global_id'])
        catch0_c = gpd.sjoin(catch0_c, reach0_c)
        catch0_c = catch0_c[['geometry']].reset_index().drop_duplicates(subset=['global_id'])
        catch_id_c = catch0_c.global_id.values.tolist()
        # dissolve
        print(f'Dissolving {catch0_c.shape[0]} geometries......')
        if catch0_c.shape[0] > 1000:
            catch0_c = step_dissolve_polygon(catch0_c, parallel = True)
        else:
            catch0_c = catch0_c.dissolve().geometry.values[0]

    # since there could be the same catch-sets matched to different reaches, so we get the unique catch-sets and then dissolve
    catch_ids = []
    segment_ids = []
    for i,global_id in enumerate(tmp0.global_id.values):
        segment_ids.append(reach.loc[global_id].segment_id)
        upstream = upstream_dict[global_id]
        # exclude upstream that already in upstream_c
        upstream = list(set(upstream)-set(upstream_c))
        # initial subset catch to reduce time of spatial join
        reach0 = reach.loc[upstream]
        reach0 = reach0[['geometry']].reset_index().drop(columns=['global_id'])
        minx,miny,maxx,maxy = reach0.total_bounds
        catch0 = catch.loc[(catch.minx<=maxx)&(catch.maxx>=minx)&(catch.miny<=maxy)&(catch.maxy>=miny),:]
        # spatial join
        catch0 = gpd.sjoin(catch0, reach0, how = 'left')
        catch0 = catch0.loc[~catch0.index_right.isna(),:]
        catch0 = catch0[['geometry']].reset_index().drop_duplicates(subset=['global_id'])
        # remove catch that already in catch_id_c
        catch0 = catch0.loc[~catch0.global_id.isin(catch_id_c),:]
        if catch0.shape[0] > 0:
            catch0_ids = set(catch0.global_id.values.tolist())
            catch_ids.append(catch0_ids)
        else:
            catch_ids.append({})

    # get unique catch-sets
    unique_catch_ids = pd.DataFrame({
        'catch_ids':catch_ids,
        'reach_ids':tmp0.global_id.values,
        'segment_ids':segment_ids,
        'distance':tmp0['distance'].values
    })
    unique_catch_ids['need_common_catch'] = False
    unique_catch_ids.loc[unique_catch_ids.reach_ids.isin(upstream_dict_1000.keys()),'need_common_catch'] = True
    unique_catch_ids = unique_catch_ids.drop_duplicates(subset=['catch_ids'])
    unique_catch_ids['catch_ids'] = unique_catch_ids['catch_ids'].apply(lambda x:list(x))

    if cfg['use_dis'] == True:
        idx_min_dis = np.argmin(unique_catch_ids.distance.values)
        unique_catch_ids = unique_catch_ids.iloc[[idx_min_dis],:]

    # loop to dissolve
    grit_darea = []
    geometries = []
    for k,row in unique_catch_ids.iterrows():
        catch0_ids = row.catch_ids
        if len(catch0_ids) > 0:
            catch0 = catch.loc[catch0_ids]
            # dissolve
            print(f'Dissolving {catch0.shape[0]} geometries......')
            if catch0.shape[0] > 1000 and catch0.shape[0] <= 100000:
                catch0 = step_dissolve_polygon(catch0, parallel = True)
            elif catch0.shape[0] > 100000:
                catch0 = step_dissolve_polygon(catch0, parallel = True)
            elif catch0.shape[0] <= 1000 and catch0.shape[0] > 1:
                catch0 = catch0.dissolve().geometry.values[0]
            elif catch0.shape[0] == 1:
                catch0 = catch0.geometry.values[0]
            # union with catch0_c
            if catch0_c is not None and row.need_common_catch is True:
                print('Dissolving catch0 with common catch......')
                catch0 = unary_union([catch0, catch0_c])
        else:
            catch0 = catch0_c
        
        # fill hole
        catch0 = fill_holes(catch0)
        geometries.append(catch0)
        grit_darea.append(catch0.area / 1000000)
        
    if np.isnan(ohdb_darea) or ohdb_darea <= 0:
        bias = 1 / np.array(grit_darea)
    else:
        bias = np.abs(np.array(grit_darea) - ohdb_darea) / ohdb_darea * 100

    idx = np.argmin(bias)
    unique_catch_ids['grit_darea'] = grit_darea
    unique_catch_ids['bias'] = bias
    unique_catch_ids['domain'] = domain
    unique_catch_ids['ohdb_darea'] = ohdb_darea
    out0 = unique_catch_ids.iloc[[idx],:]
    out0 = gpd.GeoDataFrame(data = out0, geometry = np.array([geometries[idx]]), crs = 'epsg:8857')
    out0['ohdb_id'] = ohdb_id
    out0 = out0.rename(columns={
        'reach_ids':'reach_id',
        'segment_ids':'segment_id',
    }).drop(columns=['need_common_catch','catch_ids'])
    write_dataframe(out0, outName)
    print(ohdb_id, 'yes', grit_darea[idx], ohdb_darea)

def get_args() -> Dict:
    """Parse input arguments

    Returns
    -------
    dict
        Dictionary containing the run config.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, required=True, choices=['NA','AF','AS','SA','EU','SI','SP'])
    parser.add_argument('--start', type=int)
    parser.add_argument('--end', type=int)
    parser.add_argument('--ohdb_id', type=str)
    parser.add_argument('--GRITv', type=str, default='GRITv06', choices = ['GRITv06', 'GRITv05'])
    parser.add_argument('--global_id', type=int)
    parser.add_argument('--lon', type=float)
    parser.add_argument('--lat', type=float)
    parser.add_argument('--distance', type=int, default=2000)
    parser.add_argument('--use_ref', 
                        action=argparse.BooleanOptionalAction, 
                        default=False, 
                        help='Whether use MERIT/OHDB catchment area as reference')
    parser.add_argument('--use_dis', 
                        action=argparse.BooleanOptionalAction, 
                        default=False, 
                        help='Whether use only the closest reach')
    cfg = vars(parser.parse_args())
    return cfg

def main(cfg):
    gdf_sta = pd.read_csv(dir0 / 'data/OHDB/OHDB_v0.2.3/OHDB_metadata/OHDB_metadata_correct_coord.csv')

    if cfg['use_ref'] == True:
        gdf_merit = pd.read_csv(
            dir0 / 'data/GRIT/full_catchment/MERIT_catchment_boundary_EPSG8857.csv'
        )
        gdf_sta = gdf_sta.merge(gdf_merit[['ohdb_id','darea']], on = 'ohdb_id', how = 'left')
        gdf_sta['ohdb_catchment_area_merit'] = np.where(
            np.isnan(gdf_sta['ohdb_catchment_area']), 
            gdf_sta['darea'],
            gdf_sta['ohdb_catchment_area']
        )
    else:
        gdf_sta['ohdb_catchment_area_merit'] = np.nan

    domain = cfg['domain']
    GRITv = cfg['GRITv']

    # select gauges within specified domain
    gdf_domain = read_dataframe(dir0 / 'data/GRIT/domains.gpkg', layer='domains')
    gdf_sta = gpd.GeoDataFrame(
        data = gdf_sta,
        geometry = gpd.points_from_xy(gdf_sta.ohdb_longitude, gdf_sta.ohdb_latitude),
        crs = 'epsg:4326'
    ).to_crs('epsg:8857')
    gdf_sta = gpd.sjoin(gdf_sta, gdf_domain.loc[gdf_domain.domain==domain,:])

    # limit to selected gagues
    df_tmp = pd.read_csv('../data/OHDB_metadata_at_least_80_complete_seasonal_records_during_1982_2023.csv')
    gdf_sta = gdf_sta.loc[gdf_sta.ohdb_id.isin(df_tmp.ohdb_id.values),:]
    tmp1 = read_dataframe('../basin_boundary/MIX_good.gpkg', read_geometry = False)
    tmp1 = tmp1.loc[~tmp1.reach_id.isna(),:]
    gdf_sta = gdf_sta.loc[~gdf_sta.ohdb_id.isin(tmp1.ohdb_id.values),:]

    # remove stations that have been already processed
    if cfg['ohdb_id'] is None:
        fnames = dir0.glob(f'data/GRIT/full_catchment/raw/GRITv06_full_catchment_EPSG8857_{domain}_OHDB*nearest.gpkg')
        ohdb_ids = [re.search(r'OHDB_\d+',str(a)).group(0) for a in fnames]
        gdf_sta = gdf_sta.loc[(~gdf_sta.ohdb_id.isin(ohdb_ids)),:]
        if gdf_sta.shape[0] == 0:
            sys.exit('All stations have been processed')
    else:
        gdf_sta = gdf_sta.loc[gdf_sta.ohdb_id==cfg['ohdb_id'],:]
        if cfg['lon'] is not None:
            gdf_sta['ohdb_longitude'] = cfg['lon']
        if cfg['lat'] is not None:
            gdf_sta['ohdb_latitude'] = cfg['lat']
        gdf_sta = gpd.GeoDataFrame(
            data = gdf_sta.drop(columns=['geometry']),
            geometry = gpd.points_from_xy(gdf_sta.ohdb_longitude, gdf_sta.ohdb_latitude),
            crs = 'epsg:4326'
        ).to_crs('epsg:8857')

    # read GRIT reach
    global reach
    fname = dir0 / f'data/GRIT/segments/{GRITv}_reaches_{domain}_EPSG8857.gpkg'
    reach = read_dataframe(fname, layer = 'lines')
    reach['global_id'] = reach['global_id'].astype(np.int64)

    # create networks
    global G
    G = reach2Graph(reach)

    global tmp
    if cfg['global_id'] is not None:
        tmp = pd.DataFrame({
            'global_id': [int(cfg['global_id'])],
            'ohdb_id': [cfg['ohdb_id']],
            'ohdb_catchment_area_merit': [gdf_sta.ohdb_catchment_area_merit.values[0]],
            'ohdb_catchment_area': [ gdf_sta.ohdb_catchment_area.values[0]],
        })
        line = reach.loc[reach.global_id==int(cfg['global_id']),:].geometry.values[0]
        point = gdf_sta.loc[gdf_sta.ohdb_id==cfg['ohdb_id'],:].geometry.values[0]
        tmp['distance'] = line.distance(point)
    else:
        dis = cfg['distance']
        tmp = gpd.sjoin_nearest(
            reach[['global_id','geometry']], 
            gdf_sta[['geometry','ohdb_id','ohdb_catchment_area_merit','ohdb_catchment_area']], 
            distance_col='distance', 
            max_distance = dis)
        if tmp.shape[0] == 0:
            sys.exit(f'No stations are within {dis} meters of GRIT reaches')

    ohdb_ids = tmp.ohdb_id.unique()
    print(f'There are {len(ohdb_ids)} stations snapped')

    np.random.seed(42)  # Set seed
    np.random.shuffle(ohdb_ids)

    start = cfg['start']
    end = cfg['end']
    if start is not None and end is not None:
        ohdb_ids = ohdb_ids[start:end]
        tmp = tmp.loc[tmp.ohdb_id.isin(ohdb_ids),:]
    else:
        pass

    reach = reach.set_index('global_id')
    reach = reach[['geometry','segment_id']]

    # get upstream reach id
    pool = mp.Pool(32)
    upstreams = pool.map(get_upstream_func, tmp.global_id.values)

    # subset reaches
    global_ids = np.hstack(upstreams)
    global_ids = np.unique(global_ids)
    reach = reach.loc[global_ids]
    reach['geometry'] = reach.simplify(10)
    print('Finish subseting reaches', reach.shape)

    # read and subset catchment to reduce the risk of out-of-memory error
    catchName = dir0 / f'data/GRIT/catchment_domain/GRITv06_reach_catchments_{domain}_EPSG8857.gpkg'
    columns = ['global_id']
    bbox = reach.total_bounds.tolist()
    bbox = (bbox[0] - 2000, bbox[1] - 2000, bbox[2] + 2000, bbox[3] + 2000)
    global catch
    catch = read_dataframe(catchName, columns = columns, bbox = bbox)
    print('Successfully read catches')
    
    catch['global_id'] = catch['global_id'].astype(np.int64)
    catch['geometry1'] = catch.geometry.values
    catch['geometry'] = catch.centroid
    tmp1 = gpd.sjoin_nearest(
        catch[['global_id','geometry']],
        reach[['geometry']].reset_index().drop(columns=['global_id']),
        distance_col = 'distance',
        max_distance = 50000
    )
    catch = catch.loc[catch.global_id.isin(tmp1.global_id.unique()),:]
    catch = catch[['global_id','geometry1']].rename(columns={'geometry1':'geometry'})
    catch[['minx','miny','maxx','maxy']] = catch.bounds
    catch = catch.set_index('global_id')
    print('Finish subseting catches', catch.shape)
    del tmp1

    # parallel setting
    ParallelPandas.initialize(n_cpu=12, split_factor=2)

    # get unique OHDB IDs to be processed
    ohdb_ids = tmp.ohdb_id.unique()
            
    number = len(ohdb_ids)
    for i,ohdb_id in enumerate(ohdb_ids):
        print(f'There are {number-i} stations to be processed')
        dissolve(ohdb_id, cfg)

if __name__ == "__main__":
    # get new args
    config = get_args()
    # print config to terminal
    for key, val in config.items():
        print(f"{key}: {val}")
    main(config)