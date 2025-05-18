from pyogrio import read_dataframe,write_dataframe
from shapely.geometry import Polygon, MultiPolygon
import geopandas as gpd
import os,glob,sys,time,re
import networkx as nx
import pandas as pd
from shapely.geometry import box
import numpy as np
import multiprocessing as mp
from parallel_pandas import ParallelPandas
from tqdm.auto import tqdm
from functools import reduce
from shapely.ops import unary_union

domain = sys.argv[1]
print('processing ', str(sys.argv))

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

def get_upstream_func(global_id):
    return get_upstream(global_id, G)

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

def step_dissolve_polygon(gdf0):
    '''dissolve for huge datasets, such as 10,000 records'''
    # Create a grid
    gdf_fishnet = create_fishnet(gdf0)
    # Assign polygons to grid cells
    gdf0_join = gpd.sjoin(gdf0, gdf_fishnet, how = 'left')
    gdf0_join = gdf0_join.loc[~gdf0_join.grid_id.isna(),:]
    gdf0_join = gdf0_join.groupby('global_id').apply(lambda x:x.iloc[0,:])
    # parallel dissolving
    gdf0_join = gdf0_join.groupby('grid_id').p_apply(lambda x:x.dissolve())
    gdf0_join['global_id'] = np.arange(gdf0_join.shape[0])
    gdf0_join = gdf0_join[['global_id','geometry']]
    gdf0_join = gdf0_join.reset_index().drop(columns=['grid_id'])
    gdf0_join = gdf0_join.set_crs('epsg:8857')

    # further dissolve parallel
    gdf_fishnet = create_fishnet(gdf0_join, nrow = 3, ncol = 3)
    gdf0_join = gpd.sjoin(gdf0_join, gdf_fishnet, how = 'left')
    gdf0_join = gdf0_join.loc[~gdf0_join.grid_id.isna(),:]
    gdf0_join = gdf0_join.groupby('global_id').apply(lambda x:x.iloc[0,:])
    gdf0_join = gdf0_join.groupby('grid_id').p_apply(lambda x:x.dissolve())

    gdf0_join = gdf0_join.dissolve()
    polygon = gdf0_join.geometry.values[0]
    return polygon

def dissolve(ohdb_id):
    if os.path.exists(f'/data/ouce-drift/cenv1021/data/GRIT/full_catchment/GRITv06_full_catchment_EPSG8857_{domain}_{ohdb_id}.gpkg'):
        print(ohdb_id, 'already has')
        return
    tmp0 = tmp.loc[tmp.ohdb_id==ohdb_id,:].reset_index()
    ohdb_darea = tmp0.ohdb_catchment_area_hydrosheds.values[0]
    # ohdb_darea_fill = tmp0.ohdb_catchment_area_hydrosheds.values[0]

    # common basin dissolve
    upstream_dict = {k:get_upstream_func(k) for k in tmp0.global_id.values}
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
            catch0_c = step_dissolve_polygon(catch0_c)
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
    unique_catch_ids = pd.DataFrame({'catch_ids':catch_ids,'reach_ids':tmp0.global_id.values,'segment_ids':segment_ids})
    unique_catch_ids = unique_catch_ids.drop_duplicates(subset=['catch_ids'])
    unique_catch_ids['catch_ids'] = unique_catch_ids['catch_ids'].apply(lambda x:list(x))

    # loop to dissolve
    grit_darea = []
    geometries = []
    for k,row in unique_catch_ids.iterrows():
        catch0_ids = row.catch_ids
        if len(catch0_ids) > 0:
            catch0 = catch.loc[catch0_ids]
            # dissolve
            print(f'Dissolving {catch0.shape[0]} geometries......')
            if catch0.shape[0] > 1000:
                catch0 = step_dissolve_polygon(catch0)
            else:
                catch0 = catch0.dissolve().geometry.values[0]
            # union with catch0_c
            if catch0_c is not None:
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
    write_dataframe(out0, f'../data/GRIT/full_catchment/GRITv06_full_catchment_EPSG8857_{domain}_{ohdb_id}.gpkg')
    print(ohdb_id, 'yes', grit_darea[idx], ohdb_darea)

# get global_id of all reaches
if __name__ == '__main__':
    gdf_sta = read_dataframe('/data/ouce-drift/cenv1021/data/OHDB/OHDB_v0.2.3/OHDB_metadata_fill_hydrosheds_fcc.gpkg')

    # remove stations that have been already processed
    fnames = glob.glob(f'/data/ouce-drift/cenv1021/data/GRIT/full_catchment/GRITv06_full_catchment_EPSG8857_{domain}_OHDB*.gpkg')
    ohdb_ids = [re.search(r'OHDB_\d+',a).group(0) for a in fnames]
    gdf_sta = gdf_sta.loc[(~gdf_sta.ohdb_id.isin(ohdb_ids))&(gdf_sta.domain==domain),:]
    if gdf_sta.shape[0] == 0:
        sys.exit('All stations have been processed')

    # read GRIT reach
    fname = f'/data/ouce-evoflood/results/global_river_topology/output_v06/reach_attributes/GRITv06_reaches_{domain}_EPSG8857.gpkg'
    reach = read_dataframe(fname, layer = 'lines')
    reach['global_id'] = reach['global_id'].astype(np.int64)
    
    # create networks
    G = reach2Graph(reach)
    tmp = gpd.sjoin_nearest(
        reach[['global_id','geometry']], 
        gdf_sta[['geometry','ohdb_id','ohdb_catchment_area_hydrosheds','ohdb_catchment_area']], 
        distance_col='distance', 
        max_distance = 2000)
    if tmp.shape[0] == 0:
        sys.exit('No stations are within 2000 meters of GRIT reaches')

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
    catch = read_dataframe(f'/data/ouce-evoflood/results/global_river_topology/output_v06/catchments/GRITv06_reach_catchments_{domain}_EPSG8857.gpkg')
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

    # get unique OHDB IDs to be processed
    ohdb_ids = tmp.ohdb_id.unique()
    
    try:
        # split the task
        start = int(sys.argv[2])
        end = int(sys.argv[3])
        ohdb_ids = ohdb_ids[start:end]
    except:
        print('do')
        
    # number = len(ohdb_ids)
    # for i,ohdb_id in enumerate(ohdb_ids):
    #     print(f'There are {number-i} stations to be processed')
    #     dissolve(ohdb_id)
    ohdb_id = 'OHDB_014022469'
    dissolve(ohdb_id)