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
from parallel_pandas import ParallelPandas
from tqdm.auto import tqdm
ParallelPandas.initialize(n_cpu=48, split_factor=4)

domain = 'EU'

def get_upstream(global_id, G):
    nodes = nx.edge_dfs(G, global_id, orientation='reverse')
    if len(list(nodes)) == 0:
        x = [global_id]
    else:
        y = list(list(zip(*(nx.edge_dfs(G, global_id, orientation='reverse'))))[0])
        x = y+[global_id] 
    return x

def get_downstream(global_id, G):
    nodes = nx.edge_dfs(G, global_id)
    if len(list(nodes)) == 0:
        x = [global_id]
    else:
        y = list(list(zip(*(nx.edge_dfs(G, global_id))))[0])
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

def calc(x):
    # x is a row in gdf_basin
    reach1 = reach.loc[reach.intersects(x),:].reset_index(drop=True)
    length = reach1.length.sum() / 1000
    G = reach2Graph(reach1)
    reach2 = reach1.loc[reach1.grwl_overlap>=0.5,:]
    for global_id in reach2.global_id.values:
        downstreams = get_downstream(global_id, G)
        reach1.loc[reach1.global_id.isin(downstreams),'grwl_overlap'] = 1
    reach1 = reach1.loc[reach1.grwl_overlap>=0.5,:]
    length_0p5 = reach1.length.sum() / 1000
    return [length, length_0p5]
    
# get global_id of all reaches
if __name__ == '__main__':
    # remove stations that have been already processed
    gdf_basin = read_dataframe(f'../basin_boundary/GRIT_full_catchment_{domain}_EPSG8857_simplify_final_125km2_subset.gpkg')

    fname = f'../../data/GRIT/segments/GRITv06_reaches_{domain}_EPSG8857.gpkg'
    reach = read_dataframe(fname, layer = 'lines')
    reach['global_id'] = reach['global_id'].astype(np.int64)
    tmp = read_dataframe(fname[:-5]+'_subset.gpkg', read_geometry = False)
    reach = reach.loc[reach.global_id.isin(tmp.global_id.values),:]

    gdf_basin['tmp'] = gdf_basin.geometry.p_apply(calc)
    gdf_basin['riv_len'] = gdf_basin.tmp.apply(lambda x:x[0])
    gdf_basin['riv_len_0p5'] = gdf_basin.tmp.apply(lambda x:x[1])
    gdf_basin['riv_len_0p5_per_darea'] = gdf_basin['riv_len_0p5'] / gdf_basin['gritDarea']

    gdf_basin[['ohdb_id','riv_len','riv_len_0p5','riv_len_0p5_per_darea']].to_csv(f'../geography/GRIT_riv_length_{domain}.csv', index = False)