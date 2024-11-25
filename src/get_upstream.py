# use predefined global_id mask netcdf to clip netcdf
import os,sys,re
import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
from pyogrio import read_dataframe
from joblib import Parallel, delayed
from tqdm import tqdm

domain = sys.argv[1]
cluster_id = int(sys.argv[2])
csvName = sys.argv[3]
method = sys.argv[4]

catchName = '../data/GRIT/catchment_domain/GRITv05_segment_catchments_%s_EPSG8857_correct_cluster_%d.gpkg'%(domain, cluster_id)
reachName = '../data/GRIT/segments/GRITv05_segments_%s_EPSG8857_correct.gpkg'%domain
connectName = '../data/GRIT/segments/GRITv05_segments_%s_EPSG8857_connect_segment_catch.csv'%domain

def get_upstream(global_id, G):
    nodes = nx.edge_dfs(G, global_id, orientation='reverse')
    if len(list(nodes)) == 0:
        x = [global_id]
    else:
        y = list(list(zip(*(nx.edge_dfs(G, global_id, orientation='reverse'))))[0])
        x = y+[global_id] 
    return x

def reach2Graph(reach, catch, connect):
    newcol = reach.downstream_line_ids.str.split(',',expand=True)
    colName = ['down%d'%(a+1) for a in range(newcol.shape[1])]
    reach[colName] = newcol
    reach[colName] = np.where(reach[colName]=='', np.nan, reach[colName])
    reach = reach.loc[reach['darea']>0,:]
    catch = catch.merge(connect, left_on = 'global_id', right_on = 'global_id_catch')
    reach = reach.merge(catch[['global_id_reach','area']], left_on = 'global_id', right_on = 'global_id_reach')
    # create graph
    G = nx.MultiDiGraph()
    for colName0 in colName:
        path1 = reach[['global_id',colName0]].dropna().astype(int)
        path1 = path1.loc[path1[colName0].isin(reach.global_id.values),:]
        G.add_edges_from(list(zip(path1.global_id, path1[colName0])))
    # add node into graph with attribute area
    nodes = reach.global_id.values
    areas = reach.apply(lambda x: {'area':x.area, 'darea':x.darea}, axis = 1)
    nodes = tuple(zip(nodes, areas))
    G.add_nodes_from(nodes)
    return G

def func_mean(G, connect, pr):
    nodes = list(nx.topological_sort(G))
    # loop catch
    areas = nx.get_node_attributes(G, 'area')
    num = len(nodes)
    connect = connect.set_index('global_id_reach')
    dict_darea = {}
    for node1 in tqdm(nodes, total = len(nodes)):   
        # get upstream nodes of node
        nodes0 = [s for s in G.predecessors(node1)]
        if len(nodes0) == 0:
            dict_darea[node1] = areas[node1]
            continue
        catch_ids0 = connect.loc[nodes0].global_id_catch.values
        cols0 = [catch_ids.index(a) for a in catch_ids0]
        w0 = [dict_darea[a] for a in nodes0]
        # in case there are multiple downstream reaches, the upstream drainage area is equally divided
        down_num = [len([a for a in G.neighbors(b)]) for b in nodes0]
        w0 = [a if down_num[i]==1 else a/down_num[i] for i,a in enumerate(w0)]
        
        catch_ids1 = connect.loc[node1].global_id_catch
        cols1 = catch_ids.index(catch_ids1)
        w1 = np.array([areas[node1]])

        pr0 = pr[:,cols0]
        pr1 = pr[:,[cols1]]
        prs = np.hstack([pr0, pr1])
        ws = np.hstack([w0, w1])
        dict_darea[node1] = np.sum(ws)
        ws = ws / np.sum(ws)
        pr[:,cols1] = np.nansum(prs * ws, axis = 1)
    return(pr)

def func_max_sum(G, connect, pr, method = 'sum'):
    nodes = list(nx.topological_sort(G))
    # loop catch
    num = len(nodes)
    connect = connect.set_index('global_id_reach')
    for node1 in tqdm(nodes, total = len(nodes)):   
        # get upstream nodes of node
        nodes0 = [s for s in G.predecessors(node1)]
        if len(nodes0) == 0:
            continue
        catch_ids0 = connect.loc[nodes0].global_id_catch.values
        cols0 = [catch_ids.index(a) for a in catch_ids0]
        catch_ids1 = connect.loc[node1].global_id_catch
        cols1 = catch_ids.index(catch_ids1)

        pr0 = pr[:,cols0]
        pr1 = pr[:,[cols1]]
        prs = np.hstack([pr0, pr1])
        if method == 'sum':
            pr[:,cols1] = np.nansum(prs, axis = 1)
        elif method == 'max':
            pr[:,cols1] = np.nanmax(prs, axis = 1)
        else:
            raise Exception('method must be sum or max')
    return(pr)


'''varName should be a dataframe (csv or parquet) with number of variables (row) X catch ID (column)'''
df = pd.read_parquet(csvName)

try:
    time = pd.to_datetime(df.time.values)
except:
    time = df.index.values

df.columns = [str(a) for a in df.columns]
catch_ids = [int(float(s)) for s in df.columns.tolist() if bool(re.search('\d+', s))]

if 'time' in df.columns:
    df = df.drop(['time'], axis = 1)

pr = df.values

# create nodes
try:
    reach = read_dataframe(reachName, read_geometry = False)
    reach['darea'] = reach[['drainage_area_out', 'drainage_area_mainstem_out']].max(axis=1)
    catch = read_dataframe(catchName, read_geometry = False)
    connect = pd.read_csv(connectName, index_col = [0])
except:
    raise Exception('reach, catch, connect is not provided')
G = reach2Graph(reach, catch, connect)
nodes = list(nx.topological_sort(G))

if method == 'mean':
    pr = func_mean(G, connect, pr)
elif method == 'max':
    pr = func_max_sum(G, connect, pr, method = 'max')
elif method == 'sum':
    pr = func_max_sum(G, connect, pr, method = 'sum')
else:
    raise Excpetion('method must be mean, max, or sum')

df_pr = pd.DataFrame(data=pr, columns = catch_ids)

# only keep those darea greater than thres
catch_ids1 = connect.loc[connect.global_id_reach.isin(nodes),'global_id_catch'].values
df_pr = df_pr[catch_ids1]
# change columns from catch id to reach id
f = lambda x: connect.loc[connect.global_id_catch==x].global_id_reach.values[0]
newCols = [str(int(f(a))) for a in df_pr.columns.values]
df_pr.columns = newCols

df_pr['time'] = time
df_pr = df_pr.set_index('time')

outName = '.'.join(csvName.split('.')[:-1]) + '_catch-%s.parquet'%method
df_pr.to_parquet(outName)