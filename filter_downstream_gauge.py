from pyogrio import read_dataframe
import numpy as np
import geopandas as gpd
import multiprocessing as mp

def func(i):
    gdf0 = gdf.iloc[[i],:].reset_index(drop=True)
    gdf1 = gdf.drop(gdf.index[i])
    sp = gpd.sjoin(gdf0,gdf1)
    sp = sp.loc[sp.gritDarea_right<=gdf0.gritDarea.values[0],:].reset_index(drop=True)
    if sp.shape[0] == 0:
        return
    try:
        sp['SParea'] = sp.ohdb_id_right.apply(lambda x: gdf0.intersection(gdf.loc[gdf.ohdb_id==x,:].reset_index(drop=True)).area.values[0]/1000000)
    except:
        return
    sp['ratio'] = sp['SParea'] / sp['gritDarea_right']
    print(i)
    if np.round(sp['ratio'].max()) == 1:
        return gdf0.ohdb_id.values[0]
    else:
        return
if __name__ == '__main__':
    gdf = read_dataframe('../basin_boundary/GRIT_full_catchment_all_EPSG8857_simplify_final_125km2_subset.gpkg')
    while True:
        pool = mp.Pool(8)
        out = pool.map(func, np.arange(gdf.shape[0]))
        out = [a for a in out if a != None]
        if len(out) == 0:
            break
        gdf = gdf.loc[~gdf.ohdb_id.isin(out),:].reset_index(drop=True)