from pyogrio import read_dataframe
import geopandas as gpd
import numpy as np
import sys,os
import numpy as np
domain = sys.argv[1]
print(domain)
fname = f'../basin_boundary/GRIT_full_catchment_{domain}_EPSG8857_simplify_final_125km2_subset.gpkg'
gdf_soil = read_dataframe('../../data/geography/GLHYMPS.gpkg')
gdf_soil = gdf_soil.to_crs('epsg:8857')
gdf = read_dataframe(fname)
chunk = 200
if gdf.shape[0] > chunk:
    gdf = [gdf.iloc[a:(a+chunk),:] for a in np.arange(0, gdf.shape[0]+chunk, chunk)[:-1]]
else:
    gdf = [gdf]

cols = ['xxx','Porosity_x','logK_Ice_x','logK_Ferr_']
for s,gdf0 in enumerate(gdf):
    if os.path.exists(f'../geography/GLHYMPS_mean_{domain}_{s}.csv'):
        continue
    out = gpd.overlay(gdf0, gdf_soil)
    out['xxx'] = out.area / 1e6
    out = out.groupby('ohdb_id')[cols].apply(lambda x:x.iloc[:,1:].mul(x.iloc[:,0],axis=0).sum()/x.iloc[:,0].sum())
    out = out.reset_index()
    out.to_csv(f'../geography/GLHYMPS_mean_{domain}_{s}.csv', index = False)
    print(s)
del gdf_soil

gdf_glim = read_dataframe('../../data/geography/soilgrids/GLiM.gpkg')
gdf_glim = gdf_glim.to_crs('epsg:8857')
for s,gdf0 in enumerate(gdf):
    if os.path.exists(f'../geography/GLiM_fraction_{domain}_{s}.csv'):
        continue
    out = gpd.overlay(gdf0, gdf_glim)
    out['xxx'] = out.area / 1e6
    out = out.groupby('ohdb_id').apply(lambda x:x.groupby('xx')['xxx'].sum()/x.xxx.sum())
    out = out.reset_index().pivot_table(index='ohdb_id',columns='xx',values='xxx')
    out = out.fillna(0).reset_index()
    out.to_csv(f'../geography/GLiM_fraction_{domain}_{s}.csv', index = False)
    print(s)