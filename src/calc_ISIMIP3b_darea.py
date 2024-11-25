# get catchment area from ISIMP3b input dataset: ddm30 flow dir nc (https://files.isimip.org/ISIMIP3b/InputData/geo_conditions/river_routing/ddm30_flowdir_cru_neva.nc)
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import networkx as nx
import os
from matplotlib.colors import LogNorm

ds = xr.open_dataset('../data/ISIMP3b/ddm30_flowdir_cru_neva.nc')
df = ds.flowdirection.to_dataframe().reset_index().dropna()
ilon = ds.lon.values.tolist()
ilat = ds.lat.values.tolist()
ds.close()
df['ilon'] = df.lon.apply(lambda x: ilon.index(x))
df['ilat'] = df.lat.apply(lambda x: ilat.index(x))
ix = {4:-1, 3:0, 2:1, 1:1, 8:1, 7:0, 6:-1, 5:-1}
iy = {4:-1, 3:-1, 2:-1, 1:0, 8:1, 7:1, 6:1, 5:0}
df['ID'] = np.arange(df.shape[0])
def get_downID(x):
    dir0 = int(x.flowdirection)
    if dir0 <= 0:
        return (-99)
    ix0 = ix[dir0] + x.ilon
    iy0 = iy[dir0] + x.ilat
    idx = (df.ilon==ix0)&(df.ilat==iy0)
    if (idx==False).all():
        return (-99)
    else:
        return (df.loc[idx,'ID'].values[0])
downID = []
for i in range(df.shape[0]):
    downID.append(get_downID(df.iloc[i,:]))
df['downID'] = downID

# create networkx from df
G = nx.MultiDiGraph()
df1 = df.loc[df.downID>=0,:]
G.add_edges_from(list(zip(df1.ID, df1.downID)))
nodes = df.ID.values
G.add_nodes_from(nodes)
def get_upstream(global_id, G):
    nodes = nx.edge_dfs(G, global_id, orientation='reverse')
    if len(list(nodes)) == 0:
        x = [global_id]
    else:
        y = list(list(zip(*(nx.edge_dfs(G, global_id, orientation='reverse'))))[0])
        x = y+[global_id] 
    return x
df['darea'] = df.ID.apply(lambda x: len(get_upstream(x, G)))
df['darea'] = df['darea'] * 55 * 55

ds = df[['lon','lat','darea']].set_index(['lat','lon']).to_xarray()
fig,ax=plt.subplots()
ds.darea.plot(ax=ax, norm = LogNorm(), cbar_kwargs={"label": "Drainagea area ($km^2$)"})
ax.set_title('ISIMIP3b drainage area')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
fig.savefig('../picture/isimip3b_darea.png',dpi=600)

ds.to_netcdf('../data/ISIMP3b/isimip3b_darea.nc')