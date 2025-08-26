import rasterio
import numpy as np
import geopandas as gpd
from rasterio.windows import Window
from rasterio.features import shapes
from rasterio.transform import rowcol,xy
from shapely.geometry import shape
import sys,os,glob
import pandas as pd
import whitebox
from pathlib import Path
from pyproj import Transformer
import multiprocessing
from functools import reduce
from shapely.geometry import Polygon, MultiPolygon

dir0 = Path('/data/ouce-drift/cenv1021')

'''
    select the grid cell with the maximum drainagea area within a distance of 60 meters
'''

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

# circle window 60 m -> 60 / 30 = 2
radius_grid = 2
radius =  60   # for gauges with reported catchment area
radius_no = 60 # for gauges without reported catchment area

# input file names
fdr_name = sys.argv[1]
upa_name = sys.argv[2]
domain = sys.argv[3]

ohdb_name = dir0 / 'data/OHDB/OHDB_v0.2.3/OHDB_metadata/OHDB_metadata_correct_coord.csv'

# clip gauges within domain
df = pd.read_csv(ohdb_name)
df = gpd.GeoDataFrame(data = df, geometry = gpd.points_from_xy(df.ohdb_longitude, df.ohdb_latitude), crs = 'epsg:4326')
df = df.to_crs('epsg:8857')
gdf0 = gpd.read_file(os.path.join(os.path.dirname(fdr_name), domain+'_dissolve.gpkg'))
df = gpd.sjoin(df, gdf0).reset_index(drop=True)
df = df.sort_values('ohdb_catchment_area',ascending=False).reset_index(drop=True)
df["x"] = df.geometry.x
df["y"] = df.geometry.y

# limit to selected gagues
df_tmp = pd.read_csv('../data/OHDB_metadata_at_least_80_complete_seasonal_records_during_1982_2023.csv')
df = df.loc[df.ohdb_id.isin(df_tmp.ohdb_id.values),:].reset_index(drop=True)

# exclude gauges that are processed already
try:
    previous_fnames = glob.glob(f'../data/GRITv06_watershed_{domain}_60m_radius_part*gpkg') + [
        f'../data/GRITv06_watershed_{domain}_60m_radius.gpkg']
    previous_fnames = [ss for ss in previous_fnames if os.path.exists(str(ss))]
    tmp = pd.concat([gpd.read_file(s) for s in previous_fnames])
    df = df.loc[~df.ohdb_id.isin(tmp.ohdb_id.values),:].reset_index(drop=True)
    previous_num = len(previous_fnames)
    if df.shape[0] == 0:
        print(f'Gauges in domain {domain} were finished')
        sys.exit(0)
except:
    previous_num = 0
    pass

num = df.shape[0]
print(f'{num} gauges are to be processed')

# get the rows and cols with the closest drainagea area to reported catchment area 
with rasterio.open(upa_name) as src:
    rows = np.ones(df.shape[0]) * np.nan
    cols = np.ones(df.shape[0]) * np.nan
    lons = np.ones(df.shape[0]) * np.nan
    lats = np.ones(df.shape[0]) * np.nan
    upas = np.ones(df.shape[0]) * np.nan
    for index, row in df.iterrows():
        lon = row['x']
        lat = row['y']
        darea = row['ohdb_catchment_area']

        # Convert (lon, lat) to (row, col)
        row, col = rowcol(src.transform, lon, lat)

        # Define window boundaries, clipped to image bounds
        row_start = max(row - radius_grid, 0)
        row_stop = min(row + radius_grid + 1, src.height)
        col_start = max(col - radius_grid, 0)
        col_stop = min(col + radius_grid + 1, src.width)

        # rectangular mask
        window = Window(col_start, row_start, col_stop - col_start, row_stop - row_start)
        data = src.read(1, window=window)

        # then circle mask
        row_off, col_off = np.meshgrid(np.arange(-radius_grid,radius_grid+1), np.arange(-radius_grid,radius_grid+1), indexing="ij")
        rows0 = row + row_off
        cols0 = col + col_off
        lons0, lats0 = xy(src.transform, rows0, cols0, offset='center')
        lons0 = np.array(lons0).ravel()
        lats0 = np.array(lats0).ravel()
        # ── Transform to EPSG:8857 (Equal Earth) ───────────────
        transformer = Transformer.from_crs(src.crs, "EPSG:8857", always_xy=True)
        x_proj, y_proj = transformer.transform(lons0, lats0)
        x_proj = x_proj.reshape(rows0.shape)
        y_proj = y_proj.reshape(rows0.shape)
        x_center = x_proj[radius_grid,radius_grid]
        y_center = y_proj[radius_grid,radius_grid]
        # calculate distance to circle center and limit to radius
        dis = np.sqrt(((x_proj-x_center)**2 + (y_proj-y_center)**2))
        circle_mask = dis <= radius
        data = np.where(circle_mask, data, np.nan)
        dis = np.where(circle_mask, dis, np.nan)

        if darea > 0:
            # if all pixels with area differences of more than 20%, then exclude gauges
            bias = np.abs(data - darea) / darea * 100
            if (bias[~np.isnan(bias)] >= 20).all():
                circle_mask = dis <= radius_no
                data = np.where(circle_mask, data, np.nan)
                
                # identify the grid with the maximum darea
                min_pos = np.unravel_index(np.nanargmax(data), data.shape)
                closest_value = data[min_pos]
            else:
                # identify the best grid with the minimum darea bias
                min_pos = np.unravel_index(np.nanargmin(bias), bias.shape)
                closest_value = data[min_pos]
        else:
            circle_mask = dis <= radius_no
            data = np.where(circle_mask, data, np.nan)
            
            # identify the grid with the maximum darea
            min_pos = np.unravel_index(np.nanargmax(data), data.shape)
            closest_value = data[min_pos]

        if closest_value == 0:
            continue

        # Convert window-relative row/col back to global row/col
        global_row = row_start + min_pos[0]
        global_col = col_start + min_pos[1]

        # Convert (col, row) to (x, y)
        lon, lat = src.transform * (global_col + 0.5, global_row + 0.5)  # center of the pixel

        lons[index] = lon
        lats[index] = lat

        rows[index] = global_row
        cols[index] = global_col

        upas[index] = closest_value

    df['rows_snapping'] = rows
    df['cols_snapping'] = cols
    df['lons_snapping'] = lons
    df['lats_snapping'] = lats
    df['upa_km2'] = upas

df = df.dropna(subset=['rows_snapping','cols_snapping']).reset_index(drop=True)
df['DN'] = np.arange(1, df.shape[0] + 1)

# if no plausible snapping is obtained, then stop
if df.shape[0] == 0:
    print('No plausible snapping is obtained')
    sys.exit(0)

# get the watershed boundary using whitebox
fdr_name1 = Path(fdr_name).resolve()
pour_pts = Path(f'../data/pour_points_{domain}.shp').resolve()
out_wshed = Path(f"../data/watershed_{domain}.tif").resolve()

wbt = whitebox.WhiteboxTools()

df_raw = df.copy()

def create_pour_pts_vector(df):
    df = df.reset_index(drop=True)
    df['DN'] = np.arange(1, df.shape[0] + 1)
    # save pour points vector
    df = gpd.GeoDataFrame(data = df, geometry = gpd.points_from_xy(df.lons_snapping, df.lats_snapping), crs = 'epsg:4326')
    df[['DN','geometry']].to_file(pour_pts)
    return df

def watershed(wbt, fdr_name, pour_pts, df):
    wbt.watershed(fdr_name, pour_pts, out_wshed)
    if not os.path.exists(out_wshed):
        sys.exit(0)
    # watershed TIFF to watershed Shapefile
    with rasterio.open(out_wshed) as src:
        image = src.read(1)  # first band
        mask = image != src.nodata
        transform = src.transform
        crs = src.crs
    results = (
        {'geometry': shape(geom), 'DN': value}
        for geom, value in shapes(image, mask=mask, transform=transform)
    )
    results = list(results)
    gdf = gpd.GeoDataFrame.from_records(results).set_geometry('geometry')
    gdf.crs = crs
    # dissolve by DN
    gdf = gdf.dissolve(by='DN').reset_index()
    # fill holes
    gdf['geometry'] = gdf.geometry.apply(fill_holes)
    # connect with pour_points to add file ohdb_id
    gdf = gdf.merge(df[['DN', 'ohdb_id', 'lons_snapping', 'lats_snapping', 'upa_km2']], on = 'DN').drop(columns=['DN'])
    gdf = gdf.to_crs('epsg:8857')
    
    gdf['darea'] = gdf.area / 1000000
    gdf['bias'] = np.abs(gdf.darea - gdf.upa_km2) / gdf.upa_km2 * 100
    gdf = gdf.sort_values('bias', ascending=True)
    return gdf

# loop to address nested basins
output = []
remaining_num = df_raw.shape[0]
index = previous_num + 1
no_change_time = 0
while True:
    df = create_pour_pts_vector(df)
    gdf0 = watershed(wbt, fdr_name1, pour_pts, df)
    gdf0 = gdf0.loc[gdf0.bias<=1,:]
    gdf0.to_file(f'../data/GRITv06_watershed_{domain}_60m_radius_part{index}.gpkg')
    index = index + 1
    output.append(gdf0)

    df1 = df_raw.loc[~df_raw.ohdb_id.isin(gdf0.ohdb_id.values),:].reset_index(drop=True)
    print(f'{df1.shape[0]} gauges are to be processed')
    df_raw = df1.copy()
    if df1.shape[0] == 0:
        break

    df1['DN'] = np.arange(1, df1.shape[0] + 1)
    if df1.shape[0] > 0:
        if df1.shape[0] < remaining_num:
            remaining_num = df1.shape[0]
            df = df1.copy()
        elif df1.shape[0] == remaining_num:
            df = df1.iloc[[np.random.choice(df1.shape[0])],:]
            print(df1)

            # sometimes catchmnt area does not equal to upa, so quit it
            no_change_time = no_change_time + 1
            if no_change_time == 3:
                break

# delete intermediate files
for fname in glob.glob(f'../data/pour_points_{domain}.*'):
    os.remove(fname)
os.remove(f'../data/watershed_{domain}.tif')