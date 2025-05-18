import numpy as np
import xarray as xr
import pandas as pd
import os,sys,glob

def func(Ta, P, S_snow0, fdd = 2, Tthr = 1):
    # fdd: melt rate set at 2.0 (mm/d/K) 
    # Tthr: temperature threshold of snowfall
    tmp = fdd * (Ta - Tthr)
    try:
        tmp0 = xr.where(tmp > S_snow0, S_snow0, tmp)
    except:
        time_offset = pd.Timedelta(days=1)
        S_snow0['time'] = S_snow0['time'] + time_offset
        tmp0 = xr.where(tmp > S_snow0, S_snow0, tmp)
    Snowmelt1 = xr.where(Ta < Tthr, 0, tmp0)

    tmp1 = S_snow0 - Snowmelt1
    S_snow1 = xr.where(Ta < Tthr, P + S_snow0, xr.where(tmp1 > 0, tmp1, 0))
    Snowfall1 = xr.where(Ta < Tthr, P, 0)
    
    return S_snow1, Snowmelt1, Snowfall1

def CalSnowmelt(year, save = False):
    # reference: Berghuijs et al. (2016) and Stein et al. (2019)
    
    # read snow at previous date
    year0 = year - 1
    if year > 1981:
        try:
            ds_snow = xr.open_dataset('../../data/MSWX/Snow/snow_%d_last_day.nc'%year0)
            S_snow0 = ds_snow.s_snow
        except:
            raise Exception('../../data/MSWX/Snow/snow_%d_last_day.nc'%year0, 'not found!')
    else:
        S_snow0 = 0
    
    S_snow = []
    Snowfall = []
    Snowmelt = []
    for date in pd.date_range('%d-1-1'%year, '%d-12-31'%year, freq = 'D'):
        year = date.year
        doy = (date - pd.to_datetime('%d-1-1'%year)).days + 1
        d = '%d%03d'%(year, doy)
        ds_P = xr.open_dataset('../../data/MSWX/P/Daily/%s.nc'%d)
        ds_Tmax = xr.open_dataset('../../data/MSWX/Tmax/Daily/%s.nc'%d)
        ds_Tmin = xr.open_dataset('../../data/MSWX/Tmin/Daily/%s.nc'%d)
        
        P = ds_P.precipitation
        Ta = (ds_Tmax.air_temperature + ds_Tmin.air_temperature) / 2
        
        S_snow1, Snowmelt1, Snowfall1 = func(Ta, P, S_snow0, fdd = 2, Tthr = 1)
        S_snow0 = S_snow1
        
        ds_P.close()
        ds_Tmax.close()
        ds_Tmin.close()
        
        if save:
            S_snow.append(S_snow1)
            Snowfall.append(Snowfall1)
            Snowmelt.append(Snowmelt1)

        print(date)

    out = xr.Dataset({'s_snow':S_snow0})
    if not os.path.exists('../../data/MSWX/Snow/snow_%d_last_day.nc'%year):
        out.to_netcdf('../../data/MSWX/Snow/snow_%d_last_day.nc'%year)

    if save:
        out = xr.concat(S_snow, dim = 'time')
        out = out.where((out.lat>=-52)&(out.lat<=75)&(out.lon>=-155),drop=True)
        out.to_netcdf('../../data/MSWX/Snow/snow_%d.nc'%year)

        out = xr.concat(Snowfall, dim = 'time')
        out.where((out.lat>=-52)&(out.lat<=75)&(out.lon>=-155),drop=True).to_netcdf('../../data/MSWX/Snow/snowfall_%d.nc'%year)

        out = xr.concat(Snowmelt, dim = 'time')
        out.where((out.lat>=-52)&(out.lat<=75)&(out.lon>=-155),drop=True).to_netcdf('../../data/MSWX/Snow/snowmelt_%d.nc'%year)

year = int(sys.argv[1])
CalSnowmelt(year, save = True)