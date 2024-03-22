#############################################
# Regrid data for the reference dataset and 
# selects square spatial regions
#############################################

import os
import numpy as np
import xarray as xr
import xesmf as xe
import regionmask
from dask.diagnostics import ProgressBar

wd = os.getcwd()
region_list = ['ALA','AMZ','AUS','CNA','EAF','MED','NAS','SAS','SEA','SSA','WAF']
start = np.datetime64('1980-01-01')
end = np.datetime64('2014-12-01')
lat_grid = 1.0
lon_grid = 1.0
regrid_alg = 'bilinear'
s = 30

reference_dataset = 'W5E5'
variable_list = ['tasmax']

for v in variable_list:
    print("==========================================================================")
    print("Current variable to be processed:", v)
    variable_id = v
    reference_dir = os.path.join(wd, 'data', reference_dataset, variable_id)
    print('Data to process will be loaded from: ', reference_dir)
    os.chdir(wd)
    os.chdir(reference_dir)
    reference = xr.open_mfdataset('*.nc', engine='netcdf4')
    
    # create target lat_grid x lon_grid degree rectilinear grid
    rg = xr.Dataset(
       {"lat": (["lat"], np.arange(-90, 90, lat_grid)),
        "lon": (["lon"], np.arange(-180, 180, lon_grid)),})
    
    reference_regridder = xe.Regridder(reference, rg, regrid_alg, periodic=True)
    rg_reference = reference_regridder(reference, keep_attrs=True)
    rg_reference_sel_time = rg_reference.sel(time=slice(start,end))

    for i,r in enumerate(regionmask.defined_regions.giorgi):
        if r.abbrev in region_list:
            print("==========================================================================")
            region = r.bounds
            region_name = r.abbrev
            print("Current region:", region_name)
            print("==========================================================================")
            
            # square regions approximately centred on giorgi region centres
            lon_c, lat_c = r.centroid
            lon_c = int(lon_c)
            lat_c = int(lat_c)

            lon_min = lon_c - s/2
            lat_min = lat_c - s/2
            lon_max = lon_min + s - 1
            lat_max = lat_min + s - 1
            
            rg_reference_sel_region = rg_reference_sel_time.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
            reference_processed = rg_reference_sel_region
            os.chdir(wd)
            print("Current directory: %s" % (os.getcwd()))
            reference_save_dir = os.path.join(wd, 'ensembles', region_name.replace(" ","") +"_"+reference_dataset)
            os.makedirs(reference_save_dir, exist_ok=True)
            print("Saving reference files to", reference_save_dir)
            years, y_datasets = zip(*reference_processed.groupby("time.year"))
            fns=[reference_dataset+'_'+variable_id+"_"+"_"+region_name.replace(" ","")+"_"+f"{y}.nc" for y in years]
            paths=[os.path.join(reference_save_dir,fn) for fn in fns]
            with ProgressBar():
                xr.save_mfdataset(y_datasets, paths, mode="w")
