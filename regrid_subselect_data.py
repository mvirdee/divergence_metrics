##################################################
# Regrid data for the list of climate models and 
# selects square spatial regions
##################################################

import os
import numpy as np
import xarray as xr
import xesmf as xe
import regionmask
from dask.diagnostics import ProgressBar
from utils import feb_29_30

wd = os.getcwd()
region_list = ['ALA','AMZ','AUS','CNA','EAF','MED','NAS','SAS','SEA','SSA','WAF']
start = np.datetime64('1980-01-01')
end = np.datetime64('2014-12-01')
lat_grid = 1.0
lon_grid = 1.0
regrid_alg = 'bilinear'
s = 30 # size of square regions

models_list = 'IPSL-CM6A-LR,GFDL-ESM4,MPI-ESM1-2-HR,MRI-ESM2-0,UKESM1-0-LL'
variable_list = ['tasmax']

for v in variable_list:
    print("==========================================================================")
    print("Current variable to be processed:", v)
    variable_id = v
    experiment_id='historical' 
    
    # dataset splitting may be different across models - group files belonging to each dataset
    model_files = {}
    for model in models_list.split(","):
        os.chdir(wd)
        print(model)
        if model == 'UKESM1-0-LL':
            member_id='r1i1p1f2'
            print(member_id)
            save_dir = os.path.join(wd, 'data', variable_id, member_id)
            print('First available ensemble member for {} is {}'.format(model, member_id))
            print('Data process will be loaded from: ', save_dir)
            model_filenames=[]
            for filename in os.listdir(save_dir):
                if model in filename:
                    model_filenames.append(save_dir+"/"+filename)
            model_files[model] = model_filenames
            
        elif model != 'UKESM1-0-LL':
            member_id='r1i1p1f1'
            save_dir = os.path.join(wd, 'data', variable_id, member_id)
            print('Data to process will be loaded from: ', save_dir)
            model_filenames=[]
            for filename in os.listdir(save_dir):
                if model in filename:
                    model_filenames.append(save_dir+"/"+filename)
            model_files[model] = model_filenames
            
    data = {}
    print("Number of files per model: ")
    for model, files in model_files.items():
        print(model, len(files))
        data[model] = xr.open_mfdataset(files, engine='netcdf4', chunks={'time': 1000})
    os.chdir(wd)

    print("==========================================================================")
    
    # create target lat_grid x lon_grid degree rectilinear grid
    rg = xr.Dataset(
       {"lat": (["lat"], np.arange(-90, 90, lat_grid)),
        "lon": (["lon"], np.arange(-180, 180, lon_grid)),})
    
    # set up regridder for each model
    regridders = {}
    for i, (model, dataset) in enumerate(data.items()):
        regridder = xe.Regridder(dataset, rg, regrid_alg, periodic=True) # periodic longitudes
        print(model, regridder, '\n')
        regridders[model] = regridder
        
    # create regridded datasets
    data_rg = {}
    for i, (model, dataset) in enumerate(data.items()):
        rg_model = regridders[model](dataset, keep_attrs=True)
        data_rg[model] = rg_model
    
    # sub-select region after regridding
    region_data_rg={}
    for i,r in enumerate(regionmask.defined_regions.giorgi):
        if r.abbrev in region_list:
            print("==========================================================================")
            region = r.bounds
            region_name = r.abbrev
            print("Current region:", region_name)
            print("==========================================================================")
            
            # new square regions approximately centred on giorgi region centres
            lon_c, lat_c = r.centroid
            lon_c = int(lon_c)
            lat_c = int(lat_c)

            lon_min = lon_c - s/2
            lat_min = lat_c - s/2
            lon_max = lon_min + s - 1
            lat_max = lat_min + s - 1
        
            for model, dataset in data_rg.items():
                # sub-select region and time, normalize time index
                select_region = dataset.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
                
                if model == 'UKESM1-0-LL': # uses a cftime 360-day calendar - drop feb 29 and 30 to enable conversion to datetimeindex
                    drop_times = select_region.sel(time=feb_29_30(select_region['time.month'], select_region['time.day'])).time
                    select_region_drop_times = select_region.drop_sel(time=drop_times)
                    select_region = select_region_drop_times
                
                if select_region.indexes['time'].dtype!='datetime64[ns]':
                    datetimeindex = select_region.indexes['time'].to_datetimeindex()
                    select_region['time'] = datetimeindex
                    
                select_region['time'] = select_region.indexes['time'].normalize()
                select_time = select_region.sel(time=slice(start,end), drop=True)
                print(select_time.dims)
                region_data_rg[model] = select_time
        
            # save sub-selected regional regridded datasets for each model 
            os.chdir(wd)
            member_id = 'r1i1p1f1' # note r1p1i1f2 for UKESM1-0-LL is saved to this folder
            ensemble_name="_".join([region_name.replace(" ",""), member_id])
            ensemble_path=os.path.join(wd, "ensembles", ensemble_name)
            
            print("Current directory: %s" % (os.getcwd()))
            if os.path.isdir(ensemble_path):
                print("Saving model files to", ensemble_path)
            else:
                print("Creating subdirectory", ensemble_path)
                os.makedirs(ensemble_path)
                print("Saving model files to",  ensemble_path)
            
            for model, dataset in region_data_rg.items():
                print("Saving ", model)
                identifier = '_'.join([variable_id,
                                    dataset.table_id,
                                    dataset.source_id,
                                    dataset.experiment_id,
                                    dataset.variant_label,
                                    region_name.replace(' ','')
                                      ])
                years, y_datasets = zip(*dataset.groupby("time.year"))
                fns=['ens_'+identifier+f'_{y}.nc' for y in years]
                paths=[os.path.join(ensemble_path,fn) for fn in fns]
                with ProgressBar():
                    xr.save_mfdataset(y_datasets, paths, mode="w")
