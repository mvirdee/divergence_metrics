##############################################################################
# This file calculates distributions vs. temporal aggregation
# for temporal aggregation method 1
##############################################################################

import os
import sys
import numpy as np
import xarray as xr
import pandas as pd
from utils import *
from distances import *
import lmoments3
import warnings
warnings.filterwarnings('ignore')

def hist_dist(data:list, bins='auto'):
    '''
    generate distributions from histograms of data
    '''
    dists=[]
    bin_range = (min(a.min() for a in data), max(a.max() for a in data))
    for array in data:
        dist = rv_histogram(np.histogram(array, bins='auto', range=bin_range), density=True)
        dists.append(dist)
    return dists

wd = os.getcwd()
## get region from command-line argument
if len(sys.argv) > 1:
    region = sys.argv[1]
    print("Region: ", region)
else:
    print("No argument provided. Specify a region.")

start = np.datetime64('1980-01-01')
end = np.datetime64('2014-12-01')
reference_dataset = 'W5E5'
ad_reference_dataset = 'MERRA2'
models_list = 'IPSL-CM6A-LR,GFDL-ESM4,MPI-ESM1-2-HR,MRI-ESM2-0,UKESM1-0-LL'
member_id='r1i1p1f1'

ensemble_name="_".join([region.replace(" ",""), member_id])
models_dir = os.path.join(wd, "ensembles", ensemble_name)
reference_dir = os.path.join(wd, 'ensembles', region.replace(" ","") +"_"+reference_dataset)
load_models = load_mf_dataset(models_dir, models_list)
ens = multimodel_ensemble(load_models, start, end)
models_df = ens.drop(['bnds','height']).to_dataframe().reset_index().set_index(['time','lat','lon'])
models_df['realization'] = models_df['realization'].map(dict([i for i in enumerate(models_list.split(','))]))
models_df = models_df.rename(columns={"realization":"model"}).set_index(['model'], append=True).unstack()

models = xr.Dataset.from_dataframe(models_df)
reference = load_reference_dataset(reference_dir, start,end)

ad_reference_dir = os.path.join(wd, 'ensembles', region.replace(" ","") +"_"+ad_reference_dataset)
ad_reference = load_reference_dataset(ad_reference_dir, start,end)

##########################################################################################
# add a step to interpolate 'NaN' values arising from different calendars used by different models
nans = models.isnull().sum()
nans_fraction = nans/models.count()

# check that NaNs arise from calendar misalignments as expected
for v in models:
    nan_mask = models[v].isnull()
    nan_coords = models[v].where(nan_mask, drop=True).reset_coords(drop=True)
    if len(nan_coords) != 0:
        print("Missing days in {}: ".format(v))
        print("Number of 'NaN' values = {}, as a fraction of total = {:.3f}".format(nans[v].item(),nans_fraction[v].item()))
        print("Set of (Month, Day) for missing values:")
        print(list(set([((date.astype('datetime64[M]').astype(int) % 12 + 1),
                       (date - date.astype('datetime64[M]')).astype('timedelta64[D]').astype(int) + 1)
                      for date in nan_coords.time.values])))
        print("============================================================================")

# interpolate any NaNs to the nearest time
models = models.interpolate_na(dim='time', method='nearest')
ad_reference = ad_reference.reindex_like(reference, method='nearest')

##########################################################################################

var_list=['tasmax']
models_list_s = models_list.split(',')
print(var_list, models_list_s)

r = 30
hr = 15

# get center of current region
lat_range = reference.lat.max().values-reference.lat.min().values
lon_range = reference.lon.max().values-reference.lon.min().values
lat_c = reference.lat.min().values + lat_range/2
lon_c = reference.lon.min().values + lon_range/2

ref = reference
mod = models
ad_ref = ad_reference

rvs_A_dict, rvs_B_dict = {}, {}
rv_names = ['A','B']
rvs = [rvs_A_dict, rvs_B_dict]

lat, lon = lat_c, lon_c

for j, v in enumerate(var_list):
    T_min = min(ref[v].min(), ad_ref[v].min(), min([mod[v, m].min() for m in models_list_s]))
    T_max = max(ref[v].max(), ad_ref[v].max(), max([mod[v, m].max() for m in models_list_s]))
    for i, model in enumerate(models_list_s+["MERRA2"]):
        v_model = (v, model) # tuple for model indexing
        print(v_model)

        rvs_A_v, rvs_B_v = [], []

        t_lim = 500
        t_step=14
        
        for t in range(1,t_lim,t_step):
            if (t-1) % (10*t_step)==0:
                print(t-1)

            A = np.array(ref[v].isel(time=slice(0,t))).flatten()
            if model != "MERRA2":
                B = np.array(mod[v_model].isel(time=slice(0,t))).flatten()
            elif model == "MERRA2":
                B = np.array(ad_ref[v].isel(time=slice(0,t))).flatten()
        
            dist_A, dist_B = hist_dist([A,B], bins='auto')
            # T_min = min(a.min() for a in [A,B])
            # T_max = max(a.max() for a in [A,B])
            X = np.linspace(T_min, T_max, 1000)
            rvs_A = dist_A.pdf(X)
            rvs_B = dist_B.pdf(X)

            rvs_A_v.append((X, rvs_A))
            rvs_B_v.append((X, rvs_B))

        rvs_A_dict[v_model] = rvs_A_v
        rvs_B_dict[v_model] = rvs_B_v

distributions = {d_n:d for d_n,d in zip(rv_names, rvs)}
distributions_df = pd.DataFrame(distributions)

os.chdir(wd)
if not os.path.exists('results'):
    os.makedirs('results')
df_names  = ['distributions']
for i, df in enumerate([distributions_df]):
    filename = 'results/'+region.replace(' ','')+'_'+df_names[i]+'_temporal_method1.parquet'
    df.to_parquet(filename)



