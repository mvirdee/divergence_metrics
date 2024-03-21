##############################################################################
# This file calculates distance metrics and L-moments vs. spatial aggregation
# for spatial aggregation method 1
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

wd = os.getcwd()
## get region from command-line argument
if len(sys.argv) > 1:
    region = sys.argv[1]
    print("Region: ", region)
else:
    print("No argument provided. Specify a region and threshold.")

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
# check how many nans per model
nans = models.isnull().sum()
nans_fraction = nans/models.count()

# check that these are calendar misalignments as expected
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

##########################################################################################

# list models, variables and indices to loop through
# var_list = [i for i in reference.data_vars]
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

ref = reference # for when using other reference datasets
mod = models
ad_ref = ad_reference

# initialise model and reference L-moment list/dicts
L1, L2, T3, T4 = {}, {}, {}, {}
L1_ref, L2_ref, T3_ref, T4_ref = {}, {}, {}, {}
Lmom_names = ['$\lambda_1$', '$\lambda_2$', '$𝜏_3$', '$𝜏_4$']
Lmoms = [L1, L2, T3, T4]
Lmoms_ref = [L1_ref, L2_ref, T3_ref, T4_ref]

# initialise divergence metric dicts
hellinger, wasserstein, intquad = {}, {}, {}
metric_names = ['Hellinger distance', 'Wasserstein distance', 'Integrated quadratic distance']
metrics = [hellinger, wasserstein, intquad]

lat, lon = lat_c, lon_c

print("============================================================================")
print("============================================================================")
print("========================== SPATIAL METHOD 1 ================================")
print("============================================================================")
print("============================================================================")

for j, v in enumerate(var_list):
    T_min = min(ref[v].min(), ad_ref[v].min(), min([mod[v, m].min() for m in models_list_s]))
    T_max = max(ref[v].max(), ad_ref[v].max(), max([mod[v, m].max() for m in models_list_s]))
    for i, model in enumerate(models_list_s+["MERRA2"]):
        v_model = (v, model) # tuple for model indexing
        print(v_model)
        
        hellinger_v, wasserstein_v, intquad_v = [], [], []
        L1_v, L2_v, T3_v, T4_v = [], [], [], []
        L1_ref_v, L2_ref_v, T3_ref_v, T4_ref_v = [], [], [], []
        
        for s in range(1,hr+1):
            print("Calculating error for spatial aggregation: ", s*2)
            A = np.array(ref[v].sel(lat=slice(lat-s,lat+s), lon=slice(lon-s, lon+s))).flatten()
            if model != "MERRA2":
                B = np.array(mod[v_model].sel(lat=slice(lat-s,lat+s), lon=slice(lon-s, lon+s))).flatten()
            elif model == "MERRA2":
                B = np.array(ad_ref[v].sel(lat=slice(lat-s,lat+s), lon=slice(lon-s, lon+s))).flatten()

            # Calculate L-moments for each model
            A_L1, A_L2, A_T3, A_T4 = lmoments3.lmom_ratios(A, nmom=4)
            B_L1, B_L2, B_T3, B_T4 = lmoments3.lmom_ratios(B, nmom=4)

            print(A_L1, B_L1)
            
            if i==0:
                L1_ref_v.append(A_L1)
                L2_ref_v.append(A_L2)
                T3_ref_v.append(A_T3)
                T4_ref_v.append(A_T4)

                L1_ref[v]=L1_ref_v
                L2_ref[v]=L2_ref_v
                T3_ref[v]=T3_ref_v
                T4_ref[v]=T4_ref_v
        
            dist_A, dist_B = hist_dist([A,B], bins='auto')
            # T_min = min(a.min() for a in [A,B])
            # T_max = max(a.max() for a in [A,B])
            X = np.linspace(T_min, T_max, 1000)
            pdf_A, pdf_B = dist_A.pdf(X), dist_B.pdf(X)

            hellinger_v.append(d_hel(pdf_A,pdf_B))
            wasserstein_v.append(d_was(X,X,pdf_A,pdf_B))
            intquad_v.append(d_iq_sp(A,B))

            L1_v.append(B_L1)
            L2_v.append(B_L2)
            T3_v.append(B_T3)
            T4_v.append(B_T4)

        hellinger[v_model]=hellinger_v
        wasserstein[v_model]=wasserstein_v
        intquad[v_model]=intquad_v

        L1[v_model]=L1_v
        L2[v_model]=L2_v
        T3[v_model]=T3_v
        T4[v_model]=T4_v


results = {m_n:m for m_n, m in zip(metric_names,metrics)}
results_df = pd.DataFrame(results)

results_Lmoms = {m_n:m for m_n, m in zip(Lmom_names,Lmoms)}
results_Lmoms_df = pd.DataFrame(results_Lmoms)

results_Lmoms_ref = {m_n:m for m_n, m in zip(Lmom_names,Lmoms_ref)}
results_Lmoms_ref_df = pd.DataFrame(results_Lmoms_ref)

os.chdir(wd)
df_names = ['divergences', 'Lmoms', 'Lmoms_ref']
for i, df in enumerate([results_df, results_Lmoms_df, results_Lmoms_ref_df]):
    filename = 'results/'+region.replace(' ', '')+"_"+df_names[i]+'_spatial_method1.parquet'
    df.to_parquet(filename)


