##############################################################################
# This file calculates distance metrics and L-moments vs. spatial aggregation
# for spatial aggregation method 3
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
# add a step to interpolate 'NaN' values arising from different calendars used by different models (easier to do this after other processing steps)
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

##########################################################################################

models = models.reindex_like(reference, method='nearest')
ad_reference = ad_reference.reindex_like(reference, method='nearest')

var_list=['tasmax']
models_list_s = models_list.split(',')
print(var_list, models_list_s)

lat_min, lat_max = reference.lat.min().values, reference.lat.max().values
lon_min, lon_max = reference.lon.min().values, reference.lon.max().values
lat_range = reference.lat.max().values-reference.lat.min().values
lon_range = reference.lon.max().values-reference.lon.min().values
lat_c = reference.lat.min().values + lat_range/2
lon_c = reference.lon.min().values + lon_range/2
r = 30 # range
hr = 15 # half-range
lat_min, lat_max = lat_c-hr, lat_c+hr
lon_min, lon_max = lon_c-hr, lon_c+hr

ref = reference
mod = models
ad_ref = ad_reference

sp_subset_sizes = range(1,r+1)
n_samples = [int(np.ceil(r**2/i**2)) for i in sp_subset_sizes]
print(n_samples)

ref_grouped = {}
mod_grouped = {}
ad_ref_grouped = {}

for i, s in enumerate(sp_subset_sizes):
    reduced_lat_max, reduced_lon_max = lat_max-s, lon_max-s
    ref_groups = [] # create list of subsets for current value of s
    mod_groups = []
    ad_ref_groups = []
    for n in range(n_samples[i]):
        # randomly select bottom left corners of n subsets of size s
        lat_blc, lon_blc = np.random.uniform(lat_min,reduced_lat_max), np.random.uniform(lon_min,reduced_lon_max)         
        rg = ref.sel(lat=slice(lat_blc,lat_blc+s), lon=slice(lon_blc,lon_blc+s))
        ref_groups.append(rg)
        print("Current subset for reference data: ")
        print("Lat x lon: ", rg.dims['lat'], "x", rg.dims['lon'])
        print("Lat range: ", rg.lat.min().item(), rg.lat.max().item())
        print("Lon range: ", rg.lon.min().item(), rg.lon.max().item())
        print("---------------------------")

        mg = mod.sel(lat=slice(lat_blc,lat_blc+s), lon=slice(lon_blc,lon_blc+s))
        mod_groups.append(mg)

        arg = ad_ref.sel(lat=slice(lat_blc,lat_blc+s), lon=slice(lon_blc,lon_blc+s))
        ad_ref_groups.append(arg)

    ref_grouped[s] = ref_groups
    mod_grouped[s] = mod_groups
    ad_ref_grouped[s] = ad_ref_groups

L1, L2, T3, T4 = {}, {}, {}, {}
L1_ref, L2_ref, T3_ref, T4_ref = {}, {}, {}, {}
Lmom_names = ['$\lambda_1$', '$\lambda_2$', '$𝜏_3$', '$𝜏_4$']
Lmoms = [L1, L2, T3, T4]
Lmoms_ref = [L1_ref, L2_ref, T3_ref, T4_ref]

hellinger, wasserstein, intquad = {}, {}, {}
metric_names = ['Hellinger distance', 'Wasserstein distance', 'Integrated quadratic distance']
metrics = [hellinger, wasserstein, intquad]

print("============================================================================")
print("============================================================================")
print("========================== SPATIAL METHOD 3 ================================")
print("============================================================================")
print("============================================================================")

for j, v in enumerate(var_list):
    T_min = min(ref[v].min(), ad_ref[v].min(), min([mod[v, m].min() for m in models_list_s]))
    T_max = max(ref[v].max(), ad_ref[v].max(), max([mod[v, m].max() for m in models_list_s]))
    for i, model in enumerate(models_list_s+["MERRA2"]):
        v_model = (v, model)
        print(v_model)
        
        hellinger_v, wasserstein_v, intquad_v = [], [], []
        L1_v, L2_v, T3_v, T4_v = [], [], [], []
        L1_ref_v, L2_ref_v, T3_ref_v, T4_ref_v = [], [], [], []
        ks_v, ad_v, cvm_v = [], [], []

        for s in sp_subset_sizes:
            print("Size of subsets: lat x lon = {} x {}".format(s,s))
            ref_groups = ref_grouped[s]
            mod_groups = mod_grouped[s]
            ad_ref_groups = ad_ref_grouped[s]

            hellinger_g, wasserstein_g, intquad_g = [], [], []
            L1_g, L2_g, T3_g, T4_g = [], [], [], []
            L1_ref_g, L2_ref_g, T3_ref_g, T4_ref_g = [], [], [], []
            
            for g, group in enumerate(ref_groups):
                A = np.array(group[v]).flatten()
                if model != "MERRA2":
                    B = np.array(mod_groups[g][v_model]).flatten()
                elif model == "MERRA2":
                    B = np.array(ad_ref_groups[g][v]).flatten()
                # Calculate L-moments for each model
                A_L1, A_L2, A_T3, A_T4 = lmoments3.lmom_ratios(A, nmom=4)
                B_L1, B_L2, B_T3, B_T4 = lmoments3.lmom_ratios(B, nmom=4)
                
                dist_A, dist_B = hist_dist([A,B], bins='auto')
                # T_min = min(a.min() for a in [A,B])
                # T_max = max(a.max() for a in [A,B])
                X = np.linspace(T_min, T_max, 1000)
                pdf_A, pdf_B = dist_A.pdf(X), dist_B.pdf(X)

                hellinger_g.append(d_hel(pdf_A, pdf_B))
                wasserstein_g.append(d_was(X,X,pdf_A,pdf_B))
                intquad_g.append(d_iq_sp(A,B))

                L1_g.append(B_L1)
                L2_g.append(B_L2)
                T3_g.append(B_T3)
                T4_g.append(B_T4)
                
                L1_ref_g.append(A_L1)
                L2_ref_g.append(A_L2)
                T3_ref_g.append(A_T3)
                T4_ref_g.append(A_T4)
                
            hellinger_v.append(np.mean(hellinger_g))
            wasserstein_v.append(np.mean(wasserstein_g))
            intquad_v.append(np.mean(intquad_g))

            L1_v.append(np.mean(L1_g))
            L2_v.append(np.mean(L2_g))
            T3_v.append(np.mean(T3_g))
            T4_v.append(np.mean(T4_g))

            if i == 0 and j ==0:
            
                L1_ref_v.append(np.mean(L1_ref_g))
                L2_ref_v.append(np.mean(L2_ref_g))
                T3_ref_v.append(np.mean(T3_ref_g))
                T4_ref_v.append(np.mean(T4_ref_g))

                L1_ref[v]=L1_ref_v
                L2_ref[v]=L2_ref_v
                T3_ref[v]=T3_ref_v
                T4_ref[v]=T4_ref_v
            
        hellinger[v_model] = hellinger_v
        wasserstein[v_model] = wasserstein_v
        intquad[v_model] = intquad_v

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
    filename = 'results/'+region.replace(' ', '')+"_"+df_names[i]+'_spatial_method3.parquet'
    print(filename)
    df.to_parquet(filename)
