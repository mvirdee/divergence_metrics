##############################################################################
# This file calculates distance metrics and L-moments vs. temporal aggregation
# for temporal aggregation method 2
##############################################################################

import os
import sys
import numpy as np
import xarray as xr
import pandas as pd
import math
import lmoments3
from utils import *
from distances import *
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

print(len(models['time']), len(reference['time']), len(ad_reference['time']))

##########################################################################################
# add a step to interpolate 'NaN' values arising from different calendars used by different models
nans = models.isnull().sum()
nans_fraction = nans/models.count()

# check that NaNs are calendar misalignments as expected
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

# get center of current region
lat_range = reference.lat.max().values-reference.lat.min().values
lon_range = reference.lon.max().values-reference.lon.min().values
lat_c = reference.lat.min().values + lat_range/2
lon_c = reference.lon.min().values + lon_range/2

ref = reference
mod = models

ad_ref = ad_reference

t_min = 0
t_max = len(reference.time)
t_subsets = [1,5,10,15,20,25,50,100,150,200,250,300,350,400,500]

ref_grouped = {}
mod_grouped = {}
ad_ref_grouped = {}

for t in t_subsets:

    print("---------------------------")
    print("The period will be divided into ", math.ceil(t_max/t), "time subsets.")
    time_string = str(int(t))+'D'
    ref_grouped[t] = ref.resample(time=time_string)
    mod_grouped[t] = mod.resample(time=time_string)

    ad_ref_grouped[t] = ad_ref.resample(time=time_string)

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
print("========================== TEMPORAL METHOD 2 ===============================")
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

        for t in t_subsets:
            print("Number of time subsets: ", math.ceil(t_max/t))
            ref_groups = [g[1] for g in ref_grouped[t]]
            mod_groups = [g[1] for g in mod_grouped[t]]
            
            ad_ref_groups = [g[1] for g in ad_ref_grouped[t]]

            print("len ref groups: ", len(ref_groups))
            print("len mod groups: ", len(mod_groups))
            print("len ad ref groups: ", len(ad_ref_groups))
            hellinger_g, wasserstein_g, intquad_g = [], [], []
            L1_g, L2_g, T3_g, T4_g = [], [], [], []
            L1_ref_g, L2_ref_g, T3_ref_g, T4_ref_g = [], [], [], []
            
            for g, group in enumerate(ref_groups):
                if g % 100 == 0:
                    print(g)
                A = np.array(group[v]).flatten()

                if model != "MERRA2":
                    B = np.array(mod_groups[g][v_model]).flatten()
                if model == "MERRA2":
                    B = np.array(ad_ref_groups[g][v]).flatten()

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
    filename = 'results/'+region.replace(' ', '')+"_"+df_names[i]+'_temporal_method2.parquet'
    print(filename)
    df.to_parquet(filename)