import os
import numpy as np
import xarray as xr
import pandas as pd
import requests
from tqdm import tqdm
from xclim.ensembles import create_ensemble
from scipy.stats import rv_histogram

def download(url, filename):
    '''
    from pandas dataframe of filenames and urls, download files
    '''
    print("Downloading ", filename)
    r = requests.get(url, stream=True)
    total_size, block_size = int(r.headers.get('content-length', 0)), 1024
    with open(filename, 'wb') as f:
        for data in tqdm(r.iter_content(block_size),
                         total=total_size//block_size,
                         unit='KiB', unit_scale=True):
            f.write(data)

    if total_size != 0 and os.path.getsize(filename) != total_size:
        print("Downloaded size does not match expected size!\n",
              "FYI, the status code was ", r.status_code)


def feb_29_30(month, day):
    f29 = (month==2) & (day==29)
    f30 = (month==2) & (day==30)
    return f29+f30

def load_mf_dataset(ensemble_path, models:list):
    '''
    given path to multi-model ensemble dataset and list of models,
    load xr files into a dict of mf_datasets for each model
    '''
    model_files = {}
    for model in models.split(","):
        model_filenames=[]
        for filename in os.listdir(ensemble_path):
            if model in filename:
                model_filenames.append(filename)
        model_files[model] = model_filenames

    os.chdir(ensemble_path)
    data = {}
    for model, files in model_files.items():
        print(model, len(files))
        data[model] = xr.open_mfdataset(files, engine='netcdf4', chunks={'time': 1000})
    return data

def load_reference_dataset(reference_path, start, end):
    '''
    load xr multi-file dataset for reference data, selecting start and end dates,
    and normalise time index
    '''
    os.chdir(reference_path)
    reference = xr.open_mfdataset('*.nc', engine='netcdf4')

    reference['time'] = reference.indexes['time'].normalize()
    reference_sl = reference.sel(time=slice(start,end))
    reference_sl_nl = reference_sl.sel(time=~((reference_sl.time.dt.month == 2) & (reference_sl.time.dt.day == 29))) # no leap
    ref = reference_sl_nl
    return ref

def multimodel_ensemble(data, start, end):
    ''' given a dict of models i.e. model_name[data]=dataset,
    create an xclim ensemble, selecting start and end dates, normalizing time
    '''
    ensemble = create_ensemble([model for model in data.values()]).load()
    ensemble.close()
    ensemble['time'] = ensemble.indexes['time'].normalize()
    ensemble_sl = ensemble.sel(time=slice(start,end))
    ens = ensemble_sl
    return ens

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
