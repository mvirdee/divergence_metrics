from pyesgf.search import SearchConnection
import os
import numpy as np
import requests
import xarray as xr
from tqdm import tqdm
import pandas as pd
from utils import download

os.environ["ESGF_PYCLIENT_NO_FACETS_STAR_WARNING"] = "on"
wd = os.getcwd()

models_list = ['IPSL-CM6A-LR','GFDL-ESM4','MPI-ESM1-2-HR','MRI-ESM2-0','UKESM1-0-LL']
variable_id = 'tasmax' 
table_id = 'day' 
experiment_id='historical'

save_dir = os.path.join(wd, 'data', variable_id, 'r1i1p1f1')
# r1i1p1f2 from UKESM1-0-LL is also saved to this folder
print("Data from search will be saved to: ", save_dir)
print("Number of models in search:", len(models_list))

conn = SearchConnection('https://esgf-node.llnl.gov/esg-search', distrib=True)

for model in models_list:
    if model == 'UKESM1-0-LL':
        member_id='r1i1p1f2'
    else:
        member_id='r1i1p1f1'    
    
    query = conn.new_context(
        latest = True,
        project='CMIP6',
        source_id=model,
        experiment_id=experiment_id,
        variable_id=variable_id,
        table_id=table_id,
        member_id=member_id,
        data_node='aims3.llnl.gov')
    
    print("Number of search results: ", query.hit_count)
    results = query.search()
    
    print("Retrieving search results...")
    files=[]
    for i in range(0, len(results)):
        files.extend(list(map(lambda f : {'filename': f.filename, 'url': f.download_url},
                                   results[i].file_context().search())))
    files = list(files)
    files = pd.DataFrame.from_dict(files)
    files.drop_duplicates('filename')
    
    os.chdir(wd)
    print("Current directory: %s" % (os.getcwd()))
    if os.path.isdir(save_dir):
        print("Saving files to ", save_dir)
    else:
        print("Creating subdirectory", save_dir)
        os.makedirs(save_dir)
        print("Saving files to", save_dir)
    
    os.chdir(save_dir)
    for index, row in files.iterrows():
        if os.path.isfile(row.filename):
            print("The file", row.filename, "already exists. Skipping.")
        else:
            print("Saving file", row.filename, "...")
            download(row.url, row.filename)
    os.chdir(wd)
