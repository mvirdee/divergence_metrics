## Code to generate results in 'Spatial and temporal limits to the simulation of temperature extremes in the CMIP6 ensemble'

### Data

- After cloning this repository, set up the conda enviroment by running `conda env create -f environment.yml`
- Code to download CMIP6 data is provided in 'download_cmip_data.py' - alternatively, download from https://esgf-index1.ceda.ac.uk/search/cmip6-ceda/. These files should be saved in 'data/<variable_name>/<member_id>'.
- W5E5 historical daily maximum temperature data can be downloaded from: https://data.isimip.org/datasets/38d4a8f4-12e8-44ff-afe3-0c7ce0e0dad6/. These files should be saved in 'data/W5E5/<variable_name>'.
- MERRA2 historical daily data can be downloaded from: https://disc.gsfc.nasa.gov/datasets/M2SDNXSLV_5.12.4/summary?keywords=T2MMAX. These files should be saved in 'data/MERRA2/<variable_name>'.

### Steps

1. Run 'process_data.py' to regrid and process data and extract geographic regions for analysis
2. Run 'calculate_errors.py' to get results for divergence metrics and moments calculated according to the different spatial and temporal aggregation methods
3. Plot results in the notebook 'plots.ipynb'
