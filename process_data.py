#################################################################
# Process model and reference datasets
# regrid and extract geographic regions
#################################################################

import subprocess

programme_paths = ['regrid_subselect_data.py',
                   'regrid_subselect_W5E5.py',
                   'regrid_subselect_MERRA2.py']

for p in programme_paths:
    cmd = f"python {p}"
    try:
        subprocess.run(cmd, shell=True, check = True)
    except subprocess.CalledProcessError as e:
        print(f"Error excecuting the programme with argument '{p}': {e}")