################################################################
# Run all spatial/temporal aggregation methods for all regions.
################################################################

import subprocess

region_list = ['ALA','AMZ','AUS','CNA','EAF','MED','NAS','SAS','SEA','SSA','WAF']

programme_paths = ['spatial_errors_method1_MERRA2.py',
                   'spatial_errors_method2_MERRA2',
                   'spatial_errors_method3_MERRA2.py',
                   'temporal_errors_method1_MERRA2.py',
                   'temporal_errors_method2_MERRA2.py',
                   'temporal_errors_method3_MERRA2.py']

for p in programme_paths:
    for r in region_list:
        cmd = f"python {p} {r}"
        try:
            subprocess.run(cmd, shell=True, check = True)
        except subprocess.CalledProcessError as e:
            print(f"Error excecuting the programme with argument '{r}': {e}")
