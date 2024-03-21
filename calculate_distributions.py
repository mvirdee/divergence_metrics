############################################################################
# Get distributions vs. aggregation 
############################################################################

import subprocess

region_list = ['ALA','AMZ','AUS','CNA','EAF','MED','NAS','SAS','SEA','SSA','WAF']
	
programme_paths = ['spatial_distributions_method1.py',
                  'temporal_distributions_method1.py']
for p in programme_paths:
    for r in region_list:
        cmd = f"python {p} {r}"
        try:
            subprocess.run(cmd, shell=True, check = True)
        except subprocess.CalledProcessError as e:
            print(f"Error excecuting the programme with argument '{r}': {e}")
