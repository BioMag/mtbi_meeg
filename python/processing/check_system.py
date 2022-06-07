"""
This script performs a series of checks on the system to see if everything is
ready to run the analysis pipeline.
"""

import os
import pkg_resources

# Check to see if the python dependencies are fullfilled.
dependencies = []
with open('./requirements.txt') as f:
    for line in f:
        line = line.strip()
        if len(line) == 0 or line.startswith('#'):
            continue
        dependencies.append(line)

# This raises errors of dependencies are not met
pkg_resources.working_set.require(dependencies)

# Check that the data is present on the system
from config_common import raw_data_dir
if not os.path.exists(raw_data_dir):
    raise ValueError(f'The `raw_data_dir` points to a directory that does not exist: {raw_data_dir}')

# Make sure the output directories exist
from config_common import processed_data_dir
os.makedirs(processed_data_dir, exist_ok=True)

# TODO: still use meg files? maybe for better pipeline?
#from meg.config_meg import fname 
#os.makedirs(fname.figures_dir, exist_ok=True)
#os.makedirs(fname.reports_dir, exist_ok=True)

from eeg.config_eeg import fname
os.makedirs(fname.figures_dir, exist_ok=True)
os.makedirs(fname.reports_dir, exist_ok=True)

# Prints some information about the system
import mne
mne.sys_info()

with open('system_check.txt', 'w') as f:
    f.write('System check OK.')

print("""
All seems to be in order.
You can now run the pipelines with:
    cd eeg
    python -m doit
and
    cd meg
    python -m doit
""")

