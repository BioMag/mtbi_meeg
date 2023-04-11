"""
This script performs a series of checks on the system to see if everything is
ready to run the analysis pipeline.
"""

import os
import pkg_resources
import mne

# Check to see if the python dependencies are fullfilled.
dependencies = []
with open('../requirements.txt') as f:
    for line in f:
        line = line.strip()
        if len(line) == 0 or line.startswith('#'):
            continue
        dependencies.append(line)

# This raises errors of dependencies are not met
try:
    pkg_resources.working_set.require(dependencies)
except pkg_resources.VersionConflict as e:
    # Get the conflicting distribution and requirement
    dist = e.dist
    req = e.req

    # Create a custom error message
    error_message = f"\nVersion conflict:Library {dist} ({dist.location}) does not meet the requirements: {req}"

    # Raise a new exception with the custom error message
    raise ValueError(error_message) from e

# Check that the data is present on the system
from config_common import raw_data_dir
if not os.path.exists(raw_data_dir):
    raise ValueError(f'The `raw_data_dir` points to a directory that does not exist: {raw_data_dir}')

# Make sure the processed data directories exist
from config_common import processed_data_dir
os.makedirs(processed_data_dir, exist_ok=True)

# TODO: still use meg files? maybe for better pipeline?
#from meg.config_meg import fname 
#os.makedirs(fname.figures_dir, exist_ok=True)
#os.makedirs(fname.reports_dir, exist_ok=True)

from config_eeg import fname
os.makedirs(fname.figures_dir, exist_ok=True)
os.makedirs(fname.reports_dir, exist_ok=True)

# Prints some information about the system
print('\nNME dependencies installed in the system\n------')
mne.sys_info()
print('-------------')

# I don't know what is the goal of this
#with open('system_check.txt', 'w') as f:
#    f.write('System check OK.')

print('INFO: Success! System requirements are met.')
print('You can run the pipelines by doing TBD')


