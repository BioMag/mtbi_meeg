"""
This script performs a series of checks on the system to see if everything is
ready to run the analysis pipeline.
"""

import os
import pkg_resources
import mne
from config_common import raw_data_dir, processed_data_dir, figures_dir, reports_dir

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

# Check that the raw data directory is present on the system and raise error if doesnt exist
if not os.path.exists(raw_data_dir):
    raise ValueError(f'The `raw_data_dir` points to a directory that does not exist: {raw_data_dir}')

# Make sure the processed data, figures and reports directories exist
if not os.path.exists(processed_data_dir):
    print(f'Creating directory {processed_data_dir}')
    os.makedirs(processed_data_dir, exist_ok=True)
if not os.path.exists(figures_dir):
    print(f'Creating directory {figures_dir}')
    os.makedirs(figures_dir, exist_ok=True)
if not os.path.exists(reports_dir):
    print(f'Creating directory {reports_dir}')
    os.makedirs(reports_dir, exist_ok=True)

# Prints some information about the system
print('\nNME dependencies installed in the system\n------')
mne.sys_info()
print('-------------')

print('INFO: Success! System requirements are met.')
print('You can run the pipelines by doing.... TBD')
