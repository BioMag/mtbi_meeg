"""
===========
Config file
===========

Configuration parameters for the study.
"""

import os
from getpass import getuser
from socket import gethostname

###############################################################################
# Determine which user is running the scripts on which machine and set the path
# where the data is stored and how many CPU cores to use.

user = getuser()  # Username of the user running the scripts
host = gethostname()  # Hostname of the machine running the scripts

# You want to add your machine to this list
# TODO: change them to match current users?
if host == 'nbe-077' and user == 'heikkiv7':
    # Verna's workstation in Aalto
    raw_data_dir = '/m/nbe/scratch/tbi-meg/verna/BIDS'
    processed_data_dir = '/m/nbe/scratch/tbi-meg/verna/k22_processed'
    reports_dir = '/m/nbe/scratch/tbi-meg/verna/reports'
    figures_dir = '/m/nbe/scratch/tbi-meg/verna/k22_processedfigures'
    n_jobs = 4
    matplotlib_backend = 'Qt5Agg'
elif host.endswith('triton.aalto.fi'):
    # Triton cluster
    raw_data_dir = '/m/nbe/scratch/brrr_fingerprinting/biomagtbi_bids/bids'
    processed_data_dir = '/m/nbe/scratch/brrr_fingerprinting/biomagtbi_bids/bids/derivatives/biomag-tbi'
    reports_dir = '/home/vanvlm1/data/biomag-tbi/reports'
    figures_dir = '/home/vanvlm1/data/biomag-tbi/figures'
    n_jobs = 1
    matplotlib_backend = 'Agg'  # No graphics on triton
elif host == 'sirius' and user == 'heikkiv' : 
    # Verna's workstation in BioMag
    raw_data_dir = '/net/theta/fishpool/projects/tbi_meg/BIDS/'
    processed_data_dir = '/net/theta/fishpool/projects/tbi_meg/k22_processed/'
    reports_dir = os.path.join('/net/tera2/home/heikkiv/work_s2022/mtbi-eeg/python/reports/',user)
    figures_dir = os.path.join('/net/tera2/home/heikkiv/work_s2022/mtbi-eeg/python/figures/',user)
    n_jobs = 4
    matplotlib_backend = 'Qt5Agg'
elif host == 'ypsilon.biomag.hus.fi' and user == 'heikkiv':
    raw_data_dir = '/net/theta/fishpool/projects/tbi_meg/BIDS/'
    processed_data_dir = '/net/theta/fishpool/projects/tbi_meg/k22_processed/'
    reports_dir = os.path.join('/net/tera2/home/heikkiv/work_s2022/mtbi-eeg/python/reports/',user)
    figures_dir = os.path.join('/net/tera2/home/heikkiv/work_s2022/mtbi-eeg/python/reports/',user)
    n_jobs = 4
    matplotlib_backend = 'Qt5Agg'
elif host == 'psi' and user == 'aino' :
    # Ainos's workstation in BioMag
    raw_data_dir = '/net/theta/fishpool/projects/tbi_meg/BIDS/'
    processed_data_dir = '/net/theta/fishpool/projects/tbi_meg/k22_processed/'
    reports_dir = os.path.join('/net/tera2/home/aino/work/mtbi-eeg/python/reports/', user)
    figures_dir = os.path.join('/net/tera2/home/aino/work/mtbi-eeg/python/figures/',user)
    n_jobs = 4
    matplotlib_backend = 'Qt5Agg'
elif host == 'rho' and user == 'portae1' :
    # Estanislao's workstation in BioMag
    raw_data_dir = '/net/theta/fishpool/projects/tbi_meg/BIDS/'
    processed_data_dir = '/net/theta/fishpool/projects/tbi_meg/k22_processed/'
    reports_dir = os.path.join('/net/tera2/home/portae1/biomag/mtbi-eeg/python/reports/', user)
    figures_dir = os.path.join('/net/tera2/home/portae1/biomag/mtbi-eeg/python/figures/',user)
    n_jobs = 4
    matplotlib_backend = 'Qt5Agg'    
else:
    raise ValueError(f'Please enter the details of your system ({user}@{host}) in config_common.py')

# For BLAS to use the right amount of cores
os.environ['OMP_NUM_THREADS'] = str(n_jobs)

# Configure the graphics backend
import matplotlib
matplotlib.use(matplotlib_backend)


###############################################################################
# These are all the relevant parameters that are common to both the EEG and MEG
# analysis pipelines.
###############################################################################

# All subjects for which there is some form of data available
all_subjects = os.listdir(raw_data_dir)
for s in all_subjects:
    s_path = os.path.join(raw_data_dir, s)
    if not os.path.isdir(s_path):
        all_subjects.remove(s) #TODO: check this! 
all_subjects.remove('participants.tsv')  
all_subjects = [x.replace('sub-', '') for x in all_subjects]
# Tasks performed in the scanner
tasks = ['ec', 'eo', 'PASAT']
