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
if host == 'nbe-024' and user == 'vanvlm1':
    # Marijn's workstation
    raw_data_dir = '/m/nbe/scratch/brrr_fingerprinting/biomagtbi_bids/bids'
    processed_data_dir = '/m/nbe/scratch/brrr_fingerprinting/biomagtbi_bids/bids/derivatives/biomag-tbi'
    reports_dir = '/m/home/home4/45/vanvlm1/data/projects/biomag-tbi/reports'
    figures_dir = '/m/home/home4/45/vanvlm1/data/projects/biomag-tbi/figures'
    n_jobs = 4  # My workstation has 4 cores
    matplotlib_backend = 'Qt5Agg'
elif user == 'wmvan':
    # Marijn's laptop
    raw_data_dir = 'S:/nbe/brrr_fingerprinting/biomagtbi_bids/bids'
    processed_data_dir = 'S:/nbe/brrr_fingerprinting/biomagtbi_bids/bids/derivatives/biomag-tbi'
    reports_dir = 'C:/Users/wmvan/projects/biomag-tbi/reports'
    figures_dir = 'C:/Users/wmvan/projects/biomag-tbi/figures'
    n_jobs = 6  # My laptop has 6 cores
    matplotlib_backend = 'Qt5Agg'
elif host.startswith('vdiubuntu') and user == 'vanvlm1':
    # Marijn's VDI machine
    raw_data_dir = '/m/nbe/scratch/brrr_fingerprinting/biomagtbi_bids/bids'
    processed_data_dir = '/m/nbe/scratch/brrr_fingerprinting/biomagtbi_bids/bids/derivatives/biomag-tbi'
    reports_dir = '/u/45/vanvlm1/unix/projects/biomag-tbi/reports'
    figures_dir = '/u/45/vanvlm1/unix/projects/biomag-tbi/figures'
    n_jobs = 2  # VDI gives you 2 cores
    matplotlib_backend = 'Qt5Agg'
elif host.endswith('triton.aalto.fi'):
    # Triton cluster
    raw_data_dir = '/m/nbe/scratch/brrr_fingerprinting/biomagtbi_bids/bids'
    processed_data_dir = '/m/nbe/scratch/brrr_fingerprinting/biomagtbi_bids/bids/derivatives/biomag-tbi'
    reports_dir = '/home/vanvlm1/data/biomag-tbi/reports'
    figures_dir = '/home/vanvlm1/data/biomag-tbi/figures'
    n_jobs = 1
    matplotlib_backend = 'Agg'  # No graphics on triton
elif host == 'nbe-065' and user == 'hkoivikk':
    # Hanna's workstation
    raw_data_dir = '/m/nbe/scratch/brrr_fingerprinting/biomagtbi_bids/bids'
    processed_data_dir = '/m/nbe/scratch/brrr_fingerprinting/biomagtbi_bids/bids/derivatives/biomag-tbi'
    reports_dir = '/m/home/home0/09/hkoivikk/unix/biomag-tbi/reports'
    figures_dir = '/m/home/home0/09/hkoivikk/unix/biomag-tbi/figures'
    n_jobs = 4
    matplotlib_backend = 'Qt5Agg'
elif host == 'sirius' and user == 'heikkiv' :
    # Verna's workstation in BioMag
    raw_data_dir = '/net/theta/fishpool/projects/FABEEG/childEEG_data/bids/'
    processed_data_dir = '/net/theta/fishpool/projects/FABEEG/childEEG_data/bids/derivatives/'
    reports_dir = os.path.join('/net/theta/fishpool/projects/FABEEG/childEEG_data/reports',user)
    figures_dir = os.path.join('/net/theta/fishpool/projects/FABEEG/childEEG_data/figures',user)
    n_jobs = 4
    matplotlib_backend = 'Qt5Agg'
elif host == 'psi' and user == 'kuusia1' :
    # Verna's workstation in BioMag
    raw_data_dir = '/net/theta/fishpool/projects/FABEEG/childEEG_data/bids/'
    processed_data_dir = '/net/theta/fishpool/projects/FABEEG/childEEG_data/bids/derivatives/'
    reports_dir = os.path.join('/net/theta/fishpool/projects/FABEEG/childEEG_data/reports',user)
    figures_dir = os.path.join('/net/theta/fishpool/projects/FABEEG/childEEG_data/figures',user)
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
all_subjects = list(range(1, 45))

# Tasks performed in the scanner
tasks = ['emptyroom', 'eyesopen', 'eyesclosed', 'pasat']
