"""Portable runtime configuration for the mTBI EEG/MEG pipeline.

Site-specific defaults are retained for existing BioMag/Aalto installations.
Every setting can be overridden without editing source code by using the
``MTBI_*`` environment variables documented in the README.
"""

import os
from getpass import getuser
from pathlib import Path
from socket import gethostname

user = getuser()
host = gethostname()

# Existing site profiles. Environment variables below always take precedence.
_profile = None
if host == "nbe-077" and user == "heikkiv7":
    _profile = {
        "raw": "/m/nbe/scratch/tbi-meg/verna/BIDS",
        "processed": "/m/nbe/scratch/tbi-meg/verna/k22_processed",
        "reports": "/m/nbe/scratch/tbi-meg/verna/reports",
        "figures": "/m/nbe/scratch/tbi-meg/verna/k22_processedfigures",
        "jobs": 4,
        "backend": "Qt5Agg",
    }
elif host.endswith("triton.aalto.fi"):
    _profile = {
        "raw": "/m/nbe/scratch/brrr_fingerprinting/biomagtbi_bids/bids",
        "processed": "/m/nbe/scratch/brrr_fingerprinting/biomagtbi_bids/bids/derivatives/biomag-tbi",
        "reports": "/home/vanvlm1/data/biomag-tbi/reports",
        "figures": "/home/vanvlm1/data/biomag-tbi/figures",
        "jobs": 1,
        "backend": "Agg",
    }
elif host in {"sirius", "ypsilon.biomag.hus.fi"} and user == "heikkiv":
    _profile = {
        "raw": "/net/theta/fishpool/projects/tbi_meg/BIDS",
        "processed": "/net/theta/fishpool/projects/tbi_meg/k22_processed",
        "reports": f"/net/tera2/home/heikkiv/work_s2022/mtbi_meeg/output/reports/{user}",
        "figures": f"/net/tera2/home/heikkiv/work_s2022/mtbi_meeg/output/figures/{user}",
        "jobs": 4,
        "backend": "Qt5Agg",
    }
elif host == "psi" and user == "aino":
    _profile = {
        "raw": "/net/theta/fishpool/projects/tbi_meg/BIDS",
        "processed": "/net/theta/fishpool/projects/tbi_meg/k22_processed",
        "reports": f"/net/tera2/home/aino/work/mtbi-eeg/src/reports/{user}",
        "figures": f"/net/tera2/home/aino/work/mtbi-eeg/src/figures/{user}",
        "jobs": 4,
        "backend": "Qt5Agg",
    }
elif host == "rho" and user == "portae1":
    _profile = {
        "raw": "/net/theta/fishpool/projects/tbi_meg/BIDS",
        "processed": "/net/theta/fishpool/projects/tbi_meg/k22_processed",
        "reports": f"/net/tera2/home/portae1/biomag/mtbi_meeg/output/reports/{user}",
        "figures": f"/net/tera2/home/portae1/biomag/mtbi_meeg/output/figures/{user}",
        "jobs": 4,
        "backend": "Qt5Agg",
    }
elif "vdiubuntu" in host and user == "portae1":
    _profile = {
        "raw": "/m/home/home2/20/portae1/unix/biomag/k22_processed",
        "processed": "/m/home/home2/20/portae1/unix/biomag/k22_processed",
        "reports": "/m/home/home2/20/portae1/unix/biomag/mtbi_meeg/output/reports",
        "figures": "/m/home/home2/20/portae1/unix/biomag/mtbi_meeg/output/figures",
        "jobs": 4,
        "backend": "Qt5Agg",
    }

_profile = _profile or {}
_project_root = Path(__file__).resolve().parents[1]
_default_output = _project_root / "output"


def _path_setting(variable, profile_key, default):
    """Resolve a path from environment, site profile, then portable default."""
    return str(Path(os.getenv(variable, _profile.get(profile_key, default))).expanduser())


raw_data_dir = _path_setting("MTBI_RAW_DATA_DIR", "raw", _project_root / "data")
processed_data_dir = _path_setting(
    "MTBI_PROCESSED_DATA_DIR", "processed", _default_output / "derivatives"
)
reports_dir = _path_setting("MTBI_REPORTS_DIR", "reports", _default_output / "reports")
figures_dir = _path_setting("MTBI_FIGURES_DIR", "figures", _default_output / "figures")
n_jobs = int(os.getenv("MTBI_N_JOBS", _profile.get("jobs", 1)))
matplotlib_backend = os.getenv(
    "MTBI_MPL_BACKEND", _profile.get("backend", "Agg" if not os.getenv("DISPLAY") else "Qt5Agg")
)

if n_jobs < 1:
    raise ValueError("MTBI_N_JOBS must be a positive integer")

os.environ["OMP_NUM_THREADS"] = str(n_jobs)

import matplotlib
matplotlib.use(matplotlib_backend)
