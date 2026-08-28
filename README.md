# mTBI-EEG

Pipeline for the processing and analysis of EEG data from mild Traumatic Brain Injuries (mTBI) patients.

The processing pipeline reads raw EEG data and processes it to obtain bandpower data.
The analysis pipeline utilizes ML models for classifying the subjects.

The preprocessing and analysis sections are run separately.

Authors: Verna Heikkinen, Mia Liljeström, Aino Kuusi, Estanislao Porta

## Diagram
The diagram of the pipeline can be seen in the following diagram:

![Pipeline diagram](/src/pipeline_diagram.png)

### Example images on the output data:
<Add images of the output control plots / ROC / metadata >

## Folder structure
The folder structure for the project is shown below. The root folder is `mtbi_eeg`. The code is under `src`. Common scripts and config files are under `src/`.
The modules for data processing and data analysis are under `src/processing/` and `src/analyisis/`.

```bash
mtbi_meeg/
├── LICENSE
├── README.md
├── requirements.txt
├── setup.py
├── src/
│   ├── check_system.py
│   ├── config_eeg.py
│   ├── config_common.py
│   ├── fnames.py
│   ├── pickle_data_handler.py
│   ├── other_files/ (some old / unused files)
│   ├── analysis/
│   │    ├── 01_read_processed_data.py
│   │    ├── 02_plot_topographic_data.py (WIP)
│   │    ├── 03_plot_processed_data.py
│   │    ├── 04_fit_classifier_and_plot.py
│   │    ├── 05_create_report.py
│   │    ├── 06_convert_reports_to_pdf.py
│   │    └── run_files.py
│   └── processing/
│        ├── 01_freqfilt.py
│        ├── 02_ica.py
│        ├── 03_psds.py
│        ├── 04_bandpower.py
│        └──run_files.py 
└── tests/
    ├── test_01_read_processed_data.py
    ├── test_02_plot_processed_data.py
    ├── test_03_fit_model_and_plot.py (WIP)
    ├── test_04_create_report.py (WIP)
    └── test_05_convert_reports_to_pdf.py (WIP)
```

## Installing the package locally in the computer
The package can be installed by cloning the repository and using pip install:

```bash
$ git clone https://github.com/BioMag/mtbi_meeg
# cd into the root directory
$ cd mtbi_meeg
$ python3 -m pip install .
```
This will install all the necessary dependencies for the package to work. 

In the case that freely installing dependencies in the local computer is not possible or not desirable, an option is to create a conda environment or start a Docker container. Instructions using these two alternative methods are descibed below. 

## Installing in a Conda enviornment
1. After cloning the repository, navigate to the root directory.
2. Create the conda environment by running,
    ```
    $ conda env create --file environment.yml
    # Note: this step might take 20+ minutes
    ```
3. Activate the created environment,
    ```
    $ conda activate mtbi_meeg_conda
    ```
4. Install the package and its dependencies,
    ```
    $ python3 -m pip install .
    ```
## Installing using Docker
This Dockerfile specifies a base image (`continuumio/miniconda3:latest`), updates conda, installs the necessary dependencies, and copies the `mtbi_meeg` package code into the container. It also sets the working directory and specifies the default command to run when the container starts.

1. Build the Docker image: Run the command `docker build -t mtbi_meeg . ` to build the Docker image. This will create a new image named `mtbi_meeg` based on the Dockerfile.

2. Run the Docker container: Run the command `docker run -it mtbi_meeg` to start the container and run your package. The default container command runs the data-free pytest suite. Override the command to run an individual pipeline script after mounting the required data directories.

## Getting started: portable configuration and system check

Existing BioMag and Aalto workstation profiles continue to work. New users no
longer need to edit Python source: set the runtime paths with environment
variables before running the pipeline.

```bash
export MTBI_RAW_DATA_DIR=/path/to/bids
export MTBI_PROCESSED_DATA_DIR=/path/to/derivatives
export MTBI_REPORTS_DIR=/path/to/reports
export MTBI_FIGURES_DIR=/path/to/figures
export MTBI_N_JOBS=4
export MTBI_MPL_BACKEND=Agg  # use Qt5Agg for an interactive desktop

cd src
python check_system.py
```

All variables are optional. On a recognized BioMag/Aalto host, existing site
values are used. Elsewhere the defaults are `data/` and `output/` inside the
repository, one worker, and a headless-safe Matplotlib backend. The system check
still requires the raw-data directory to exist before preprocessing and creates
the output directories when needed.

The raw directory is expected to follow BIDS subject naming (`sub-01C`,
`sub-01P`, etc.). Subject discovery is lazy and therefore importing analysis
helpers or running unit tests does not require access to clinical data.

### Running tests without clinical data

The test suite uses generated arrays and temporary directories only:

```bash
python -m pip install . pytest
python -m pytest -q
```

Pull requests run the same suite on Python 3.10, 3.11, and 3.12 through GitHub
Actions.

#### Version Conflict errors
If there is an issue with packages or versions, you will see a message indicating the library with a Version Conflict. Please update the package
```bash
# For updating using pip,
$ python3 -m pip upgrade <package-name>
# For updating using conda,
$ conda update <package-name>
```
#### Missing raw data dir errors
 If raw data is missing, the repository cannot be used.


# Running the pipelines
Once you have added the required information in config_common and checked that all the dependencies are met, you can run the `preprocessing` or `analysis` sections. If you haven't, please follow the instructions described above.

## Preprocessing
The preprocessing pipeline can be found in `src/processing/`. The aim of this pipeline is to clean up the data and extract useful features, so data can be used by the classifiers in the analysis section.

**Files:**
- `01_freqfilt.py`: applies frequency filtering
- `02_ica.py`: removes ocular & heartbeat artefacts with independent component analysis
- `03_psds.py`: computes the PSDs over all channels and saves them as h5 files
- `04_bandpower.py`: calculates band power for each subject and creates a spatial frequency matrix that is then vectorized for later analysis.

**Inputs:**
- Raw data (in folder `raw_data_dir`)
- Parameters defined in config_eeg.py 
- Subject(s)

**Outputs:**
- Processed files: CSV files with bandpower data (in folder `processed_data_dir`)
- Reports

#### How to run:
Go to the folder `src/processing`. Make sure that file `subjects.txt` exists in the folder.

You can run one file at a time using `python3 <filename> <arguments>`.
Alternatively, you can run the pipeline using the `run_files.py` file. It loops over all steps of the pipeline, using one subject at a time. This means that it will re-run the pipeline as many times as subjects there are.
Since running all the steps for one subject might take a couple of minutes, there's an option to run a test run with only two subjects by modifying the boolean `TEST_RUN` to True in the `run_files.py` file.

```bash
$ cd src/processing/
# Note: Due to the extensive time that running the processing pipeline for each subject takes,
# one can modify the boolean `TEST_RUN` to True in the file `run_files.py` (using, e.g., `nano run_files.py`).
# This will run the whole processing pipeline for only 2 subjects, which takes around 3min or 1.5min per subject.
$ python3 run_files.py
```

## Analysis pipeline
The data analysis is done using the scripts in the folder `src/analysis`. The aim is to use different classifiers (LR, LDA, SVM and RF) to differentiate between patients and controls. A file `subjects.txt` is expected in this folder

**Files:**
- `01_read_processed_data.py`: Reads in EEG bandpower data from CSV files into a dataframe. The dataframe and the arguments used to run the script are added to a pickle object.
- `02_plot_processed_data.py`: (optional step) Plots the processed EEG data of the PSD intensity for visual assessment.
- `03_fit_model_and_plot.py`: Fits ML classifiers using the processed data, performs cross validation and evaluates the performance of the classification using ROC curves. Outputs a CSV file with the classification results, plots and saves plots to disk and adds the information to a metadata file.
- `04_create_report.py`: (optional) Creates an HTML report with the figures created in step 03.
- `05_convert_report_to_pdf.py`: (optional) Bundle up all htmls into one with a certain cover and create a PDF from it.
- `run_files.py`:

**Inputs:**
- Processed files: CSV files with bandpower data (in folder `processed_data_dir`)
- List of subjects in 
- parameters defined in `config_eeg.py`

**Outputs:**
- CSV file with results from ML classification accuracy metrics
- PNG figures
- HTML reports
- PDF with all the HTML reports

#### How to run it:
Go to the folder `src/analysis`. Make sure that file `subjects.txt` exists in the folder.

You can run one file at a time using `python3 <filename> <arguments>`.
Alternatively, you can run the whole pipeline using the `run_files.py` file. It loops over all steps of the pipeline, using all the list of subjects for each steps, but iterating over the four different tasks: eyes open (EO), eyes closed (EC), Paced Auditory Serial Addition Test 1 or 2 (PASAT_1 or PASAT_2). This means that you will run the whole pipeline four times. 

```bash
$ cd src/analysis/
# Note: Make sure that subjects.txt exists here
$ python3 run_files.py
```

## Subjects as arguments
The current way of defining the subjects to be processed is either via command line arguments and running each step of the pipeline separately or by using `run_files.py`, along with a file where all the subjects to be processed exist, named `subjects.txt`.

An example of the file can be seen below:
```python
# subjects.txt
01C
01P
02C
02C
# etc
```

# Instructions for running it in Aalto's HPC
asdf

## Things that are yet to be implemented:
- [x] config file for analysis? 
- [x] model fitting
- [ ] hyperparameter optimization, triton-compatible
- [x] model validation
- [x] visualizations
- [x] statistics


## Git stuff

You can ```git clone``` the repo using HTTPS address.
-  [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
-  [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd ~/mtbi-eeg
git pull origin main #maybe...
git branch <your branch name>
git push
```
It is **strongly recommended** to work on separate branches which are later merged to main by the owner of the repo.  
Merge requests are handled together! Check the branch you are currently on by typing ```git status```. 
You can change the branch you are one with the command ```git checkout -b <your branch name>```.
After pushing, create a merge request and assign someone to go through your code.

>NOTE: Before creating a new branch, ALWAYS pull the main branch from the remote repository!


## Collaboration

-  [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
-  [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
-  [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)


## License
Project under MIT License
