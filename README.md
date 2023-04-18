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

Instead of installing the package system-wide, it is instead recommended to use a [virtual environment](https://docs.python.org/3/library/venv.html)
```bash
$ git clone https://github.com/BioMag/mtbi_meeg
$ cd mtbi_meeg
$ python3 -m venv .venv
$ source .venv/bin/activate
$ python3 -m pip install .
```
or a [conda environment](https://docs.conda.io/en/latest/)
```bash
$ git clone https://github.com/BioMag/mtbi_meeg
$ cd mtbi_meeg
$ conda create -f environment.yml
$ conda activate mtbi_meeg_conda
$ python3 -m pip install .
```



## Installing using Docker
This Dockerfile specifies a base image (`continuumio/miniconda3:latest`), updates conda, installs the necessary dependencies, and copies the `mtbi_meeg` package code into the container. It also sets the working directory and specifies the default command to run when the container starts.

1. Build the Docker image: Run the command `docker build -t mtbi_meeg . ` to build the Docker image. This will create a new image named `mtbi_meeg` based on the Dockerfile.

2. Run the Docker container: Run the command `docker run -it mtbi_meeg` to start the container and run your package. This will start a new container based on the my_package image and run the default command specified in the Dockerfile.

## Getting started: config_common and system_check

Before the first time you execute the scripts in this repository, you must edit the file `src/config_common.py` and include a new block with information about your user, workstation, data directories, and matplotlib backend:

1. Go to folder `src` and open the file `config_common.py` for editing. From terminal,
    ```bash
    $ cd src
    $ nano config_common.py
    # Note: you can also use other editors. For using VS Code, type `code config_common.py`
    ```
2. Edit the file: add your `user` and `host` to the list. There's a commented block you can use as example [here](https://version.aalto.fi/gitlab/heikkiv7/mtbi-eeg/-/blob/main/python/processing/config_common.py#L78). Copy the block and edit it.
3. Add the path where `raw_data_dir` is expected ('/net/theta/fishpool/projects/tbi_meg/BIDS' in BioMag)
4. Add the path where `processed_data_dir` is expected (Be mindful which folder you choose, as you may overwrite other people's data)
5. Add the paths where figures and reports will be created into (you can use the directories in this repository or other)
6. Add the matplotlib backend (TBC)
7. Check that your system has the required dependencies by running the script `check_system.py`. From terminal, 
    ```bash
    $ python3 check_system.py
    ```
If there's no problems, you should see a message saying that 'System requirements are ok' and you should be ok to run the pipeline now.

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
