# mTBI-EEG

Git repos building on the BioMag pipelines for analyzing the EEG data of mTBI patients.
Preprocessing and analysis are currently on separate folders.


## Getting started

In order to execute the scripts in this repository, you should first configure your user in the `python/processing/config_common.py` file. The information is used for system requirements check and also to define the paths where the input and output data is expected.

Edit the `config_common` by doing the following,

1. Edit the file `python/processing/config_common.py` and add your user and host to the list. There's a commented block you can use as example in line 70.
2. Add the path for input data and the target path where to put processed data (*only for the the preprocessing pipeline. WIP to include also the output from the analysis pipeline*).
3. Check your system by running the script `python/processing/check_system.py`.

If everything looks ok, you should be ready to execute the pipeline.

## Data preprocessing pipeline

The preprocessing pipeline can be found in `python/processing/`. The scripts are under the folder `python/processing/eeg`. The aim of this pipeline is to clean up the data and extract useful features, so data can be used by the classifiers in the analysis section.

The list of files is described below:
- `config_eeg.py`: contains the information of bad channels for the dataset k22
- `00_maxfilter.py`: applies max filtering (not applicable to this data)
- `01_freqfilt.py`: applies frequency filtering
- `02_ica.py`: removes ocular & heartbeat artefacts with independent component analysis
- `03_psds.py`: computes the PSDs over all channels and saves them as h5 files
- `04_bandpower.py`: calculates band power for each subject and creates a spatial frequency matrix that is then vectorized for later analysis.

To run the pipeline, go to `python/processing/eeg` and do, **Is this working?**
```
doit
```
or run,
```
python -m doit
```
This will run the `dodo.py` file, which triggers the exection of all the scripts in this folders. Currently also `runall.sh` is being used (TBC)

> NOTE: these can be also ran separately for each individual!
> NOTE: another possibility is to use shell script that loops through the subjects

## Data analysis pipeline

The data analysis is done using the scripts in the folder `python/analysis`. The aim is to use different classifiers (LR, LDA, SVM and RF) to differentiate between patients and controls. 

1. Read in the data from ```readdata.py```
2. Fit and plot the ROC curves of various models with ```ROC_AUC.py``` (accepts commandline arguments)

### Things that are yet to be implemented:

- [ ] config file for analysis? 
- [x] model fitting
- [ ] hyperparameter optimization, triton-compatible
- [ ] model validation
- [x] visualizations
- [ ] statistics


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
You can change the branch you are one with the command ```git checkout <your branch name>```.
After pushing, create a merge request and assign someone to go through your code.

>NOTE: Before creating a new branch, ALWAYS pull the main branch from the remote repository!


## Collaboration

-  [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
-  [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
-  [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)


## License
Project under MIT License
