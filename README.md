# mTBI-EEG

Git repo building on the BioMag pipelines for analyzing the EEG data of mTBI patients.
Preprocessing and analysis are currently on separate folders.


## Getting started

The preprocessing pipeline is organized as follows:

1. Check your system with check_system.py
2. Configure user settings in ```config_common.py``` & ```config_eeg.py```
3. Using ```python -m doit``` while in eeg/ -directory, run the preprocessing of all files
    - ```01_freqfilt.py``` applies frequency filtering
    - ```02_ica.py``` removes ocular artefacts with independent component analysis
    - ```03_psds.py``` computes the PSDs over all channels and saves them as h5 files

> NOTE: these can be also ran separately for each individual!

## Git stuff

-  [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
-  [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd ~/mtbi-eeg
git branch configurations
git push
```
It is **strongly recommended** to work on separate branches which are later merged to main by the owner of the repo.  
Merge requests are handled together!

## Collaboration

-  [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
-  [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
-  [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
-  [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
-  [Automatically merge when pipeline succeeds](https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)


## License
Project under MIT License
