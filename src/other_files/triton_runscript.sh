#!/bin/bash
#SBATCH --time=01:40:00
#SBATCH --mem=900M
#SBATCH --job-name=cv_roc-hardcoded
#SBATCH --output=cv_roc-hardcoded_%a.out
#SBATCH --array=0-2 

case $SLURM_ARRAY_TASK_ID in
   0)  CLF=LDAoas ;;
   1)  CLF=SVM  ;;
   2)  CLF=LR  ;;
esac



module load anaconda

python ROC_AUC.py --iter 3 --drop Band_only --band all --sensor grad --eyes ec --location BioMag --clf $CLF > ROC_AUC_$CLF.json


