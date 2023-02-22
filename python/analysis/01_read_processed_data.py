#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 10:21:38 2022

@author: aino

Reads bandpower data from csv files and creates a matrix whose rows represent each subject. 
"""
import numpy as np
import csv
import pandas as pd
import argparse
from tqdm import tqdm 

import sys
sys.path.append('../processing/')
from config_common import processed_data_dir

def define_subtasks(task):
    """
    Define the subtasks to be used for the analysis
    
    
    Input parameters
    ---------
    - task: chosen task (eyes open, eyes closed, Paced Auditory Serial Addition Test 1 or PASAT 2)
    
    Returns
    -------
    - chosen_tasks: The list of chosen subtasks
    """
    tasks = [['ec_1', 'ec_2', 'ec_3'], 
             ['eo_1', 'eo_2', 'eo_3'], 
             ['PASAT_run1_1', 'PASAT_run1_2'], 
             ['PASAT_run2_1', 'PASAT_run2_2']]
       
    # Define which files to read for each subject
    if task == 'ec':
        chosen_tasks = tasks[0]
    elif task == 'eo':
        chosen_tasks = tasks[1]
    elif task == 'PASAT_1':
        chosen_tasks = tasks[2]
    elif task == 'PASAT_2': 
        chosen_tasks = tasks[3]
    else:
        raise("Incorrect task")
    
    
    return chosen_tasks

def create_subjects_and_tasks(chosen_tasks):
    """
    Read in the list of subjects, and combines it with the chosen task
    Create a list of subjects_and_tasks
    

    Parameters
    ----------
    - chosen_tasks: list of subtaks pertaining to each task 
    
    Returns
    -------
    - subjects_and_tasks: a list with 2-uples formed by all the combinations of (subjects, tasks)

    """
    # List of extra controls, dismissed so we'd have equal number of P vs C.
    to_exclude = ['32C', '33C', '34C', '35C', '36C', '37C', '38C', '39C', '40C', '41C']
             
    # Get the list of subjects. File expected in same directory    
    with open('subjects.txt', 'r') as subjects_file:
        subjects = subjects_file.readlines()
        subjects_file.close()
    subjects = [x.rstrip() for x in subjects]
    
    # Excluse subjects with errors
    for i in to_exclude:
        subjects.remove(i)
        
    # Define a list with 2-uples formed by all the combinations of (subjects, tasks)
    subjects_and_tasks = [(x,y) for x in subjects for y in chosen_tasks]
    # NOTE: shape(subjects_and_tasks) = n x 2, 
    #       where n = (elements in subjects * columns in chosen_tasks) 
    #print(f'There are {len(subjects_and_tasks)} subject_and_task combinations.')
    
    return subjects_and_tasks


def read_processed_data(subjects_and_tasks, freq_bands, normalization):

    """
    Read in processed bandpower data for each subject_and_tasks from files
    Create an array
    
    Create a dataframe to be used by the ROC_AUC.py script
    
    
    Input parameters
    ----------------
    subjects_and_tasks: 
        
    Output
    ------
    - dataframe: panda dataframe that contains in each row the the bandpower measurements per each combination of (subject, tasks)
    """
    
    # List of freq. indices (Note: if the bands are changed in 04_bandpower.py, these need to be modified too.)
    #TODO: Could we use these from config_eeg? YEs, I can change it
    wide_bands = [(0,3), (3,7), (7,11), (11,34), (34,40), (40,90)] 
 
        
    # Initialize a list to store processed data for each unique subject+sub_task combination 
    all_bands_vectors = [] 
    # NOTE: shape(all_bands_vectors) =  n vectors of length m
    #       where n = (elements in subjects * columns in chosen_tasks)
    #       and m = (number of channels [64] * number of frequency bands [89 when using 'thin' bands, or 6 when using 'wide' bands]) = 5696 when using thin bands or 384 when using wide bands 
    
    # Iterate over all combinations of (subject, subtask) and populate 'all_bands_vectors' with numpy array 'sub_bands_array' containing processed data for each subject_and_tasks
    for pair in subjects_and_tasks:
        
        # Construct the path pointing to where processed data for (subject,task) is stored         
        # TODO: could we change the name of the variable a bit? e.g., path_to_processed_data
        subject, task = pair[0].rstrip(), pair[1] 
        bandpower_file = f'{processed_data_dir}sub-{subject}/ses-01/eeg/bandpowers/{task}.csv'

        # Create a 2D list to which the read data will be added
        sub_bands_list = []
        
        # Read csv file and save the data to f_bands_list

        with open(bandpower_file, 'r') as file:
            reader = csv.reader(file)
            for f_band in reader: #Goes through each frequency band. 
                sub_bands_list.append([float(f) for f in f_band])  
            file.close()
            
        
        # Convert list to array
        # TODO: Is the 0:90 assuming 'thin' bands?
        sub_bands_array = np.array(sub_bands_list)[0:90, :] # m x n matrix (m = frequency bands, n=channels)
        
        # Aggregate 1 Hz freq bands to conventional delta, theta, alpha, etc.
        if freq_bands == 'wide':
            sub_bands_list = []
            sub_bands_list.append([np.sum(sub_bands_array[slice(*t),:], axis=0) for t in wide_bands])
            #create array again
            sub_bands_array = np.array(sub_bands_list)[0] #apparently there is annyoing extra dimension
        
        # Normalize each band
        if normalization: 
            ch_tot_powers = np.sum(sub_bands_array, axis = 0)
            sub_bands_array = sub_bands_array / ch_tot_powers[None,:]
        
        sub_bands_vec = np.concatenate(sub_bands_array.transpose())
        
        if (len(sub_bands_vec) != 5696):
            print(f'Processed data for subject {subject} does not meet the expected format.')
            #raise ValueError(f'Processed data for subject {subject} is does not meet the expected format. \nDifferent band width might have been used for processing.\n####\n')
            #sys.exit(1)    
            
        # Add vector to matrix
        all_bands_vectors.append(sub_bands_vec)    
        
    #print(f'Shape of \'all_bands_vectors\' is {len(all_bands_vectors)} x {len(all_bands_vectors[0])} (note that this is only the length of first column and first row, it does not account for shorter/longer rows/columns).')
    return all_bands_vectors

def create_data_frame(all_bands_vectors, subjects_and_tasks):
    """
    Creating a data frame
    """
    
    # Create indices for dataframe
    indices = []
    for i in subjects_and_tasks:
        j = i[0].rstrip()+'_'+i[1]
        indices.append(j)
    
    # Convert list to numpy array to dataframe 
    dataframe = pd.DataFrame(np.array(all_bands_vectors, dtype = object), indices ) 
    
    # Add column 'Group'
    groups = []
    for subject in indices:
        if 'P' in subject[2]:
            groups.append(1)
        elif 'C' in subject[2]:
            groups.append(0)
        else:
            groups.append(2) # In case there is a problem
    dataframe.insert(0, 'Group', groups)
    #TODO: horrible bubble-gum quickfix for CV problem
    #fixed the line above so that it works for all tasks
    subs = np.array([s.split('_'+chosen_tasks[0][0:3])[0] for s in indices]) 
    dataframe.insert(1, 'Subject', subs)
    
    return dataframe

    
if __name__ == '__main__':
    
    CV = True
    
    # Add arguments to be parsed from command line    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, help="ec, eo, PASAT_1 or PASAT_2", default="ec")
    parser.add_argument('--freq_bands', type=str, help="Define the frequency bands. 'thin' are 1hz bands from 1 to 90hz. 'wide' are conventional delta, theta, etc. Default is 'thin'.", default="thin")
    parser.add_argument('--normalization', type=bool, help='Normalizing of the data from the channels', default=False)
    #parser.add_argument('--threads', type=int, help="Number of threads, using multiprocessing", default=1) #skipped for now
    args = parser.parse_args()
    
    # Print out the chosen configuration
    print(f"Reading in data from task {args.task}, using {args.freq_bands} frequency bands... \n")
    
                
    # Define subtasks according to input arguments
    chosen_tasks = define_subtasks(args.task)
    
    # Read in list of subjects from file and create subjects_and_tasks list
    subjects_and_tasks = create_subjects_and_tasks(chosen_tasks)
    
    # Read in processed data from file and create list
    all_bands_vectors = read_processed_data(subjects_and_tasks, args.freq_bands, args.normalization)
    
    # Create dataframe
    dataframe = create_data_frame(all_bands_vectors, subjects_and_tasks)

    # Outputs the dataframe file that is needed by the ROC_AUC.py
    #TODO: Add a path to config_common for this folder? Or if data frame is not needed, remove the creation of a file, and rather return a value to be consumed by the ROC function?
    dataframe.to_csv('dataframe.csv', index_label='Index')
    print('\n###\nDataframe has been created to file dataframe.csv, in current directory.')    
        