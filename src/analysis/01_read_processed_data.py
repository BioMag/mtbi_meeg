#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#############################
# 01_read_processed_data.py #
#############################

@authors: Verna Heikkinen, Aino Kuusi, Estanislao Porta

Reads in EEG data from CSV files into a dataframe
Each rows contains bandpower data for each channel and frequency band.
The dataframe and the arguments used to run the script are added to a pickle object.

Arguments
---------
    - task : str
        Each of the four tasks that have been measured for this experiment:
        Eyes Closed (ec), Eyes Open (eo),
        Paced Auditory Serial Addition Test 1 or 2 (PASAT_1 or PASAT_2)
    - freq_band_type : str
        Frequency bands used in the binning of the subject information.
        Thin bands are 1hz bands from 1 to 90hz. 
        Wide bands are conventional Delta, Theta, Alpha, Beta, Gamma
    - normalization : bool
        Defines whether channel data is normalized for all the channels

Returns
-------
    - eeg_tmp_data.pickle : pickle object 
        Object of pickle format containing the dataframe with the data
        and the metadata with the information about the arguments
        used to run this script.

# TODO: Define use of thinbands
# TODO: Add number of subjects and number of features to metadata
"""

import os
import sys
import argparse
import time
import re

import csv
import numpy as np
import pandas as pd

processing_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(processing_dir)
from config_common import processed_data_dir, user, host
from config_eeg import thin_bands, wide_bands, select_task_segments, channels
from pickle_data_handler import PickleDataHandler

def read_subjects():
    """
    Reads in the list of subjects from file subjects.txt. Asserts format to contain two digits and then a letter P or C for Patients or Controls. 
        
    Returns
    -------
    - subjects: a list with all the subjects
 
    """
     # List of extra controls, dismissed so we'd have equal number of P vs C
    to_exclude = ['32C', '33C', '34C', '35C', '36C', '37C', '38C', '39C', '40C', '41C', '12P']

    subject_pattern = r'^\d{2}[PC]'   
    try:
        with open('subjects.txt', 'r') as subjects_file:
            subjects = [line.rstrip() for line in subjects_file.readlines()]
            # Assert that each line has the expected format
            for line in subjects:
                assert re.match(subject_pattern, line), f"Subject '{line}' does not have the expected format."
    except FileNotFoundError as error_warning:
        print("The file 'subjects.txt' does not exist in the current directory. The program will exit.")
        raise error_warning
    
    # Excluse subjects with errors
    for i in to_exclude:
        subjects.remove(i)
    
    return subjects

def create_subjects_and_tasks(chosen_tasks, subjects):

    """
    Combines the subjects and with the chosen tasks and creates a list of subjects_and_tasks
    

    Arguments
    ---------
    - chosen_tasks: list of subtasks pertaining to each task 
    - subjects: list of all the subjects
    
    Returns
    -------
    - subjects_and_tasks: a list with 2-uples formed by all the combinations of (subjects, tasks)

    """
   
    subjects_and_tasks = [(x, y) for x in subjects for y in chosen_tasks]
    print(f'INFO: There are {len(subjects_and_tasks)} subject_and_task combinations.')
    
    return subjects_and_tasks

def read_data(subjects_and_tasks, freq_band_type, normalization, processed_data_dir):

    """
    Read in processed bandpower data for each subject_and_tasks from files
    Creates an array of np with PSD data
    
    Arguments
    ---------
    - subjects_and_tasks: list of 2-uples
            Contains the combinations of subjects and segments (e.g., (Subject1, Task1_segment1), (Subject1, Task1_segment2), ...)
    - freq_band_type: str
            Frequency bins, 'thin' or 'wide'
    - normalization: boolean
            If True, normalization of the PSD data for all channels will be performed   
    - processed_data_dir: str
            path to the processed data directory as defined in config_common

    Returns
    -----
    - all_bands_vector: list of np arrays
            Each row contains the PSD data (for the chosen frquency bands and for all channels) per subject_and_tasks
    """
    
    # Initialize a list to store processed data for each unique subject+segment combination 
    all_bands_vectors = [] 

    # Iterate over all combinations of (subject, subtask) and populate 'all_bands_vectors' with numpy array 'sub_bands_array' containing processed data for each subject_and_tasks
    for pair in subjects_and_tasks:  
        # Construct the path pointing to where processed data for (subject,task) is stored         
        subject, task = pair[0].rstrip(), pair[1] 
        path_to_processed_data = os.path.join(f'{processed_data_dir}', f'sub-{subject}', 'ses-01', 'eeg', 'bandpowers', f'{freq_band_type}_{task}.csv')
        
        # Create a 2D list to which the read data will be added
        subject_and_task_bands_list = []
        
        # Read csv file and saves each the data to f_bands_list
        with open(path_to_processed_data, 'r') as file:
            reader = csv.reader(file)
            for frequency_band in reader:  
                try:
                    subject_and_task_bands_list.append([float(f) for f in frequency_band])
                except ValueError as e:
                    print("Error: Invalid data, could not convert to float")              
                    raise e
                    
        # Convert list to array
        if freq_band_type == 'thin':
            subject_and_task_bands_array = np.array(subject_and_task_bands_list[0:40])
        else:
            subject_and_task_bands_array = np.array(subject_and_task_bands_list)
        
        # Normalize each band
        if normalization: 
            ch_tot_powers = np.sum(subject_and_task_bands_array, axis=0)
            subject_and_task_bands_array = subject_and_task_bands_array / ch_tot_powers[None, :]
        
        subject_and_task_bands_vector = np.concatenate(subject_and_task_bands_array.transpose())
        
#      Validate subject_and_task_bands_vector length:
        if freq_band_type == 'thin':
            assert len(subject_and_task_bands_vector) == (channels * 40), f"Processed data for subject {subject} does not have the expected length when using thin frequency bands."
            #assert len(subject_and_task_bands_vector) == (channels * len(thin_bands)), f"Processed data for subject {subject} does not have the expected length when using thin frequency bands."
        elif freq_band_type == 'wide':
            assert len(subject_and_task_bands_vector) == (channels * len(wide_bands)), f'Processed data for subject {subject} does not have the expected length when using wide frequency bands.'
            
        # Add vector to matrix
        all_bands_vectors.append(subject_and_task_bands_vector)    

    print(f'INFO: Success! Shape of \'all_bands_vectors\' is {len(all_bands_vectors)} x {len(all_bands_vectors[0])}, as expected.')
    return all_bands_vectors

def create_data_frame(subjects_and_tasks, all_bands_vectors):
    """
    Create a dataframe structure to be used by the model_testing and ROC_AUC.py scripts
        
    Arguments
    ---------
    - all_bands_vector: list of np arrays
            Each row contains the PSD data (for the chosen frquency bands and for all channels) per subject_and_tasks
    - subjects_and_tasks: list of 2-uples
            Contains the combinations of subjects and segments (e.g., (Subject1, Task1_segment1), (Subject1, Task1_segment2), ...) 
    
    Returns
    ------
    - dataframe: panda dataframe
            Each row contains the subject_and_task label, the group which it belongs to, and the PSD data (for the chosen frquency bands and for all channels) per subject_and_tasks
    """
    if not subjects_and_tasks:
        raise ValueError("The list of subject-task combinations cannot be empty.")
    if len(all_bands_vectors) == 0:
        raise ValueError("The list of PSD data cannot be empty.")        
    
    # Create a list of indices of format 'subject_segment'
    indices = [i[0].rstrip() + '_' + i[1] for i in subjects_and_tasks]
  
    # Convert list to numpy array to dataframe 
    dataframe = pd.DataFrame(np.array(all_bands_vectors, dtype=object), index=indices) 
    
    groups = []
    subs = []
    for subject, _ in subjects_and_tasks:
        subs.append(subject)
        if 'P' in subject:
            groups.append(1)
        elif 'C' in subject:
            groups.append(0)
        else:
            groups.append(2) # In case there is a problem
    dataframe.insert(0, 'Group', groups)
    dataframe.insert(1, 'Subject', subs)
    
    return dataframe 

def initialize_argparser_and_metadata():
    """ Initialize argparser and add args to metadata."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, help="ec, eo, PASAT_1 or PASAT_2", default="PASAT_1")
    parser.add_argument('--freq_band_type', type=str, help="Define the frequency bands. 'thin' are 1hz bands from 1 to 40hz. 'wide' are conventional delta, theta, etc. Default is 'thin'.", default="thin")
    parser.add_argument('--normalization', type=bool, help='Normalizing of the data from the channels', default=True)
    #parser.add_argument('--threads', type=int, help="Number of threads, using multiprocessing", default=1) #skipped for now
    args = parser.parse_args()
    
    # Create dictonary with metadata information 
    # NOTE: It is important that it is CREATED here and not that stuff gets appended
    metadata = {"task": args.task, "freq_band_type": args.freq_band_type, "normalization": args.normalization}
    # Define the number of segments per task
    if metadata["task"] in ('eo', 'ec'):
        segments = 3
    elif metadata["task"] in ('PASAT_1', 'PASAT_2'):
        segments = 2
    metadata["segments"] = segments
    
    print('######## \nINFO: Starting to run 01_read_processed_data.py')
    
    # Print out the chosen configuration
    if args.normalization:
        print(f"\nINFO: Reading in data from task {args.task}, using {args.freq_band_type} frequency bands. Data will be normalized. \n")
    else:
        print(f"\nINFO: Reading in data from task {args.task}, using {args.freq_band_type} frequency bands. Data will NOT be normalized. \n")
    
    return metadata, args

if __name__ == '__main__':
    # Save time of beginning of the execution to measure running time
    start_time = time.time()

    # 1 - Initialize command line arguments and save arguments to metadata
    metadata, args = initialize_argparser_and_metadata()

    # 2 - Define subtasks according to input arguments
    chosen_tasks = select_task_segments(args.task)

    # 3 - Read in the list of subjects from file subjects.txt
    subjects = read_subjects()

    # 4 - Read in list of subjects from file and create subjects_and_tasks list
    subjects_and_tasks = create_subjects_and_tasks(chosen_tasks, subjects)

    # 5 - Create list: each row contains all frequency bands and all channels per subject_and_task
    all_bands_vectors = read_data(subjects_and_tasks, args.freq_band_type, args.normalization, processed_data_dir)

    # 6 - Create dataframe
    dataframe = create_data_frame(subjects_and_tasks, all_bands_vectors)

    # 7 - Add info to metadata
    if "k22" in processed_data_dir:
        metadata["dataset"] = "k22"
    metadata["user"] = f'{user}@{host}'
    metadata["license"] = "MIT License"

    # 8 - Outputs the pickle object composed by the dataframe file and metadata to be used by 02_plot_processed_data.py and 03_fit_classifier_and_plot.py
    handler = PickleDataHandler()
    handler.export_data(dataframe=dataframe, metadata=metadata)
    
    # Calculate time that the script takes to run
    execution_time = (time.time() - start_time)
    print('\n###################################################\n')
    print(f'Execution time of 01_read_processed_data.py is: {round(execution_time,2)} seconds\n')
    print('###################################################\n')
