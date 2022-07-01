#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 11:11:53 2022

@author: aino

Script that helps (while bidsing) to check that each subject has a unique case number and whether there are some files missing
"""
import os


cases = {
    '01C': 'case_5887',
    '02C': 'name1',
    '03C': 'name2',
    '04C': 'name3',
    '05C': 'name4',
    '06C': 'name5',
    '07C': 'case_5897',
    '08C': 'case_5898',
    '09C': 'case_5900',
    '10C': 'case_5901',
    '11C': 'case_5902',
    '12C': 'case_5903',
    '13C': 'case_5904',
    '14C': 'name6',
    '15C': 'case_5927',
    '16C': 'case_5946',
    '17C': 'case_5948',
    '18C': 'case_5969',
    '19C': 'case_5972',
    '20C': 'case_5973',
    '21C': 'case_5979',
    '22C': 'case_5980',
    '23C': 'case_5983',
    '24C': 'case_5985',
    '25C': 'case_5990',
    '26C': 'case_5908',
    '27C': 'case_5947',
    '28C': 'case_5967',
    '29C': 'case_5968',
    '30C': 'case_5970',
    '31C': 'case_5971',
    '32C': 'case_5974',
    '33C': 'case_5975',
    '34C': 'case_5984',
    '35C': 'case_5989',
    '36C': 'case_5992',
    '37C': 'case_5993', 
    '38C': 'case_5994', 
    '39C': 'case_5995',
    '40C': 'case_5999',
    '41C': 'case_6004',
    '01P': 'case_5780',
    '02P': 'case_5837', 
    '03P': 'case_5793',
    '04P': 'case_5798',
    '05P': 'case_5809',
    '06P': 'case_5817',
    '07P': 'case_5821',
    '08P': 'case_5824',
    '09P': 'case_5828',
    '10P': 'case_5835',
    '11P': 'case_5836',
    '12P': 'case_5852',
    '13P': 'case_5857', #or case_5914?
    '14P': 'case_5859',
    '15P': 'case_5860',
    '16P': 'case_5864', #or case_5949?
    '17P': 'case_5865', #case_5940?
    '18P': 'case_5867',
    '19P': 'case_5871',
    '20P': 'case_5872',
    '21P': 'case_5877',
    '22P': 'case_5878',
    '23P': 'case_5879',
    '24P': 'case_5932',
    '25P': 'case_5939',
    '26P': 'case_5991',
    '27P': 'case_6006',
    '28P': 'case_6010',
    '29P': 'case_6035',
    '30P': 'case_6050',
    '31P': 'case_6078'
        } 
key_list = list(cases.keys())
value_list = list(cases.values())


# Check for duplicates in cases dictionary (empty)
cases_2 = {}
for key, value in cases.items():
    cases_2.setdefault(value, set()).add(key)
duplicates = [key for key, values in cases_2.items() if len(values)<1]
all_cases = [key for key, values in cases_2.items()]


# Get subdirectories (case numbers) from ec_eo_for_BIDS directory 
# List: subdirectories_cases
directory = '/net/theta/fishpool/projects/tbi_meg/eo_ec_for_BIDS/'
sub_directories_tuples = os.walk(directory)
sub_directories = [x[0] for x in sub_directories_tuples]
subdirectories = []
subdirectories_cases = []
for i in range(len(sub_directories)):
    subdirectories.append(sub_directories[i].removeprefix(directory))
for i in range(len(subdirectories)):
    if len(subdirectories[i]) == 9:
        subdirectories_cases.append(subdirectories[i])
        
not_in_dir = []

# Get cases which do not have data in ec_eo_for_BIDS directory
# List: not_in_dir
for i in all_cases:
    if i not in subdirectories_cases:
        not_in_dir.append(i)

# Get subjects who do not have data in ec_eo_for_BIDS directory
# List: not_in_dir_subs
not_in_dir_subs = []
for i in not_in_dir:
    position = value_list.index(i)
    not_in_dir_subs.append(key_list[position])

# Get unknown subjects from ec_eo_for_BIDS directory (i.e. case numbers that dont correspond to any subjects)
# List: unknown_subjects_in_dir
unknown_subjects_in_dir = []
for i in subdirectories_cases:
    if i not in value_list:
        unknown_subjects_in_dir.append(i)
    
