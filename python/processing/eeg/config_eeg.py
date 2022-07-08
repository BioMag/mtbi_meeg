"""
These are all the relevant parameters that are unique to the EEG analysis
pipeline.
"""

# Some relevant files are in the parent folder.
import sys
sys.path.append('../')

from fnames import FileNames

from config_common import (raw_data_dir, processed_data_dir, figures_dir,
                           reports_dir, all_subjects, tasks)


###############################################################################
# Parameters that should be mentioned in the paper

# Highpass filter above 1Hz. This is needed for the ICA to perform well
# later on. Lowpass filter below 100Hz to get rid of the signal produced by
# the cHPI coils. Notch filters at 50Hz and 100Hz to get rid of powerline.
fmin = 1
fmax = 90
fnotch = [50, 100]

# Computation of the PSDs
n_fft = 2048  # Higher number means more resolution at the lower frequencies
#1024?4096?


###############################################################################
# Parameters pertaining to the subjects

# Subjects removed from the MEG analysis because of some problem
# TODO: check these 
# bad_subjects = [
#     24, 29,  # No EEG data present
# ]
bad_subjects= []
# Analysis is performed on these subjects
subjects = [subject for subject in all_subjects if subject not in bad_subjects]

ecg_channel = 'ECG002'
eog_channel = 'EOG001'
bads={}

#Bad EEG channels for each subject and each task
ec_bads = {
        '01C': ['EEG033', 'EEG025', 'EEG023'],
        '02C': [],
        '03C': ['EEG044'],
        '04C': ['EEG026', 'EEG045', 'EEG027', 'EEG049'],
        '05C': ['EEG003', 'EEG010', 'EEG057'],
        '06C':['EEG001', 'EEG013', 'EEG020', 'EEG022', 'EEG027', 'EEG026', 'EEG023', 'EEG018', 'EEG019', 'EEG012', 'EEG011'],
        '07C': [],
        '08C': ['EEG003', 'EEG007', 'EEG004', 'EEG008'],
        '09C': ['EEG011', 'EEG018'],
        '10C':['EEG019', 'EEG018', 'EEG011'],
        '11C': ['EEG029'],
        '12C':['EEG016', 'EEG033', 'EEG023'],
        '13C': ['EEG019'],
        '14C':['EEG003', 'EEG018', 'EEG007', 'EEG008', 'EEG004', 'EEG014', 'EEG015', 'EEG033', 'EEG023'],
        '15C':['EEG018', 'EEG019', 'EEG011', 'EEG033', 'EEG023', 'EEG055', 'EEG063'],
        '16C':['EEG043', 'EEG018', 'EEG019'],
        '17C': ['EEG005', 'EEG020', 'EEG026', 'EEG028', 'EEG023', 'EEG027', 'EEG017'],
        '18C':['EEG007', 'EEG008', 'EEG004', 'EEG034'],
        '19C':['EEG004', 'EEG008', 'EEG007', 'EEG022', 'EEG043'],
        '20C':['EEG044', 'EEG045', 'EEG052', 'EEG059'],
        '21C':['EEG003', 'EEG010', 'EEG012', 'EEG019', 'EEG029', 'EEG030', 'EEG042', 'EEG046', 'EEG059', 'EEG062', 'EEG061'],
        '22C': [],#really bad eeg
        '23C':['EEG003', 'EEG018', 'EEG027', 'EEG025', 'EEG037'],
        '24C':['EEG013', 'EEG020', 'EEG001', 'EEG022', 'EEG027', 'EEG029', 'EEG028', 'EEG047'],
        '25C':['EEG001', 'EEG002', 'EEG013', 'EEG017', 'EEG022', 'EEG023', 'EEG026', 'EEG027', 'EEG028'],
        '26C':['EEG034', 'EEG035', 'EEG037', 'EEG050', 'EEG048', 'EEG042', 'EEG043', 'EEG049', 'EEG047'],
        '27C':['EEG033', 'EEG058'],
        '28C':['EEG019', 'EEG056'],#first 30 s bad
        '29C':['EEG023', 'EEG033'],
        '30C':[],
        '31C':['EEG001', 'EEG002', 'EEG013', 'EEG032', 'EEG028', 'EEG027', 'EEG026', 'EEG023', 'EEG022', 'EEG050', 'EEG003'],
        '32C':['EEG003', 'EEG060'],
        '33C':['EEG001', 'EEG003', 'EEG020', 'EEG026', 'EEG028', 'EEG027', 'EEG056'],
        '34C':[],
        '35C':['EEG006', 'EEG003', 'EEG028'],
        '36C':['EEG003', 'EEG006'],
        '37C':['EEG003', 'EEG001', 'EEG017', 'EEG013', 'EEG005', 'EEG020', 'EEG014', 'EEG022', 'EEG023', 'EEG027', 'EEG028'],
        '38C':['EEG019', 'EEG022', 'EEG023'],
        '39C':['EEG018', 'EEG011', 'EEG010'],
        '40C':[],
        '41C':['EEG007', 'EEG063', 'EEG062', 'EEG064', 'EEG055'],#"karvaisia kanavia"
        '01P':[],
        '02P':['EEG013', 'EEG017', 'EEG022', 'EEG027'],
        '03P':['EEG005', 'EEG022', 'EEG023', 'EEG027', 'EEG032', 'EEG028'],
        '04P':['EEG018'],
        '05P':[],
        '06P':['EEG013', 'EEG012', 'EEG032', 'EEG026', 'EEG022', 'EEG023'],
        '07P':[],
        '08P':['EEG027', 'EEG042'],
        '09P':['EEG009', 'EEG013', 'EEG035'],
        '10P': [], #a lot of eye movement and heart artifacts
        '11P':[],
        '12P':[],
        '13P':['EEG029', 'EEG038', 'EEG042'],
        '14P':['EEG018', 'EEG007', 'EEG014', 'EEG027', 'EEG026', 'EEG043', 'EEG048'],
        '15P':['EEG039', 'EEG044', 'EEG056', 'EEG059', 'EEG060', 'EEG046', 'EEG063'],
        '16P':['EEG017', 'EEG016', 'EEG021', 'EEG028'],
        '17P':['EEG017'],#001-007 "karvaisia"
        '18P':['EEG017', 'EEG020', 'EEG009', 'EEG022', 'EEG023', 'EEG026', 'EEG028', 'EEG048', 'EEG047'],
        '19P':['EEG007', 'EEG014', 'EEG008', 'EEG015'],
        '20P':['EEG010', 'EEG014', 'EEG007', 'EEG008', 'EEG009'],
        '21P':['EEG004', 'EEG030', 'EEG052', 'EEG061'],#a lot of eye movements
        '22P':['EEG017', 'EEG020', 'EEG019', 'EEG013', 'EEG018', 'EEG022', 'EEG028', 'EEG026'],
        '23P':['EEG007', 'EEG004', 'EEG003', 'EEG014'],
        '24P':['EEG023', 'EEG033', 'EEG035', 'EEG042'],
        '25P':['EEG004', 'EEG007', 'EEG023', 'EEG033'],
        '26P':['EEG007', 'EEG004', 'EEG029', 'EEG050'],
        '27P':['EEG010', 'EEG027'],
        '28P':['EEG003'],
        '29P':['EEG001', 'EEG003', 'EEG020', 'EEG019', 'EEG013', 'EEG005', 'EEG027', 'EEG022'],
        '30P':['EEG003', 'EEG004', 'EEG007', 'EEG008', 'EEG026', 'EEG029'],
        '31P':['EEG003', 'EEG007', 'EEG011', 'EEG028', 'EEG027', 'EEG034'],
        }#channels 11, 18, 19 were quite flat in general

eo_bads = {
        "01C": ['EEG023', 'EEG025', 'EEG033'],
        '02C': ['EEG040'],
        '03C': ['EEG044'],
        '04C':['EEG026', 'EEG036', 'EEG028', 'EEG027', 'EEG032', 'EEG045', 'EEG047', 'EEG043', 'EEG054', 'EEG049'],
        '05C':['EEG003', 'EEG010', 'EEG057', 'EEG053', 'EEG062'],
        '06C': ['EEG001', 'EEG013', 'EEG020', 'EEG022', 'EEG023', 'EEG026', 'EEG027'],
        '07C':['EEG032', 'EEG041'],
        '08C':['EEG003', 'EEG028'],#a lot of eye movements 
        '09C':['EEG020'],
        '10C':['EEG014'],
        '11C':['EEG007', 'EEG004', 'EEG003', 'EEG008', 'EEG015', 'EEG029', 'EEG032'],
        '12C':['EEG016', 'EEG008', 'EEG023', 'EEG033'],
        '13C':['EEG042'],
        '14C':['EEG003', 'EEG014', 'EEG023', 'EEG033'],
        '15C':['EEG023', 'EEG033', 'EEG055'],
        '16C':['EEG012', 'EEG043'],
        '17C':['EEG003', 'EEG005', 'EEG017', 'EEG026', 'EEG023', 'EEG028', 'EEG027'],
        '18C': ['EEG034'],
        '19C':['EEG022', 'EEG043'],
        '20C':['EEG012', 'EEG032', 'EEG044', 'EEG045', 'EEG059', 'EEG052', 'EEG058', 'EEG053', 'EEG054', 'EEG064'],
        '21C':['EEG003', 'EEG010', 'EEG012', 'EEG019', 'EEG005', 'EEG007', 'EEG029', 'EEG030', 'EEG024', 'EEG042', 'EEG046', 'EEG059', 'EEG062', 'EEG053'],
        '22C':[], #very bad eeg
        '23C':['EEG018', 'EEG027', 'EEG025', 'EEG037', 'EEG034'],
        '24C':['EEG017', 'EEG013', 'EEG020', 'EEG003', 'EEG001', 'EEG027', 'EEG022', 'EEG029', 'EEG028', 'EEG047'],
        '25C':['EEG013', 'EEG001', 'EEG002', 'EEG022', 'EEG023', 'EEG026', 'EEG027', 'EEG028', 'EEG048', 'EEG049'],
        '26C':['EEG035', 'EEG034', 'EEG037', 'EEG042', 'EEG043', 'EEG048', 'EEG050', 'EEG047', 'EEG049', 'EEG056'],
        '27C':['EEG033', 'EEG058'],
        '28C': ['EEG019', 'EEG013', 'EEG028', 'EEG058'],
        '29C':['EEG007', 'EEG018', 'EEG009', 'EEG023', 'EEG033', 'EEG032'],
        '30C':[],
        '31C':['EEG001', 'EEG002', 'EEG013', 'EEG003', 'EEG022', 'EEG023', 'EEG026', 'EEG027', 'EEG028', 'EEG032', 'EEG050'],
        '32C':['EEG003', 'EEG060'],
        '33C':['EEG001', 'EEG003', 'EEG020', 'EEG013', 'EEG026', 'EEG028', 'EEG027', 'EEG056'],
        '34C':[],
        '35C':['EEG013', 'EEG007', 'EEG008', 'EEG034', 'EEG032', 'EEG043', 'EEG047'],#ekg?
        '36C':['EEG006', 'EEG003', 'EEG028'],
        '37C': ['EEG017', 'EEG013', 'EEG005', 'EEG020', 'EEG003', 'EEG001', 'EEG027', 'EEG023', 'EEG022'],
        '38C':['EEG001', 'EEG008', 'EEG015', 'EEG035', 'EEG023'],
        '39C':['EEG018', 'EEG015', 'EEG002', 'EEG010', 'EEG009', 'EEG011'],
        '40C':[],
        '41C':['EEG064', 'EEG063', 'EEG062'],
        '01P':['EEG004', 'EEG017'],
        '02P':['EEG017', 'EEG003', 'EEG013', 'EEG022', 'EEG027', 'EEG061', 'EEG056'],
        '03P':['EEG005', 'EEG013', 'EEG022', 'EEG023', 'EEG027', 'EEG032', 'EEG038'],
        '04P':['EEG018', 'EEG003', 'EEG024', 'EEG032', 'EEG044', 'EEG055', 'EEG062'],
        '05P':['EEG014', 'EEG032'],
        '06P':['EEG013', 'EEG012', 'EEG022', 'EEG023', 'EEG026', 'EEG032'],
        '07P':['EEG008'],
        '08P':['EEG027', 'EEG024', 'EEG042'],
        '09P':['EEG009', 'EEG035'],
        '10P':[], #heart artefact
        '11P':[],
        '12P':[],
        '13P':['EEG013', 'EEG038'],
        '14P':['EEG018', 'EEG027', 'EEG043', 'EEG048'],
        '15P':['EEG015', 'EEG014', 'EEG044', 'EEG056', 'EEG059', 'EEG060', 'EEG046', 'EEG063'],
        '16P':['EEG016', 'EEG017', 'EEG032'],
        '17P':['EEG017'],#001-007 "karvaisia"
        '18P':['EEG017', 'EEG020', 'EEG001', 'EEG003', 'EEG026', 'EEG023', 'EEG022', 'EEG028', 'EEG047', 'EEG048'],
        '19P':[], #a lot of blinking
        '20P':['EEG014', 'EEG027', 'EEG061'],
        '21P':['EEG052'], #a lot of eye movements
        '22P':['EEG017', 'EEG019', 'EEG020', 'EEG018', 'EEG013', 'EEG022', 'EEG028', 'EEG041'],
        '23P':[],
        '24P':['EEG023', 'EEG033', 'EEG035'],
        '25P':['EEG023', 'EEG033'], #001-007 "karvaisia"
        '26P':[],
        '27P':['EEG027'],
        '28P':['EEG003'],
        '29P':['EEG001', 'EEG003', 'EEG005', 'EEG019', 'EEG020', 'EEG026', 'EEG027', 'EEG022', 'EEG023', 'EEG048', 'EEG042'],
        '30P':[], #"karvaisia kanavia"
        '31P':['EEG003', 'EEG007', 'EEG027', 'EEG028', 'EEG045'] #a lot of blinking       
        }
pasat1_bads = {#pasats are shorter
        '01C': ['EEG033', 'EEG025', 'EEG023'],
        '02C': ['EEG016', 'EEG053', 'EEG054'],
        '03C': ['EEG044'],
        '04C': ['EEG049', 'EEG045', 'EEG043', 'EEG038'], #a lot of bad data
        '05C': [],
        '06C':['EEG001', 'EEG020', 'EEG027', 'EEG023', 'EEG022', 'EEG026'],
        '07C': [],
        '08C': ['EEG003'],
        '09C': ['EEG027'],
        '10C':[],
        '11C': ['EEG029', 'EEG032'],#karvaisia kanavia 1-7, bad eog, weird ecg
        '12C':['EEG016', 'EEG033', 'EEG027', 'EEG023'],
        '13C':  ['EEG003', 'EEG007'],
        '14C':['EEG033', 'EEG023', 'EEG063'],
        '15C':['EEG023', 'EEG033', 'EEG055'],
        '16C':[],
        '17C': ['EEG005', 'EEG017', 'EEG020', 'EEG027', 'EEG028', 'EEG026', 'EEG023'],
        '18C':[],
        '19C':['EEG011', 'EEG043'],
        '20C': ['EEG033', 'EEG044', 'EEG045', 'EEG059', 'EEG052'],
        '21C':['EEG012', 'EEG010', 'EEG007', 'EEG003', 'EEG030', 'EEG029', 'EEG024', 'EEG046', 
               'EEG059', 'EEG042', 'EEG062'],
        '22C': [],#really bad eeg
        '23C':['EEG003', 'EEG018', 'EEG025', 'EEG037', 'EEG027'],
        '24C':['EEG001', 'EEG017', 'EEG013', 'EEG020', 'EEG022', 'EEG027', 'EEG029', 'EEG028'],
        '25C':['EEG013', 'EEG001', 'EEG002', 'EEG017', 'EEG028', 'EEG027', 'EEG026', 'EEG023', 'EEG022'],
        '26C':['EEG034', 'EEG035', 'EEG037', 'EEG042', 'EEG043', 'EEG048', 'EEG050'],
        '27C':['EEG033', 'EEG063'],
        '28C':['EEG019', 'EEG038'],
        '29C':['EEG009', 'EEG023', 'EEG033'],
        '30C':[],
        '31C':['EEG017', 'EEG001', 'EEG002', 'EEG003', 'EEG032', 'EEG022', 'EEG023', 
               'EEG027', 'EEG026', 'EEG028', 'EEG050'],#lot of blinks/otherwise quite bad data
        '32C':['EEG003'],
        '33C':['EEG001', 'EEG003', 'EEG020', 'EEG002', 'EEG026', 'EEG028', 'EEG027', 
               'EEG022', 'EEG023', 'EEG056', 'EEG060'],
        '34C':[],
        '35C':['EEG013', 'EEG012', 'EEG034', 'EEG030', 'EEG043', 'EEG047'],#eog, ecg wrong labels
        '36C':['EEG003', 'EEG006'],
        '37C':['EEG005', 'EEG017', 'EEG013', 'EEG002', 'EEG001', 'EEG003', 'EEG023', 'EEG022', 'EEG027'],
        '38C':[],
        '39C':['EEG018'],
        '40C':[],
        '41C':[],
        '01P':[],
        '02P':['EEG003', 'EEG013', 'EEG017', 'EEG022', 'EEG027', 'EEG061'],
        '03P':['EEG005', 'EEG001', 'EEG032', 'EEG027', 'EEG023', 'EEG022'],
        '04P':['EEG018'],
        '05P':[],
        '06P':['EEG013', 'EEG022', 'EEG023', 'EEG026', 'EEG032', 'EEG056'],
        '07P':[],
        '08P':['EEG027', 'EEG042', 'EEG063'],#karvaisia kanavia
        '09P':['EEG009', 'EEG035'],
        '10P': ['EEG006'], #a lot of eye movement and heart artifacts
        '11P':[],
        '12P':[],
        '13P': ['EEG038'],
        '14P':['EEG018', 'EEG027', 'EEG043', 'EEG048'],
        '15P':['EEG056', 'EEG059', 'EEG060', 'EEG044', 'EEG046', 'EEG057', 'EEG045', 'EEG063'],
        '16P':[],
        '17P':['EEG017'],#001-007 "karvaisia"
        '18P':['EEG017', 'EEG020', 'EEG001', 'EEG026', 'EEG022', 'EEG027', 'EEG023', 
               'EEG039', 'EEG028', 'EEG037', 'EEG047'],
        '19P':[],
        '20P':['EEG061'],
        '21P':[],
        '22P':['EEG001', 'EEG017', 'EEG019', 'EEG020', 'EEG013', 'EEG027', 'EEG028', 'EEG048'],
        '23P':[],#karvaisia kanavia
        '24P':['EEG023', 'EEG033', 'EEG032'],#karvaisia kanavia
        '25P':['EEG003', 'EEG023', 'EEG033'],
        '26P':['EEG003'],
        '27P':['EEG027', 'EEG037', 'EEG049', 'EEG056'],
        '28P':['EEG003', 'EEG007', 'EEG024'],
        '29P':['EEG005', 'EEG001', 'EEG019', 'EEG020', 'EEG003', 'EEG022', 'EEG023', 
               'EEG026', 'EEG027', 'EEG063'],
        '30P':['EEG058'],
        '31P':['EEG003', 'EEG011', 'EEG007', 'EEG027', 'EEG046'],
    }
pasat2_bads = {
        '01C': ['EEG033', 'EEG025', 'EEG023'],
        '02C': [],
        '03C': ['EEG044'],
        '04C': ['EEG026', 'EEG028', 'EEG038', 'EEG027', 'EEG045', 'EEG049', 'EEG043', 'EEG057', 'EEG064'], 
        '05C': ['EEG010'],
        '06C':['EEG001', 'EEG020', 'EEG022', 'EEG023', 'EEG026', 'EEG027', 'EEG028'],
        '07C': [],
        '08C': ['EEG003'],
        '09C': ['EEG027'], #horrible eog!!
        '10C':[],
        '11C': ['EEG029'],#karvaisia kanavia 1-7, dead eog, weird ecg
        '12C':['EEG016', 'EEG023', 'EEG033'],
        '13C':  ['EEG003'],
        '14C':['EEG023', 'EEG033'],
        '15C':['EEG023', 'EEG033', 'EEG055'],
        '16C':[],
        '17C': ['EEG005', 'EEG017', 'EEG023', 'EEG026', 'EEG027', 'EEG028', 'EEG029'],
        '18C':[],
        '19C':['EEG043'],
        '20C':  ['EEG033', 'EEG044', 'EEG045', 'EEG059'],
        '21C': ['EEG003', 'EEG010', 'EEG012', 'EEG019', 'EEG007', 'EEG030', 'EEG029', 
                'EEG039', 'EEG024', 'EEG046', 'EEG042', 'EEG059', 'EEG064', 'EEG062'],
        '22C': [],#really bad eeg
        '23C':['EEG018', 'EEG025', 'EEG027', 'EEG037'],
        '24C':['EEG001', 'EEG013', 'EEG017', 'EEG027', 'EEG022', 'EEG028', 'EEG029', 'EEG047'],
        '25C':['EEG013', 'EEG002', 'EEG001', 'EEG023', 'EEG026', 'EEG027', 'EEG028'],#two first seconds bad
        '26C':['EEG018', 'EEG034', 'EEG035', 'EEG037', 'EEG042', 'EEG043', 'EEG048', 'EEG050', 'EEG049'],
        '27C':['EEG003', 'EEG033'],
        '28C':['EEG019'],
        '29C':['EEG023', 'EEG033'],
        '30C':[],
        '31C':['EEG001', 'EEG002', 'EEG017', 'EEG022', 'EEG023', 'EEG026', 'EEG027', 
               'EEG028', 'EEG032', 'EEG050'],
        '32C':['EEG003'],
        '33C':['EEG001', 'EEG003', 'EEG013', 'EEG020', 'EEG023', 'EEG026', 'EEG027', 
               'EEG028', 'EEG022', 'EEG056'],
        '34C':[],
        '35C':['EEG013', 'EEG034'],#eog, ecg wrong labels
        '36C':['EEG003', 'EEG006'],
        '37C':['EEG005', 'EEG013', 'EEG017', 'EEG002', 'EEG022', 'EEG023', 'EEG027', 'EEG028'],
        '38C':[],
        '39C':[],
        '40C':[],
        '41C':['EEG014'],
        '01P':[],
        '02P':['EEG013', 'EEG017', 'EEG022', 'EEG027'],
        '03P':['EEG003', 'EEG013', 'EEG032', 'EEG022', 'EEG023', 'EEG027'],
        '04P':['EEG018'],
        '05P':[],
        '06P':['EEG013', 'EEG012', 'EEG022', 'EEG023', 'EEG026', 'EEG032'],
        '07P':[],
        '08P':['EEG027', 'EEG042'],#karvaisia kanavia
        '09P':['EEG009', 'EEG018', 'EEG035'],
        '10P':[], #a lot of eye movement and heart artifacts
        '11P':[],
        '12P':[],
        '13P': ['EEG038'],
        '14P':['EEG018', 'EEG027', 'EEG043', 'EEG048'],#a lot of eye artefacts
        '15P':['EEG044', 'EEG056', 'EEG059', 'EEG060', 'EEG063'],
        '16P':[],
        '17P':['EEG017'],#karvaisia kanavia
        '18P':['EEG020', 'EEG017', 'EEG026', 'EEG028', 'EEG022', 'EEG047'],
        '19P':['EEG015'],
        '20P':['EEG014', 'EEG061'],
        '21P':[],
        '22P':['EEG017', 'EEG019', 'EEG020', 'EEG022', 'EEG028', 'EEG048'],
        '23P':[],
        '24P':['EEG023', 'EEG033'],#a lot of eye artefacts
        '25P':['EEG003', 'EEG023', 'EEG033'],
        '26P':['EEG003'],
        '27P':['EEG027', 'EEG056', 'EEG064', 'EEG061'],
        '28P':['EEG003', 'EEG007', 'EEG028'],
        '29P':['EEG001', 'EEG020', 'EEG005', 'EEG019', 'EEG022', 'EEG023', 'EEG026', 'EEG027'],
        '30P':[],
        '31P':['EEG003', 'EEG011'],
    }
###############################################################################
# Templates for filenames
#
# This part of the config file uses the FileNames class. It provides a small
# wrapper around string.format() to keep track of a list of filenames.
# See fnames.py for details on how this class works.
fname = FileNames()

# Some directories
fname.add('raw_data_dir', raw_data_dir)
fname.add('processed_data_dir', processed_data_dir)

# Continuous data
fname.add('raw', '{raw_data_dir}/sub-{subject}/ses-01/meg/sub-{subject}_ses-01_task-{task}_run-0{run}_proc-raw_meg.fif')
fname.add('filt', '{processed_data_dir}/sub-{subject}/ses-01/eeg/sub-{subject}_ses-01_task-{task}_run-0{run}_filt.fif')
fname.add('clean', '{processed_data_dir}/sub-{subject}/ses-01/eeg/sub-{subject}_ses-01_task-{task}_run-0{run}_clean.fif')

# Maxfilter
fname.add('tsss', '{raw_data_dir}/sub-{subject}/ses-01/meg/sub-{subject}_ses-01_task-{task}_run-0{run}_proc-raw_meg_mc_tsss.fif')
fname.add('pos', '{raw_data_dir}/sub-{subject}/ses-01/meg/sub-{subject}_ses-01_task-{task}_run-0{run}_movecomp.pos') 
fname.add('tsss_log', '{raw_data_dir}/sub-{subject}/ses-01/meg/sub-{subject}_ses-01_task-{task}_run-0{run}_tsss_log.log')


# Files used during EOG and ECG artifact suppression
fname.add('ica', '{processed_data_dir}/sub-{subject}/ses-01/eeg/sub-{subject}_ses-01_task-{task}_run-0{run}_ica.h5')

# PSD files
fname.add('psds', '{processed_data_dir}/sub-{subject}/ses-01/eeg/sub-{subject}_psds.h5')

# Band power files
fname.add('bandpower', '{processed_data_dir}/sub-{subject}/ses-01/eeg/sub-{subject}_bandpower.csv')

# Filenames for MNE reports
fname.add('reports_dir', f'{reports_dir}')
fname.add('report', '{reports_dir}/sub-{subject}-report.h5')
fname.add('report_html', '{reports_dir}/sub-{subject}-report.html')

# Filenames for figures
fname.add('figures_dir', f'{figures_dir}')
fname.add('figure_psds', '{figures_dir}/psds.pdf')


def get_all_fnames(subject, kind, exclude=None):
    """Get all filenames for a given subject of a given kind.

    Not all subjects have exactly the same files. For example, subject 1 does
    not have an emptyroom recording, while subject 4 has 2 runs of emptyroom.
    Use this function to get a list of the the files that are present for a
    given subject. It will check which raw files there are and based on that
    will generate a list with corresponding filenames of the given kind.

    You can exclude the recordings for one or more tasks with the ``exclude``
    parameter. For example, to skip the emptyroom recordings, set
    ``exclude='emptyroom'``.

    Parameters
    ----------
    subject : str
        The subject to get the names of the raw files for.
    kind : 'raw' | 'tsss' | 'filt' | 'eog_ecg_events' | 'ica'
        The kind of files to return the filenames for.
    exclude : None | str | list of str
        The tasks to exclude from the list.
        Defaults to not excluding anything.

    Returns
    -------
    all_fnames : list of str
        The names of the files of the given kind.
    """
    import os.path as op

    if exclude is None:
        exclude = []
    elif type(exclude) == str:
        exclude = [exclude]
    elif type(exclude) != list:
        raise TypeError('The `exclude` parameter should be None, str or list')

    all_fnames = list()
    #print('Looking for: ' + str(fname.raw(subject=subject)))
    for task in tasks:
        if task in exclude:
            continue
        for run in [1, 2]:
            if op.exists(fname.raw(subject=subject, task=task, run=run)):
                all_fnames.append(fname.files()[f'{kind}'](subject=subject, task=task, run=run))
    return all_fnames


def task_from_fname(fname):
    """Extract task name from a BIDS filename."""
    import re
    match = re.search(r'task-([^_]+)_run-(\d\d)', str(fname))
    task = match.group(1)
    run = int(match.group(2))
    if task == 'PASAT':
        return f'{task}_run{run}'
    else:
        return task
