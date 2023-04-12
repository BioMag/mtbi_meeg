#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 12:20:55 2023

@author: portae1
"""

import os.path
import datetime

def is_file_recent(file_path, time_threshold):
    """
    Checks if a file exists and was modified within the specified time threshold.

    Parameters:
    file_path (str): The path to the file to check.
    time_threshold (float): The time threshold in seconds.

    Returns:
    bool: True if the file exists and was modified within the time threshold, False otherwise.
    """
    if not os.path.exists(file_path):
        return False

    time_diff = datetime.datetime.now() - datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
    
    return time_diff.total_seconds() <= time_threshold

if is_file_recent("path/to/file.txt", time_threshold=1):
    # do something with the file
    pass
else:
    # file doesn't exist or was not modified recently
    pass

