#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 12:39:16 2023

@author: portae1
"""

import os

def count_lines_of_code(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    count_code = 0
    count_comments = 0
    count_docstrings = 0
    in_multiline_string = False
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith('"""') or stripped_line.startswith("'''"):
            if not in_multiline_string:
                count_docstrings += 1
            in_multiline_string = not in_multiline_string
        elif stripped_line.startswith('#') or in_multiline_string:
            count_comments += 1
        elif stripped_line:
            count_code += 1
    return count_code, count_comments, count_docstrings
# Example usage:
file_path = ['01_read_processed_data.py', '02_plot_processed_data.py', '03_fit_classifier_and_plot.py', '04_create_report.py']
num_code_lines, num_comments, num_docstrings = count_lines_of_code(file_path[2])
print(f'Number of lines of code: {num_code_lines}')
print(f'Number of comments: {num_comments}')
print(f'Number of docstrings: {num_docstrings}')