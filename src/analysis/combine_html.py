#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 13:47:09 2023
Bundle up all htmls into one with a certain cover

Then print it as a PDF
@author: portae1
"""
from weasyprint import HTML, CSS
import os
import sys
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_dir)
from config_common import reports_dir

# Get list of HTML files in directory
html_files = [os.path.join(reports_dir, f) for f in os.listdir(reports_dir) if f.endswith('.html')]

# Load all HTML files and concatenate into a single HTML string
html_string = ''
for html_file in html_files:
    with open(html_file, 'r') as f:
        html_string += f.read()

# Combine all the HTML content into a single string
base_url = 'file:///m/home/home2/20/portae1/unix/biomag/mtbi-eeg/src/figures/'
# Initialize a WeasyPrint document
pdf_document = HTML(string=html_string, base_url=base_url)

# Define the CSS styles to apply to the PDF document
css = """
img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
    }
    """
# Set the page size and margins using CSS
pdf_document.write_pdf('mTBI-eeg_normalized_not-scaled.pdf', presentational_hints=True, stylesheets=[CSS(string=css)])