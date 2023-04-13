#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bundle up all HTMLs from the `reports_dir` into one PDF naamed 'mtbi_meeg_report.pdf'
Be careful because it will overwrite files with the same name

@author: Estanislao Porta
"""
from weasyprint import HTML, CSS
import os
import sys
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(SRC_DIR)
from config_common import reports_dir, figures_dir

# Get list of HTML files in directory
html_files = [os.path.join(reports_dir, f) for f in os.listdir(reports_dir) if f.endswith('.html')]

# Load all HTML files and concatenate into a single HTML string
html_string = ''
for html_file in html_files:
    with open(html_file, 'r') as f:
        html = f.read()
        if html_string:
            # Add a page break before all HTML files except the first one
            html_string += f'<div style="page-break-before: always;"></div>{html}'
        else:
            html_string += html

# Used to resolve relative paths from the HTMLs
base_url =f'file://{figures_dir}'
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
# Write the html into a PDF
filename='mtbi_meeg_report.pdf'
pdf_document.write_pdf(os.path.join(reports_dir, filename), presentational_hints=True, stylesheets=[CSS(string=css)])
print(f"INFO: Success! All the HTML reports from {reports_dir} have been combined into one PDF named '{filename}'" )