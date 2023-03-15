#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 15:00:27 2023
Creates an HTML report with the images created in the previous step of the pipeline
@author: portae1
"""

# HTML report
report = open('report.html', 'w')
report.write('''
<!DOCTYPE html>
<html>
<head>
	<title>mTBI-EEG - Analysis</title>
</head>
<body>
	<h1>My Report</h1>
	<p>Here are my figures:</p>
	<img src="output_papa.png">
	<img src="figure2.png">
</body>
</html>
''')
report.close()