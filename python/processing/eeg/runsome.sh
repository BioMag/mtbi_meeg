#!/bin/bash


while read LINE; do
    #echo ${LINE}
    python3 03_psds.py $LINE
done < some_subjects.txt
