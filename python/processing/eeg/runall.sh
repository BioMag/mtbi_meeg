#!/bin/bash


while read LINE; do
    #echo ${LINE}
    python3 02_ica.py $LINE
    python3 03_psds.py $LINE
    python3 04_bandpower.py $LINE
done < subjects.txt


