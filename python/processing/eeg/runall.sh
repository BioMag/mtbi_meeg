#!/bin/bash


while read LINE; do
    #echo ${LINE}
    python3 01_freqfilt.py $LINE
done < subjects.txt


