#!/bin/bash

#run in the same folder as subjects.txt!
while read LINE; do
    echo ${LINE}
    python3 01_freqfilt.py $LINE
done < subjects.txt


