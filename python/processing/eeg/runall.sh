#!/bin/bash


while read LINE; do
    #echo ${LINE}
    python3 00_maxfilter.py $LINE
done < subjects_test.txt


