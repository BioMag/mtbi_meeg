#!/bin/bash


while read LINE; do
    #echo ${LINE}
    python3 04_bandpower.py $LINE
done < subjects.txt


