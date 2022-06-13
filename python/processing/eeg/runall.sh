#!/bin/bash


while read LINE; do
    #echo ${LINE}
    python3 02_ica.py $LINE
done < subjects.txt


