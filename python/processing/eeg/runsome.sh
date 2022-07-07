#!/bin/bash

<<<<<<< HEAD
for sub in 10C 15P 31P 36C 37C
do
  python3 00_maxfilter.py $sub
done
=======

while read LINE; do
    #echo ${LINE}
    python3 03_psds.py $LINE
done < some_subjects.txt
>>>>>>> 78e40629578456819550f19cd18d0c3931d03722
