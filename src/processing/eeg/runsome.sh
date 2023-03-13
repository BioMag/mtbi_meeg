#!/bin/bash
for sub in 10C 15P 31P 36C
do
  python3 01_freqfilt.py $sub
done
