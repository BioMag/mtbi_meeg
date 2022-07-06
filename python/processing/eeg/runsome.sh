#!/bin/bash

for sub in 10C 15P 31P 36C 37C
do
  python3 00_maxfilter.py $sub
done
