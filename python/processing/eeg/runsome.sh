                   
#!/bin/bash


while read LINE; do
    #echo ${LINE}
    python3 ica_without_ecg_ch.py $LINE
done < subjects_without_ecg.txt
