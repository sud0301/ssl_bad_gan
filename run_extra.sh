#!/bin/bash

#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l mem=8000mb
#PBS -l walltime=20:00:00
#PBS -e myprog.err
#PBS -o myprog.out
#PBS -q default-gpu
source ~/.bashrc
workon apple

cd /misc/lmbraid19/mittal/ssl_baselines/forked/ssl_bad_gan
python pr2_trainer.py >> output_35k_filtered_cups.txt
