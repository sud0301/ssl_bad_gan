#!/bin/bash

#PBS -l nodes=1:ppn=2:gpus=1
#PBS -l mem=8000mb
#PBS -l walltime=24:00:00
#PBS -e myprog.err
#PBS -o myprog.out
#PBS -q default-cpu
source ~/.bashrc
workon apple

cd /misc/lmbraid19/mittal/ssl_baselines/forked/ssl_bad_gan
python pr2_trainer.py >> output_20180202_1049_clean.txt