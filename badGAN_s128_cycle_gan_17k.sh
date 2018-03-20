#!/bin/bash

#PBS -l nodes=1:ppn=1:gpus=1:nvidiaTITANX
#PBS -l mem=12gb
#PBS -l walltime=24:00:00
#PBS -e myprog.err
#PBS -o myprog.out
#PBS -q default-gpu
source ~/.bashrc
workon apple

cd /misc/lmbraid19/mittal/ssl_baselines/forked/ssl_bad_gan
python pr2_trainer_128x128.py >> ./records/output_badGAN_128x128_tr_1_te_1_17k_400_7.txt


