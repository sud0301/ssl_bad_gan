#!/bin/bash

#PBS -l nodes=1:ppn=1:gpus=1:nvidiaTITANX
#PBS -l mem=20gb
#PBS -l walltime=24:00:00
#PBS -e myprog.err
#PBS -o myprog.out
#PBS -q default-gpu
source ~/.bashrc
workon apple

cd /misc/lmbraid19/mittal/ssl_baselines/forked/ssl_bad_gan
python pr2_trainer_224x224.py >> ./records/output_badGAN_224x224_tr_1_te_1_25k_400_7_n128_cont2.txt

