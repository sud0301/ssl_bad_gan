#!/bin/bash

#PBS -l nodes=1:ppn=1:gpus=1:nvidiaTITANX
#PBS -l mem=25gb
#PBS -l walltime=24:00:00
#PBS -e myprog.err
#PBS -o myprog.out
#PBS -q default-gpu
source ~/.bashrc
workon apple

cd /misc/lmbraid19/mittal/ssl_baselines/forked/ssl_bad_gan
python pr2_trainer_vgg.py >> ./records/output_badGAN_224x224_tr_1_te_1_24k_400_13_vgg11_fcn.txt

