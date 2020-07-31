#!/bin/bash

#SBATCH --job-name=test_seq
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=4  #This is the number of cores reserved
#SBATCH --mem-per-cpu=4G   #This is the memory reserved per core.
#Total memory reserved: 16GBG
#SBATCH --qos=1week
#SBATCH --mail-user=normand.overney@stud.unibas.ch 
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --output=run_all_seq.txt

module load CUDA
source $HOME/venv-pytorch/bin/activate 
cd $HOME/EEG_Dev
python -u run_all_seq.py
