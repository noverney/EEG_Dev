#!/bin/bash

#SBATCH --job-name=GPU_VANILLA
#SBATCH --time=7-00:00:00
#SBATCH --qos=1week
#SBATCH --cpus-per-task=4   #This is the number of cores reserved
#SBATCH --mem-per-cpu=8G    #This is the memory reserved per core.
#Total memory reserved: 32GB4
#SBATCH --partition=pascal     # or pascal / k80
#SBATCH --gres=gpu:1        # --gres=gpu:2 for two GPU, aso.
#SBATCH --mail-user=normand.overney@stud.unibas.ch 
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --output=run_vanilla_15_dropout_0,05.txt

module load CUDA
source $HOME/venv-pytorch/bin/activate 
cd $HOME/EEG_Dev
python -u run.py
