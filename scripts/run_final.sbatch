#!/bin/bash

#SBATCH --job-name=GPU_LSTM_JOB
#SBATCH --time=2-00:00:00
#SBATCH --qos=1week
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=pascal     # or pascal / k80
#SBATCH --gres=gpu:1        # --gres=gpu:2 for two GPU, aso.
#SBATCH --mail-user=normand.overney@stud.unibas.ch 
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --output=generate_samples_take8.txt

module load CUDA
source $HOME/venv-pytorch/bin/activate 
cd $HOME/EEG_Dev
python -u generate_samples.py
