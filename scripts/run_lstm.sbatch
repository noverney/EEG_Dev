#!/bin/bash

#SBATCH --job-name=GPU_JOB
#SBATCH --time=01:00:00
#SBATCH --qos=6hours
#SBATCH --mem-per-cpu=4G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=pascal     # or pascal / k80
#SBATCH --gres=gpu:1        # --gres=gpu:2 for two GPU, aso.
#SBATCH --mail-user=normand.overney@stud.unibas.ch 
#SBATCH --mail-type=END
#SBATCH --output={output_name}

module load CUDA
source $HOME/venv-pytorch/bin/activate 
cd $HOME/EEG_Dev
python test_lstm.py {filepath} {num_epoches} {hidden_units} {prefix} {random_state} 
