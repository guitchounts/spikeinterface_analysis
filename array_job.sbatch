#!/bin/bash

#SBATCH --job-name=dgmou32_101921_7
#SBATCH --output=/n/groups/datta/guitchounts/ephys_results/spike_sorting_results/gmou32_101921/logs/dgmou32_101921_%a_7.out
#SBATCH --error=/n/groups/datta/guitchounts/ephys_results/spike_sorting_results/gmou32_101921/logs/dgmou32_101921_%a_7.err
#SBATCH -t 0-6:00
#SBATCH -p short
#SBATCH -c 8
#SBATCH -N 1
#SBATCH --mem=32000

cd /home/gg121/code/spikeinterface_analysis
conda activate spikeinterface 
python run_sorting.py -paramfile params_waveform_save_gmou32_101921.txt -line ${SLURM_ARRAY_TASK_ID}
