#!/bin/sh
# Grid Engine options (lines prefixed with #$)         
#$ -wd /exports/eddie/scratch/s2591023/egnn_halographnet_mlp_egnn_vel_128            
#$ -l h_rt=05:00:00 
#$ -l h_vmem=20G
# Request one GPU in the gpu queue:
#$ -q gpu
#$ -pe gpu-a100 1
#  These options are:
#  job name: -N
#  use the current working directory: -cwd
#  runtime limit of 5 minutes: -l h_rt
#  memory limit of 1 Gbyte: -l h_vmem
# Initialise the environment modules
. /etc/profile.d/modules.sh

# Load Python
module load anaconda
conda activate mlp

# Initialise the environment modules and load CUDA version 11.0.2
. /etc/profile.d/modules.sh
module load cuda

# Run the program
source /exports/applications/support/set_cuda_visible_devices.sh

## If the following gives you nothing then something has gone wrong and you haven't been allocated a GPU
echo $CUDA_VISIBLE_DEVICES

## find out where the CUDA code actually is (nvcc is the CUDA compiler which you won't need to use directly): 
which nvcc 
python main_TNG_LH.py
