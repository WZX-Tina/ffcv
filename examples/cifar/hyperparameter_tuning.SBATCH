#!/bin/bash

# SLURM script configuration
#SBATCH --job-name=my_job
#SBATCH --output=output_%A_%a.txt
#SBATCH --array=1-3
#SBATCH --time=00:10:00

# Load necessary modules (if needed)
# module load python

# Set the paths to the Python scripts and files
read_p_script="read_p.py"
write_datasets_script="write_datasets_100.py"
train_cifar_script="train_cifar_100.py"
existing_param_file="existing_param.txt"
temp_config_file="temp_config.yaml"

# Generate a unique config file for each iteration
python $read_p_script $existing_param_file $temp_config_file

# Run write_datasets_100.py using the generated config file
python $write_datasets_script --config-file $temp_config_file

# Run train_cifar_100.py using the same config file
python $train_cifar_script --config-file $temp_config_file
