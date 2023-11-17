#!/bin/bash
#SBATCH--nodes=1
#SBATCH--ntasks-per-node=1
#SBATCH--cpus-per-task=12
#SBATCH--time=20:00:00
#SBATCH--mem=128GB
#SBATCH--gres=gpu
#SBATCH--job-name=128GB
#SBATCH --array=1-64

DELAY=$(($SLURM_ARRAY_TASK_ID * 60))
sleep $DELAY

SEED=$1
CONFIG="config/temp_config-${SEED}-${SLURM_ARRAY_TASK_ID}.yaml"

module purge
singularity exec --nv\
    --overlay ffcv.ext3:ro\
    /scratch/work/public/singularity/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif\
    /bin/bash -c "source /ext3/env.sh; \
     conda activate ffcv; \
     cd /scratch/zw3720/ffcv/examples/cifar;\

python train_cifar_100.py --config-file $CONFIG;"
