#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -q regular
#SBATCH -J der_code_eval
#SBATCH --mail-user=varun.m.iitkgp@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -t 01:00:00
#SBATCH -A m4140_g

# OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=true

# Experiment logs
CONFIG_PATH="configs/is2re/10k/cgcnn-evidential/cgcnn-evidential.yml"
LOG_DIR="der_code_eval_logs/"

# run the application:
srun python scripts/save_experiment_details.py --config_path $CONFIG_PATH --log_dir $LOG_DIR --job_id $SLURM_JOBID
srun -n 4 -c 32 --cpu_bind=cores -G 4 --gpu-bind=single:1 python -u -m torch.distributed.launch --nproc_per_node=8 main.py --distributed --num-gpus 4 --mode train --config-yml $CONFIG_PATH