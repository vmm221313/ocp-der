#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -q regular
#SBATCH -J der-main
#SBATCH --mail-user=varun.m.iitkgp@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -t 04:00:00
#SBATCH -A m4140_g
#SBATCH -o /global/homes/v/varunvm/ocp/slurm-logs-all/slurm-%x-%j.out

# OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=true

# Experiment logs 
CONFIG_PATH="configs/is2re/all/cgcnn-evidential/cgcnn-evidential.yml"
PREV_TIMESTAMP="2022-08-06-07-32-32"
OUTPUT_PATH="slurm-logs-all/slurm-der-main-$SLURM_JOBID.out"
LOG_DIR="der-all-logs/"
MODE="train"

# module load python
# conda activate /global/homes/v/varunvm/.conda/envs/ocp-models

echo $CONDA_DEFAULT_ENV
echo $CONDA_PREFIX

# run the application:
# srun python main.py --mode train --config-yml $CONFIG_PATH
# srun -n 4 -c 32 --cpu_bind=cores -G 4 --gpu-bind=single:1 python main.py --mode train --config-yml $CONFIG_PATH
# srun -n 4 -c 32 --cpu_bind=cores -G 4 --gpu-bind=single:1 python -u -m torch.distributed.launch --nproc_per_node=8 main.py --distributed --num-gpus 4 --mode train --config-yml $CONFIG_PATH

# run the below command for distributed training 
srun python -u -m torch.distributed.launch --nproc_per_node=4 main.py --distributed --num-gpus 4 --mode train --config-yml $CONFIG_PATH --checkpoint checkpoints/$PREV_TIMESTAMP/checkpoint.pt
srun python scripts/save_experiment_details.py --output_path $OUTPUT_PATH --log_dir $LOG_DIR --job_id $SLURM_JOBID --mode $MODE