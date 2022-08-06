#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -q regular
#SBATCH -J der-main-predict
#SBATCH --mail-user=varun.m.iitkgp@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -t 00:10:00
#SBATCH -A m4140_g
#SBATCH -o /global/homes/v/varunvm/ocp/slurm-logs-all/slurm-%x-%j.out

# OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=true

# Experiment logs
CONFIG_PATH="configs/is2re/all/cgcnn-evidential/cgcnn-evidential.yml"
TIMESTAMP="2022-08-05-02-40-16"
OUTPUT_PATH="slurm-logs-all/slurm-der-main-predict-$SLURM_JOBID.out"
LOG_DIR="der-all-logs/"
MODE="predict"

# module load python
# conda activate /global/homes/v/varunvm/.conda/envs/ocp-models

echo $CONDA_DEFAULT_ENV
echo $CONDA_PREFIX

# training
# srun python main.py --mode train --config-yml $CONFIG_PATH
# srun -n 4 -c 32 --cpu_bind=cores -G 4 --gpu-bind=single:1 python main.py --mode train --config-yml $CONFIG_PATH
# srun -n 4 -c 32 --cpu_bind=cores -G 4 --gpu-bind=single:1 python -u -m torch.distributed.launch --nproc_per_node=8 main.py --distributed --num-gpus 4 --mode train --config-yml $CONFIG_PATH
# srun python -u -m torch.distributed.launch --nproc_per_node=4 main.py --distributed --num-gpus 4 --mode train --config-yml $CONFIG_PATH
# srun python scripts/save_experiment_details.py --output_path $OUTPUT_PATH --log_dir $LOG_DIR --job_id $SLURM_JOBID

# prediction
srun python main.py --mode predict --config-yml $CONFIG_PATH --checkpoint checkpoints/$TIMESTAMP/checkpoint.pt
srun python scripts/save_experiment_details.py --output_path $OUTPUT_PATH --log_dir $LOG_DIR --job_id $SLURM_JOBID --mode $MODE

# python main.py --mode predict --config-yml configs/is2re/all/cgcnn-evidential/cgcnn-evidential.yml --checkpoint checkpoints/2022-07-23-23-09-04/checkpoint.pt