#!/bin/bash
### General options
### -- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J run_analysis_test
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 GPU in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm -- maximum 24 hours for GPU queues
#BSUB -W 1:00
### -- request 16GB of system-memory --
#BSUB -R "rusage[mem=16GB]"
### -- set the email address --
##BSUB -u s223730@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion --
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
#BSUB -o output_%J.out
#BSUB -e error_%J.err
# -- end of LSF options --

# Load necessary modules
module load python3/3.10.7
module load cuda/11.6

# Activate your virtual environment if needed
source source venv/bin/activate

# Start nvidia-smi logging in the background
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv -l 1 > nvidia_smi_log_${LSB_JOBID}.txt &

# Save the PID of the background nvidia-smi process
NVSMI_PID=$!

# Run your Python script
python scripts/analysis/run_analysis.py

# After the script completes, kill the nvidia-smi logging process
kill $NVSMI_PID