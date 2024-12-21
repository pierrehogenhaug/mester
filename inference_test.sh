#!/bin/bash
### General options
### -- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J run_analysis_test
### -- ask for number of cores
#BSUB -n 4
### -- Select the resources: 1 GPU in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm -- maximum 24 hours for GPU queues
#BSUB -W 1:00
### -- request 16GB of system-memory --
#BSUB -R "rusage[mem=8GB]"
### -- Specify the output and error file. %J is the job-id --
#BSUB -o output_%J.out
#BSUB -e error_%J.err
# -- end of LSF options --

# Load necessary modules
module load python3/3.10.15
module load cuda/12.5
module load cmake
module load make

# Activate your virtual environment if needed
source ../llama_env/bin/activate

# Run your Python script with the model_id argument
python scripts/analysis/run_simple_analysis_llamacpp.py