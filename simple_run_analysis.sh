#!/bin/bash
### -- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J run_analysis_test
### -- ask for number of cores (default: 1) -- minimum 4 per gpu
#BSUB -n 4
### -- Select the resources: 1 GPU in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm -- maximum 24 hours for GPU queues
#BSUB -W 10:00
### -- request 8GB of system-memory --
#BSUB -R "rusage[mem=1GB]"
### -- Specify the output and error file. %J is the job-id --
#BSUB -o output_%J.out
#BSUB -e error_%J.err
# -- end of LSF options --

module load python3/3.10.14
module load cuda

# Activate virtual environment
source ../llama_env/bin/activate

# Run the updated script
python scripts/analysis/run_analysis_cpp.py