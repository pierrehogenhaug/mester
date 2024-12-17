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
#BSUB -W 0:15
### -- request 4GB / 16GB of system-memory -- #BSUB -R "select[avx512]"
#BSUB -R "rusage[mem=4GB]"
### -- Specify the output and error file. %J is the job-id --
#BSUB -o output_%J.out
#BSUB -e error_%J.err
# -- end of LSF options --

# Load necessary modules
module load python3/3.10.15
module load cuda/12.3.2 || { echo "Failed to load CUDA module"; exit 1; }

# Activate your virtual environment if needed
source llamacpp_test/bin/activate

# export MODEL_PATH="./data/gguf_folder/Meta-Llama-3-8B-Instruct.Q8_0.gguf"
# export MODEL_PATH="./data/gguf_folder/llama-2-7b.Q4_0.gguf"
# export MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
# export MODEL_ID="meta-llama/Llama-3.2-3B-Instruct"

# Check if MODEL_PATH is set
# if [ -z "$MODEL_PATH" ]; then
#     echo "MODEL_PATH is not set. Please set the MODEL_PATH environment variable before submitting the job."
#     exit 1
# fi

# echo "Using model: $MODEL_PATH"

# Run your Python script with the model_id argument
python scripts/analysis/run_simple_analysis_llamacpp.py
