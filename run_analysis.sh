#!/bin/bash
### General options
### -- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J run_analysis_test
### -- ask for number of cores (default: 1) -- minimum 4 per gpu
#BSUB -n 4
### -- Select the resources: 1 GPU in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm -- maximum 24 hours for GPU queues
#BSUB -W 12:00
### -- request 4GB / 16GB of system-memory --
#BSUB -R "rusage[mem=8GB]"
### -- set the email address --
##BSUB -u s223730@dtu.dk
### -- send notification at start --#BSUB -B
### -- send notification at completion --#BSUB -N
### -- Specify the output and error file. %J is the job-id --
#BSUB -o output_%J.out
#BSUB -e error_%J.err
# -- end of LSF options --

# Load necessary modules
# module unload gcc
# module load gcc/10.3.0
module load python3/3.10.14
module load cuda

# Activate your virtual environment if needed
source test_env/bin/activate

# Start nvidia-smi logging in the background
#nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv -l 1 > nvidia_smi_log_${LSB_JOBID}.txt &
# Save the PID of the background nvidia-smi process
#NVSMI_PID=$!

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Choose which prompt template to use:
# Options: "YES_NO_COT_PROMPT_TEMPLATE" or "YES_NO_BASE_PROMPT_TEMPLATE"
export PROMPT_TEMPLATE="YES_NO_BASE_PROMPT_TEMPLATE"

# export MODEL_ID="microsoft/Phi-3.5-mini-instruct"
export MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
# export MODEL_ID="meta-llama/Llama-3.2-3B-Instruct"
# export MODEL_ID="meta-llama/Llama-3.2-1B-Instruct"
# Set SAMPLE to "true" to enable sampling, or leave unset/false to process full data
export SAMPLE=true  # Set to true or false

# Check if MODEL_ID is set
if [ -z "$MODEL_ID" ]; then
    echo "MODEL_ID is not set. Please set the MODEL_ID environment variable before submitting the job."
    #kill $NVSMI_PID
    exit 1
fi

echo "Using model: $MODEL_ID"

# Determine if the --sample flag should be included
if [ "$SAMPLE" = "true" ]; then
    SAMPLE_FLAG="--sample"
    echo "Sampling enabled: Processing 100 unique Prospectus IDs."
else
    SAMPLE_FLAG=""
    echo "Sampling disabled: Processing the full dataset."
fi

# python scripts/analysis/run_analysis.py --model_id "$MODEL_ID"
# Run your Python script with the model_id and prompt_template arguments
python scripts/analysis/run_analysis_sampled.py --model_id "$MODEL_ID" --prompt_template "$PROMPT_TEMPLATE" $SAMPLE_FLAG

# After the script completes, kill the nvidia-smi logging process
#kill $NVSMI_PID