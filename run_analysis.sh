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

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Choose which prompt template to use # YES_NO_COT_PROMPT_TEMPLATE or # YES_NO_BASE_PROMPT_TEMPLATE or YES_NO_FEW_SHOT_PROMPT_TEMPLATE
export PROMPT_TEMPLATE="YES_NO_BASE_PROMPT_TEMPLATE"
# export PROMPT_TEMPLATE="ENHANCED_PROMPT_TEMPLATE"
# export PROMPT_TEMPLATE="YES_NO_COT_PROMPT_TEMPLATE"
# export PROMPT_TEMPLATE="YES_NO_FEW_SHOT_PROMPT_TEMPLATE"

# Example local gguf model path
# export MODEL_ID="../huggingface/gguf_folder/Llama-3.2-3B-Instruct-Q8_0.gguf"
export MODEL_ID="../huggingface/gguf_folder/phi-4-Q4_K_M.gguf"
# export MODEL_ID="../huggingface/gguf_folder/Llama-3.2-1B-Instruct-Q8_0.gguf"
# export MODEL_ID="../huggingface/gguf_folder/mistral-7b-instruct-v0.2.Q8_0.gguf"



# Enable sampling or not
export SAMPLE=true  

# Check if MODEL_ID is set
if [ -z "$MODEL_ID" ]; then
    echo "MODEL_ID is not set. Please set the MODEL_ID environment variable."
    exit 1
fi

echo "Using llama-cpp model: $MODEL_ID"

# Determine if the --sample flag should be included
if [ "$SAMPLE" = "true" ]; then
    SAMPLE_FLAG="--sample"
    echo "Sampling enabled: Processing 100 unique Prospectus IDs."
else
    SAMPLE_FLAG=""
    echo "Sampling disabled: Processing the full dataset."
fi

# Run the updated script
python scripts/analysis/run_analysis_cpp.py \
  --model_id "$MODEL_ID" \
  --prompt_template "$PROMPT_TEMPLATE" \
  $SAMPLE_FLAG