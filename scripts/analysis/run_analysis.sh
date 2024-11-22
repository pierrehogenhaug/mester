#!/bin/bash
#SBATCH --job-name=run_analysis
#SBATCH --output=run_analysis_%j.log
#SBATCH --error=run_analysis_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Load necessary modules (adjust according to your HPC environment)
module load python/3.8

# Activate virtual environment if needed
# source /path/to/your/venv/bin/activate

# Navigate to the project directory
cd $SLURM_SUBMIT_DIR

# Run the analysis script
python scripts/analysis/run_analysis.py