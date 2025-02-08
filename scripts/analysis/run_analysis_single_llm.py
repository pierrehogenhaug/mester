import os
import sys

# Make sure your local paths are set up properly:
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.analysis.analysis_single_llm import main

"""
Run Guides for LLM Detection Script

This script processes CSV files using an LLM. You can customize its behavior
with various command-line flags.

Usage Examples:

1. Run with the default OpenAI model:
   $ python run_llm_detection.py --model_type openai

2. Run with a local LLM model:
   $ python run_llm_detection.py --model_type local --local_model_path /path/to/local/model

3. Disable sampling (process all CSVs instead of a sample of 40):
   $ python run_llm_detection.py --sample 0


Note:
- The default model type is "openai". To switch to a local model, set --model_type local and provide a valid --local_model_path.
- The --sample flag controls how many unique Prospectus IDs to process (default is 40; use 0 to disable sampling).
"""


if __name__ == "__main__":
    main()