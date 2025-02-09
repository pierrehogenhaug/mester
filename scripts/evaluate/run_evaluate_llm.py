import os
import sys
import re
import json
import argparse
import pandas as pd

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.evaluation.evaluate_llm import evaluate_models

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Pass all CSV file paths as a list to the module function
        evaluate_models(sys.argv[1:])
    else:
        print("Usage: python run_evaluation.py <processed_csv_path1> [<processed_csv_path2> ...]")