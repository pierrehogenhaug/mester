import os
import sys
import re
import json
import argparse
import pandas as pd

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.evaluation.evaluate_llm_intra_annotator import main

if __name__ == "__main__":
    main()