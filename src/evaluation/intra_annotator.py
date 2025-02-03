"""
This script walks through the processed dataset, finds groups of CSV files
that follow the pattern:
    <BASE>_parsed.csv, <BASE>_parsed_1.csv, and <BASE>_parsed_2.csv
for each BASE, and then computes intra‐annotator agreement (Fleiss’ Kappa)
for each question column.
Usage:
    python run_intra_annotator_all.py --processed_root /path/to/data/processed \
        [--question_cols "Market Dynamics - a" "Intra-Industry Competition - a" ...]
"""

import argparse
import json
import os
import pandas as pd
from typing import List, Optional
from tqdm import tqdm
import numpy as np

# Statsmodels for Fleiss' Kappa
from statsmodels.stats.inter_rater import fleiss_kappa


# ----------------------------
# Helper: Group CSV Files
# ----------------------------
def get_csv_triples(processed_root: str) -> dict:
    """
    Walk the processed_root directory (only in folders containing 'as_expected')
    and group CSV files by their base name.
    We look for files matching:
         <BASE>_parsed.csv
         <BASE>_parsed_1.csv
         <BASE>_parsed_2.csv
    and return a dictionary keyed by BASE with a dict mapping suffix -> full file path.
    Only groups having all three files are returned.
    """
    csv_groups = {}
    for root, dirs, files in os.walk(processed_root):
        if "as_expected" not in root:
            continue
        for file in files:
            # Match file names that end with _parsed.csv, _parsed_1.csv, or _parsed_2.csv
            m = re.match(r"(.+)_parsed(?:_(\d+))?\.csv$", file, flags=re.IGNORECASE)
            if not m:
                continue
            base = m.group(1)
            suffix = m.group(2)
            suffix = int(suffix) if suffix is not None else 0  # default _parsed.csv => suffix 0
            full_path = os.path.join(root, file)
            if base not in csv_groups:
                csv_groups[base] = {}
            csv_groups[base][suffix] = full_path

    # Only keep groups that have all three expected files (suffix 0, 1, and 2)
    triple_groups = {base: paths for base, paths in csv_groups.items() if all(s in paths for s in [0, 1, 2])}
    return triple_groups


# def parse_arguments():
#     parser = argparse.ArgumentParser(
#         description="Compute Fleiss' Kappa for LLM intra-annotator agreement."
#     )
#     parser.add_argument("--csv1", required=True, help="Path to first CSV (_parsed.csv)")
#     parser.add_argument("--csv2", required=True, help="Path to second CSV (_parsed_1.csv)")
#     parser.add_argument("--csv3", required=True, help="Path to third CSV (_parsed_2.csv)")
#     parser.add_argument(
#         "--question_cols",
#         nargs="+",
#         default=[
#             "Market Dynamics - a",
#             "Intra-Industry Competition - a",
#             "Technology Risk - a",
#             "Regulatory Framework - a",
#         ],
#         help=(
#             "List of question columns to evaluate. Defaults to the four typical columns "
#             "from your run_detection.py example."
#         ),
#     )
#     return parser.parse_args()

def extract_detection_answer(cell_str: str) -> Optional[str]:
    """
    Extract the 'detection_step_parsed' -> 'answer' (Yes/No) from the JSON string 
    stored in each CSV cell. 
    If it doesn't parse or is missing, return None.
    """
    if not cell_str.strip():
        return None
    try:
        data = json.loads(cell_str)
    except json.JSONDecodeError:
        return None
    
    # The detection script typically stores "answer" at the top level 
    # or inside "detection_step_parsed" => "answer". 
    # By default, run_detection.py puts "answer" at top-level
    # AND also inside detection_step_parsed["answer"].
    # We'll try the top-level "answer" first:
    ans = data.get("answer", None)
    if isinstance(ans, str) and ans in ("Yes", "No"):
        return ans

    # If we need to parse from detection_step_parsed:
    det_parsed = data.get("detection_step_parsed", {})
    ans2 = det_parsed.get("answer", None)
    if isinstance(ans2, str) and ans2 in ("Yes", "No"):
        return ans2
    
    # Otherwise no valid answer
    return None

def extract_evaluation_answer(cell_str: str) -> Optional[str]:
    """
    Extract the 'evaluation_result.answer' (Yes/No) from the JSON string.
    In run_detection.py, that is stored in:
       detection_dict["evaluation_answer"]  (top-level)
    or inside "evaluation_step_parsed" => "answer".
    """
    if not cell_str.strip():
        return None
    try:
        data = json.loads(cell_str)
    except json.JSONDecodeError:
        return None

    # The script typically sets "evaluation_answer" at top-level:
    ans = data.get("evaluation_answer", None)
    if isinstance(ans, str) and ans in ("Yes", "No"):
        return ans

    # Alternatively, we can parse from "evaluation_step_parsed":
    eval_parsed = data.get("evaluation_step_parsed", {})
    ans2 = eval_parsed.get("answer", None)
    if isinstance(ans2, str) and ans2 in ("Yes", "No"):
        return ans2

    return None

def make_fleiss_kappa_table(ratings: List[str]) -> List[int]:
    """
    Given a list of rating labels from multiple annotators for a single item,
    e.g. ["Yes", "Yes", "No"], 
    returns [count_of_No, count_of_Yes] for that item 
    so it can be used in fleiss_kappa with 2 categories (No/Yes).
    The order of categories is up to you, but keep it consistent.
    
    We'll do: index 0 => "No", 1 => "Yes"
    """
    n_yes = sum(r == "Yes" for r in ratings if r is not None)
    n_no = sum(r == "No" for r in ratings if r is not None)
    return [n_no, n_yes]

def compute_fleiss_kappa_for_column(
    df0: pd.DataFrame,
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    column_name: str,
    mode: str = "detection"
) -> float:
    """
    Compute Fleiss' Kappa for a single question column across the three dataframes.
      - mode='detection' => use extract_detection_answer
      - mode='evaluation' => use extract_evaluation_answer
    """
    # Choose the extractor function
    if mode == "detection":
        extractor = extract_detection_answer
    else:
        extractor = extract_evaluation_answer

    # Build the table of shape (N_items, 2_categories)
    # We'll skip any row that does not have valid answers in all 3 CSVs 
    # or at least, we skip if we want to enforce that all raters produce a rating.
    # Alternatively, you can decide to skip only rows that fail for any CSV. 
    # Here we assume "If any rater is missing, skip that row."
    table = []
    n_rows = len(df0)
    for i in range(n_rows):
        cell1 = str(df0.at[i, column_name]) if column_name in df0.columns else ""
        cell2 = str(df1.at[i, column_name]) if column_name in df1.columns else ""
        cell3 = str(df2.at[i, column_name]) if column_name in df2.columns else ""

        r1 = extractor(cell1)
        r2 = extractor(cell2)
        r3 = extractor(cell3)

        # If all are valid (Yes/No), we add a row to the table
        if r1 in ("Yes", "No") and r2 in ("Yes", "No") and r3 in ("Yes", "No"):
            row_counts = make_fleiss_kappa_table([r1, r2, r3])
            table.append(row_counts)
        # else skip

    if not table:
        # No valid rows => kappa is not definable
        return float("nan")

    # Convert to np array for statsmodels
    freq_array = np.array(table)

    # Now compute Fleiss Kappa
    kappa_val = fleiss_kappa(freq_array)
    return kappa_val

def main():
    parser = argparse.ArgumentParser(
        description="Compute intra-annotator agreement (Fleiss' Kappa) over all CSV triple groups."
    )
    parser.add_argument(
        "--processed_root",
        required=True,
        help="Path to the processed root folder (e.g., data/processed)"
    )
    parser.add_argument(
        "--question_cols",
        nargs="+",
        default=[
            "Market Dynamics - a",
            "Intra-Industry Competition - a",
            "Technology Risk - a",
            "Regulatory Framework - a",
        ],
        help=(
            "List of question columns to evaluate. Defaults to four example columns."
        )
    )
    args = parser.parse_args()

    triple_groups = get_csv_triples(args.processed_root)
    if not triple_groups:
        print("No CSV triple groups found. Check your processed_root path and naming conventions.")
        return

    # We'll store results per base (group)
    results = {}

    # Process each group (i.e. each base name)
    for base, paths in tqdm(triple_groups.items(), desc="Processing CSV groups"):
        try:
            df0 = pd.read_csv(paths[0], dtype=str)
            df1 = pd.read_csv(paths[1], dtype=str)
            df2 = pd.read_csv(paths[2], dtype=str)
        except Exception as e:
            print(f"Error reading CSV files for group {base}: {e}")
            continue

        group_results = {}
        for col in args.question_cols:
            kappa_detection = compute_fleiss_kappa_for_column(df0, df1, df2, col, mode="detection")
            kappa_evaluation = compute_fleiss_kappa_for_column(df0, df1, df2, col, mode="evaluation")
            group_results[col] = {
                "detection": kappa_detection,
                "evaluation": kappa_evaluation,
            }
        results[base] = group_results

    # Print a summary report for each group
    print("\n---------- Intra-Annotator Agreement (Fleiss’ Kappa) ----------")
    for base, group_results in results.items():
        print(f"\nGroup (Base): {base}")
        for col, kappas in group_results.items():
            print(f"  Column: {col}")
            print(f"    Detection-level Fleiss' Kappa:  {kappas['detection']:.4f}")
            print(f"    Evaluation-level Fleiss' Kappa: {kappas['evaluation']:.4f}")
    print("---------------------------------------------------------------")

if __name__ == "__main__":
    main()