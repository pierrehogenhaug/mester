import argparse
import json
import os
import re
import pandas as pd
from typing import List, Optional
from tqdm import tqdm
import numpy as np

# Statsmodels for Fleiss' Kappa
from statsmodels.stats.inter_rater import fleiss_kappa

# List of RMS ids to process
SAMPLED_RMS_IDS = [
    '367', '999', '1609', '625', '1108', '673', '219', '440', '328', '355',
    '139', '1629', '1074', '352', '1052', '946', '1897', '317', '653', '642',
    '1525', '1277', '935', '433', '153', '221', '1261', '199', '130', '252',
    '377', '84', '518', '201', '989', '1069', '1727', '1739', '258', '1127',
    '1182', '1096', '311', '1765', '661', '990', '251', '1136', '257', '398'
]


# ----------------------------
# Helper: Group CSV Files
# ----------------------------
def get_csv_triples(processed_root: str) -> dict:
    """
    Walk the processed_root directory and group CSV files by their base name,
    but now using a composite key of (rms_id, base) so that only files
    within the same 'as_expected' subfolder (i.e., same rms_id) are grouped together.
    
    We only process CSV files in directories named "as_expected" whose parent folder name
    is one of the sampled RMS IDs. We look for files matching:
         <BASE>_parsed.csv, <BASE>_parsed_1.csv, <BASE>_parsed_2.csv
    Only groups having all three files are returned.
    """
    csv_groups = {}
    print(f"DEBUG: Starting os.walk on processed_root: {processed_root}")
    for root, dirs, files in os.walk(processed_root):
        # Only process directories whose immediate name is exactly "as_expected"
        if os.path.basename(root) != "as_expected":
            continue

        # Check that the parent folder (presumably the RMS id) is in our sampled list.
        parent_folder = os.path.basename(os.path.dirname(root))
        if parent_folder not in SAMPLED_RMS_IDS:
            # print(f"DEBUG: Skipping directory: {root} (parent folder '{parent_folder}' not in sampled list)")
            continue

        # print(f"DEBUG: Processing directory: {root} with files: {files}")
        for file in files:
            # Match file names that end with _parsed.csv, _parsed_1.csv, or _parsed_2.csv
            m = re.match(r"(.+)_parsed(?:_(\d+))?\.csv$", file, flags=re.IGNORECASE)
            if not m:
                # print(f"DEBUG: File {file} in {root} did not match expected pattern.")
                continue
            base = m.group(1)
            suffix = m.group(2)
            suffix = int(suffix) if suffix is not None else 0  # default _parsed.csv => suffix 0
            full_path = os.path.join(root, file)
            # print(f"DEBUG: Found matching file: {full_path}, base: '{base}', suffix: {suffix}")
            # Create a composite key using the parent folder (rms_id) and the base filename
            group_key = (parent_folder, base)
            if group_key not in csv_groups:
                csv_groups[group_key] = {}
            csv_groups[group_key][suffix] = full_path

    # Only keep groups that have all three expected files (suffix 0, 1, and 2)
    triple_groups = {group_key: paths for group_key, paths in csv_groups.items() if all(s in paths for s in [0, 1, 2])}
    print(f"DEBUG: Found {len(triple_groups)} CSV triple group(s) after filtering for complete groups.")
    return triple_groups


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
    
    # Try top-level "answer" first
    ans = data.get("answer", None)
    if isinstance(ans, str) and ans in ("Yes", "No"):
        return ans

    # If not, try from detection_step_parsed
    det_parsed = data.get("detection_step_parsed", {})
    ans2 = det_parsed.get("answer", None)
    if isinstance(ans2, str) and ans2 in ("Yes", "No"):
        return ans2
    
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

    # Try top-level "evaluation_answer" first
    ans = data.get("evaluation_answer", None)
    if isinstance(ans, str) and ans in ("Yes", "No"):
        return ans

    # Alternatively, try from "evaluation_step_parsed"
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
    We'll do: index 0 => "No", 1 => "Yes"
    """
    n_yes = sum(r == "Yes" for r in ratings if r is not None)
    n_no = sum(r == "No" for r in ratings if r is not None)
    return [n_no, n_yes]


def get_rating_table_for_column(
    df0: pd.DataFrame,
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    column_name: str,
    mode: str = "detection"
) -> List[List[int]]:
    """
    For a given column (and mode) across three DataFrames, iterate through
    the rows, applying the appropriate extractor (detection or evaluation)
    and build the frequency table (one row per item) that will later be passed
    to fleiss_kappa.
    """
    extractor = extract_detection_answer if mode == "detection" else extract_evaluation_answer

    # Use the minimum number of rows to avoid index errors if the CSVs have different lengths
    n_rows = min(len(df0), len(df1), len(df2))
    table = []
    for i in range(n_rows):
        # If all three dataframes contain the "Subsection Title" column, check for a match.
        if "Subsection Title" in df0.columns and "Subsection Title" in df1.columns and "Subsection Title" in df2.columns:
            sub0 = df0.iloc[i]["Subsection Title"]
            sub1 = df1.iloc[i]["Subsection Title"]
            sub2 = df2.iloc[i]["Subsection Title"]
            if not (sub0 == sub1 == sub2):
                # Skip rows that do not match on "Subsection Title"
                continue

        # Extract the cell content for the target question column from each dataframe
        cell1 = str(df0.iloc[i][column_name]) if column_name in df0.columns else ""
        cell2 = str(df1.iloc[i][column_name]) if column_name in df1.columns else ""
        cell3 = str(df2.iloc[i][column_name]) if column_name in df2.columns else ""

        r1 = extractor(cell1)
        r2 = extractor(cell2)
        r3 = extractor(cell3)

        # Only add the row if all three cells have valid answers
        if r1 in ("Yes", "No") and r2 in ("Yes", "No") and r3 in ("Yes", "No"):
            row_counts = make_fleiss_kappa_table([r1, r2, r3])
            table.append(row_counts)
    return table


def compute_fleiss_kappa_for_column(
    df0: pd.DataFrame,
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    column_name: str,
    mode: str = "detection"
) -> float:
    """
    Compute Fleiss' Kappa for a single question column across the three dataframes.
    """
    table = get_rating_table_for_column(df0, df1, df2, column_name, mode)
    if not table:
        return float("nan")
    freq_array = np.array(table)
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
        help="List of question columns to evaluate. Defaults to four example columns."
    )
    args = parser.parse_args()

    print(f"DEBUG: Looking for CSV groups in: {args.processed_root}")
    triple_groups = get_csv_triples(args.processed_root)
    if not triple_groups:
        print("No CSV triple groups found. Check your processed_root path and naming conventions.")
        return

    # Updated: Unpack the composite key (rms_id, base) for clarity.
    print(f"DEBUG: CSV triple groups found: {list(triple_groups.keys())}")
    results = {}

    # Create dictionaries to accumulate frequency tables for aggregated kappa per column.
    aggregated_tables = {col: {"detection": [], "evaluation": []} for col in args.question_cols}

    # ----------------------------
    # Accumulate evaluation and detection answer distributions per annotator (0, 1, 2)
    # ----------------------------
    evaluation_distribution = {
        0: {col: {"Yes": 0, "No": 0} for col in args.question_cols},
        1: {col: {"Yes": 0, "No": 0} for col in args.question_cols},
        2: {col: {"Yes": 0, "No": 0} for col in args.question_cols},
    }
    detection_distribution = {
        0: {col: {"Yes": 0, "No": 0} for col in args.question_cols},
        1: {col: {"Yes": 0, "No": 0} for col in args.question_cols},
        2: {col: {"Yes": 0, "No": 0} for col in args.question_cols},
    }

    for (rms_id, base), paths in tqdm(triple_groups.items(), desc="Processing CSV groups"):
        print(f"DEBUG: Processing group with RMS id '{rms_id}' and base '{base}' with files: {paths}")
        try:
            df0 = pd.read_csv(paths[0], dtype=str, on_bad_lines='skip')
            df1 = pd.read_csv(paths[1], dtype=str, on_bad_lines='skip')
            df2 = pd.read_csv(paths[2], dtype=str, on_bad_lines='skip')
        except Exception as e:
            print(f"Error reading CSV files for group {base} (RMS id: {rms_id}): {e}")
            continue

        # ----------------------------
        # Update evaluation and detection distribution counts for each annotator and question column
        # ----------------------------
        for annotator_index, df in enumerate([df0, df1, df2]):
            for col in args.question_cols:
                if col in df.columns:
                    # Update evaluation answer counts
                    for cell in df[col]:
                        answer_eval = extract_evaluation_answer(str(cell))
                        if answer_eval in ("Yes", "No"):
                            evaluation_distribution[annotator_index][col][answer_eval] += 1
                    # Update detection answer counts
                    for cell in df[col]:
                        answer_detect = extract_detection_answer(str(cell))
                        if answer_detect in ("Yes", "No"):
                            detection_distribution[annotator_index][col][answer_detect] += 1

        group_results = {}
        for col in args.question_cols:
            if col in df0.columns and col in df1.columns and col in df2.columns:
                print(f"DEBUG: Group (RMS id: {rms_id}, Base: {base}) has column '{col}' in all files. Computing kappa.")
                # Compute kappa for detection and evaluation modes
                kappa_detection = compute_fleiss_kappa_for_column(df0, df1, df2, col, mode="detection")
                kappa_evaluation = compute_fleiss_kappa_for_column(df0, df1, df2, col, mode="evaluation")
                group_results[col] = {
                    "detection": kappa_detection,
                    "evaluation": kappa_evaluation,
                }
                # Also accumulate the underlying frequency tables for aggregated kappa
                table_detection = get_rating_table_for_column(df0, df1, df2, col, mode="detection")
                table_evaluation = get_rating_table_for_column(df0, df1, df2, col, mode="evaluation")
                aggregated_tables[col]["detection"].extend(table_detection)
                aggregated_tables[col]["evaluation"].extend(table_evaluation)
            else:
                group_results[col] = {
                    "detection": float("nan"),
                    "evaluation": float("nan")
                }
                print(f"Warning: Group (RMS id: {rms_id}, Base: {base}) is missing required column '{col}' in one of the files. Skipping agreement computation for this column.")

        results[(rms_id, base)] = group_results

    print("\n---------- Intra-Annotator Agreement (Fleiss’ Kappa) per Group ----------")
    for (rms_id, base), group_results in results.items():
        print(f"\nGroup (RMS id: {rms_id}, Base: {base})")
        for col, kappas in group_results.items():
            print(f"  Column: {col}")
            print(f"    Detection-level Fleiss' Kappa:  {kappas['detection']:.4f}")
            print(f"    Evaluation-level Fleiss' Kappa: {kappas['evaluation']:.4f}")
    print("-----------------------------------------------------------------------")

    # ----------------------------
    # Print Evaluation Answer Distribution per Annotator
    # ----------------------------
    print("\n---------- Evaluation Answer Distribution per Annotator ----------")
    for annotator_index in sorted(evaluation_distribution.keys()):
        print(f"\nAnnotator {annotator_index + 1}:")
        for col in args.question_cols:
            counts = evaluation_distribution[annotator_index][col]
            total = counts["Yes"] + counts["No"]
            print(f"  Column: {col} -> Yes: {counts['Yes']}, No: {counts['No']} (Total valid answers: {total})")
    print("-----------------------------------------------------------------------")

    # ----------------------------
    # Print Detection Answer Distribution per Annotator
    # ----------------------------
    print("\n---------- Detection Answer Distribution per Annotator ----------")
    for annotator_index in sorted(detection_distribution.keys()):
        print(f"\nAnnotator {annotator_index + 1}:")
        for col in args.question_cols:
            counts = detection_distribution[annotator_index][col]
            total = counts["Yes"] + counts["No"]
            print(f"  Column: {col} -> Yes: {counts['Yes']}, No: {counts['No']} (Total valid answers: {total})")
    print("-----------------------------------------------------------------------")

    # Now compute the aggregated Fleiss' Kappa per column over all groups.
    print("\n---------- Aggregated Intra-Annotator Agreement (Fleiss’ Kappa) ----------")
    for col in args.question_cols:
        agg_detection_table = aggregated_tables[col]["detection"]
        agg_evaluation_table = aggregated_tables[col]["evaluation"]
        if agg_detection_table:
            agg_kappa_detection = fleiss_kappa(np.array(agg_detection_table))
        else:
            agg_kappa_detection = float("nan")
        if agg_evaluation_table:
            agg_kappa_evaluation = fleiss_kappa(np.array(agg_evaluation_table))
        else:
            agg_kappa_evaluation = float("nan")
        print(f"\nColumn: {col}")
        print(f"  Aggregated Detection-level Fleiss' Kappa:  {agg_kappa_detection:.4f}")
        print(f"  Aggregated Evaluation-level Fleiss' Kappa: {agg_kappa_evaluation:.4f}")
    print("-----------------------------------------------------------------------")

if __name__ == "__main__":
    main()