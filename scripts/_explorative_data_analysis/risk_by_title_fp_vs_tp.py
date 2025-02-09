import os
import sys
import csv
import json
import re
import ast
import string
import pandas as pd
from fuzzywuzzy import fuzz

########################################
# Helper Functions (from evaluation.py)
########################################

def fuzzy_match_answer(answer: str, target: str = "yes", threshold: int = 90) -> bool:
    """
    Returns True if the fuzzy match score between answer and target is at or above the threshold.
    This allows for minor misspellings or extra whitespace.
    """
    if not answer or not answer.strip():
        return False
    score = fuzz.ratio(answer.strip().lower(), target.lower())
    return score >= threshold

def clean_text(text: str) -> str:
    """Removes newlines and extra whitespace from text."""
    return text.replace('\n', '').replace('\r', '').strip()

def parse_tagged_characteristics(s: str) -> list:
    """Safely parse a string representing a list (using ast.literal_eval)."""
    try:
        return ast.literal_eval(s)
    except Exception:
        return []

def assign_letters(group: pd.DataFrame, positive_letters: list, negative_letters: list, group_name: tuple) -> pd.DataFrame:
    """
    Given a group of rows (for a given Category and CharacteristicInfluence),
    assign letters (from positive or negative letters) and create a new Label column.
    """
    characteristic_influence = group_name[1]  # e.g. 'Positive' or 'Negative'
    n = len(group)
    if characteristic_influence == 'Positive':
        letters = positive_letters[:n]
    else:
        letters = negative_letters[:n]
    group = group.copy()
    group['letter'] = letters
    group['Label'] = group['Category'] + '.' + group['letter']
    return group

def column_to_label(col_name: str) -> str:
    """
    Converts a column name such as 'Market Dynamics - a' into a label like 'Market Dynamics.a'
    """
    return col_name.replace(' - ', '.').strip()

########################################
# Ground Truth Loading & Processing
########################################

def load_ground_truth():
    """
    Loads the two CSV files used for ground truth (rms_with_fundamental_score.csv and unique_score_combinations.csv)
    and processes them in the same manner as evaluation.py so that we can create a mapping from RmsId to
    the set of analyst-assigned labels.
    
    Returns:
      analyst_labels_dict: dict mapping RmsId (as a string) to a set of labels (e.g. {"Market Dynamics.a", ...})
      label_mapping: dict mapping each risk column to its label (e.g. "Market Dynamics - a" -> "Market Dynamics.a")
      risk_columns: list of the four risk columns
    """
    try:
        rms_with_fundamental_score = pd.read_csv('./data/rms_with_fundamental_score.csv')
    except Exception as e:
        print("Error loading rms_with_fundamental_score.csv:", e)
        sys.exit(1)
    try:
        df_labels = pd.read_csv('./data/unique_score_combinations.csv')
    except Exception as e:
        print("Error loading unique_score_combinations.csv:", e)
        sys.exit(1)
    
    # Clean and assign letters (to create the label strings)
    df_labels['TaggedCharacteristics'] = df_labels['TaggedCharacteristics'].apply(clean_text)
    positive_letters = list(string.ascii_uppercase)
    negative_letters = list(string.ascii_lowercase)
    groups = df_labels.groupby(['Category', 'CharacteristicInfluence'])
    processed_groups = [assign_letters(group, positive_letters, negative_letters, name) for name, group in groups]
    df_labels = pd.concat(processed_groups).reset_index(drop=True)
    
    # Process the ground truth file
    df = rms_with_fundamental_score.copy()
    df['TaggedCharacteristics'] = df['TaggedCharacteristics'].apply(parse_tagged_characteristics)
    df = df.explode('TaggedCharacteristics')
    df['CharacteristicText'] = df['TaggedCharacteristics'].apply(
        lambda x: x.get('CharacteristicText', '') if isinstance(x, dict) else ''
    )
    df['CharacteristicInfluence'] = df['TaggedCharacteristics'].apply(
        lambda x: x.get('CharacteristicInfluence', '') if isinstance(x, dict) else ''
    )
    df['CharacteristicText'] = df['CharacteristicText'].apply(clean_text)
    
    merged_df = pd.merge(
        df,
        df_labels,
        left_on=['Category', 'CharacteristicText', 'CharacteristicInfluence'],
        right_on=['Category', 'TaggedCharacteristics', 'CharacteristicInfluence'],
        how='left'
    )
    
    # Group by RmsId (and ScoringDate) and collect the unique analyst-assigned labels
    grouped_df = merged_df.groupby(['RmsId', 'ScoringDate'])['Label'].apply(lambda x: x.dropna().unique().tolist()).reset_index()
    
    # Define the risk columns and corresponding labels
    risk_columns = [
        "Market Dynamics - a",
        "Intra-Industry Competition - a",
        "Technology Risk - a",
        "Regulatory Framework - a"
    ]
    label_mapping = {col: column_to_label(col) for col in risk_columns}
    all_labels = list(label_mapping.values())
    
    # For each grouped row, filter the labels so that only those corresponding to our risk columns remain.
    grouped_df['Analyst_Labels'] = grouped_df['Label'].apply(
        lambda labels: [lbl for lbl in labels if lbl in all_labels]
    )
    
    # Build a dictionary mapping RmsId (as a string) to the set of analyst-assigned labels.
    analyst_labels_dict = {}
    for _, row in grouped_df.iterrows():
        rms = str(row['RmsId'])
        labels = set(row['Analyst_Labels'])
        if rms in analyst_labels_dict:
            analyst_labels_dict[rms].update(labels)
        else:
            analyst_labels_dict[rms] = labels
    return analyst_labels_dict, label_mapping, risk_columns

########################################
# Main Function: TP vs FP per Section Title
########################################

def main():
    # Increase CSV field size limit to handle very large fields.
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int / 10)
    
    project_root = "./"
    processed_root = os.path.join(project_root, "data", "processed")
    
    # Compile regex to match filenames ending in '_parsed.csv'
    csv_file_pattern = re.compile(r'_parsed\.csv$', re.IGNORECASE)
    
    # Gather all CSV file paths under folders that include "as_expected"
    all_csv_paths = []
    for root, dirs, files in os.walk(processed_root):
        if "as_expected" in root:
            for file in files:
                if csv_file_pattern.search(file):
                    csv_path = os.path.join(root, file)
                    all_csv_paths.append(csv_path)
    
    # Load the ground truth data to get the analyst-assigned labels.
    analyst_labels_dict, label_mapping, risk_columns = load_ground_truth()
    
    # Initialize a dictionary to store counts per Section Title.
    # The structure will be:
    # {
    #    "Section Title": {
    #         <risk_column>: {"TP": count, "FP": count},
    #         "Overall_TP": count,
    #         "Overall_FP": count
    #    },
    #    ...
    # }
    section_counts = {}
    
    # Process each CSV file.
    # (Each CSV file is assumed to be under: ./data/processed/{rms_id}/as_expected/*_parsed.csv)
    for csv_path in all_csv_paths:
        # Extract the rms_id from the file path.
        # Expected structure: ./data/processed/{rms_id}/as_expected/filename_parsed.csv
        relative_path = os.path.relpath(csv_path, processed_root)
        parts = relative_path.split(os.sep)
        if len(parts) < 3:
            continue
        rms_id = str(parts[0])  # the first part is the rms_id
        
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                section_title = row.get("Section Title", "Unknown")
                if section_title not in section_counts:
                    # Initialize counts for each risk column
                    section_counts[section_title] = {col: {"TP": 0, "FP": 0} for col in risk_columns}
                    section_counts[section_title]["Overall_TP"] = 0
                    section_counts[section_title]["Overall_FP"] = 0
                
                # For each risk column, parse its contents and if evaluation_answer is (fuzzy) "Yes" then decide TP vs FP.
                for col in risk_columns:
                    risk_cell = row.get(col, "")
                    try:
                        risk_data = json.loads(risk_cell) if risk_cell.strip() else {}
                    except Exception:
                        risk_data = {}
                    
                    evaluation_answer = risk_data.get("evaluation_answer", "")
                    if fuzzy_match_answer(evaluation_answer, "yes"):
                        # The model detected the risk in the evaluation step.
                        # Get the corresponding label for this risk column.
                        label = label_mapping.get(col, col)
                        # Look up the analyst-assigned labels (ground truth) for this rms_id.
                        analyst_labels = analyst_labels_dict.get(rms_id, set())
                        if label in analyst_labels:
                            section_counts[section_title][col]["TP"] += 1
                            section_counts[section_title]["Overall_TP"] += 1
                        else:
                            section_counts[section_title][col]["FP"] += 1
                            section_counts[section_title]["Overall_FP"] += 1
    
    # Sort the sections by the overall number of detected risks (TP + FP) in descending order.
    sorted_sections = sorted(
        section_counts.items(),
        key=lambda item: (item[1]["Overall_TP"] + item[1]["Overall_FP"]),
        reverse=True
    )
    
    # Define the output CSV file name.
    output_file = 'risk_by_title_tp_fp_output.csv'
    
    # Prepare the header for the CSV.
    # The header will include: Section Title, then for each risk column two columns (e.g. "Market Dynamics - a TP", "Market Dynamics - a FP"),
    # and finally overall totals.
    header = ["Section Title"]
    for col in risk_columns:
        header.append(f"{col} TP")
        header.append(f"{col} FP")
    header.extend(["Overall TP", "Overall FP", "Overall Total"])
    
    # Write the aggregated results to the output CSV.
    with open(output_file, 'w', newline='', encoding='utf-8') as out_csv:
        writer = csv.writer(out_csv)
        writer.writerow(header)
        for section, counts in sorted_sections:
            row_items = [section]
            for col in risk_columns:
                row_items.append(str(counts[col]["TP"]))
                row_items.append(str(counts[col]["FP"]))
            overall_tp = counts["Overall_TP"]
            overall_fp = counts["Overall_FP"]
            overall_total = overall_tp + overall_fp
            row_items.extend([str(overall_tp), str(overall_fp), str(overall_total)])
            writer.writerow(row_items)
    
    print("Output written to", output_file)

if __name__ == "__main__":
    main()