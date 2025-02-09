import os
import sys
import csv
import json
import re
import ast
import string
import pandas as pd

########################################
# Helper Functions (modified: no fuzzy matching)
########################################

def is_yes(answer: str) -> bool:
    """
    Returns True if the answer, after stripping whitespace and converting to lower-case, equals "yes".
    """
    return answer.strip().lower() == "yes" if answer else False

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
# New Helper Function for Word Count
########################################

def count_words(text: str) -> int:
    """A simple heuristic: count words by splitting on whitespace."""
    return len(text.split())

########################################
# Ground Truth Loading & Processing
########################################

def load_ground_truth():
    """
    Loads the two CSV files used for ground truth and processes them
    so that we can create a mapping from RmsId to the set of analyst-assigned labels.
    
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
# Main Function: TP vs FP per Section Title + Section Text Length (Words Only)
########################################

def main():
    # List of sampled RMS IDs to process.
    sampled_rms_ids = ['367', '999', '1609', '625', '1108', '673', '219', '440', '328', '355', 
                         '139', '1629', '1074', '352', '1052', '946', '1897', '317', '653', 
                         '642', '1525', '1277', '935', '433', '153', '221', '1261', '199', 
                         '130', '252', '377', '84', '518', '201', '989', '1069', '1727', '1739', 
                         '258', '1127', '1182', '1096', '311', '1765', '661', '990', '251', 
                         '1136', '257', '398']
    
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
    # Structure:
    # { "Section Title": {
    #         <risk_column>: {"TP": count, "FP": count},
    #         "Overall_TP": count,
    #         "Overall_FP": count
    #   },
    #   ...
    # }
    section_counts = {}
    # Initialize a dictionary to accumulate text per Section Title (per prospectus).
    # Structure: { "Section Title": { rms_id: concatenated_text } }
    section_texts = {}
    
    # Process each CSV file.
    # (Each CSV file is assumed to be under: ./data/processed/{rms_id}/as_expected/*_parsed.csv)
    for csv_path in all_csv_paths:
        # Extract the rms_id from the file path.
        # Expected structure: ./data/processed/{rms_id}/as_expected/filename_parsed.csv
        relative_path = os.path.relpath(csv_path, processed_root)
        parts = relative_path.split(os.sep)
        if len(parts) < 3:
            continue
        rms_id = str(parts[0])
        # Only process files for RMS IDs in the sampled list.
        if rms_id not in sampled_rms_ids:
            continue
        
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                section_title = row.get("Section Title", "Unknown")
                # Initialize risk counts for this section if not already done.
                if section_title not in section_counts:
                    section_counts[section_title] = {col: {"TP": 0, "FP": 0} for col in risk_columns}
                    section_counts[section_title]["Overall_TP"] = 0
                    section_counts[section_title]["Overall_FP"] = 0
                
                # Process each risk column: if evaluation_answer is "yes" then decide TP vs FP.
                for col in risk_columns:
                    risk_cell = row.get(col, "")
                    try:
                        risk_data = json.loads(risk_cell) if risk_cell.strip() else {}
                    except Exception:
                        risk_data = {}
                    
                    evaluation_answer = risk_data.get("evaluation_answer", "")
                    if is_yes(evaluation_answer):
                        label = label_mapping.get(col, col)
                        analyst_labels = analyst_labels_dict.get(rms_id, set())
                        if label in analyst_labels:
                            section_counts[section_title][col]["TP"] += 1
                            section_counts[section_title]["Overall_TP"] += 1
                        else:
                            section_counts[section_title][col]["FP"] += 1
                            section_counts[section_title]["Overall_FP"] += 1
                
                # --- Aggregate the text from both "Subsubsection Title" and "Subsubsection Text" ---
                subsec_title = row.get("Subsubsection Title", "")
                subsec_text = row.get("Subsubsection Text", "")
                combined_text = " " + subsec_title + " " + subsec_text
                if section_title not in section_texts:
                    section_texts[section_title] = {}
                if rms_id not in section_texts[section_title]:
                    section_texts[section_title][rms_id] = ""
                section_texts[section_title][rms_id] += combined_text
    
    # Sort the sections by the overall number of detected risks (TP + FP) in descending order.
    sorted_sections = sorted(
        section_counts.items(),
        key=lambda item: (item[1]["Overall_TP"] + item[1]["Overall_FP"]),
        reverse=True
    )
    
    # Compute text statistics for each Section Title based on word counts.
    section_text_stats = {}
    for section, prospectus_texts in section_texts.items():
        total_words = 0
        num_prospectuses = len(prospectus_texts)
        for pid, combined_text in prospectus_texts.items():
            words = count_words(combined_text)
            total_words += words
        avg_words = total_words / num_prospectuses if num_prospectuses > 0 else 0
        section_text_stats[section] = {
            "total_words": total_words,
            "avg_words": avg_words,
            "prospectus_count": num_prospectuses
        }
    
    # Calculate the grand total of words across all sections.
    grand_text_total = sum(stats["total_words"] for stats in section_text_stats.values())
    
    # Calculate the grand total of overall risk counts across all sections.
    grand_risk_total = sum((counts["Overall_TP"] + counts["Overall_FP"]) for section, counts in sorted_sections)
    
    # Define the output CSV file name.
    output_file = 'risk_by_title_tp_fp_output.csv'
    
    # Prepare the header for the CSV.
    header = ["Section Title"]
    for col in risk_columns:
        header.append(f"{col} TP")
        header.append(f"{col} FP")
    header.extend(["Overall TP", "Overall FP", "Overall Total"])
    header.extend([
        "Total Text Length (words)",
        "Avg Text Length (words)",
        "Prospectus Count",
        "Text Percentage (%)",
        "Overall Risk Percentage (%)",
        "Accumulated Text Percentage (%)",
        "Accumulated Overall Risk Percentage (%)"
    ])
    
    # Initialize accumulators for the running percentages.
    accumulated_text_percentage = 0.0
    accumulated_risk_percentage = 0.0

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
            
            # Look up text statistics for this section.
            stats = section_text_stats.get(section, {"total_words": 0, "avg_words": 0, "prospectus_count": 0})
            row_items.append(str(stats["total_words"]))
            row_items.append(f"{stats['avg_words']:.2f}")
            row_items.append(str(stats["prospectus_count"]))
            
            # Compute the percentage of the grand total text that this section represents.
            text_percentage = (stats["total_words"] / grand_text_total * 100) if grand_text_total > 0 else 0
            row_items.append(f"{text_percentage:.2f}%")
            
            # Compute the percentage of the grand overall risk total that this section represents.
            risk_percentage = (overall_total / grand_risk_total * 100) if grand_risk_total > 0 else 0
            row_items.append(f"{risk_percentage:.2f}%")
            
            # Update the accumulated percentages.
            accumulated_text_percentage += text_percentage
            accumulated_risk_percentage += risk_percentage
            row_items.append(f"{accumulated_text_percentage:.2f}%")
            row_items.append(f"{accumulated_risk_percentage:.2f}%")
            
            writer.writerow(row_items)
    
    print("Output written to", output_file)

if __name__ == "__main__":
    main()