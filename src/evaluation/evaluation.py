########################################
# Imports
########################################
import os
import ast
import json
import string
import numpy as np
import pandas as pd


########################################
# Helper Functions
########################################

def clean_text(text: str) -> str:
    """Cleans the input text by removing newline characters and stripping whitespace."""
    return text.replace('\n', '').replace('\r', '').strip()


def parse_tagged_characteristics(s: str) -> list:
    """Safely parses a string representing a list of tagged characteristics."""
    try:
        return ast.literal_eval(s)
    except:
        return []


def assign_letters(group: pd.DataFrame, positive_letters: list, negative_letters: list) -> pd.DataFrame:
    """
    Assigns letters (uppercase for Positive, lowercase for Negative) to each row in 'group'.
    Assumes the group is partitioned by (Category, CharacteristicInfluence).
    """
    n = len(group)
    if group.name[1] == 'Positive':
        letters = positive_letters[:n]
    else:
        letters = negative_letters[:n]
    group = group.copy()
    group['letter'] = letters
    group['Label'] = group['Category'] + '.' + group['letter']
    return group


def column_to_label(col_name: str) -> str:
    """
    Converts a column name like 'Market Dynamics - a' into the label format 'Market Dynamics.a'.
    (Used to build the mapping between DataFrame columns and label strings.)
    """
    return col_name.replace(' - ', '.').strip()


def compute_metrics(tp: int, fp: int, fn: int, tn: int):
    """
    Given TP, FP, FN, TN counts, computes Precision, Recall, F1 Score, and Accuracy.
    Returns: (precision, recall, f1_score, accuracy)
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else np.nan
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else np.nan
    return precision, recall, f1_score, accuracy


########################################
# Helper: Compute Parsing Statistics
########################################

def compute_parsing_statistics(df: pd.DataFrame, specified_columns: list) -> pd.DataFrame:
    """
    Computes the proportion of successful parsings vs. parsing errors for each column
    in specified_columns.

    Returns a DataFrame with columns:
      - column_name
      - total_valid_rows (number of rows with non-empty, non-skipped output)
      - parse_errors
      - parse_successes
      - success_rate (parse_successes / total_valid_rows)
    """
    stats = []
    for col_name in specified_columns:
        total_valid_rows = 0
        parse_errors = 0
        for val in df[col_name].fillna("").tolist():
            # Ignore empty or skipped
            if not val or "Skipped processing due to length" in val:
                continue

            # Otherwise, attempt JSON parse
            total_valid_rows += 1
            try:
                parsed_json = json.loads(val)
                parsed_response = parsed_json.get("parsed_response", "")
                if parsed_response == "Parsing Error":
                    parse_errors += 1
            except Exception:
                parse_errors += 1

        parse_successes = total_valid_rows - parse_errors
        success_rate = None
        if total_valid_rows > 0:
            success_rate = parse_successes / total_valid_rows

        stats.append({
            "column_name": col_name,
            "total_valid_rows": total_valid_rows,
            "parse_errors": parse_errors,
            "parse_successes": parse_successes,
            "success_rate": success_rate
        })
    return pd.DataFrame(stats)


########################################
# Helper: Get LLM-Assigned Labels
########################################

def get_LLM_labels_for_prospectus(df: pd.DataFrame, label_columns: list, label_mapping: dict) -> set:
    """
    Returns a set of labels assigned by the LLM for all rows in `df`
    for the columns in `label_columns`.
    
    - Skips rows that contain a "Parsing Error" or are "Skipped".
    - If the LLM said "Yes" in any valid row for a column, that label is considered assigned.
    """
    assigned_labels = set()
    
    for col in label_columns:
        label = label_mapping[col]
        found_yes = False
        
        for val in df[col].fillna("").tolist():
            # Ignore empty or skipped
            if (not val) or ("Skipped processing due to length" in val):
                continue
            
            # Try to parse
            try:
                parsed_json = json.loads(val)
                parsed_response = parsed_json.get("parsed_response", "")
                
                # If "parsed_response" is "Yes" or "Yes: {something}"
                if parsed_response.lower().startswith("yes"):
                    found_yes = True
                    break  # No need to check more rows for this column
                # If parsed_response is "Parsing Error", we skip it.
            except Exception:
                # If JSON can't parse, skip it
                continue
        
        if found_yes:
            assigned_labels.add(label)
    
    return assigned_labels


########################################
# Main Function: evaluate_model
########################################

def evaluate_model(processed_file_path: str):
    """
    Loads both the ground-truth Analyst data (rms_with_fundamental_score, df_labels)
    and the LLM output (df_LLM) to:
      1) Compute parsing statistics for the relevant columns,
      2) Merge and transform the ground truth to create per-RmsId assigned labels,
      3) Derive LLM-assigned labels and compare to ground truth,
      4) Produce and print confusion matrix metrics.
    """

    # ------------------------------------
    # 1) Load the LLM output DataFrame
    # ------------------------------------
    df_LLM = pd.read_csv(processed_file_path)

    # Columns we care about
    specified_columns = [
        'Market Dynamics - a' 
        # Additional columns can be uncommented or added here as needed:
        # 'Market Dynamics - b', 'Market Dynamics - c',
        # 'Intra-Industry Competition - a', ...
    ]

    # ------------------------------------
    # 2) Print parsing stats
    # ------------------------------------
    parse_stats_df = compute_parsing_statistics(df_LLM, specified_columns)
    print("\n=== Parsing Success Statistics ===")
    print(parse_stats_df)
    
    # ------------------------------------
    # 3) Load analyst ground-truth data
    # ------------------------------------
    rms_with_fundamental_score = pd.read_csv('./data/rms_with_fundamental_score.csv')
    df_labels = pd.read_csv('./data/unique_score_combinations.csv')

    # Clean the TaggedCharacteristics in df_labels
    df_labels['TaggedCharacteristics'] = df_labels['TaggedCharacteristics'].apply(clean_text)

    # Assign letters to labels
    positive_letters = list(string.ascii_uppercase)
    negative_letters = list(string.ascii_lowercase)
    
    df_labels = df_labels.groupby(['Category', 'CharacteristicInfluence']).apply(
        lambda g: assign_letters(g, positive_letters, negative_letters)
    ).reset_index(drop=True)

    # ------------------------------------
    # 4) Process and merge ground-truth for analyst-assigned labels
    # ------------------------------------
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

    grouped_df = merged_df.groupby(['RmsId', 'ScoringDate'])['Label'] \
                          .apply(lambda x: x.dropna().unique().tolist()) \
                          .reset_index()

    # Build label mapping for columns -> 'Category.letter'
    label_mapping = {col: column_to_label(col) for col in specified_columns}
    all_labels = list(label_mapping.values())

    # Build a mapping from RmsId to Analyst-Assigned Labels
    grouped_df['Analyst_Labels'] = grouped_df['Label'].apply(
        lambda labels: [lbl for lbl in labels if lbl in all_labels]
    )
    analyst_labels_dict = dict(zip(grouped_df['RmsId'], grouped_df['Analyst_Labels']))

    # ------------------------------------
    # 5) Derive LLM-assigned labels (using new robust JSON parsing)
    # ------------------------------------
    LLM_labels_dict = {}
    for prospectus_id, group in df_LLM.groupby('Prospectus ID'):
        assigned_labels = get_LLM_labels_for_prospectus(group, specified_columns, label_mapping)
        LLM_labels_dict[prospectus_id] = assigned_labels

    # Helper to map Prospectus ID -> RmsId
    def get_RmsId_from_ProspectusID(prospectus_id):
        return int(str(prospectus_id).split('_')[0])

    # ------------------------------------
    # 6) Construct DataFrame to compute Confusion Matrix
    # ------------------------------------
    data = []
    for prospectus_id, llm_labels in LLM_labels_dict.items():
        rms_id = get_RmsId_from_ProspectusID(prospectus_id)
        analyst_labels = set(analyst_labels_dict.get(rms_id, []))
        for label in all_labels:
            llm_assigned = label in llm_labels
            analyst_assigned = label in analyst_labels
            data.append({
                'RmsId': rms_id,
                'Prospectus ID': prospectus_id,
                'Label': label,
                'LLM_Assigned': llm_assigned,
                'Analyst_Assigned': analyst_assigned
            })

    df_confusion = pd.DataFrame(data)

    # ------------------------------------
    # 7) Compute Confusion Matrix & Metrics
    # ------------------------------------
    confusion_matrix = df_confusion.groupby('Label').apply(
        lambda x: pd.Series({
            'TP': ((x['LLM_Assigned'] == True) & (x['Analyst_Assigned'] == True)).sum(),
            'FP': ((x['LLM_Assigned'] == True) & (x['Analyst_Assigned'] == False)).sum(),
            'FN': ((x['LLM_Assigned'] == False) & (x['Analyst_Assigned'] == True)).sum(),
            'TN': ((x['LLM_Assigned'] == False) & (x['Analyst_Assigned'] == False)).sum()
        })
    ).reset_index()

    confusion_matrix[['Precision', 'Recall', 'F1 Score', 'Accuracy']] = confusion_matrix.apply(
        lambda row: compute_metrics(row['TP'], row['FP'], row['FN'], row['TN']),
        axis=1,
        result_type='expand'
    )

    TP = ((df_confusion['LLM_Assigned'] == True) & (df_confusion['Analyst_Assigned'] == True)).sum()
    FP = ((df_confusion['LLM_Assigned'] == True) & (df_confusion['Analyst_Assigned'] == False)).sum()
    FN = ((df_confusion['LLM_Assigned'] == False) & (df_confusion['Analyst_Assigned'] == True)).sum()
    TN = ((df_confusion['LLM_Assigned'] == False) & (df_confusion['Analyst_Assigned'] == False)).sum()

    overall_precision, overall_recall, overall_f1, overall_accuracy = compute_metrics(TP, FP, FN, TN)
    overall_confusion = pd.DataFrame({
        'Metric': ['TP', 'FP', 'FN', 'TN', 'Precision', 'Recall', 'F1 Score', 'Accuracy'],
        'Value': [TP, FP, FN, TN, overall_precision, overall_recall, overall_f1, overall_accuracy]
    })

    # ------------------------------------
    # 8) Print the Results
    # ------------------------------------
    print("\n=== Per-Label Confusion Matrix with Metrics ===")
    print(confusion_matrix)

    print("\n=== Overall Confusion Matrix and Metrics ===")
    print(overall_confusion)