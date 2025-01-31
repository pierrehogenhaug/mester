########################################
# evaluation.py
########################################
import os
import sys
import ast
import json
import string
import numpy as np
import pandas as pd
from typing import List, Union


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


def assign_letters(group: pd.DataFrame, positive_letters: list, negative_letters: list, group_name: tuple) -> pd.DataFrame:
    """
    Assigns letters to each row in 'group' based on 'CharacteristicInfluence'.
    """
    characteristic_influence = group_name[1]  # 'Positive' or 'Negative'
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
    Converts a column name like 'Market Dynamics - a' into 'Market Dynamics.a'.
    """
    return col_name.replace(' - ', '.').strip()


def compute_metrics(tp: int, fp: int, fn: int, tn: int):
    """
    Given TP, FP, FN, TN counts, compute Precision, Recall, F1 Score, and Accuracy.
    Returns: (precision, recall, f1_score, accuracy)
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else np.nan
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else np.nan
    return precision, recall, f1_score, accuracy


########################################
# Helper: Get LLM-Assigned Labels
########################################

def get_LLM_labels_for_prospectus(df: pd.DataFrame, label_columns: list, label_mapping: dict) -> set:
    """
    Returns a set of labels assigned by the LLM for all rows in `df` for the columns in `label_columns`.
    
    - If the JSON indicates 'answer' or 'Answer' == 'Yes', we consider that label assigned.
    - Ignores rows that can't be parsed or have 'No' answers.
    """
    assigned_labels = set()
    
    for col in label_columns:
        label = label_mapping[col]
        found_yes = False
        
        for val in df[col].fillna("").tolist():
            # Ignore empty
            if not val.strip():
                continue
            
            try:
                parsed_json = json.loads(val)
                # If we stored detection in "answer" or "detection_step_parsed"...
                # Adjust these lines to your exact JSON structure from run_analysis_llm.py
                # For instance, you might have:
                #   parsed_json["answer"] in {"Yes", "No"}
                
                # We'll check 'answer' if it exists:
                answer_value = parsed_json.get("answer", "")  
                if isinstance(answer_value, str) and answer_value.lower() == "yes":
                    found_yes = True
                    break
            except Exception:
                continue  # skip unparseable
                
        if found_yes:
            assigned_labels.add(label)
    
    return assigned_labels


########################################
# Main Function: evaluate_model
########################################

def evaluate_models(processed_file_paths: Union[str, List[str]]):
    """
    Loads and aggregates LLM output from one or multiple CSVs,
    then compares them to the ground-truth data in 'rms_with_fundamental_score'
    to produce aggregated confusion matrices and metrics.

    :param processed_file_paths: Single path (str) or list of paths to the processed CSV files.
    """
    # Ensure we have a list
    if isinstance(processed_file_paths, str):
        processed_file_paths = [processed_file_paths]

    # ------------------------------------
    # 1) Concatenate the LLM output DataFrames
    # ------------------------------------
    df_list = []
    for path in processed_file_paths:
        try:
            tmp = pd.read_csv(path)
            # If needed, handle missing columns gracefully here:
            df_list.append(tmp)
        except Exception as e:
            print(f"[ERROR] Could not read {path}: {e}")
    if not df_list:
        print("[ERROR] No valid dataframes to process in evaluate_model.")
        return

    df_LLM = pd.concat(df_list, ignore_index=True)

    # Columns we care about (must match the columns you wrote to in run_analysis_llm.py)
    specified_columns = [
        'Market Dynamics - a',
        'Intra-Industry Competition - a',
        # Add more if you want them evaluated:
        'Regulatory Framework - a',
        'Technology Risk - a',
    ]

    # Ensure missing columns exist (filled with "")
    for col in specified_columns:
        if col not in df_LLM.columns:
            df_LLM[col] = ""

    # ------------------------------------
    # 2) Load analyst ground-truth data
    # ------------------------------------
    rms_with_fundamental_score = pd.read_csv('./data/rms_with_fundamental_score.csv')
    df_labels = pd.read_csv('./data/unique_score_combinations.csv')

    # Clean the TaggedCharacteristics in df_labels
    df_labels['TaggedCharacteristics'] = df_labels['TaggedCharacteristics'].apply(clean_text)

    # Assign letters to labels
    positive_letters = list(string.ascii_uppercase)
    negative_letters = list(string.ascii_lowercase)
    
    groups = df_labels.groupby(['Category', 'CharacteristicInfluence'])
    processed_groups = [
        assign_letters(group, positive_letters, negative_letters, name)
        for name, group in groups
    ]
    df_labels = pd.concat(processed_groups).reset_index(drop=True)

    # ------------------------------------
    # 3) Process & merge ground-truth
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

    grouped_df = (
        merged_df.groupby(['RmsId', 'ScoringDate'])['Label']
        .apply(lambda x: x.dropna().unique().tolist())
        .reset_index()
    )

    # Build label mapping for columns -> 'Category.letter'
    label_mapping = {col: column_to_label(col) for col in specified_columns}
    all_labels = list(label_mapping.values())

    # Map RmsId -> Analyst-Assigned Labels
    grouped_df['Analyst_Labels'] = grouped_df['Label'].apply(
        lambda labels: [lbl for lbl in labels if lbl in all_labels]
    )
    analyst_labels_dict = dict(zip(grouped_df['RmsId'], grouped_df['Analyst_Labels']))

    # ------------------------------------
    # 4) Derive LLM-assigned labels
    # ------------------------------------
    LLM_labels_dict = {}
    for prospectus_id, group in df_LLM.groupby('Prospectus ID'):
        assigned_labels = get_LLM_labels_for_prospectus(
            group, specified_columns, label_mapping
        )
        LLM_labels_dict[prospectus_id] = assigned_labels

    # Helper to map Prospectus ID -> RmsId
    def get_RmsId_from_ProspectusID(prospectus_id):
        return int(str(prospectus_id).split('_')[0])

    # ------------------------------------
    # 5) Construct DataFrame for confusion matrix
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
    # 6) Compute Confusion Matrix & Metrics
    # ------------------------------------
    label_groups = df_confusion.groupby('Label')
    confusion_data = []
    for label, group in label_groups:
        TP = ((group['LLM_Assigned'] == True) & (group['Analyst_Assigned'] == True)).sum()
        FP = ((group['LLM_Assigned'] == True) & (group['Analyst_Assigned'] == False)).sum()
        FN = ((group['LLM_Assigned'] == False) & (group['Analyst_Assigned'] == True)).sum()
        TN = ((group['LLM_Assigned'] == False) & (group['Analyst_Assigned'] == False)).sum()
        
        precision, recall, f1_score, accuracy = compute_metrics(TP, FP, FN, TN)
        
        confusion_data.append({
            'Label': label,
            'TP': TP,
            'FP': FP,
            'FN': FN,
            'TN': TN,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1_score,
            'Accuracy': accuracy
        })

    confusion_matrix = pd.DataFrame(confusion_data)

    # Overall confusion metrics
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
    # 7) Print the Results
    # ------------------------------------
    print("\n=== Per-Label Confusion Matrix with Metrics ===")
    print(confusion_matrix)

    print("\n=== Overall Confusion Matrix and Metrics ===")
    print(overall_confusion)


if __name__ == "__main__":
    """
    If you want to test from CLI: 
      python evaluation.py path_to_processed.csv
      or 
      python evaluation.py path1.csv path2.csv ...
    """
    if len(sys.argv) > 1:
        evaluate_models(sys.argv[1:])
    else:
        print("Usage: python evaluation.py <processed_csv_path1> [<processed_csv_path2> ...]")