# src/evaluation/evaluation.py

import pandas as pd
import numpy as np
import ast
import string
import os


def evaluate_model(processed_file_path):
    # Load other necessary data
    rms_with_fundamental_score = pd.read_pickle('./data/rms_with_fundamental_score.pkl')
    df_labels = pd.read_csv('./data/unique_score_combinations.csv')

    # Clean the TaggedCharacteristics in df_labels
    def clean_text(text):
        return text.replace('\n', '').replace('\r', '').strip()

    df_labels['TaggedCharacteristics'] = df_labels['TaggedCharacteristics'].apply(clean_text)

    # Assign letters to labels
    positive_letters = list(string.ascii_uppercase)
    negative_letters = list(string.ascii_lowercase)

    def assign_letters(group):
        n = len(group)
        if group.name[1] == 'Positive':
            letters = positive_letters[:n]
        else:
            letters = negative_letters[:n]
        group = group.copy()
        group['letter'] = letters
        group['Label'] = group['Category'] + '.' + group['letter']
        return group

    df_labels = df_labels.groupby(['Category', 'CharacteristicInfluence']).apply(assign_letters).reset_index(drop=True)

    # Process rms_with_fundamental_score to get analyst labels
    df = rms_with_fundamental_score.copy()

    # Parse the TaggedCharacteristics column
    def parse_tagged_characteristics(s):
        try:
            return ast.literal_eval(s)
        except:
            return []

    df['TaggedCharacteristics'] = df['TaggedCharacteristics'].apply(parse_tagged_characteristics)

    # Explode the TaggedCharacteristics
    df = df.explode('TaggedCharacteristics')

    # Extract CharacteristicText and CharacteristicInfluence
    df['CharacteristicText'] = df['TaggedCharacteristics'].apply(lambda x: x.get('CharacteristicText', '') if isinstance(x, dict) else '')
    df['CharacteristicInfluence'] = df['TaggedCharacteristics'].apply(lambda x: x.get('CharacteristicInfluence', '') if isinstance(x, dict) else '')

    # Clean the CharacteristicText
    df['CharacteristicText'] = df['CharacteristicText'].apply(clean_text)

    # Merge with df_labels
    merged_df = pd.merge(df, df_labels,
                         left_on=['Category', 'CharacteristicText', 'CharacteristicInfluence'],
                         right_on=['Category', 'TaggedCharacteristics', 'CharacteristicInfluence'],
                         how='left')

    # Group by RmsId and ScoringDate and collect the labels
    grouped_df = merged_df.groupby(['RmsId', 'ScoringDate'])['Label'].apply(lambda x: x.dropna().unique().tolist()).reset_index()

    # The labels corresponding to the specified columns
    specified_columns = [
        'Market Dynamics - a', 'Market Dynamics - b', 'Market Dynamics - c',
        'Intra-Industry Competition - a', 'Intra-Industry Competition - b', 'Intra-Industry Competition - c',
        'Regulatory Framework - a', 'Regulatory Framework - b',
        'Technology Risk - a', 'Technology Risk - b'
    ]

    # Map df_LLM columns to labels in the format 'Category.letter'
    def column_to_label(col_name):
        return col_name.replace(' - ', '.').strip()

    label_mapping = {col: column_to_label(col) for col in specified_columns}
    all_labels = list(label_mapping.values())

    # Build a mapping from RmsId to Analyst-Assigned Labels
    grouped_df['Analyst_Labels'] = grouped_df['Label'].apply(lambda labels: [label for label in labels if label in all_labels])
    analyst_labels_dict = dict(zip(grouped_df['RmsId'], grouped_df['Analyst_Labels']))

    # Load df_LLM from the processed file
    df_LLM = pd.read_csv(processed_file_path)

    # Process df_LLM to extract LLM-assigned labels
    def get_LLM_labels_for_prospectus(df, label_columns, label_mapping):
        assigned_labels = set()
        for col in label_columns:
            label = label_mapping[col]
            # Check if any row has 'Highly Relevant' or 'Somewhat Relevant' for this label
            relevant = df[col].astype(str).str.startswith('Highly Relevant').any()
            if relevant:
                assigned_labels.add(label)
        return assigned_labels

    # Build a dictionary mapping Prospectus ID to LLM-assigned labels
    LLM_labels_dict = {}
    for prospectus_id, group in df_LLM.groupby('Prospectus ID'):
        assigned_labels = get_LLM_labels_for_prospectus(group, specified_columns, label_mapping)
        LLM_labels_dict[prospectus_id] = assigned_labels

    # Map Prospectus ID to RmsId
    def get_RmsId_from_ProspectusID(prospectus_id):
        return int(str(prospectus_id).split('_')[0])

    # Construct DataFrame for Confusion Matrix Calculation
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

    # Calculate the Confusion Matrix and Metrics
    # Function to compute metrics
    def compute_metrics(tp, fp, fn, tn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else np.nan
        accuracy = (tp + tn) / (tp + fp + fn + tn)
        return precision, recall, f1_score, accuracy

    # Compute per-label confusion matrix and metrics
    confusion_matrix = df_confusion.groupby('Label').apply(
        lambda x: pd.Series({
            'TP': ((x['LLM_Assigned'] == True) & (x['Analyst_Assigned'] == True)).sum(),
            'FP': ((x['LLM_Assigned'] == True) & (x['Analyst_Assigned'] == False)).sum(),
            'FN': ((x['LLM_Assigned'] == False) & (x['Analyst_Assigned'] == True)).sum(),
            'TN': ((x['LLM_Assigned'] == False) & (x['Analyst_Assigned'] == False)).sum()
        })
    ).reset_index()

    confusion_matrix[['Precision', 'Recall', 'F1 Score', 'Accuracy']] = confusion_matrix.apply(
        lambda row: compute_metrics(row['TP'], row['FP'], row['FN'], row['TN']), axis=1, result_type='expand'
    )

    # Compute overall confusion matrix and metrics
    TP = ((df_confusion['LLM_Assigned'] == True) & (df_confusion['Analyst_Assigned'] == True)).sum()
    FP = ((df_confusion['LLM_Assigned'] == True) & (df_confusion['Analyst_Assigned'] == False)).sum()
    FN = ((df_confusion['LLM_Assigned'] == False) & (df_confusion['Analyst_Assigned'] == True)).sum()
    TN = ((df_confusion['LLM_Assigned'] == False) & (df_confusion['Analyst_Assigned'] == False)).sum()

    overall_precision, overall_recall, overall_f1, overall_accuracy = compute_metrics(TP, FP, FN, TN)

    overall_confusion = pd.DataFrame({
        'Metric': ['TP', 'FP', 'FN', 'TN', 'Precision', 'Recall', 'F1 Score', 'Accuracy'],
        'Value': [TP, FP, FN, TN, overall_precision, overall_recall, overall_f1, overall_accuracy]
    })

    # Display the Results
    print("Per-Label Confusion Matrix with Metrics:")
    print(confusion_matrix)

    print("\nOverall Confusion Matrix and Metrics:")
    print(overall_confusion)