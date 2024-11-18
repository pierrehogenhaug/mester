import sys
print(sys.executable)
print(sys.path)

from langchain_ollama import OllamaLLM
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import json
import re
import ast
import string

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Add the project root to sys.path
sys.path.insert(0, project_root)

from src.analysis.prospectus_analyzer import ProspectusAnalyzer

def main():
    # Initialize the LLM
    llm = OllamaLLM(model="llama3.2")

    # Initialize the analyzer
    analyzer = ProspectusAnalyzer(llm_model=llm)

    # Load the data
    processed_file_path = './data/prospectuses_data_processed.csv'
    raw_file_path = './data/prospectuses_data.csv'

    # Check if processed file exists
    if os.path.exists(processed_file_path):
        df_LLM = pd.read_csv(processed_file_path)
    else:
        print("Processed file not found. Processing raw data...")
        df_LLM = pd.read_csv(raw_file_path)
        # Filter out rows that have "failed parsing" in the Section ID column
        df_LLM = df_LLM[df_LLM['Section ID'] != "failed parsing"]

    # Ensure the relevance and evidence columns are created with a compatible data type
    specified_columns = [
        'Market Dynamics - a', 'Market Dynamics - b', 'Market Dynamics - c',
        'Intra-Industry Competition - a', 'Intra-Industry Competition - b', 'Intra-Industry Competition - c',
        'Regulatory Framework - a', 'Regulatory Framework - b',
        'Technology Risk - a', 'Technology Risk - b'
    ]

    for column_name in specified_columns:
        if column_name in df_LLM.columns:
            df_LLM[column_name] = df_LLM[column_name].astype('string')
        else:
            df_LLM[column_name] = ""

    # Prepare the questions
    questions_market_dynamics = {
        "Market Dynamics - a": "Does the text mention that the company is exposed to risks associated with cyclical products?",
        "Market Dynamics - b": "Does the text mention risks related to demographic or structural trends affecting the market?",
        "Market Dynamics - c": "Does the text mention risks due to seasonal volatility in the industry?"
    }
    questions_intra_industry_competition = {
        "Intra-Industry Competition - a": "Does the text mention that market pricing for the company's products or services is irrational or not based on fundamental factors?",
        "Intra-Industry Competition - b": "Does the text mention that the market is highly fragmented with no clear leader or that there is only one dominant leader?",
        "Intra-Industry Competition - c": "Does the text mention low barriers to entry in the industry, making it easy for new competitors to enter the market?"
    }
    questions_regulatory_framework = {
        "Regulatory Framework - a": "Does the text mention that the industry is subject to a high degree of regulatory scrutiny?",
        "Regulatory Framework - b": "Does the text mention a high dependency on regulation or being a beneficiary from regulation in an unstable regulatory environment?"
    }
    questions_technology_risk = {
        "Technology Risk - a": "Does the text mention that the industry is susceptible to rapid technological advances or innovations?",
        "Technology Risk - b": "Does the text mention that the company is perceived as a disruptor or is threatened by emerging technological changes?"
    }

    all_question_dicts = [
        questions_market_dynamics,
        questions_intra_industry_competition,
        questions_regulatory_framework,
        questions_technology_risk
    ]

    # Initialize counter for new rows processed
    new_rows_processed = 0

    # Iterate over each row in the DataFrame with a progress bar
    for index, row in tqdm(df_LLM.iterrows(), total=df_LLM.shape[0], desc="Processing Rows"):
        row_processed = False  # Flag to check if we processed any new data in this row

        for question_dict in all_question_dicts:
            for column_name, question in question_dict.items():
                # Check if the answer column is already filled
                if pd.notnull(df_LLM.at[index, column_name]) and df_LLM.at[index, column_name] != "":
                    # Skip processing this row for this question
                    continue
                combined_answer = analyzer.analyze_row_single_question(row, question)
                df_LLM.at[index, column_name] = combined_answer
                row_processed = True  # We processed new data in this row

        if row_processed:
            new_rows_processed += 1

        # Save progress every 50 rows
        if (index + 1) % 50 == 0:
            df_LLM.to_csv(processed_file_path, index=False)
            # print(f"Progress saved at row {index + 1}")

        # After processing 10 new rows, sleep for 30 seconds
        if new_rows_processed >= 10:
            df_LLM.to_csv(processed_file_path, index=False)  # Save before sleeping
            print(f"Processed 10 new rows. Pausing for 30 seconds.")
            # time.sleep(30)
            new_rows_processed = 0  # Reset counter

    # Save the final DataFrame after processing all rows
    df_LLM.to_csv(processed_file_path, index=False)
    print("All rows have been processed and saved.")

    # Proceed to compute confusion matrices and metrics
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
    # Map df_LLM columns to labels in the format 'Category.letter'
    def column_to_label(col_name):
        return col_name.replace(' - ', '.').strip()

    label_mapping = {col: column_to_label(col) for col in specified_columns}
    all_labels = list(label_mapping.values())

    # Build a mapping from RmsId to Analyst-Assigned Labels
    grouped_df['Analyst_Labels'] = grouped_df['Label'].apply(lambda labels: [label for label in labels if label in all_labels])
    analyst_labels_dict = dict(zip(grouped_df['RmsId'], grouped_df['Analyst_Labels']))

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

if __name__ == "__main__":
    main()