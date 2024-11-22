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
from src.evaluation import evaluate_model  

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

        # After processing 10 new rows, pause if necessary
        if new_rows_processed >= 10:
            df_LLM.to_csv(processed_file_path, index=False)  # Save before pausing
            print(f"Processed 10 new rows. Pausing for 30 seconds.")
            # time.sleep(30)
            new_rows_processed = 0  # Reset counter

    # Save the final DataFrame after processing all rows
    df_LLM.to_csv(processed_file_path, index=False)
    print("All rows have been processed and saved.")

    # Call the evaluation function
    evaluate_model(processed_file_path)

if __name__ == "__main__":
    main()