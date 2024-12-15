from huggingface_hub import login
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm

import json
import torch
import argparse
import glob
import os
import pandas as pd
import re 
import sys
import time

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Login to the Hugging Face Hub
login(token="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")

from src.analysis.prospectus_analyzer import ProspectusAnalyzer
from src.evaluation.evaluation import evaluate_model


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run analysis with specified HuggingFace model.")
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="HuggingFace model identifier or local path (e.g., 'meta-llama/Llama-3.2-3B-Instruct')."
    )
    args = parser.parse_args()

    # Initialize the LLM (Hugging Face) with the provided model_id
    model_id = args.model_id
    # model_id = "meta-llama/Llama-3.2-1B-Instruct"
    print(f"Loading model: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=True)
    # Ensure pad_token is set if not present
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_hf = AutoModelForCausalLM.from_pretrained(model_id, token=True)
    model_hf.generation_config.pad_token_id = tokenizer.pad_token_id

    # Create a text-generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model_hf,
        tokenizer=tokenizer,
        device = 0 if torch.cuda.is_available() else (torch.device("mps") if torch.backends.mps.is_available() else -1),
        max_new_tokens=256
    )

    # Initialize the LLM with the pipeline
    llm_hf = HuggingFacePipeline(pipeline=pipe)

    # Initialize the analyzer with the new LLM
    analyzer_hf = ProspectusAnalyzer(llm_model=llm_hf)

    # Define output directory and file paths based on MODEL_ID
    output_dir = os.path.join('./data', model_id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Processed file path (no suffix increments)
    processed_file_path = os.path.join(output_dir, 'prospectuses_data_processed.csv')

    # Check if a processed file exists
    if os.path.exists(processed_file_path):
        # Load the existing processed file
        df_LLM = pd.read_csv(processed_file_path)
        print(f"Loaded existing processed data from {processed_file_path}")
    else:
        # No processed file found, load raw data
        raw_file_path = './data/prospectuses_data.csv'
        print("Processed file not found. Loading raw data...")
        df_LLM = pd.read_csv(raw_file_path)
        # Filter out rows that have "failed parsing"
        df_LLM = df_LLM[df_LLM['Section ID'] != "failed parsing"]
        # Save initially as the "processed" file, even if no analysis done yet
        df_LLM.to_csv(processed_file_path, index=False)

    # Limit to first 100 rows for testing
    # df_LLM = df_LLM.head(5)
    
    # Ensure the relevance and evidence columns are created with a compatible data type
    specified_columns = [
        'Market Dynamics - a'#, 'Market Dynamics - b', 'Market Dynamics - c'
    ]

    for column_name in specified_columns:
        if column_name not in df_LLM.columns:
            df_LLM[column_name] = ""
        df_LLM[column_name] = df_LLM[column_name].astype('string')

    # Prepare the questions
    questions_market_dynamics = {
        "Market Dynamics - a": "Does the text mention that the company is exposed to risks associated with cyclical products?"
        # ,"Market Dynamics - b": "Does the text mention risks related to demographic or structural trends affecting the market?"
        # ,"Market Dynamics - c": "Does the text mention risks due to seasonal volatility in the industry?"
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
        questions_market_dynamics
        # ,questions_intra_industry_competition,
        # questions_regulatory_framework,
        # questions_technology_risk
    ]

    # Determine where to start processing:
    # A row is considered "fully processed" if all specified columns are non-empty.
    def row_fully_processed(row):
        for c in specified_columns:
            if pd.isnull(row[c]) or row[c].strip() == "":
                return False
        return True

    # Find the first unprocessed row index
    # If all rows processed, start_index = df_LLM.shape[0] (nothing to do)
    start_index = 0
    for i, r in df_LLM.iterrows():
        if not row_fully_processed(r):
            start_index = i
            break
    else:
        # If loop completes with no break, all rows are processed
        start_index = df_LLM.shape[0]

    if start_index >= df_LLM.shape[0]:
        print("All rows have already been processed.")
        evaluate_model(processed_file_path)
        return

    print(f"Resuming processing from row {start_index}...")

    # Initialize counters
    new_rows_processed = 0

    # Iterate from the first unprocessed row
    for index in tqdm(range(start_index, df_LLM.shape[0]), desc="Processing Rows"):
        row = df_LLM.iloc[index]
        row_dict = row.to_dict()
        row_processed = False

        # Check if row is already processed (possibly due to partial previous attempts)
        if row_fully_processed(row):
            # Skip since it's already done
            continue

        for question_dict in all_question_dicts:
            for column_name, question in question_dict.items():
                # Only process if the column is empty
                if pd.isnull(df_LLM.iloc[index, column_name]) or df_LLM.iloc[index, column_name].strip() == "":
                    answers = analyzer_hf.analyze_rows_yes_no([row_dict], question)
                    answer_dict = answers[0]
                    df_LLM.iloc[index, column_name] = json.dumps(answer_dict)
                    row_processed = True

        # If we processed new data in this row
        if row_processed:
            new_rows_processed += 1

        # Save progress every row
        if row_processed:
            new_rows_processed += 1
            # Save after each processed row to ensure progress is recorded
            df_LLM.to_csv(processed_file_path, index=False)

    # Save the final DataFrame after processing all rows
    df_LLM.to_csv(processed_file_path, index=False)
    print("All rows have been processed and saved.")

    # Call the evaluation function
    evaluate_model(processed_file_path)

if __name__ == "__main__":
    main()