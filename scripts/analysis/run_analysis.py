from huggingface_hub import login
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm

# import torch

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


def get_latest_processed_file(processed_file_base):
    existing_files = glob.glob(processed_file_base + '*.csv')
    if not existing_files:
        return None, 0
    else:
        suffixes = []
        for fname in existing_files:
            base_name = os.path.basename(fname)
            match = re.match(r'prospectuses_data_processed(?:_(\d+))?\.csv', base_name)
            if match:
                if match.group(1):
                    suffixes.append(int(match.group(1)))
                else:
                    suffixes.append(0)
        if not suffixes:
            return None, 0
        max_suffix = max(suffixes)
        if max_suffix == 0:
            latest_file = processed_file_base + '.csv'
        else:
            latest_file = f"{processed_file_base}_{max_suffix}.csv"
        return latest_file, max_suffix


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
    # model_id = "meta-llama/Llama-3.2-3B-Instruct"
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
        max_new_tokens=128,
    )

    # Initialize the LLM with the pipeline
    llm_hf = HuggingFacePipeline(pipeline=pipe)

    # Initialize the analyzer with the new LLM
    analyzer_hf = ProspectusAnalyzer(llm_model=llm_hf)

    # Define output directory and file paths based on MODEL_ID
    output_dir = os.path.join('./data', model_id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define base file name
    processed_file_base = os.path.join(output_dir, 'prospectuses_data_processed')

    # Function to get the latest processed file
    latest_file, max_suffix = get_latest_processed_file(processed_file_base)
    if latest_file:
        # Load data from latest processed file
        df_LLM = pd.read_csv(latest_file)
        print(f"Loaded data from {latest_file}")
    else:
        # No existing processed files
        raw_file_path = './data/prospectuses_data.csv'
        print("Processed file not found. Processing raw data...")
        df_LLM = pd.read_csv(raw_file_path)
        # Filter out rows that have "failed parsing" in the Section ID column
        df_LLM = df_LLM[df_LLM['Section ID'] != "failed parsing"]

    # Set new processed_file_path
    new_suffix = max_suffix + 1
    if new_suffix == 0:
        processed_file_path = processed_file_base + '.csv'
    else:
        processed_file_path = f"{processed_file_base}_{new_suffix}.csv"

    # Limit to first 100 rows for testing
    df_LLM = df_LLM.head(3)
    
    # Ensure the relevance and evidence columns are created with a compatible data type
    specified_columns = [
        'Market Dynamics - a'#, 'Market Dynamics - b', 'Market Dynamics - c'
    ]

    for column_name in specified_columns:
        if column_name in df_LLM.columns:
            df_LLM[column_name] = df_LLM[column_name].astype('string')
        else:
            df_LLM[column_name] = ""

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

    # Initialize counters for new rows processed
    new_rows_processed = 0

    # Iterate over each row in the DataFrame
    for index, row in tqdm(df_LLM.iterrows(), total=df_LLM.shape[0], desc="Processing Rows"):
        row_dict = row.to_dict()
        row_processed = False

        for question_dict in all_question_dicts:
            for column_name, question in question_dict.items():
                # Check if the answer column is already filled
                if pd.notnull(df_LLM.at[index, column_name]) and df_LLM.at[index, column_name] != "":
                    # Already answered, skip
                    continue
                else:
                    answers = analyzer_hf.analyze_rows_yes_no([row_dict], question)
                    answer = answers[0]
                    df_LLM.at[index, column_name] = answer
                    row_processed = True

        # If we processed new data in this row
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