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

    parser.add_argument(
        "--prompt_template",
        type=str,
        default="YES_NO_COT_PROMPT_TEMPLATE",
        help="Prompt template to use: 'YES_NO_COT_PROMPT_TEMPLATE' or 'YES_NO_BASE_PROMPT_TEMPLATE'."
    )

    parser.add_argument(
        "--sample",
        action='store_true',
        help="Enable sampling of 100 unique Prospectus IDs. If not set, process the full dataset."
    )
    
    args = parser.parse_args()
    perform_sampling = args.sample  # Boolean flag to determine sampling
    # perform_sampling = True  # Boolean flag to determine sampling
    prompt_template = args.prompt_template

    # Initialize variables
    model_id = args.model_id
    # model_id = "meta-llama/Llama-3.2-1B-Instruct"
    sample_size = 100
    random_seed = 42

    print(f"Loading model: {model_id}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)
    
    # Ensure pad_token is set if not present
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Initialize model
    model_hf = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=True)
    model_hf.generation_config.pad_token_id = tokenizer.pad_token_id

    # Create a text-generation pipeline
    device = 0 if torch.cuda.is_available() else (torch.device("mps") if torch.backends.mps.is_available() else -1)
    pipe = pipeline(
        "text-generation",
        model=model_hf,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=256
    )

    # Initialize the LLM with the pipeline
    llm_hf = HuggingFacePipeline(pipeline=pipe)

    # Initialize the analyzer with the new LLM
    analyzer_hf = ProspectusAnalyzer(llm_model=llm_hf, prompt_template=prompt_template)

    # Define output directory and file paths based on MODEL_ID
    output_dir = os.path.join('./data', model_id.replace('/', '_'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Incorporate the prompt_template name into the file name
    suffix = prompt_template
    if perform_sampling:
        processed_file_path = os.path.join(output_dir, f'prospectuses_data_processed_sampled_{suffix}.csv')
    else:
        processed_file_path = os.path.join(output_dir, f'prospectuses_data_processed_full_{suffix}.csv')

    # Check if a processed file exists
    if os.path.exists(processed_file_path):
        # Load the existing processed file
        df_LLM = pd.read_csv(processed_file_path)
        print(f"Loaded existing processed data from {processed_file_path}")
    else:
        # No processed file found, load raw data
        raw_file_path = './data/prospectuses_data.csv'
        if not os.path.exists(raw_file_path):
            print(f"Raw data file not found at {raw_file_path}. Exiting.")
            sys.exit(1)
        print("Processed file not found. Loading raw data...")
        df_LLM = pd.read_csv(raw_file_path)
        
        # Filter out rows that have "failed parsing"
        if 'Section ID' in df_LLM.columns:
            df_LLM = df_LLM[df_LLM['Section ID'] != "failed parsing"]
        else:
            print("Column 'Section ID' not found in the data. Proceeding without this filter.")

        # Check if 'Prospectus ID' column exists
        if 'Prospectus ID' not in df_LLM.columns:
            print("Column 'Prospectus ID' not found in the data. Please ensure the correct column name.")
            sys.exit(1)

        if perform_sampling:
            # Sample unique Prospectus IDs
            unique_ids = df_LLM['Prospectus ID'].dropna().unique()
            if len(unique_ids) < sample_size:
                print(f"Not enough unique Prospectus IDs to sample {sample_size}. Available: {len(unique_ids)}.")
                sys.exit(1)
            
            sampled_ids = pd.Series(unique_ids).sample(n=sample_size, random_state=random_seed).tolist()
            print(f"Sampled {len(sampled_ids)} Prospectus IDs.")

            # Filter the dataframe to include only the sampled Prospectus IDs
            df_LLM = df_LLM[df_LLM['Prospectus ID'].isin(sampled_ids)].copy()
            print(f"Filtered data to include only sampled Prospectus IDs.")

        # Save the processed data
        df_LLM.to_csv(processed_file_path, index=False)
        print(f"Saved processed data to {processed_file_path}")

    # Define columns to process
    specified_columns = [
        'Market Dynamics - a'  # , 'Market Dynamics - b', 'Market Dynamics - c'
    ]

    # Ensure the specified columns exist and have the correct data type
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
                col_idx = df_LLM.columns.get_loc(column_name)
                if pd.isnull(df_LLM.iloc[index, col_idx]) or df_LLM.iloc[index, col_idx].strip() == "":
                    answers = analyzer_hf.analyze_rows_yes_no([row_dict], question)
                    answer_dict = answers[0]
                    df_LLM.iloc[index, col_idx] = json.dumps(answer_dict)
                    row_processed = True

        # If we processed new data in this row
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