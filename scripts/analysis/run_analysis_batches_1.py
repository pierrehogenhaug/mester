from huggingface_hub import login
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm

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
    print(f"Loading model: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=True)
    # Check if pad_token_id is set; if not, set it to eos_token_id
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id


    model_hf = AutoModelForCausalLM.from_pretrained(model_id, token=True).to('cuda')
    model_hf.generation_config.pad_token_id = tokenizer.pad_token_id

    # Create a text-generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model_hf,
        tokenizer=tokenizer,
        device=0,
        max_length=4096
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
        raw_file_path = './data/prospectuses_data_test.csv'
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
    df_LLM = df_LLM.head(100)

    # Ensure the relevance and evidence columns are created with a compatible data type
    specified_columns = [
        'Market Dynamics - a', 'Market Dynamics - b', 'Market Dynamics - c'
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

    all_question_dicts = [
        questions_market_dynamics
    ]

    initial_batch_size = 8  # Or any suitable initial batch size
    batch_size = initial_batch_size
    batch_data = []

    # Iterate over each row in the DataFrame with a progress bar
    for index, row in tqdm(df_LLM.iterrows(), total=df_LLM.shape[0], desc="Processing Rows"):
        for question_dict in all_question_dicts:
            for column_name, question in question_dict.items():
                # Check if the answer column is already filled
                if pd.notnull(df_LLM.at[index, column_name]) and df_LLM.at[index, column_name] != "":
                    # Skip processing this row for this question
                    continue
                else:
                    # Add to batch
                    batch_data.append({'index': index, 'row': row, 'question': question, 'column_name': column_name})

                    # If batch size reached, process the batch
                    while len(batch_data) >= batch_size:
                        try:
                            # Process the batch
                            current_batch = batch_data[:batch_size]
                            rows_in_batch = [item['row'] for item in current_batch]
                            questions_in_batch = [item['question'] for item in current_batch]
                            combined_answers = analyzer_hf.analyze_rows_batch(rows_in_batch, questions_in_batch)

                            # Update the DataFrame
                            for item, answer in zip(current_batch, combined_answers):
                                idx = item['index']
                                col_name = item['column_name']
                                df_LLM.at[idx, col_name] = answer

                            # Remove processed items from batch_data
                            batch_data = batch_data[batch_size:]

                            # Save progress
                            df_LLM.to_csv(processed_file_path, index=False)

                            # Reset batch_size to initial value if it was reduced
                            if batch_size < initial_batch_size:
                                batch_size = initial_batch_size
                        except torch.cuda.OutOfMemoryError:
                            print("Out of memory error encountered. Reducing batch size and retrying.")
                            torch.cuda.empty_cache()
                            prev_batch_size = batch_size
                            batch_size = max(1, batch_size // 2)
                            print(f"New batch size: {batch_size}")
                            if batch_size == prev_batch_size and batch_size == 1:
                                # Cannot reduce batch size further
                                print("Batch size is already at minimum. Cannot process further.")
                                raise

    # After the loop, process any remaining items in batch_data
    while batch_data:
        try:
            current_batch = batch_data[:batch_size]
            rows_in_batch = [item['row'] for item in current_batch]
            questions_in_batch = [item['question'] for item in current_batch]
            combined_answers = analyzer_hf.analyze_rows_batch(rows_in_batch, questions_in_batch)

            # Update the DataFrame
            for item, answer in zip(current_batch, combined_answers):
                idx = item['index']
                col_name = item['column_name']
                df_LLM.at[idx, col_name] = answer

            # Remove processed items from batch_data
            batch_data = batch_data[batch_size:]

            # Save progress
            df_LLM.to_csv(processed_file_path, index=False)

            # Reset batch_size to initial value if it was reduced
            if batch_size < initial_batch_size:
                batch_size = initial_batch_size
        except torch.cuda.OutOfMemoryError:
            print("Out of memory error encountered. Reducing batch size and retrying.")
            torch.cuda.empty_cache()
            prev_batch_size = batch_size
            batch_size = max(1, batch_size // 2)
            print(f"New batch size: {batch_size}")
            if batch_size == prev_batch_size and batch_size == 1:
                # Cannot reduce batch size further
                print("Batch size is already at minimum. Cannot process further.")
                raise

    print("All rows have been processed and saved.")

    # Call the evaluation function
    evaluate_model(processed_file_path)

if __name__ == "__main__":
    main()