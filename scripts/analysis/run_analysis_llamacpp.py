import json
import argparse
import glob
import os
import pandas as pd
import re
import sys
import time
from tqdm import tqdm

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.analysis.prospectus_analyzer import ProspectusAnalyzer
from src.evaluation.evaluation import evaluate_model

# Import LlamaCpp from langchain
from langchain_community.llms import LlamaCpp

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run analysis with specified LLaMA model (llama.cpp).")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the local gguf file (e.g., './data/gguf_folder/ggml-model-q4_0.gguf')."
    )
    args = parser.parse_args()

    model_path = args.model_path
    print(f"Loading model from: {model_path}")

    # Handle multiple gguf files if necessary
    # If model_path is a directory containing multiple gguf files, ensure all parts are present
    if os.path.isdir(model_path):
        # Assuming the main gguf file is named in a standard way, e.g., 'ggml-model.gguf'
        gguf_files = glob.glob(os.path.join(model_path, "*.gguf"))
        if not gguf_files:
            print(f"No gguf files found in directory: {model_path}")
            sys.exit(1)
        # Sort the files to ensure correct loading order if required
        gguf_files.sort()
        # For LlamaCpp, typically you point to the main gguf file
        # Assuming the main file is the first one; adjust as needed
        main_gguf = gguf_files[0]
        print(f"Using main gguf file: {main_gguf}")
        model_path_to_use = main_gguf
    else:
        # model_path is assumed to be a path to a single gguf file
        if not model_path.endswith(".gguf"):
            print("Model path should point to a .gguf file or a directory containing gguf files.")
            sys.exit(1)
        model_path_to_use = model_path

    # Initialize the LLM using LlamaCpp with specified configurations
    llm_hf = LlamaCpp(
        model_path=model_path_to_use
        ,max_tokens=256          # Max new tokens
        ,n_ctx=4096              # Standard context window
        # ,temperature=0.7         # Standard temperature
        # ,top_p=0.95              # Standard top_p
        # ,verbose=True            # Enable verbose logging for debugging
        # You can adjust other parameters as needed
    )

    # Initialize the analyzer with the new LLM
    analyzer_hf = ProspectusAnalyzer(llm_model=llm_hf)

    # Define output directory and file paths based on the model's basename
    model_basename = os.path.splitext(os.path.basename(model_path_to_use))[0]
    output_dir = os.path.join('./data', model_basename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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

    # Ensure the relevance and evidence columns are created with a compatible data type
    specified_columns = [
        'Market Dynamics - a'
    ]

    for column_name in specified_columns:
        if column_name not in df_LLM.columns:
            df_LLM[column_name] = ""
        df_LLM[column_name] = df_LLM[column_name].astype('string')

    # Prepare the questions
    questions_market_dynamics = {
        "Market Dynamics - a": "Does the text mention that the company is exposed to risks associated with cyclical products?"
    }

    all_question_dicts = [
        questions_market_dynamics
    ]

    # Function to check if a row is fully processed
    def row_fully_processed(row):
        for c in specified_columns:
            if pd.isnull(row[c]) or row[c].strip() == "":
                return False
        return True

    # Find the first unprocessed row index
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

        # Check if row is already processed
        if row_fully_processed(row):
            continue

        for question_dict in all_question_dicts:
            for column_name, question in question_dict.items():
                # Only process if the column is empty
                if pd.isnull(df_LLM.iloc[index][column_name]) or df_LLM.iloc[index][column_name].strip() == "":
                    try:
                        answers = analyzer_hf.analyze_rows_yes_no([row_dict], question)
                        answer_dict = answers[0]
                        df_LLM.at[index, column_name] = json.dumps(answer_dict)
                        row_processed = True
                    except Exception as e:
                        print(f"Error processing row {index}: {e}")
                        continue

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