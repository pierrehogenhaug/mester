import json
import torch
import argparse
import glob
import os
import pandas as pd
import re
import sys
import time
from tqdm import tqdm

# If you use LangChain for LlamaCpp
from langchain.llms import LlamaCpp

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.analysis.prospectus_analyzer import ProspectusAnalyzer
from src.evaluation.evaluation import evaluate_model


def approximate_token_count(text: str) -> int:
    """
    Very rough heuristic to measure token length by splitting on whitespace.
    Replace with a real tokenizer if accurate counting is needed.
    """
    return len(text.split())

    
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run analysis with specified local Llama model using llama-cpp-python.")
    
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Local path to your GGUF model (e.g., './data/gguf_folder/llama-2-7b.Q4_0.gguf')."
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
    perform_sampling = args.sample
    prompt_template = args.prompt_template

    # We’re now using a local llama-cpp-based model
    model_path = args.model_id
    print(f"Loading local llama-cpp model: {model_path}")

    # Initialize LlamaCpp from LangChain
    # Adjust parameters (context window, gpu layers, threads, etc.) for your environment
    llm_hf = LlamaCpp(
        model_path=model_path,
        n_ctx=4096,           # Adjust if your model supports more tokens
        n_gpu_layers=35,      # Adjust based on your VRAM
        temperature=0.2,
        max_tokens=256       # Similar to 'max_new_tokens' in HF
        # ,top_p=0.95,
        # top_k=40,
        # repeat_penalty=1.0,
        # n_threads=8,        # Optional: set number of CPU threads
        # Other llama-cpp-python params as needed
    )

    # Initialize the analyzer with the new LLM
    analyzer_hf = ProspectusAnalyzer(llm_model=llm_hf, prompt_template=prompt_template)

    # Define output directory and file paths based on MODEL_ID
    output_dir = os.path.join('./data', model_path.replace('/', '_'))
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
            sample_size = 100
            random_seed = 42
            unique_ids = df_LLM['Prospectus ID'].dropna().unique()
            if len(unique_ids) < sample_size:
                print(f"Not enough unique Prospectus IDs to sample {sample_size}. Available: {len(unique_ids)}.")
                sys.exit(1)
            
            sampled_ids = pd.Series(unique_ids).sample(n=sample_size, random_state=random_seed).tolist()
            print(f"Sampled {len(sampled_ids)} Prospectus IDs.")

            # Filter the dataframe to include only the sampled Prospectus IDs
            df_LLM = df_LLM[df_LLM['Prospectus ID'].isin(sampled_ids)].copy()
            print(f"Filtered data to include only sampled Prospectus IDs.")

            # Reset the index
            df_LLM.reset_index(drop=True, inplace=True)

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
        # "Market Dynamics - b": "...",
        # "Market Dynamics - c": "...",
    }

    all_question_dicts = [
        questions_market_dynamics
    ]

    # Helper to determine if a row is fully processed
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

    # The maximum token budget for prompt (after subtracting 256 for the answer)
    MAX_PROMPT_TOKENS = 4096 - 256  # 3840

    for index in tqdm(range(start_index, df_LLM.shape[0]), desc="Processing Rows"):
        row = df_LLM.iloc[index]
        row_dict = row.to_dict()
        row_processed = False

        # Skip row if already fully processed
        if row_fully_processed(row):
            continue

        # Iterate through each set of questions
        for question_dict in all_question_dicts:
            for column_name, question in question_dict.items():
                # Only process if empty
                col_idx = df_LLM.columns.get_loc(column_name)
                if pd.isnull(df_LLM.iloc[index, col_idx]) or df_LLM.iloc[index, col_idx].strip() == "":
                    
                    # ---- NEW: Build prompt and check approximate length ----
                    prompt_text = analyzer_hf.build_prompt(
                        question=question,
                        subsection_title=row['Subsubsection Title'],
                        subsection_text=row['Subsubsection Text']
                    )
                    n_tokens = approximate_token_count(prompt_text)

                    # If prompt is too long, set the column to skip message and continue
                    if n_tokens > MAX_PROMPT_TOKENS:
                        skip_message = "Skipped processing due to length."
                        print(f"Row {index}, Column '{column_name}': Prompt length ({n_tokens}) exceeds {MAX_PROMPT_TOKENS} tokens. Marking as skipped.")
                        df_LLM.at[index, column_name] = skip_message
                        continue  # Move to the next column or row

                    # Otherwise, proceed with analysis
                    answers = analyzer_hf.analyze_rows_yes_no([row_dict], question)
                    answer_dict = answers[0]
                    df_LLM.at[index, column_name] = json.dumps(answer_dict)
                    row_processed = True

        if row_processed:
            new_rows_processed += 1
            # Save after processing each row
            df_LLM.to_csv(processed_file_path, index=False)

    # Final save
    df_LLM.to_csv(processed_file_path, index=False)
    print("All rows have been processed and saved.")

    # Evaluate the model output
    evaluate_model(processed_file_path)

if __name__ == "__main__":
    main()