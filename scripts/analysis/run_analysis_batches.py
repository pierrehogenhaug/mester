from huggingface_hub import login
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm

import os
import pandas as pd
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
    # Initialize the LLM (Hugging Face)
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)
    model_hf = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=True).to('cuda')
    model_hf.generation_config.pad_token_id = tokenizer.pad_token_id

    # Create a text-generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model_hf,
        tokenizer=tokenizer,
        device=0,
        max_length=2048,
        temperature=0.1,
    )

    # Initialize the LLM with the pipeline
    llm_hf = HuggingFacePipeline(pipeline=pipe)

    # Initialize the analyzer with the new LLM
    analyzer_hf = ProspectusAnalyzer(llm_model=llm_hf)

    # Load the data
    processed_file_path = './data/prospectuses_data_processed_test.csv'
    raw_file_path = './data/prospectuses_data.csv'

    # Check if processed file exists
    if os.path.exists(processed_file_path):
        df_LLM = pd.read_csv(processed_file_path)
    else:
        print("Processed file not found. Processing raw data...")
        df_LLM = pd.read_csv(raw_file_path)
        # Filter out rows that have "failed parsing" in the Section ID column
        df_LLM = df_LLM[df_LLM['Section ID'] != "failed parsing"]

    # Limit to first 10 rows for testing
    df_LLM = df_LLM.head(10)
    
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

    batch_size = 1  # Or any suitable batch size
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
                if len(batch_data) >= batch_size:
                    # Process the batch
                    rows_in_batch = [item['row'] for item in batch_data]
                    questions_in_batch = [item['question'] for item in batch_data]
                    combined_answers = analyzer_hf.analyze_rows_batch(rows_in_batch, questions_in_batch)

                    # Update the DataFrame
                    for item, answer in zip(batch_data, combined_answers):
                        idx = item['index']
                        col_name = item['column_name']
                        df_LLM.at[idx, col_name] = answer

                    # Clear the batch_data
                    batch_data = []

                    # Save progress
                    df_LLM.to_csv(processed_file_path, index=False)

    # After the loop, process any remaining items in the batch
    if batch_data:
        rows_in_batch = [item['row'] for item in batch_data]
        questions_in_batch = [item['question'] for item in batch_data]
        combined_answers = analyzer_hf.analyze_rows_batch(rows_in_batch, questions_in_batch)

        # Update the DataFrame
        for item, answer in zip(batch_data, combined_answers):
            idx = item['index']
            col_name = item['column_name']
            df_LLM.at[idx, col_name] = answer

        # Save progress
        df_LLM.to_csv(processed_file_path, index=False)

    print("All rows have been processed and saved.")

    # Call the evaluation function
    evaluate_model(processed_file_path)

if __name__ == "__main__":
    main()