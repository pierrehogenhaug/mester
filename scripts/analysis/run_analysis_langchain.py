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

# If you want to log experiments with Weights & Biases:
import wandb

# Add the project root directory to sys.path if needed.
# This ensures Python can find your custom modules in src/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# -----------------------------------------------------------------
# 1) Import the *redesigned* ProspectusAnalyzer from the new file:
# -----------------------------------------------------------------
from src.analysis.prospectus_analyzer import ProspectusAnalyzer

# Still import your evaluation modules
from src.evaluation.evaluation import evaluate_model
from src.evaluation.check_progress import get_progress_metrics

# If you have a local Llama-Cpp wrapper in langchain_community:
from langchain_community.llms import LlamaCpp


def approximate_token_count(text: str) -> int:
    """
    Very rough heuristic to measure token length by splitting on whitespace.
    Replace with a real tokenizer if accurate counting is needed.
    """
    return len(text.split())

    
def main():
    # ---------------------------------------------------------
    # 2) Parse CLI arguments
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser(description="Run analysis with specified local Llama model using llama-cpp-python.")
    
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Local path to GGUF model (e.g., './data/gguf_folder/llama-2-7b.Q4_0.gguf')."
    )

    parser.add_argument(
        "--prompt_template",
        type=str,
        default="YES_NO_BASE_PROMPT_TEMPLATE",
        help="Which prompt style to use. For example: "
             "'YES_NO_BASE_PROMPT_TEMPLATE', 'YES_NO_FEW_SHOT_PROMPT_TEMPLATE', or 'YES_NO_COT_PROMPT_TEMPLATE'."
    )

    parser.add_argument(
        "--sample",
        action='store_true',
        help="Enable sampling of 100 unique Prospectus IDs. If not set, process the full dataset."
    )
    
    args = parser.parse_args()
    perform_sampling = args.sample
    prompt_template_arg = args.prompt_template

    # ---------------------------------------------------------
    # 3) Initialize LLM (LlamaCpp from LangChain)
    # ---------------------------------------------------------
    model_path = args.model_id
    print(f"Loading local llama-cpp model: {model_path}")
    llm_hf = LlamaCpp(
        model_path=model_path,
        n_ctx=4096,    # Adjust if model supports more tokens
        n_gpu_layers=35,
        max_tokens=256
    )

    # Initialize W&B (optional)
    wandb.login(key="28e0a54f934e056ba846e10f3460b100aa61283c")
    wandb.init(
        project="MSc@DTU",
        name=f"analysis-run-{model_path}",  
        config={
            "model_path": model_path,
            "prompt_template": prompt_template_arg,
            "sample": perform_sampling,
        }
    )

    # ---------------------------------------------------------
    # 4) Map the user’s prompt_template arg to new booleans
    # ---------------------------------------------------------
    #   - If you have more custom logic, you can expand here.
    #   - We'll treat "YES_NO_COT_PROMPT_TEMPLATE" as an example
    #     of using 'enhanced' instructions.
    use_enhanced = False
    use_few_shot = False

    if prompt_template_arg == "YES_NO_BASE_PROMPT_TEMPLATE":
        use_enhanced = False
        use_few_shot = False
    elif prompt_template_arg == "YES_NO_FEW_SHOT_PROMPT_TEMPLATE":
        use_enhanced = False
        use_few_shot = True
    elif prompt_template_arg == "YES_NO_COT_PROMPT_TEMPLATE":
        # Or adapt as needed
        use_enhanced = True
        use_few_shot = False
    else:
        # Default fallback
        use_enhanced = False
        use_few_shot = False

    # ---------------------------------------------------------
    # 5) Instantiate your redesigned ProspectusAnalyzer
    # ---------------------------------------------------------
    analyzer_hf = ProspectusAnalyzer(
        llm_model=llm_hf,
        use_enhanced_prompt=use_enhanced,
        use_few_shot_prompt=use_few_shot
    )

    # ---------------------------------------------------------
    # 6) Prepare input data (same as your original script)
    # ---------------------------------------------------------
    output_dir = os.path.join('./data', model_path.replace('/', '_'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    suffix = prompt_template_arg
    if perform_sampling:
        processed_file_path = os.path.join(output_dir, f'prospectuses_data_processed_sampled_{suffix}.csv')
    else:
        processed_file_path = os.path.join(output_dir, f'prospectuses_data_processed_full_{suffix}.csv')

    if os.path.exists(processed_file_path):
        df_LLM = pd.read_csv(processed_file_path)
        print(f"Loaded existing processed data from {processed_file_path}")
    else:
        raw_file_path = './data/prospectuses_data.csv'
        if not os.path.exists(raw_file_path):
            print(f"Raw data file not found at {raw_file_path}. Exiting.")
            sys.exit(1)
        print("Processed file not found. Loading raw data...")
        df_LLM = pd.read_csv(raw_file_path)
        
        if 'Section ID' in df_LLM.columns:
            df_LLM = df_LLM[df_LLM['Section ID'] != "failed parsing"]

        if 'Prospectus ID' not in df_LLM.columns:
            print("Column 'Prospectus ID' not found in the data. Please ensure the correct column name.")
            sys.exit(1)

        if perform_sampling:
            sample_size = 100
            random_seed = 42
            unique_ids = df_LLM['Prospectus ID'].dropna().unique()
            if len(unique_ids) < sample_size:
                print(f"Not enough unique Prospectus IDs to sample {sample_size}. Available: {len(unique_ids)}.")
                sys.exit(1)
            
            sampled_ids = pd.Series(unique_ids).sample(n=sample_size, random_state=random_seed).tolist()
            print(f"Sampled {len(sampled_ids)} Prospectus IDs.")
            df_LLM = df_LLM[df_LLM['Prospectus ID'].isin(sampled_ids)].copy()
            df_LLM.reset_index(drop=True, inplace=True)

        df_LLM.to_csv(processed_file_path, index=False)
        print(f"Saved processed data to {processed_file_path}")

    specified_columns = [
        'Market Dynamics - a',
        'Intra-Industry Competition - a',
        'Regulatory Framework - a',
        'Technology Risk - a'
    ]

    for column_name in specified_columns:
        if column_name not in df_LLM.columns:
            df_LLM[column_name] = ""
        df_LLM[column_name] = df_LLM[column_name].astype('string')

    questions_market_dynamics = {
        "Market Dynamics - a": "Does the text mention that the company is exposed to risks associated with cyclical products?"
    }
    questions_intra_industry_competition = {
        "Intra-Industry Competition - a": "Does the text mention that market pricing for the company's products or services is irrational or not based on fundamental factors?"
    }
    questions_regulatory_framework = {
        "Regulatory Framework - a": "Does the text mention that the industry is subject to a high degree of regulatory scrutiny?"
    }
    questions_technology_risk = {
        "Technology Risk - a": "Does the text mention that the industry is susceptible to rapid technological advances or innovations?"
    }

    all_question_dicts = [
        questions_market_dynamics,
        questions_intra_industry_competition,
        questions_regulatory_framework,
        questions_technology_risk
    ]

    def row_fully_processed(row):
        for c in specified_columns:
            if pd.isnull(row[c]) or row[c].strip() == "":
                return False
        return True

    start_index = 0
    for i, r in df_LLM.iterrows():
        if not row_fully_processed(r):
            start_index = i
            break
    else:
        start_index = df_LLM.shape[0]

    if start_index >= df_LLM.shape[0]:
        print("All rows have already been processed.")
        evaluate_model(processed_file_path)
        return

    print(f"Resuming processing from row {start_index}...")

    new_rows_processed = 0
    MAX_PROMPT_TOKENS = 4096 - 256  # e.g., 3840
    parsing_errors = []

    # ---------------------------------------------------------
    # 7) Main loop: Use the new .analyze_rows(...) method
    # ---------------------------------------------------------
    for index in tqdm(range(start_index, df_LLM.shape[0]), desc="Processing Rows"):
        row = df_LLM.iloc[index]
        row_dict = row.to_dict()

        if row_fully_processed(row):
            continue

        for question_dict in all_question_dicts:
            for column_name, question in question_dict.items():
                # Only process if it's empty or previously skipped
                if pd.isnull(df_LLM.at[index, column_name]) or df_LLM.at[index, column_name].strip() == "":
                    
                    # We do a quick length check so we don't blow up the context window
                    # We'll approximate the prompt by merging question+title+text
                    prompt_text = (
                        f"Question: {question}\n"
                        f"Title: {row['Subsubsection Title']}\n"
                        f"Text: {row['Subsubsection Text']}"
                    )
                    n_tokens = approximate_token_count(prompt_text)

                    skip_message = "Skipped processing due to length."
                    if n_tokens > MAX_PROMPT_TOKENS:
                        df_LLM.at[index, column_name] = skip_message
                        print(f"Row {index}, column '{column_name}' => Prompt length {n_tokens} > {MAX_PROMPT_TOKENS}. Marked as skipped.")
                    else:
                        try:
                            # The new method: analyze a list of row dicts with a single question
                            answers = analyzer_hf.analyze_rows([row_dict], question)
                            # We expect exactly one answer back
                            answer_dict = answers[0]
                            df_LLM.at[index, column_name] = json.dumps(answer_dict)

                            # Check for parsing errors
                            parsed_response = answer_dict["parsed_response"]
                            raw_response = answer_dict["raw_response"]
                            if "Parsing Error" in parsed_response:
                                parsing_errors.append({
                                    "row_index": index,
                                    "column_name": column_name,
                                    "parsed_response": parsed_response,
                                    "raw_response": raw_response,
                                })
                            
                        except Exception as e:
                            df_LLM.at[index, column_name] = skip_message
                            print(f"Row {index}, column '{column_name}' => LLM error '{str(e)}'. Marked as skipped.")

        # Save after each row
        df_LLM.to_csv(processed_file_path, index=False)
        new_rows_processed += 1

        # Periodically log progress to W&B
        if new_rows_processed % 10 == 0:
            progress_dict = get_progress_metrics(df_LLM)
            wandb.log(progress_dict)
            wandb.log({"current_row_index": index})

            if parsing_errors:
                table = wandb.Table(columns=["row_index", "column_name", "parsed_response", "raw_response"])
                for err in parsing_errors:
                    table.add_data(
                        err["row_index"],
                        err["column_name"],
                        err["parsed_response"],
                        err["raw_response"]
                    )
                wandb.log({"parsing_errors": table})

                # Optionally clear the error list after logging
                # parsing_errors.clear()

            print(f"Logged progress and parsing errors at row {index}.")

    # Final save
    df_LLM.to_csv(processed_file_path, index=False)
    print("All rows processed and saved.")

    # Final progress log
    progress_dict = get_progress_metrics(df_LLM)
    wandb.log(progress_dict)
    wandb.log({"current_row_index": df_LLM.shape[0] - 1})
    print("Final progress logged.")

    # Evaluate the model output
    evaluate_model(processed_file_path)

    wandb.finish()


if __name__ == "__main__":
    main()