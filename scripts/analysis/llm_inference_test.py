import argparse
import logging
import os
import pandas as pd
import sys
import time

import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline

# Insert project root if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.analysis.prospectus_analyzer import ProspectusAnalyzer

def main():
    parser = argparse.ArgumentParser(description="Run LLM analysis on prospectus data.")
    parser.add_argument("--model_id", type=str, required=True, help="Hugging Face model ID/path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    model_id = args.model_id
    raw_file_path = os.path.join(project_root, 'data', 'prospectuses_data.csv')

    # Load a small subset of the data for demonstration.
    df = pd.read_csv(raw_file_path).head()
    df = df[~(df["Section ID"] == "failed parsing")]

    questions_market_dynamics = {
        "Market Dynamics - a": "Does the text mention that the company is exposed to risks associated with cyclical products?",
        "Market Dynamics - b": "Does the text mention risks related to demographic or structural trends affecting the market?"
    }

    question = questions_market_dynamics["Market Dynamics - a"]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model_hf = AutoModelForCausalLM.from_pretrained(model_id)
    model_hf.generation_config.pad_token_id = tokenizer.pad_token_id

    # Different configurations we want to try
    configurations = [
        {"max_new_tokens": 256},  # short response
        {"max_new_tokens": 512}   # longer response
    ]

    rows = df.to_dict('records')

    for i, config in enumerate(tqdm(configurations, desc="Processing Configurations")):
        if i > 0:
            print("\n")
        print(f"=== Configuration {i} ===")
        print(config)

        pipe = pipeline(
            "text-generation",
            model=model_hf,
            tokenizer=tokenizer,
            device=0,
            max_new_tokens=config["max_new_tokens"]
        )

        llm_hf = HuggingFacePipeline(pipeline=pipe)
        analyzer_hf = ProspectusAnalyzer(llm_model=llm_hf)

        # Timing the inference
        start_time = time.time()
        # Depending on your actual method name, here we use analyze_rows_yes_no (change if needed)
        combined_answers = analyzer_hf.analyze_rows_yes_no(rows, question)
        end_time = time.time()

        duration = end_time - start_time
        print(f"Total time taken for configuration {i}: {duration:.2f} seconds.")

        # If token usage can be tracked, print it here:
        # e.g. if your pipeline returns token usage somehow or you measure it internally
        # total_tokens = ... # obtain this from model output if possible
        # print(f"Tokens generated: {total_tokens}, Tokens/sec: {total_tokens/duration:.2f}")

        df[f"Answer_Config_{i}"] = combined_answers

        # If you only want to run the first configuration, break early
        # if i == 1:
        #     break

    # Optionally save results:
    output_path = os.path.join(project_root, "data", "analysis_results.csv")
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()