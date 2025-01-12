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

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from langchain_community.llms import LlamaCpp
from src.analysis.prospectus_analyzer import ProspectusAnalyzer


def main():
    model_path = "../huggingface/gguf_folder/Llama-3.2-1B-Instruct-Q8_0.gguf"
    
    prompt_template = """{question}
    Title: {subsection_title}
    Text: {subsection_text}

    Provide your answer in the following JSON format:
    {{"Answer": "Yes" or "No",
    "Evidence": "The exact sentences from the document that support your answer; otherwise, leave blank."}}
    """

    llm_hf = LlamaCpp(
        model_path=model_path,
        n_ctx=4096,           
        n_gpu_layers=35,      
        max_tokens=256       
    )

    analyzer_hf = ProspectusAnalyzer(
        llm_model=llm_hf,
        prompt_template=prompt_template
    )

    output_dir = os.path.join('./data', model_path.replace('/', '_'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    processed_file_path = os.path.join(output_dir, 'prospectuses_data_processed.csv')

    if os.path.exists(processed_file_path):
        df_LLM = pd.read_csv(processed_file_path)
    else:
        raw_file_path = './data/prospectuses_data.csv'
        if not os.path.exists(raw_file_path):
            sys.exit(1)
        df_LLM = pd.read_csv(raw_file_path)
        
        if 'Section ID' in df_LLM.columns:
            df_LLM = df_LLM[df_LLM['Section ID'] != "failed parsing"]
        else:
            print("Column 'Section ID' not found in the data. Proceeding without this filter.")

        if 'Prospectus ID' not in df_LLM.columns:
            sys.exit(1)

        df_LLM.to_csv(processed_file_path, index=False)

    specified_columns = [
        'Market Dynamics - a',
        'Intra-Industry Competition - a'
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

    all_question_dicts = [
        questions_market_dynamics,
        questions_intra_industry_competition
    ]

    start_index = 0
    parsing_errors = []
    
    for index in tqdm(range(start_index, df_LLM.shape[0]), desc="Processing Rows"):
        row = df_LLM.iloc[index]
        row_dict = row.to_dict()

        for question_dict in all_question_dicts:
            for column_name, question in question_dict.items():
                if pd.isnull(df_LLM.at[index, column_name]) or df_LLM.at[index, column_name].strip() == "":
                    
                    prompt_text = analyzer_hf.build_prompt(
                        question=question,
                        subsection_title=row['Subsubsection Title'],
                        subsection_text=row['Subsubsection Text']
                    )

                    skip_message = "Skipped processing due to error."

                    try:
                        answers = analyzer_hf.analyze_rows_yes_no([row_dict], question)
                        answer_dict = answers[0]
                        df_LLM.at[index, column_name] = json.dumps(answer_dict)

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

        df_LLM.to_csv(processed_file_path, index=False)

    df_LLM.to_csv(processed_file_path, index=False)

if __name__ == "__main__":
    main()