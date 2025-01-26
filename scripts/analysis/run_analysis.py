import json
import os
import sys
import re
import pandas as pd
import argparse
from tqdm import tqdm
from typing import Optional, Dict, Any, List, Callable

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, ValidationError

# Make sure your local paths are set up properly:
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

# ------------------------------------------------------------------------------
# 0. Run Guide
# ------------------------------------------------------------------------------
# Example usage from the command line:
# python run_analysis.py --model_type=local --local_model_path=../MyLocalModel.gguf --sample
# python run_analysis.py --model_type=local --sample


# ------------------------------------------------------------------------------
# 1. JSON Extraction Helper
# ------------------------------------------------------------------------------
def extract_first_json_object(llm_output: str) -> str:
    """
    Attempts to extract the first JSON object from the LLM output
    using a non-greedy regex. If a JSON object is found, return it.
    Otherwise, return the original string.
    """
    match = re.search(r'\{.*?\}', llm_output, flags=re.DOTALL)
    if match:
        return match.group(0)
    return llm_output


# ------------------------------------------------------------------------------
# 2. Pydantic Model (Detection)
# ------------------------------------------------------------------------------
class LlmDetectionResponse(BaseModel):
    # Accept both "answer" and "Answer" via alias
    answer: str = Field(..., alias="Answer")
    # Accept both "evidence" and "Evidence" via alias
    evidence: Optional[str] = Field(None, alias="Evidence")

    class Config:
        populate_by_name = True

    @field_validator("answer", mode="before")
    def normalize_answer(cls, v):
        if isinstance(v, str):
            lower = v.strip().lower()
            if lower == "yes":
                return "Yes"
            elif lower == "no":
                return "No"
        raise ValueError("answer must be 'yes' or 'no' (case insensitive)")

    @field_validator("evidence")
    def evidence_when_yes(cls, v, info):
        """
        If answer=Yes, we enforce that 'evidence' is not empty.
        """
        answer = info.data.get("answer")
        if answer == "Yes" and not v:
            raise ValueError("evidence is required if answer is Yes")
        return v


# ------------------------------------------------------------------------------
# 3. LLM Setup
# ------------------------------------------------------------------------------
def get_llm(model_type: str, local_model_path: str):
    """
    Returns the appropriate LLM instance based on the specified model_type:
      - "openai" uses ChatOpenAI (requires OPENAI_API_KEY).
      - "local" uses LlamaCpp from langchain_community.llms.
    """
    if model_type == "openai":
        # If you have a custom `langchain_openai.py`, import from there
        from langchain_openai import ChatOpenAI

        load_dotenv()  # Loads OPENAI_API_KEY from .env
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if not OPENAI_API_KEY:
            print("OPENAI_API_KEY not found. Make sure it's set in your .env file.")
            sys.exit(1)

        # Adjust model_name as you wish (gpt-3.5-turbo, gpt-4, etc.)
        return ChatOpenAI(
            model_name="gpt-4o-mini",
            openai_api_key=OPENAI_API_KEY,
            max_tokens=256,
        )

    elif model_type == "local":
        from langchain_community.llms import LlamaCpp
        return LlamaCpp(
            model_path=local_model_path,
            n_ctx=4096,
            n_gpu_layers=35,
            max_tokens=256,
        )

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def call_llm(llm, prompt_str: str) -> str:
    """
    Uniform wrapper to call either an OpenAI-based or a local LlamaCpp-based LLM,
    returning the text response as a string.
    """
    response = llm.invoke(prompt_str)
    if hasattr(response, "content"):
        return response.content
    else:
        return response


# ------------------------------------------------------------------------------
# 4. Define Prompt & Helper
# ------------------------------------------------------------------------------
DETECTION_PROMPT = """
You are determining if the following text indicates the presence of the risk factor below.

Question: {question}
Title: {subsection_title}
Text: {subsection_text}

Provide your answer in JSON with no extra commentary, using this format:
{{
  "Answer": "Yes" or "No",
  "Evidence": "Exact sentences or phrases from the text that support the answer (blank if No)"
}}
"""

def build_detection_prompt(input_data: Dict[str, Any]) -> str:
    return DETECTION_PROMPT.format(
        question=input_data["question"],
        subsection_title=input_data["subsection_title"],
        subsection_text=input_data["subsection_text"],
    )


# ------------------------------------------------------------------------------
# 5. Detection Step
# ------------------------------------------------------------------------------
def run_detection_step(
    llm,
    question: str,
    subsection_title: str,
    subsection_text: str,
    max_retries: int = 3,
):
    """
    Runs the detection step with multiple retries, returning either a valid
    LlmDetectionResponse or None if it fails all attempts.
    """
    step1_input = {
        "question": question,
        "subsection_title": subsection_title,
        "subsection_text": subsection_text,
    }
    prompt_str = build_detection_prompt(step1_input)

    for attempt in range(1, max_retries + 1):
        try:
            raw_response = call_llm(llm, prompt_str)
            extracted_json = extract_first_json_object(raw_response)
            parsed_detection = json.loads(extracted_json)
            detection_result = LlmDetectionResponse.model_validate(parsed_detection)
            return (detection_result, raw_response, attempt, prompt_str)
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"[Detection] Attempt {attempt} for question '{question}' failed parsing/validation: {e}")
        except Exception as ex:
            print(f"[Detection] Unexpected error: {ex}")

    # Could not parse after max_retries
    return (None, None, max_retries, prompt_str)


# ------------------------------------------------------------------------------
# 6. Reusable Function to Analyze a DataFrame
# ------------------------------------------------------------------------------
def analyze_prospectus_dataframe(
    df: pd.DataFrame,
    model_type: str = "openai",
    local_model_path: str = "../Llama-3.2-1B-Instruct-Q8_0.gguf",
    questions_dict_list: Optional[List[Dict[str, str]]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """
    Analyze the given DataFrame of bond prospectus sections using an LLM.

    :param df: Input DataFrame with at least 'Subsubsection Title' and 'Subsubsection Text'.
    :param model_type: "openai" or "local".
    :param local_model_path: Path to the local LLaMA model if model_type="local".
    :param questions_dict_list: List of question dicts (keys=column names, vals=questions).
    :param progress_callback: Function called after each row (index, total_rows).
    :return: Updated DataFrame with new columns containing detection JSON results.
    """

    # 1) Initialize LLM
    llm = get_llm(model_type, local_model_path)

    # 2) If no custom questions were provided, define them here
    if not questions_dict_list:
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

        questions_dict_list = [
            questions_market_dynamics,
            questions_intra_industry_competition,
            questions_regulatory_framework,
            questions_technology_risk,
        ]

    # Flatten the columns we will fill
    columns_to_process = []
    for qd in questions_dict_list:
        columns_to_process.extend(list(qd.keys()))

    # Ensure columns exist in the DataFrame
    for column_name in columns_to_process:
        if column_name not in df.columns:
            df[column_name] = ""
        # convert to string to avoid type issues
        df[column_name] = df[column_name].astype("string")

    def row_fully_processed(row, columns):
        """
        Returns True if *all* relevant columns for a row are non-empty.
        """
        for c in columns:
            val = row[c]
            if pd.isnull(val) or val.strip() == "":
                return False
        return True

    # 4) Iterate through rows, detect risk factors
    total_rows = len(df)
    for i, row in df.iterrows():
        # Optional progress callback
        if progress_callback:
            progress_callback(i, total_rows)

        if row_fully_processed(row, columns_to_process):
            # Already processed
            continue

        subsection_title = str(row.get("Subsubsection Title", ""))
        subsection_text = str(row.get("Subsubsection Text", ""))

        # If there's no text, skip
        if not subsection_text.strip():
            continue

        # For each set of questions
        for question_dict in questions_dict_list:
            for column_name, question in question_dict.items():
                current_value = df.at[i, column_name]
                if pd.isnull(current_value) or current_value.strip() == "":
                    detection_result, detection_raw, detection_attempt, detection_prompt_str = run_detection_step(
                        llm=llm,
                        question=question,
                        subsection_title=subsection_title,
                        subsection_text=subsection_text,
                        max_retries=3
                    )

                    if not detection_result:
                        # Could not parse after retries
                        failure_dict = {
                            "detection_step_attempt_count": detection_attempt,
                            "detection_step_raw": detection_raw or "No valid output",
                            "error": "Detection failed",
                            "prompt_str": detection_prompt_str,
                        }
                        df.at[i, column_name] = json.dumps(failure_dict)
                    else:
                        # Store detection results
                        detection_dict = {
                            "detection_step_attempt_count": detection_attempt,
                            "detection_step_raw": detection_raw,
                            "detection_step_parsed": detection_result.model_dump(),
                            "answer": detection_result.answer,
                            "evidence": detection_result.evidence,
                            "prompt_str": detection_prompt_str,
                        }
                        df.at[i, column_name] = json.dumps(detection_dict)

    return df


# ------------------------------------------------------------------------------
# 7. Main function (Command-line entry point)
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Process prospectus data with a chosen LLM (OpenAI or local)."
    )
    parser.add_argument(
        "--model_type",
        choices=["openai", "local"],
        default="openai",
        help="Which LLM to use. 'openai' uses GPT-4/ChatOpenAI; 'local' uses a local LlamaCpp model."
    )
    parser.add_argument(
        "--local_model_path",
        default="../Llama-3.2-1B-Instruct-Q8_0.gguf",
        help="Path to the local LLaMA model if --model_type=local."
    )
    parser.add_argument(
        "--sample",
        action='store_true',
        help="Enable sampling (e.g. 100 random rows). If not set, process the full dataset."
    )
    args = parser.parse_args()

    model_type = args.model_type
    local_model_path = args.local_model_path
    perform_sampling = args.sample

    # Decide on folder for saving output
    model_id_for_path = (
        "openai_gpt-4o_mini" if model_type == "openai"
        else os.path.basename(local_model_path).replace('.', '_')
    )
    output_dir = os.path.join(".", "data", model_id_for_path.replace("/", "_"))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Name of final output CSV
    if perform_sampling:
        processed_file_path = os.path.join(output_dir, "prospectuses_data_processed_single_llm_sampled.csv")
    else:
        processed_file_path = os.path.join(output_dir, "prospectuses_data_processed_single_llm.csv")

    raw_file_path = "./data/prospectuses_data.csv"
    if not os.path.exists(raw_file_path):
        print("Raw CSV does not exist. Exiting.")
        sys.exit(1)

    # If the processed file exists, load it; otherwise load raw
    if os.path.exists(processed_file_path):
        df_LLM = pd.read_csv(processed_file_path)
        df_LLM.reset_index(drop=True, inplace=True)
        print(f"Loaded existing processed CSV: {processed_file_path}")
    else:
        df_LLM = pd.read_csv(raw_file_path)
        # Filter out rows where 'Section ID' == 'failed parsing', if present
        if "Section ID" in df_LLM.columns:
            df_LLM = df_LLM[df_LLM["Section ID"] != "failed parsing"].copy()
        df_LLM.reset_index(drop=True, inplace=True)

    # Make sure the column "Prospectus ID" is present
    if "Prospectus ID" not in df_LLM.columns:
        print("Column 'Prospectus ID' not found. Exiting.")
        sys.exit(1)

    # Optionally reduce the dataset if sampling
    if perform_sampling:
        unique_ids = df_LLM["Prospectus ID"].dropna().unique()
        sample_size = min(len(unique_ids), 100)
        if sample_size < 1:
            print("No valid prospectus IDs found to sample. Exiting.")
            sys.exit(1)
        import random
        random.seed(42)
        sampled_ids = random.sample(list(unique_ids), sample_size)
        df_LLM = df_LLM[df_LLM["Prospectus ID"].isin(sampled_ids)].copy()
        df_LLM.reset_index(drop=True, inplace=True)
        print(f"Sampled {sample_size} unique Prospectus IDs.")

        # Save this subset as the initial processed file
        df_LLM.to_csv(processed_file_path, index=False)

    # ---------------
    # Call our new, refactored function to do the actual analysis in memory
    # ---------------
    def cli_progress_callback(idx, total):
        # If you'd like, you can update a tqdm or some print statements
        # Here, we just do a print every 10 rows for demonstration
        if idx % 10 == 0:
            print(f"Processing row {idx+1} of {total}...")

    print("Starting LLM analysis...")
    df_LLM = analyze_prospectus_dataframe(
        df=df_LLM,
        model_type=model_type,
        local_model_path=local_model_path,
        sample=False,  # we've already done any sampling above
        progress_callback=cli_progress_callback,
    )
    print("LLM analysis complete. Saving results...")

    # Save final results
    df_LLM.to_csv(processed_file_path, index=False)
    print(f"Processing complete. Results saved to: {processed_file_path}")


if __name__ == "__main__":
    main()