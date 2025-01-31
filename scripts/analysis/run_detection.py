import json
import os
import sys
import re
import pandas as pd
import argparse
import random
from tqdm import tqdm
from typing import Optional, Dict, Any

from dotenv import load_dotenv
#from langchain.chat_models import ChatOpenAI
from pydantic import BaseModel, Field, field_validator, ValidationError

# Make sure your local paths are set up properly:
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.evaluation.evaluation_1 import evaluate_models

# --------------------------------
# 0. Run Guide
# --------------------------------
# python scripts/analysis/run_detection.py --csv_path "/path/to/some_parsed.csv"
# python scripts/analysis/run_detection.py --rms_id ABC123
# python scripts/analysis/run_detection.py --sample 3


# --------------------------------
# 1. JSON Extraction Helper
# --------------------------------
def extract_first_json_object(llm_output: str) -> str:
    match = re.search(r'\{.*?\}', llm_output, flags=re.DOTALL)
    if match:
        return match.group(0)
    return llm_output


# --------------------------------
# 2. Pydantic Model (Detection)
# --------------------------------
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
        answer = info.data.get("answer")
        if answer == "Yes" and not v:
            raise ValueError("evidence is required if answer is Yes")
        return v


# --------------------------------
# 3. LLM Setup
# --------------------------------
def get_llm(model_type: str, local_model_path: str):
    """
    Returns the appropriate LLM instance based on the specified model_type:
      - "openai" uses ChatOpenAI (requires OPENAI_API_KEY).
      - "local" uses LlamaCpp from langchain_community.llms.
    """
    if model_type == "openai":
        # Example with (non-official) ChatOpenAI wrapper
        from langchain_openai import ChatOpenAI
        
        load_dotenv()  # Load OPENAI_API_KEY from .env
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if not OPENAI_API_KEY:
            print("OPENAI_API_KEY not found. Make sure it's set in your .env file.")
            sys.exit(1)

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
    returning the text response.
    """
    response = llm.invoke(prompt_str)
    # If it's a ChatMessage, return response.content; otherwise, just return the raw string
    if hasattr(response, "content"):
        return response.content
    else:
        return response


# --------------------------------
# 4. Define Prompt & Helper
# --------------------------------
DETECTION_PROMPT = """
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
        subsection_text=input_data["subsection_text"]
    )


# --------------------------------
# 5. Detection Step
# --------------------------------
def run_detection_step(llm, question: str, subsection_title: str, subsection_text: str, max_retries=3):
    """
    Runs the detection step with retries, returning either a valid LlmDetectionResponse
    or None if it fails all attempts.
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
            # Attempt to parse as JSON
            extracted_json = extract_first_json_object(raw_response)
            parsed_detection = json.loads(extracted_json)
            detection_result = LlmDetectionResponse.model_validate(parsed_detection)
            return (detection_result, raw_response, attempt, prompt_str)
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"[Detection] Attempt {attempt} for question '{question}' failed parsing/validation: {e}")
        except Exception as ex:
            print(f"[Detection] Unexpected error: {ex}")

    # If all retries fail
    return (None, None, max_retries, prompt_str)


# --------------------------------
# 6. Utility to check if row is fully processed
# --------------------------------
def row_fully_processed(row, columns):
    """
    Returns True if *all* relevant columns for a row are non-empty.
    """
    for c in columns:
        val = row.get(c, "")
        if pd.isnull(val) or str(val).strip() == "":
            return False
    return True


# --------------------------------
# 7. Function to process a single CSV
# --------------------------------
def process_single_csv(
    csv_path: str,
    llm,
    all_question_dicts: list[Dict[str, str]],
    partial_save: bool = True
):
    """
    Reads a single 'parsed.csv' file, checks if already fully processed, and if not:
      - adds columns if missing
      - calls the LLM row by row
      - partially saves after each row
    """
    # Read the CSV into df
    try:
        df = pd.read_csv(csv_path, dtype=str)
    except Exception as e:
        print(f"[ERROR] Failed to read {csv_path}: {e}")
        return

    # Collect all question columns
    columns_to_process = []
    for question_dict in all_question_dicts:
        columns_to_process.extend(list(question_dict.keys()))

    # Ensure those columns exist in df
    for col in columns_to_process:
        if col not in df.columns:
            df[col] = ""

    # Check if entire CSV is fully processed
    all_rows_done = True
    for _, row in df.iterrows():
        if not row_fully_processed(row, columns_to_process):
            all_rows_done = False
            break

    if all_rows_done:
        print(f"  -> All rows in {os.path.basename(csv_path)} are already processed. Skipping.")
        return

    # If not fully processed, do it row-by-row
    print(f"  -> Processing {csv_path} ...")
    for index in tqdm(range(len(df)), desc=f"  Rows in {os.path.basename(csv_path)}"):
        row = df.iloc[index]
        if row_fully_processed(row, columns_to_process):
            continue  # skip this row

        subsection_title = str(row.get("Subsubsection Title", ""))
        subsection_text = str(row.get("Subsubsection Text", ""))

        # If the subsection text is empty, skip
        if not subsection_text.strip():
            continue

        # For each question, if the cell is blank, run detection
        for question_dict in all_question_dicts:
            for column_name, question_str in question_dict.items():
                current_value = row.get(column_name, "")
                if pd.isnull(current_value) or str(current_value).strip() == "":
                    # call the detection step
                    detection_result, detection_raw, detection_attempt, detection_prompt_str = run_detection_step(
                        llm=llm,
                        question=question_str,
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
                            "prompt_str": detection_prompt_str
                        }
                        df.at[index, column_name] = json.dumps(failure_dict)
                    else:
                        # Store detection result
                        detection_dict = {
                            "detection_step_attempt_count": detection_attempt,
                            "detection_step_raw": detection_raw,
                            "detection_step_parsed": detection_result.model_dump(),
                            "answer": detection_result.answer,
                            "evidence": detection_result.evidence,
                            "prompt_str": detection_prompt_str
                        }
                        df.at[index, column_name] = json.dumps(detection_dict)

        # Partial save after each row if desired
        if partial_save:
            df.to_csv(csv_path, index=False)

    # Final save
    df.to_csv(csv_path, index=False)
    print(f"  -> Done processing {os.path.basename(csv_path)}")


# --------------------------------
# 8. Helper to parse RMS ID & suffix
# --------------------------------
def parse_rms_id_and_suffix(prospectus_id_raw: str):
    """
    Returns (base_rms_id, numeric_suffix).
    Example:
      "ABC123"   -> ("ABC123", 0)
      "ABC123_1" -> ("ABC123", 1)
      "ABC123_2" -> ("ABC123", 2)
      etc.
    """
    parts = prospectus_id_raw.split("_", 1)
    base_id = parts[0]
    if len(parts) == 1:
        # no underscore
        return base_id, 0
    else:
        # parse numeric suffix
        suffix_str = parts[1]
        try:
            suffix_val = int(suffix_str)
        except ValueError:
            # If the suffix is not a valid integer, treat as 99999 or skip
            suffix_val = 99999
        return base_id, suffix_val


# --------------------------------
# 9. Main script
# --------------------------------
def main():
    parser = argparse.ArgumentParser(description="Process each *_parsed.csv with an LLM.")
    parser.add_argument(
        "--model_type",
        choices=["openai", "local"],
        default="openai",
        help="Which LLM to use ('openai' or 'local')."
    )
    parser.add_argument(
        "--local_model_path",
        type=str,
        default="",
        help="If --model_type=local, path to your local Llama model."
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=5,
        help="Number of unique Prospectus IDs to sample when sampling is enabled (0 to disable)."
    )
    parser.add_argument(
        "--rms_id",
        type=str,
        default=None,
        help="(Optional) Specific RMS ID to process (only the first suffix if multiple)."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="(Optional) Direct path to a single CSV to process."
    )

    args = parser.parse_args()

    # Fix the random seed for reproducible sampling
    random_seed = 42
    random.seed(random_seed)

    # 1) Get the LLM
    llm = get_llm(args.model_type, args.local_model_path)

    # 2) Define your question sets exactly as before
    questions_market_dynamics = {
        "Market Dynamics - a": "Does the text indicate that the company is exposed to risks associated with cyclical products?"
    }
    questions_intra_industry_competition = {
        "Intra-Industry Competition - a": "Does the text indicate that market pricing for the company's products or services is irrational or not based on fundamental factors?"
    }
    all_question_dicts = [
        questions_market_dynamics,
        # Add more question dictionaries if needed:
        # questions_intra_industry_competition,
    ]

    # If a direct CSV path is provided, process it immediately and exit
    if args.csv_path:
        print(f"Processing single CSV path: {args.csv_path}")
        process_single_csv(
            csv_path=args.csv_path,
            llm=llm,
            all_question_dicts=all_question_dicts,
            partial_save=True
        )
        print("Finished processing. Now evaluating this single CSV...")

        # Evaluate only that one file
        evaluate_models(args.csv_path)
        
        print("All done (single CSV).")
        sys.exit(0)

    # 3) Build a dictionary of { base_rms_id -> [(suffix_val, csv_path), ...] }
    processed_root = os.path.join(project_root, "data", "processed")
    rms_id_to_csvs = {}

    for root, dirs, files in os.walk(processed_root):
        # We only care about folders that contain "as_expected"
        if "as_expected" in root:
            # Find all CSVs that end in _parsed.csv
            csv_files = [f for f in files if f.lower().endswith("_parsed.csv")]
            for csv_file in csv_files:
                csv_path = os.path.join(root, csv_file)
                # Quickly read just the first row to get "PDF Page Count" and "Prospectus ID"
                try:
                    df_firstrow = pd.read_csv(csv_path, dtype=str, nrows=1)
                except Exception as e:
                    print(f"[ERROR] Failed to read first row from {csv_path}: {e}")
                    continue

                if df_firstrow.empty:
                    # no rows, skip
                    continue

                # The "Prospectus ID" column (e.g., "ABC123", "ABC123_1", etc.)
                prospectus_id_raw = df_firstrow.iloc[0].get("Prospectus ID", "")
                if not prospectus_id_raw:
                    continue

                # PDF Page Count check
                pdf_page_count_str = df_firstrow.iloc[0].get("PDF Page Count", "")
                try:
                    pdf_page_count = int(pdf_page_count_str)
                except ValueError:
                    print(f"[WARN] PDF Page Count not valid integer in {csv_path}, skipping.")
                    continue

                if pdf_page_count > 600:
                    # skip if over limit
                    continue

                # Parse base RMS ID and suffix
                base_rms_id, suffix_val = parse_rms_id_and_suffix(prospectus_id_raw)

                # Add to dictionary
                if base_rms_id not in rms_id_to_csvs:
                    rms_id_to_csvs[base_rms_id] = []
                rms_id_to_csvs[base_rms_id].append((suffix_val, csv_path))

    # We'll collect all the CSVs we actually process in this list
    processed_csv_paths = []

    # If a single RMS ID is requested, process only that one (lowest suffix if multiple)
    if args.rms_id:
        # Make sure we handle the case where it might have underscores
        requested_base_id, _ = parse_rms_id_and_suffix(args.rms_id)
        if requested_base_id not in rms_id_to_csvs:
            print(f"No matching RMS ID found for {args.rms_id}. Exiting.")
            sys.exit(0)
        # Sort by suffix, pick the first
        csv_entries = rms_id_to_csvs[requested_base_id]
        csv_entries.sort(key=lambda x: x[0])  # sort by suffix_val
        chosen_suffix, chosen_csv_path = csv_entries[0]
        print(f"Found RMS ID {requested_base_id}, using suffix={chosen_suffix} -> {chosen_csv_path}")

        process_single_csv(
            csv_path=chosen_csv_path,
            llm=llm,
            all_question_dicts=all_question_dicts,
            partial_save=True
        )
        processed_csv_paths.append(chosen_csv_path)

        # Evaluate just this one
        print(f"Evaluating {chosen_csv_path} now...")
        evaluate_models(chosen_csv_path)
        print("All done (single RMS ID).")
        sys.exit(0)

    # 4) Otherwise, do the normal sampling logic (if sample > 0) or process all
    sample_size = args.sample
    all_rms_ids = list(rms_id_to_csvs.keys())

    if sample_size > 0:
        print(f"Sampling is enabled. We will sample {sample_size} unique base RMS IDs (seed={random_seed}).")
        # If there aren't enough RMS IDs, sample what we can
        if len(all_rms_ids) > sample_size:
            sampled_rms_ids = random.sample(all_rms_ids, sample_size)
        else:
            sampled_rms_ids = all_rms_ids
    else:
        # No sampling => process all RMS IDs
        sampled_rms_ids = all_rms_ids

    # 5) For each sampled base RMS ID, pick the CSV with the smallest suffix
    for base_rms_id in sampled_rms_ids:
        csv_entries = rms_id_to_csvs[base_rms_id]
        if not csv_entries:
            continue
        # Sort by suffix so that we pick the lowest suffix first
        csv_entries.sort(key=lambda x: x[0])  # (suffix_val, csv_path)
        chosen_suffix, chosen_csv_path = csv_entries[0]

        # Process that single CSV
        print(f"\n--- Processing {base_rms_id}, chosen suffix={chosen_suffix} -> {chosen_csv_path} ---")
        process_single_csv(
            csv_path=chosen_csv_path,
            llm=llm,
            all_question_dicts=all_question_dicts,
            partial_save=True
        )

        processed_csv_paths.append(chosen_csv_path)

    print("Finished processing all CSVs in this run.")

    # Now do an aggregated evaluation over all processed CSVs
    if processed_csv_paths:
        print("\nRunning aggregated evaluation over all processed CSVs...")
        evaluate_models(processed_csv_paths)
    

if __name__ == "__main__":
    main()