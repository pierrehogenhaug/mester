import json
import os
import sys
import re
import pandas as pd
import argparse
import random
from tqdm import tqdm
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
#from langchain.chat_models import ChatOpenAI
from pydantic import BaseModel, Field, field_validator, ValidationError

# Make sure your local paths are set up properly:
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.evaluation.evaluation_2 import evaluate_models

# --------------------------------
# 0. Run Guide
# --------------------------------
# python scripts/analysis/run_evaluation.py --csv_path "/path/to/some_parsed.csv"
# python scripts/analysis/run_evaluation.py --rms_id ABC123
# python scripts/analysis/run_evaluation.py --sample 3

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
# 2b. Pydantic Model (Evaluation)
# --------------------------------
class LlmEvaluationResponse(BaseModel):
    # Identical structure to detection for consistency
    answer: str = Field(..., alias="Answer")
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
# 4. Detection Prompt & Helper
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
# 5. Evaluation Prompt & Helper
# --------------------------------
EVALUATION_PROMPT = """
You are responsible for verifying the evaluation of the following risk factor: {question}
Text Under Review:
{subsection_title}
{subsection_text}

A previous evaluation concluded:
Answer: {answer}
Evidence: {evidence}

We have two reference cases for context:
(1) Example where the risk factor is not present:
{negative_case}
(2) Example where the risk factor is present:
{positive_case}

Your Task:
1. Analyze the Evidence:
   - Review the "Evidence" provided in the context of the "Text Under Review".
   
2. Compare with Reference Cases:
   - Positive Reference: Determine if the evidence aligns closely with case (1)
   - Negative Reference: Assess whether the evidence is more consistent with the case (2) or if it lacks sufficient support.

3. Decide on the Final Answer:
   - Confirm "Yes": If the evidence matches the Positive Case, indicating the presence of the risk factor.
   - Override to "No": If the evidence is weak, inconsistent with the Positive Case, or aligns more with the Negative Case.

4. Provide Reasoning:
   - Offer a brief explanation supporting your decision.

Please provide your evaluation in JSON with the following structure:
{{
  "Answer": "Yes" or "No",
  "Reasoning": "A brief explanation of your decision."
}}
"""

def build_evaluation_prompt(input_data: Dict[str, Any]) -> str:
    return EVALUATION_PROMPT.format(
        question=input_data["question"],
        subsection_title=input_data["subsection_title"],
        subsection_text=input_data["subsection_text"],
        answer=input_data["answer"],
        evidence=input_data["evidence"],
        negative_case=input_data["negative_case"],
        positive_case=input_data["positive_case"]
    )


# --------------------------------
# 6. Detection Step
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
# 7. Evaluation Step
# --------------------------------
def run_evaluation_step(
    llm,
    question_str: str,
    subsection_title: str,
    subsection_text: str,
    refs: Dict[str, str],
    detection_answer: str,
    detection_evidence: str,
    max_retries: int = 3
):
    """
    Runs the evaluation step with retries, returning either a valid LlmEvaluationResponse
    or None if it fails all attempts.
    """
    # Build the input for step 2
    step2_input = {
        "question": question_str,
        "subsection_title": subsection_title,
        "subsection_text": subsection_text,
        "negative_case": refs["negative_case"],
        "positive_case": refs["positive_case"],
        "answer": detection_answer,
        "evidence": detection_evidence or "",
    }
    prompt_str_evaluation = build_evaluation_prompt(step2_input)

    for attempt in range(1, max_retries + 1):
        try:
            raw_response = call_llm(llm, prompt_str_evaluation)
            # Attempt to parse as JSON
            extracted_json = extract_first_json_object(raw_response)
            parsed_evaluation = json.loads(extracted_json)
            evaluation_result = LlmEvaluationResponse.model_validate(parsed_evaluation)
            return (evaluation_result, raw_response, attempt, prompt_str_evaluation)
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"[Evaluation] Attempt {attempt} failed parsing/validation: {e}")
        except Exception as ex:
            print(f"[Evaluation] Unexpected error: {ex}")

    # If all retries fail
    return (None, None, max_retries, prompt_str_evaluation)

# --------------------------------
# 8. Utility to check if row is fully processed
# --------------------------------
def row_fully_processed(row, columns):
    """
    Returns True if *all* relevant columns for a row are non-empty.
    This only checks that some JSON exists, but does not check if evaluation
    was done. Adjust logic if needed.
    """
    for c in columns:
        val = row.get(c, "")
        if pd.isnull(val) or str(val).strip() == "":
            return False
    return True

# --------------------------------
# 9. Reference Cases
# --------------------------------
# Example references for cyclical product risk
cyclical_product_risk_negative = """
Example 1 (Negative Case)
Scenario: Submetering is an infrastructure-like business with long-term contractual agreements and non-discretionary demand. 
Observations: 80% of revenue is recurring; stable demand even during recessions. 
Model Decision: {"Answer": "No", "Evidence": "This business does not show cyclical product risk due to its long-term, stable nature."}
"""

cyclical_product_risk_positive = """
Example 2 (Positive Case)
Scenario: Construction equipment rented day-to-day. 
Rationale: 57% of revenue from construction equipment; weak macroeconomic conditions historically had a high impact on business (-25% EBITDA in 2009). 
Model Decision: {"Answer": "Yes", "Evidence": "The reliance on construction equipment and historical downturn in EBITDA indicates exposure to cyclical product risks."}
"""

# Example references for intra-industry competition
competition_negative = """
Example 1 (Negative Case)
Scenario: High market concentration and the dominance of well-established brands versus private labels.
Observations: Historically, leading players have been able to increase prices through product innovation.
Model Decision: {"Answer": "No", "Evidence": "While there is mention of branding and price increases, it does not suggest an irrational or fundamentally unsustainable level of competition."}
"""

competition_positive = """
Example 2 (Positive Case)
Scenario: A highly fragmented product offering dominated by non-branded or generic products.
Observations: The pricing environment has been highly competitive in recent years with competitors driving down prices. As a result, the company’s market share declined when prices were kept unchanged to protect profitability.
Model Decision: {"Answer": "Yes", "Evidence": "A highly fragmented market with downward price pressure indicates intra-industry competition that is not based on underlying fundamentals."}
"""

technology_risk_negative = """
Example 1 (Negative Case)
Scenario: A leading cloud computing provider offering scalable storage and computing solutions to businesses worldwide.
Observations: Continuous investment in cutting-edge technology, strong R&D capabilities, and a robust infrastructure that supports high availability and security standards.
Model Decision: {"Answer": "No", "Evidence": "The company's proactive investment in technology and strong infrastructure mitigate technology risk, positioning it as a resilient leader in the cloud computing industry."}
"""

technology_risk_positive = """
Example 2 (Positive Case)
Scenario: Traditional brick-and-mortar retail chain specializing in apparel with minimal online presence.
Observations: Limited adoption of e-commerce platforms, outdated inventory management systems, and slow response to digital marketing trends.
Model Decision: {"Answer": "Yes", "Evidence": "The company's lack of technological advancement and slow adaptation to digital retail trends expose it to significant technology risk, making it vulnerable to competition from more technologically adept retailers."}
"""

regulatory_framework_negative = """
Example 1 (Negative Case)
Scenario: Ongoing planning for elderly care is regulated at a national level with a high degree of reliance on the governmental level.
Observations: Elderly care is considered political goodwill, but the French state directs individuals to the most economical solutions in terms of public funding (i.e., homecare for the least dependent and medicalized nursing homes for others). The company offers both services and is therefore shielded from this strategic government focus.
Model Decision: {"Answer": "No", "Evidence": "The company's diversified service offerings protect it from targeted regulatory pressures, indicating a low regulatory framework risk."}
"""

regulatory_framework_positive = """
Example 2 (Positive Case)
Scenario: Collections are regulated by various authorities and according to various statutes in each European country, with all countries aiming for customers to be treated “fairly”.
Observations: The company profits from the most vulnerable group and is under political scrutiny. There is a trend in laws, rules, and regulations requiring increased availability of historic information about receivables for collection purposes, along with a higher degree of consumer protection.
Model Decision: {"Answer": "Yes", "Evidence": "The company operates under stringent and evolving regulations aimed at protecting vulnerable consumers, increasing its exposure to regulatory framework risks."}
"""

reference_cases_for_cyclical = {
    "negative_case": cyclical_product_risk_negative,
    "positive_case": cyclical_product_risk_positive
}

reference_cases_for_competition = {
    "negative_case": competition_negative,
    "positive_case": competition_positive
}

reference_cases_for_technology = {
    "negative_case": technology_risk_negative,
    "positive_case": technology_risk_positive
}

reference_cases_for_regulatory = {
    "negative_case": regulatory_framework_negative,
    "positive_case": regulatory_framework_positive
}

references_dict: Dict[str, Dict[str, str]] = {
    "Market Dynamics - a": reference_cases_for_cyclical,
    "Intra-Industry Competition - a": reference_cases_for_competition,
    "Technology Risk - a": reference_cases_for_technology,
    "Regulatory Framework - a": reference_cases_for_regulatory,
    # Add more if needed
}


# --------------------------------
# 10. Function to process a single CSV (Detection + Evaluation)
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
      - calls the LLM row by row for detection (if needed) then evaluation (if needed)
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

    # Check if entire CSV is fully processed (detection at least)
    all_rows_done = True
    for _, row in df.iterrows():
        if not row_fully_processed(row, columns_to_process):
            all_rows_done = False
            break

    if all_rows_done:
        print(f"  -> All rows in {os.path.basename(csv_path)} are already processed (detection). Checking evaluation step next...")

    # Process row-by-row
    print(f"  -> Processing {csv_path} (Detection + Evaluation) ...")
    for index in tqdm(range(len(df)), desc=f"  Rows in {os.path.basename(csv_path)}"):
        row = df.iloc[index]

        subsection_title = str(row.get("Subsubsection Title", ""))
        subsection_text = str(row.get("Subsubsection Text", ""))

        # If the subsection text is empty, skip
        if not subsection_text.strip():
            continue

        # For each question, detect if needed, evaluate if needed
        for question_dict in all_question_dicts:
            for column_name, question_str in question_dict.items():
                current_value = row.get(column_name, "").strip()

                # If there's nothing at all in the cell, we need to run detection
                if not current_value:
                    # -- DETECTION --
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
                        detection_dict = {
                            "detection_step_attempt_count": detection_attempt,
                            "detection_step_raw": detection_raw,
                            "detection_step_parsed": detection_result.model_dump(),
                            "answer": detection_result.answer,
                            "evidence": detection_result.evidence,
                            "prompt_str": detection_prompt_str
                        }
                        df.at[index, column_name] = json.dumps(detection_dict)

                    # Save partial
                    if partial_save:
                        df.to_csv(csv_path, index=False)

                # Re-fetch updated JSON after detection step
                updated_value = df.at[index, column_name]
                if not updated_value:
                    # Something went wrong, skip evaluation
                    continue

                # Attempt to parse existing detection JSON
                try:
                    detection_dict = json.loads(updated_value)
                except json.JSONDecodeError:
                    # Not valid JSON => skip
                    continue

                # If detection step was not successful, skip evaluation
                if "detection_step_parsed" not in detection_dict:
                    # Means detection completely failed
                    continue

                # Check if evaluation step has already been run
                if "evaluation_step_parsed" in detection_dict:
                    # If the evaluation results have already been saved under the new keys, skip.
                    if "evaluation_answer" in detection_dict and "evaluation_evidence" in detection_dict:
                        continue
                    else:
                        # Otherwise, reconstruct the correct keys using the parsed data.
                        # Restore the detection result (if available) from detection_step_parsed.
                        detection_parsed = detection_dict.get("detection_step_parsed", {})
                        if isinstance(detection_parsed, dict):
                            detection_dict["answer"] = detection_parsed.get("answer", "")
                            detection_dict["evidence"] = detection_parsed.get("evidence", "")
                        # Extract evaluation results from evaluation_step_parsed.
                        evaluation_parsed = detection_dict.get("evaluation_step_parsed", {})
                        if isinstance(evaluation_parsed, dict):
                            detection_dict["evaluation_answer"] = evaluation_parsed.get("answer", "")
                            detection_dict["evaluation_evidence"] = evaluation_parsed.get("evidence", "")
                        # Save the fixed cell back to the DataFrame.
                        df.at[index, column_name] = json.dumps(detection_dict)
                        continue     

                # -- EVALUATION --
                # Grab references for this column (if any), else blank
                refs = references_dict.get(column_name, {
                    "negative_case": "",
                    "positive_case": ""
                })

                detection_answer = detection_dict.get("answer", "")
                detection_evidence = detection_dict.get("evidence", "")
                
                evaluation_result, evaluation_raw, evaluation_attempt, evaluation_prompt_str = run_evaluation_step(
                    llm=llm,
                    question_str=question_str,
                    subsection_title=subsection_title,
                    subsection_text=subsection_text,
                    refs=refs,
                    detection_answer=detection_answer,
                    detection_evidence=detection_evidence,
                    max_retries=3
                )
                if not evaluation_result:
                    # Could not parse after retries
                    detection_dict["evaluation_step_attempt_count"] = evaluation_attempt
                    detection_dict["evaluation_step_raw"] = evaluation_raw or "No valid output"
                    detection_dict["error_evaluation"] = "Evaluation failed"
                    detection_dict["prompt_str_evaluation"] = evaluation_prompt_str
                else:
                    # Append to detection_dict
                    detection_dict["evaluation_step_attempt_count"] = evaluation_attempt
                    detection_dict["evaluation_step_raw"] = evaluation_raw
                    detection_dict["evaluation_step_parsed"] = evaluation_result.model_dump()
                    detection_dict["evaluation_answer"] = evaluation_result.answer
                    detection_dict["evaluation_evidence"] = evaluation_result.evidence
                    detection_dict["prompt_str_evaluation"] = evaluation_prompt_str

                # Save back
                df.at[index, column_name] = json.dumps(detection_dict)

                # Partial save after each column
                if partial_save:
                    df.to_csv(csv_path, index=False)

    # Final save
    df.to_csv(csv_path, index=False)
    print(f"  -> Done processing {os.path.basename(csv_path)}")

# --------------------------------
# 11. Helper to parse RMS ID & suffix
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
# 12. Main script
# --------------------------------
def main():
    parser = argparse.ArgumentParser(description="Process each *_parsed.csv with an LLM (detection + evaluation).")
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

    # 1) Get the LLM for detection + evaluation
    llm = get_llm(args.model_type, args.local_model_path)

    # 2) Define your question sets (unchanged from detection usage)
    questions_market_dynamics = {
        "Market Dynamics - a": "Does the text indicate that the company is exposed to risks associated with cyclical products?"
    }
    questions_intra_industry_competition = {
        "Intra-Industry Competition - a": "Does the text indicate that market pricing for the company's products or services is irrational or not based on fundamental factors?"
    }
    questions_technology_risk = {
        "Technology Risk - a": "Does the text indicate that the industry is susceptible to rapid technological advances or innovations?"
        # ,"Technology Risk - b": "Does the text indicate that the company is perceived as a disruptor or is threatened by emerging technological changes?"
    }
    questions_regulatory_framework = {
        "Regulatory Framework - a": "Does the text indicate that the industry is subject to a high degree of regulatory scrutiny?"
        # ,"Regulatory Framework - b": "Does the text indicate a high dependency on regulation or being a beneficiary from regulation in an unstable regulatory environment?"
    }

    all_question_dicts = [
        questions_market_dynamics,
        questions_intra_industry_competition,
        questions_regulatory_framework,
        questions_technology_risk
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
        print("Finished detection+evaluation. Now evaluating this single CSV...")

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

                # The "Prospectus ID" column
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
        if len(all_rms_ids) > sample_size and sample_size == 40 and random_seed == 42:
            sampled_rms_ids = ['367', '999', '1609', '625', '1108', '673', '219', '440', '328', '355', '139', '1629', '1074', '352', '1052', '946', '1897', '317', '653', '642', '1525', '1277', '935', '433', '153', '221', '1261', '199', '130', '252', '377', '84', '518', '201', '989', '1069', '1727', '1739', '258', '1127']
            print(f"Using pre-selected RMS IDs for consistent sampling.: {sampled_rms_ids}")
        elif len(all_rms_ids) > sample_size:
            sampled_rms_ids = random.sample(all_rms_ids, sample_size)
            print(f"Sampled RMS IDs: {sampled_rms_ids}")
        else:
            sampled_rms_ids = all_rms_ids
            print(f"Less than {sample_size} unique RMS IDs found. Processing all.")
    else:
        # No sampling => process all RMS IDs
        sampled_rms_ids = all_rms_ids
        print(f"Sampled RMS IDs: {sampled_rms_ids}")

    # 5) For each sampled base RMS ID, pick the CSV with the smallest suffix
    chosen_csv_paths = []
    for base_rms_id in sampled_rms_ids:
        csv_entries = rms_id_to_csvs[base_rms_id]
        if not csv_entries:
            continue
        # Sort by suffix so that we pick the lowest suffix first
        csv_entries.sort(key=lambda x: x[0])  # (suffix_val, csv_path)
        chosen_suffix, chosen_csv_path = csv_entries[0]
        chosen_csv_paths.append(chosen_csv_path)

    # 6) Process the chosen CSVs in parallel
    max_workers = 5  # or however many threads you want
    print(f"\nProcessing up to {max_workers} CSVs in parallel...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_csv = {}
        for cpath in chosen_csv_paths:
            # Submit each CSV for processing
            future = executor.submit(
                process_single_csv,
                cpath,
                llm,
                all_question_dicts,
                True  # partial_save
            )
            future_to_csv[future] = cpath

        for future in as_completed(future_to_csv):
            cpath = future_to_csv[future]
            try:
                future.result()  # raises if exception
            except Exception as exc:
                print(f"[ERROR] {cpath} generated an exception: {exc}")
            else:
                processed_csv_paths.append(cpath)
                print(f"[INFO] Completed: {cpath}")

    print("Finished detection+evaluation for all sampled CSVs in this run.")

    # Now do an aggregated evaluation over all processed CSVs
    if processed_csv_paths:
        print("\nRunning aggregated evaluation over all processed CSVs...")
        evaluate_models(processed_csv_paths)
        print("All done.")
    else:
        print("No CSVs were processed. Exiting.")


if __name__ == "__main__":
    main()