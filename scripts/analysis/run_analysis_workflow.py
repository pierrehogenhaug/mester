import json
import os
import sys
import re
import pandas as pd
import argparse
from tqdm import tqdm
from typing import Optional, Dict, Any

from dotenv import load_dotenv
#from langchain.chat_models import ChatOpenAI
from pydantic import BaseModel, Field, field_validator, ValidationError

# -----------------------------
# 0. Run Guide
# -----------------------------
# python run_analysis_workflow.py --model_type=local --local_model_path=../MyLocalModel.gguf --sample
# python run_analysis_workflow.py --model_type=local --sample
# python run_analysis_workflow.py --sample


# Make sure your local paths are set up properly:
# (Adjust if your script is nested in subfolders)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# -----------------------------
# 1. JSON Extraction Helper
# -----------------------------
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

# -----------------------------
# 2. Pydantic Models
# -----------------------------
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


class LlmEvaluationResponse(BaseModel):
    # Accept both "final_answer" and "FinalAnswer"
    final_answer: str = Field(..., alias="FinalAnswer")
    # Accept both "reasoning" and "Reasoning"
    reasoning: Optional[str] = Field(None, alias="Reasoning")

    class Config:
        populate_by_name = True

    @field_validator("final_answer", mode="before")
    def normalize_final_answer(cls, v):
        if isinstance(v, str):
            lower = v.strip().lower()
            if lower == "yes":
                return "Yes"
            elif lower == "no":
                return "No"
        raise ValueError("final_answer must be 'yes' or 'no' (case insensitive)")
        return v


# -----------------------------
# 3. LLM Setup
# -----------------------------
def get_llm(model_type: str, local_model_path: str):
    """
    Returns the appropriate LLM instance based on the specified model_type:
      - "openai" uses ChatOpenAI (requires OPENAI_API_KEY).
      - "local" uses LlamaCpp from langchain_community.llms.
    """
    if model_type == "openai":
        # (If you have a custom `langchain_openai.py`, import from there instead)
        # from langchain.chat_models import ChatOpenAI
        from langchain_openai import ChatOpenAI

        
        load_dotenv()  # Loads OPENAI_API_KEY from .env
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if not OPENAI_API_KEY:
            print("OPENAI_API_KEY not found. Make sure it's set in your .env file.")
            sys.exit(1)

        return ChatOpenAI(
            model_name="gpt-4o-mini",  # or "gpt-3.5-turbo", etc.
            openai_api_key=OPENAI_API_KEY,
            max_tokens=256,
        )

    elif model_type == "local":
        # Import from langchain_community
        from langchain_community.llms import LlamaCpp
        return LlamaCpp(
            model_path=local_model_path,
            n_ctx=4096,
            n_gpu_layers=35,
            max_tokens=256,
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

# We'll define a wrapper function that calls the LLM
# (This is just a stylistic choice; you can also directly do llm.call_as_llm() in your steps)
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

# load_dotenv()  # Loads OPENAI_API_KEY from .env
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# if not OPENAI_API_KEY:
#     print("OPENAI_API_KEY not found. Make sure it's set in your .env file.")
#     sys.exit(1)

# # Using LangChain's ChatOpenAI
# llm_openai = ChatOpenAI(
#     model_name="gpt-4o-mini",  # or "gpt-3.5-turbo", etc.
#     openai_api_key=OPENAI_API_KEY,
#     max_tokens=256,
# )

# -----------------------------
# 4. Define Prompts & Chains
# -----------------------------
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
        subsection_text=input_data["subsection_text"]
    )

EVALUATION_PROMPT = """
You are double-checking the detection of a risk factor.

We have two reference cases: one negative example and one positive example.

Negative Case:
{negative_case}

Positive Case:
{positive_case}

---

Prospectus text (the text in question):
{prospectus_text}

Detection step gave:
Answer = {answer},
Evidence = {evidence}

Instruction:
1. Compare the prospectus text and the evidence to the reference cases.
2. If the evidence is weak or does not truly match the reference risk factor—or if the text does not appear sufficiently similar to the positive reference—then override to "No".
3. If it seems consistent with the positive reference case, confirm "Yes".
4. Provide a brief reasoning.

Output in JSON with no extra commentary, using this format:
{{
  "FinalAnswer": "Yes" or "No",
  "Reasoning": "A brief reasoning."
}}
"""

def build_evaluation_prompt(input_data: Dict[str, Any]) -> str:
    return EVALUATION_PROMPT.format(**input_data)


# -----------------------------
# 5. Reference Cases
# -----------------------------
# Example references for cyclical product risk
cyclical_product_risk_negative = """
### Example 1 (Negative Case)
**Scenario:** Submetering is an infrastructure-like business with long-term contractual agreements and non-discretionary demand. 
**Observations:** 80% of revenue is recurring; stable demand even during recessions. 
Model Decision: {"Answer": "No", "Evidence": "This business does not show cyclical product risk due to its long-term, stable nature."}
"""

cyclical_product_risk_positive = """
### Example 2 (Positive Case)
**Scenario:** Construction equipment rented day-to-day. 
**Rationale:** 57% of revenue from construction equipment; weak macroeconomic conditions historically had a high impact on business (-25% EBITDA in 2009). 
Model Decision: {"Answer": "Yes", "Evidence": "The reliance on construction equipment and historical downturn in EBITDA indicates exposure to cyclical product risks."}
"""

# Example references for intra-industry competition
competition_negative = """
### Example 1 (Negative Case)
**Scenario:** High market concentration and the dominance of well-established brands versus private labels.
**Observations:** Historically, leading players have been able to increase prices through product innovation.
Model Decision: {"Answer": "No", "Evidence": "While there is mention of branding and price increases, it does not suggest an irrational or fundamentally unsustainable level of competition."}
"""

competition_positive = """
### Example 2 (Positive Case)
**Scenario:** A highly fragmented product offering dominated by non-branded or generic products.
**Observations:** The pricing environment has been highly competitive in recent years with competitors driving down prices. As a result, the company’s market share declined when prices were kept unchanged to protect profitability.
Model Decision: {"Answer": "Yes", "Evidence": "A highly fragmented market with downward price pressure indicates intra-industry competition that is not based on underlying fundamentals."}
"""

reference_cases_for_cyclical = {
    "negative_case": cyclical_product_risk_negative,
    "positive_case": cyclical_product_risk_positive
}

reference_cases_for_competition = {
    "negative_case": competition_negative,
    "positive_case": competition_positive
}

# -----------------------------
# 6. Two-step LLM invocation
# -----------------------------
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
            # raw_response = llm_openai.invoke(prompt_str)
            # response = llm_openai.invoke(prompt_str)
            # raw_response = response.content
            
            # raw_response = llm_openai.call_as_llm(prompt_str)
 
            # Attempt to parse as JSON
            extracted_json = extract_first_json_object(raw_response)
            parsed_detection = json.loads(extracted_json)
            detection_result = LlmDetectionResponse.model_validate(parsed_detection)
            return (detection_result, raw_response, attempt, prompt_str)
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"[Detection] Attempt {attempt} for question '{question}' failed parsing/validation: {e}")
        except Exception as ex:
            print(f"[Detection] Unexpected error: {ex}")
    return (None, None, max_retries, prompt_str)


def run_evaluation_step(
    llm,
    detection_result: LlmDetectionResponse,
    subsection_text: str,
    column_name: str,
    references_dict: Dict[str, Dict[str, str]],
    max_retries=3
):
    """
    Runs the evaluation step with retries, returning either a valid LlmEvaluationResponse
    or None if it fails all attempts.
    """
    
    # 1) Find the correct references for this column/question:
    if column_name in references_dict:
        refs = references_dict[column_name]
    else:
        # Fallback: if column name not found, default to cyclical references, or raise error
        refs = reference_cases_for_cyclical

    step2_input = {
        "negative_case": refs["negative_case"],
        "positive_case": refs["positive_case"],
        "prospectus_text": subsection_text,
        "answer": detection_result.answer,
        "evidence": detection_result.evidence or "",
    }
    
    prompt_str = build_evaluation_prompt(step2_input)

    for attempt in range(1, max_retries + 1):
        try:
            raw_response = call_llm(llm, prompt_str)
            # response = llm_openai.invoke(prompt_str)
            # raw_response = response.content
            # raw_response = llm_openai.call_as_llm(prompt_str)
 
            # Attempt to parse as JSON
            extracted_json = extract_first_json_object(raw_response)
            parsed_evaluation = json.loads(extracted_json)
            evaluation_result = LlmEvaluationResponse.model_validate(parsed_evaluation)
            return (evaluation_result, raw_response, attempt, prompt_str)
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"[Evaluation] Attempt {attempt} failed parsing/validation: {e}")
        except Exception as ex:
            print(f"[Evaluation] Unexpected error: {ex}")

    return (None, None, max_retries, prompt_str)

# -----------------------------
# 7. Main logic: CSV reading/writing and iteration
# -----------------------------
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
        default="../Llama-3.2-1B-Instruct-Q8_0.gguf", # local model path
        # default="../../Llama-3.2-1B-Instruct-Q8_0.gguf", # hpc model path?
        help="Path to the local LLaMA model if --model_type=local."
    )
    parser.add_argument(
        "--sample",
        action='store_true',
        help="Enable sampling of 100 unique Prospectus IDs. If not set, process the full dataset."
    )
    args = parser.parse_args()

    # Get the appropriate LLM
    llm = get_llm(args.model_type, args.local_model_path)
    
    # Use args.sample to decide if we do sampling
    perform_sampling = args.sample

    # -- File paths
    model_path = (
        "openai_gpt-4o_mini" if args.model_type == "openai"
        else os.path.basename(args.local_model_path).replace('.', '_')
    )

    # -- File paths
    # model_path = "openai_gpt-4o_mini"
    output_dir = os.path.join('.', 'data', model_path.replace('/', '_'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if perform_sampling:
        processed_file_path = os.path.join(output_dir, 'prospectuses_data_processed_workflow_sampled.csv')
    else:
        processed_file_path = os.path.join(output_dir, 'prospectuses_data_processed_workflow.csv')
    raw_file_path = './data/prospectuses_data.csv'

    # Load or create processed df
    if os.path.exists(processed_file_path):
        df_LLM = pd.read_csv(processed_file_path)
        df_LLM.reset_index(drop=True, inplace=True)
        print(f"Loaded existing processed CSV: {processed_file_path}")
    else:
        if not os.path.exists(raw_file_path):
            print("Raw CSV does not exist. Exiting.")
            sys.exit(1)
        df_LLM = pd.read_csv(raw_file_path)
        
        # Filter out rows where 'Section ID' == 'failed parsing'
        if 'Section ID' in df_LLM.columns:
            df_LLM = df_LLM[df_LLM['Section ID'] != "failed parsing"]
        else:
            print("Column 'Section ID' not found. Skipping filter.")

        # Basic check for required columns
        if 'Prospectus ID' not in df_LLM.columns:
            print("Column 'Prospectus ID' not found. Exiting.")
            sys.exit(1)

        # -------------------------------------------------------
        # PERFORM SAMPLING IF REQUESTED
        # -------------------------------------------------------
        if perform_sampling:
            sample_size = 100
            random_seed = 42
            unique_ids = df_LLM['Prospectus ID'].dropna().unique()
            if len(unique_ids) < sample_size:
                print(f"Not enough unique Prospectus IDs to sample {sample_size}. "
                      f"Available: {len(unique_ids)}.")
                sys.exit(1)

            sampled_ids = pd.Series(unique_ids).sample(
                n=sample_size, random_state=random_seed
            ).tolist()
            print(f"Sampled {len(sampled_ids)} Prospectus IDs.")

            df_LLM = df_LLM[df_LLM['Prospectus ID'].isin(sampled_ids)].copy()
            df_LLM.reset_index(drop=True, inplace=True)
            print("Filtered data to include only sampled Prospectus IDs.")

        # Ensure the DataFrame is written once initially
        df_LLM.to_csv(processed_file_path, index=False)
        df_LLM.reset_index(drop=True, inplace=True)


    # -- Define questions (same columns as example, though only detection is used)
    questions_market_dynamics = {
        "Market Dynamics - a": "Does the text mention that the company is exposed to risks associated with cyclical products?"
    }
    questions_intra_industry_competition = {
        "Intra-Industry Competition - a": "Does the text mention that market pricing for the company's products or services is irrational or not based on fundamental factors?"
    }

    questions_regulatory_framework = {
        "Regulatory Framework - a": "Does the text mention that the industry is subject to a high degree of regulatory scrutiny?"
        # ,"Regulatory Framework - b": "Does the text mention a high dependency on regulation or being a beneficiary from regulation in an unstable regulatory environment?"
    }
    questions_technology_risk = {
        "Technology Risk - a": "Does the text mention that the industry is susceptible to rapid technological advances or innovations?"
        # ,"Technology Risk - b": "Does the text mention that the company is perceived as a disruptor or is threatened by emerging technological changes?"
    }

    all_question_dicts = [
        questions_market_dynamics
        ,questions_intra_industry_competition
        # ,questions_regulatory_framework,
        # questions_technology_risk
    ]

    # Collect all columns we intend to fill:
    columns_to_process = []
    for qd in all_question_dicts:
        columns_to_process.extend(list(qd.keys()))

    # Ensure columns exist in the dataframe
    for column_name in columns_to_process:
        if column_name not in df_LLM.columns:
            df_LLM[column_name] = ""
        # Convert to string to avoid type issues
        df_LLM[column_name] = df_LLM[column_name].astype("string")

    # -------------
    # Logic to skip overwriting if everything is already processed
    # -------------
    def row_fully_processed(row, columns):
        """
        Returns True if *all* relevant columns for a row are non-empty.
        """
        for c in columns:
            if pd.isnull(row[c]) or row[c].strip() == "":
                return False
        return True

    # Check if the entire dataframe is fully processed
    all_rows_done = True
    for i, row in df_LLM.iterrows():
        if not row_fully_processed(row, columns_to_process):
            all_rows_done = False
            break

    if all_rows_done:
        print("All rows have already been processed. Exiting without overwriting.")
        sys.exit(0)


    # -------------
    # test-run on the first few rows
    # -------------
    # num_test_rows = 3
    # total_rows = df_LLM.shape[0]
    # if total_rows < num_test_rows:
    #     num_test_rows = total_rows

    # for index in tqdm(range(0, num_test_rows), desc="Processing Test Rows"):
    #     row = df_LLM.iloc[index]

    # -------------
    # normal run on all rows
    # -------------
    start_index = 0
    for index in tqdm(range(start_index, df_LLM.shape[0]), desc="Processing Rows"):

        # If the row is already fully processed, skip it
        if row_fully_processed(row, columns_to_process):
            continue

        subsection_title = str(row.get("Subsubsection Title", ""))
        subsection_text = str(row.get("Subsubsection Text", ""))


        # If these are blank or invalid, you could skip:
        if not subsection_text.strip():
            continue

        for question_dict in all_question_dicts:
            for column_name, question in question_dict.items():
                current_value = df_LLM.at[index, column_name]
                # If the cell is empty or unprocessed, run the two-step pipeline
                if pd.isnull(current_value) or current_value.strip() == "":
                    
                    references_dict = {
                        "Market Dynamics - a": reference_cases_for_cyclical,
                        "Intra-Industry Competition - a": reference_cases_for_competition
                    }

                    # STEP 1: Detection
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
                            "prompt_str": detection_prompt_str
                        }
                        df_LLM.at[index, column_name] = json.dumps(failure_dict)
                        continue

                    # STEP 2: Evaluation
                    evaluation_result, evaluation_raw, evaluation_attempt, evaluation_prompt_str = run_evaluation_step(
                        llm=llm,
                        detection_result=detection_result,
                        subsection_text=subsection_text,
                        column_name=column_name,
                        references_dict=references_dict,
                        max_retries=3,
                    )

                    if not evaluation_result:
                        # Could not parse after retries
                        failure_dict = {
                            "detection_step_attempt_count": detection_attempt,
                            "detection_step_raw": detection_raw,
                            "detection_step_parsed": detection_result.model_dump(),
                            "evaluation_step_attempt_count": evaluation_attempt,
                            "evaluation_step_raw": evaluation_raw or "No valid output",
                            "error": "Evaluation failed",
                            "prompt_str": evaluation_prompt_str
                        }
                        df_LLM.at[index, column_name] = json.dumps(failure_dict)
                        continue

                    # Store combined results in a single dictionary
                    combined_dict = {
                        "detection_step_attempt_count": detection_attempt,
                        "detection_step_raw": detection_raw,
                        "detection_step_parsed": detection_result.model_dump(),
                        "evaluation_step_attempt_count": evaluation_attempt,
                        "evaluation_step_raw": evaluation_raw,
                        "evaluation_step_parsed": evaluation_result.model_dump(),
                        "final_decision": evaluation_result.final_answer,
                        "reasoning": evaluation_result.reasoning,
                        "prompt_str": evaluation_prompt_str
                    }
                    df_LLM.at[index, column_name] = json.dumps(combined_dict)

        # Save partial progress
        df_LLM.to_csv(processed_file_path, index=False)

    # Final save
    df_LLM.to_csv(processed_file_path, index=False)
    print("Processing complete. Results saved to:", processed_file_path)


if __name__ == "__main__":
    main()