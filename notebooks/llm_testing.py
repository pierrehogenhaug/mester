from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_community.llms import LlamaCpp

# ---- 1. Your inputs & prompt ----

subsection_title = "Risks related to the ADLER Group’s Business Activities and Industry"
subsection_text = """Our business is significantly dependent on our ability to generate rental income. Our rental income and funds from operations could particularly be negatively affected by a potential increase in vacancy rates."""

questions_market_dynamics = {
    "Market Dynamics - a": "Does the text mention that the company is exposed to risks associated with cyclical products?"
}

questions_intra_industry_competition = {
    "Intra-Industry Competition - a": "Does the text mention that market pricing for the company's products is irrational?"
}

all_question_dicts = [
    questions_market_dynamics,
    questions_intra_industry_competition
]

###############################################################################
# LLM INITIALIZATION
###############################################################################
model_path = "../../Llama-3.2-3B-Instruct-Q8_0.gguf"
# model_path = "../../phi-4-Q4_K_M.gguf"

llm_hf_1 = LlamaCpp(model_path=model_path, n_ctx=4096, n_gpu_layers=35, max_tokens=256)
llm_hf_2 = LlamaCpp(model_path=model_path, n_ctx=4096, n_gpu_layers=35, max_tokens=256)
llm_hf_3 = LlamaCpp(model_path=model_path, n_ctx=4096, n_gpu_layers=35, max_tokens=256)


###############################################################################
# STEP 1: CLASSIFICATION PROMPT & RUNNABLE
###############################################################################
classification_prompt = PromptTemplate.from_template(
    """You are an expert in risk analysis.
Question: {question}

Title: {section_title}
Text: {section_text}

Answer only "Yes" or "No". 
If the text does not explicitly mention it, answer "No".
"""
)

classification_prompt_runnable = RunnableLambda(
    lambda inputs: classification_prompt.format(
        question=inputs["question"],
        section_title=inputs["subsection_title"],
        section_text=inputs["subsection_text"]
    )
)

classification_chain = classification_prompt_runnable | llm_hf_1

###############################################################################
# STEP 2: EVIDENCE PROMPT & RUNNABLE
###############################################################################
evidence_prompt = PromptTemplate.from_template(
    """You are an expert in extracting evidence from text.

Question: {question}
Title: {section_title}
Text: {section_text}

If any sentence(s) support a "Yes" answer, quote them exactly.
If there's no support, return an empty string.
"""
)

evidence_prompt_runnable = RunnableLambda(
    lambda inputs: evidence_prompt.format(
        question=inputs["question"],
        section_title=inputs["subsection_title"],
        section_text=inputs["subsection_text"]
    )
)

evidence_chain = evidence_prompt_runnable | llm_hf_2

###############################################################################
# STEP 3: JSON FORMAT PROMPT & RUNNABLE
###############################################################################
json_prompt = PromptTemplate.from_template(
    """You are given two pieces of info:
1. Classification: {classification}
2. Evidence: {evidence}

Combine them into JSON with the exact fields:
{{
  "Answer": "Yes" or "No",
  "Evidence": "Any sentence(s) from the text that support the answer; otherwise, blank."
}}
"""
)

json_prompt_runnable = RunnableLambda(
    lambda inputs: json_prompt.format(
        classification=inputs["classification"],
        evidence=inputs["evidence"]
    )
)

json_formatter_chain = json_prompt_runnable | llm_hf_3

###############################################################################
# BUILD A SEQUENTIAL CHAIN
###############################################################################
# We want to run (1) classification => store it in classification
# then (2) evidence => store it in evidence,
# finally (3) produce the final JSON using both fields.

# Step A: produce + store classification
step_a = RunnablePassthrough.assign(classification=classification_chain)

# Step B: produce + store evidence
step_b = RunnablePassthrough.assign(evidence=evidence_chain)

# Step C: final JSON output
# (the chain json_formatter_chain expects "classification" and "evidence" in the input dict)
final_chain = step_a | step_b | json_formatter_chain

###############################################################################
# RUN
###############################################################################

if __name__ == "__main__":
    for question_dict in all_question_dicts:
        for column_name, question in question_dict.items():
            # Prepare the input
            input_data = {
                "subsection_title": subsection_title,
                "subsection_text": subsection_text,
                "question": question,
            }

            # Invoke the final chain SEQUENTIALLY
            output = final_chain.invoke(input_data)

            print(f"Column Name: {column_name}")
            print("FINAL JSON Output:", output)
            print("--" * 50)


# if __name__ == "__main__":
#     for question_dict in all_question_dicts:
#         for column_name, question in question_dict.items():
#             # Prepare input data for classification and evidence steps.
#             input_data = {
#                 "subsection_title": subsection_title,
#                 "subsection_text": subsection_text,
#                 "question": question,
#             }
#             print("--" * 50)
#             print(f"Column Name: {column_name}")
#             print("Input for Classification & Evidence Step:")
#             print(input_data)
            
#             # Step 1: Get classification output.
#             classification = classification_chain.invoke(input_data)
#             print("\nIntermediate Classification Output:")
#             print(classification)

#             # Step 2: Get evidence output.
#             evidence = evidence_chain.invoke(input_data)
#             print("\nIntermediate Evidence Output:")
#             print(evidence)

#             # Step 3: Pass both intermediate outputs to the final JSON formatter.
#             json_input = {
#                 "classification": classification,
#                 "evidence": evidence
#             }
#             print("\nInput for JSON Formatter Step:")
#             print(json_input)
#             final_output = json_formatter_chain.invoke(json_input)
#             print("\nFINAL JSON Output:")
#             print(final_output)
#             print("--" * 50)