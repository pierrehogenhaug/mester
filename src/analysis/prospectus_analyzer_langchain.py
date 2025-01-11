import time
from typing import Dict, Any, List

import pandas as pd
from pydantic import BaseModel
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.output_parsers import StructuredOutputParser
from langchain.schema.runnable import RunnableMap, RunnableLambda
from langchain.llms.base import LLM

###############################################################################
# 1. Define Your Prompt Templates (Base and Enhanced)
###############################################################################
# Instead of storing them as raw strings, we wrap them in PromptTemplate objects.

base_prompt = PromptTemplate(
    input_variables=["question", "subsection_title", "subsection_text"],
    template="""
{question}

Title: {subsection_title}
Text: {subsection_text}

Provide your answer in the following JSON format:
{{
  "Answer": "Yes" or "No",
  "Evidence": "The exact sentences from the document that support your answer; otherwise, leave blank."
}}
"""
)

enhanced_prompt = PromptTemplate(
    input_variables=["question", "subsection_title", "subsection_text"],
    template="""
[SYSTEM MESSAGE]
You are a JSON generator. You must not include any additional text or commentary. 
Your output must strictly follow this structure:
{{
  "Answer": "Yes" or "No",
  "Evidence": "The exact sentences from the document that support your answer; otherwise, leave blank."
}}

[FEW-SHOT EXAMPLE]
Question: "Does the text mention that the company is exposed to foreign currency risk?"
Title: "Currency Risks"
Text: "If the currency devalues against the euro, our revenue may be adversely affected."

Correct JSON Output:
{{
  "Answer": "Yes",
  "Evidence": "If the currency devalues against the euro, our revenue may be adversely affected."
}}

[USER PROMPT]
Question: {question}

Title: {subsection_title}
Text: {subsection_text}

Provide only valid JSON.
"""
)

###############################################################################
# 2. (Optional) Few-Shot Prompt for More Examples
###############################################################################
example_prompt = PromptTemplate(
    input_variables=["example_question", "example_title", "example_text", "example_answer", "example_evidence"],
    template="""
Question: "{example_question}"
Title: "{example_title}"
Text: "{example_text}"

Correct JSON Output:
{{
  "Answer": "{example_answer}",
  "Evidence": "{example_evidence}"
}}
"""
)

examples = [
    {
        "example_question": "Does the text mention that the company is exposed to foreign currency risk?",
        "example_title": "Currency Risks",
        "example_text": "If the currency devalues against the euro, our revenue may be adversely affected.",
        "example_answer": "Yes",
        "example_evidence": "If the currency devalues against the euro, our revenue may be adversely affected."
    }
]

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="""
[SYSTEM MESSAGE]
You are a JSON generator. 
Output must strictly follow the JSON structure below:
{
  "Answer": "Yes" or "No",
  "Evidence": "The exact sentences from the document that support your answer; otherwise, leave blank."
}
""",
    suffix="{user_section}",
    input_variables=["user_section"]
)

###############################################################################
# 3. Define a Pydantic Model for Parsing
###############################################################################
class RiskResponse(BaseModel):
    Answer: str
    Evidence: str

structured_parser = StructuredOutputParser.from_pydantic(RiskResponse)

###############################################################################
# 4. ProspectusAnalyzer Class
###############################################################################
class ProspectusAnalyzer:
    """
    A class to analyze bond prospectuses using a language model and LangChain's new features.
    """ 
    def __init__(self, llm_model: LLM, use_enhanced_prompt: bool = False, use_few_shot_prompt: bool = False):
        """
        Parameters:
        -----------
        llm_model : LLM
            The LangChain LLM instance (e.g., LlamaCpp or HuggingFaceHub).
        use_enhanced_prompt : bool
            If True, use the 'enhanced' prompt template. Otherwise, use the base prompt.
        use_few_shot_prompt : bool
            If True, use the few-shot prompt approach. 
            Note: This is just a demonstration—normally you would choose either 'enhanced' or 'few-shot.'
        """
        self.llm = llm_model
        self.use_enhanced_prompt = use_enhanced_prompt
        self.use_few_shot_prompt = use_few_shot_prompt
        
        # Decide which prompt to use by default
        if self.use_few_shot_prompt:
            # We'll use the FewShotPromptTemplate approach
            self.active_prompt = few_shot_prompt
        elif self.use_enhanced_prompt:
            self.active_prompt = enhanced_prompt
        else:
            self.active_prompt = base_prompt

    def _build_pipeline(self):
        """
        Build a Runnable pipeline that:
        1) Takes user input and formats the prompt
        2) Calls the LLM
        3) Parses the JSON output using a structured parser
        """
        # If using a few-shot prompt, the pipeline input variable changes slightly.
        # For the base or enhanced prompt, the template uses question/subsection_title/subsection_text directly.
        # For the few-shot prompt, we feed a single user_section string. 
        # We'll handle that in the 'analyze_rows()' method below.
        return RunnableMap({
            # Step 1: Create the formatted prompt
            "formatted_prompt": self.active_prompt,
            # Step 2: Send to LLM
            "llm_response": self.llm,
            # Step 3: Parse the JSON output 
            "parsed_output": RunnableLambda(lambda inputs: structured_parser.parse(inputs["llm_response"]))
        })

    def analyze_rows(self, rows: List[Dict[str, Any]], question: str) -> List[Dict[str, str]]:
        """
        Analyze each row (subsection) with exactly one LLM call.
        
        Parameters:
        -----------
        rows : List of dict
            Each dict must include 'Subsubsection Title' and 'Subsubsection Text'.
        question : str
            The risk-factor question to be answered.

        Returns:
        --------
        List[Dict[str, str]]
            Each dict in the returned list corresponds to a row, containing:
                - parsed_response
                - raw_response
                - answer (Yes/No/Parsing Error)
                - evidence (string)
        """
        runnable = self._build_pipeline()
        results = []

        for idx, row in enumerate(rows):
            print(f"=== Processing Row {idx} ===")

            # Prepare the correct input for the chosen prompt approach
            if self.use_few_shot_prompt:
                # For FewShotPromptTemplate, we have a single variable 'user_section'.
                # We'll manually render the final user_section string from a separate prompt or code:
                user_section = f"""
                Question: {question}
                Title: {row['Subsubsection Title']}
                Text: {row['Subsubsection Text']}
                """

                # Then invoke the pipeline
                input_data = {
                    "user_section": user_section
                }
            else:
                # For the base or enhanced prompt, we just use the standard variables.
                input_data = {
                    "question": question,
                    "subsection_title": row['Subsubsection Title'],
                    "subsection_text": row['Subsubsection Text']
                }

            start_time = time.time()
            try:
                # Run the pipeline
                result = runnable.invoke(input_data)

                # Extract outputs
                llm_response = result["llm_response"]
                parsed_output = result["parsed_output"]
                end_time = time.time()

                print(f"LLM response (Row {idx}): {llm_response}")
                print(f"Time taken: {end_time - start_time:.2f} seconds.\n")

                # Build the final entry
                # parsed_output is a Pydantic object: RiskResponse(Answer=..., Evidence=...)
                answer = parsed_output.Answer.strip()
                evidence = parsed_output.Evidence.strip()

                # For consistent string representation
                if answer.lower() == "yes":
                    if evidence:
                        parsed_answer = f"Yes: {evidence}"
                    else:
                        parsed_answer = "Yes"
                elif answer.lower() == "no":
                    parsed_answer = "No"
                else:
                    parsed_answer = "Parsing Error"

                results.append({
                    "parsed_response": parsed_answer,
                    "raw_response": llm_response,
                    "answer": answer,
                    "evidence": evidence
                })
            except Exception as e:
                # Capture any parsing or LLM failure
                results.append({
                    "parsed_response": "Parsing Error",
                    "raw_response": "",
                    "answer": "Parsing Error",
                    "evidence": ""
                })
                print(f"Error processing row {idx}: {str(e)}")

        return results