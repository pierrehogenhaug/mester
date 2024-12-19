import json
import re
import time


class ProspectusAnalyzer:
    """
    A class to analyze bond prospectuses using a language model.
    """ 

    # Define prompt templates as class-level constants


    BINARY_PROMPT = """
    You are tasked with assessing the relevance of a given text to a question and providing a structured JSON response.
    Instructions:
    1. Review the question and the provided text.
    2. Judge the relevance of the text to the question using one of the following labels:
    - "Relevant"
    - "Not Relevant"
    3. Identify the exact phrases or sentences from the provided text that support your assessment. If no supporting evidence exists, explicitly state: "No relevant evidence found."

    Question:
    {question}

    Text:
    Title: {subsection_title}
    Text: {subsection_text}

    Provide your response in the following JSON format, without any additional text or commentary:
    {{
        "Relevance": "<Relevant | Not Relevant>",
        "Evidence": "<Exact phrases or sentences from the text | 'No relevant evidence found'>"
    }}
    """

    THREE_LEVEL_PROMPT = """
    You are tasked with assessing the relevance of a given text to a question and providing a structured JSON response.
    Instructions:
    1. Review the question and the provided text.
    2. Judge the relevance of the text to the question using one of the following labels:
    - "Highly Relevant"
    - "Somewhat Relevant"
    - "Not Relevant"
    3. Identify the exact phrases or sentences from the provided text that support your assessment. If no supporting evidence exists, explicitly state: "No relevant evidence found."

    Question:
    {question}

    Text:
    Title: {subsection_title}
    Text: {subsection_text}

    Provide your response in the following JSON format, without any additional text or commentary:
    {{
        "Relevance": "<Highly Relevant | Somewhat Relevant | Not Relevant>",
        "Evidence": "<Exact phrases or sentences from the text | 'No relevant evidence found'>"
    }}
    """

    SINGLE_QUESTION_PROMPT_TEMPLATE = """
    For the following question and text, judge whether the text is "Highly Relevant", "Somewhat Relevant", or "Not Relevant".

    Question:
    {question}

    Text:
    Title: {subsection_title}
    Text: {subsection_text}

    Please provide your answer in the following JSON format:

    {{
    "Relevance": "Highly Relevant", "Somewhat Relevant", or "Not Relevant",
    "Evidence": "The exact phrases or sentences from the document that support your assessment; otherwise, leave blank."
    }}

    Note: Only provide the JSON response without any additional text.
    """

    YES_NO_PROMPT_TEMPLATE = """
    For the following question and text, answer "Yes" or "No".

    Question:
    {question}

    Text:
    Title: {subsection_title}
    Text: {subsection_text}

    Please provide your answer in the following JSON format:

    {{"Answer": "Yes" or "No",
    "Evidence": "The exact sentences from the document that support your answer; otherwise, leave blank."}}
    """

    YES_NO_BASE_PROMPT_TEMPLATE = """{question}

    Title: {subsection_title}
    Text: {subsection_text}

    Provide your answer in the following JSON format:
    {{"Answer": "Yes" or "No",
    "Evidence": "The exact sentences from the document that support your answer; otherwise, leave blank."}}
    """

    YES_NO_COT_PROMPT_TEMPLATE = """{question}

    Title: {subsection_title}
    Text: {subsection_text}

    Think step by step and provide your answer in the following JSON format:
    {{"Answer": "Yes" or "No",
    "Evidence": "The exact sentences from the document that support your answer; otherwise, leave blank."}}
    """

    YES_NO_FEW_SHOT_PROMPT_TEMPLATE = """{question}

    Title: {subsection_title}
    Text: {subsection_text}

    Provide your answer in the following JSON format:
    {{"Answer": "Yes" or "No",
    "Evidence": "The exact sentences from the document that support your answer; otherwise, leave blank."}}

    Below are two examples demonstrating how to respond:

    ### Example 1 (Negative Case)
    Company: A  
    Background: Submetering is an infrastructure-like business with long-term contractual agreements and non-discretionary demand.  
    Rationale: 80% of revenue is recurring; high customer loyalty and low churn rates. Consistent EBITDA growth during financial crises. 20+ year average contracts.

    Model Decision:
    {{
    "Answer": "No",
    "Evidence": "Although the text mentions long-term contracts and stable, non-discretionary demand, there is no indication that this business faces cyclical product risks."
    }}

    ### Example 2 (Positive Case)
    Company: B
    Background: Construction equipment rented day-to-day.
    Rationale: 57% of revenue from construction equipment; weak macroeconomic conditions historically had a high impact on business (-25% EBITDA in 2009).

    Model Decision:
    {{
    "Answer": "Yes",
    "Evidence": "The company’s reliance on construction equipment and historical downturn in EBITDA during weak macroeconomic conditions indicate exposure to cyclical product risks."
    }}
    """
    
    def __init__(self, llm_model, prompt_template="YES_NO_COT_PROMPT_TEMPLATE"):
        """
        Initialize the ProspectusAnalyzer with a language model.

        Parameters:
        llm_model: The language model to use for analysis.
        """
        self.llm = llm_model
        self.prompt_template = prompt_template


    def extract_fields(self, response, answer_key="Relevance", evidence_key="Evidence"):
        """
        Extract the 'Relevance' (or 'Answer') and 'Evidence' fields from the model's response.

        Parameters:
        response (str): The response string from the language model.
        answer_key (str): The key for the answer field (default 'Relevance').
        evidence_key (str): The key for the evidence field (default 'Evidence').

        Returns:
        Tuple[str, List[str]]: A tuple containing the answer and a list of evidence strings.
        """
        # Remove any newlines and extra spaces
        response = ' '.join(response.strip().split())
        
        # Identify the last JSON-like block in the response.
        # The model's answer is expected to be in JSON format as per the prompt.
        json_start = response.rfind('{')
        json_end = response.rfind('}')
        if json_start != -1 and json_end != -1 and json_end > json_start:
            json_str = response[json_start:json_end + 1].strip()
        else:
            # If we cannot find a JSON block, we cannot reliably parse.
            return "Parsing Error", []

        # Extract the answer field from the identified JSON block
        answer_match = re.search(rf'"{answer_key}"\s*:\s*"([^"]+)"', json_str)
        if answer_match:
            answer = answer_match.group(1).strip()
        else:
            answer = "Parsing Error"

        # Extract the evidence field(s) from the identified JSON block
        evidence_match = re.search(rf'"{evidence_key}"\s*:\s*"([^"]*)"', json_str)
        if evidence_match:
            evidence_str = evidence_match.group(1).strip()
            evidence = [evidence_str] if evidence_str else []
        else:
            evidence = []

        return answer, evidence


    def analyze_rows_yes_no(self, rows, question):
        """
        Analyze a batch of rows with a yes/no question.
        """
        # Choose template based on self.prompt_template
        if self.prompt_template == "YES_NO_BASE_PROMPT_TEMPLATE":
            chosen_template = self.YES_NO_BASE_PROMPT_TEMPLATE
        else:
            chosen_template = self.YES_NO_COT_PROMPT_TEMPLATE

        prompts = [
            self.YES_NO_COT_PROMPT_TEMPLATE.format(
                question=question,
                subsection_title=row['Subsubsection Title'],
                subsection_text=row['Subsubsection Text']
            )
            for row in rows
        ]

        # Print information about prompts to diagnose issues
        for i, prompt in enumerate(prompts):
            print(f"=== Prompt {i} ===")
            # print(prompt)
            print(f"Prompt length (chars): {len(prompt)}")

        start_time = time.time()
        # Run the batch of prompts through the model
        print("Sending prompts to the model...")
        responses = self.llm.generate(prompts)
        end_time = time.time()
        print(f"Model response received. Time taken: {end_time - start_time:.2f} seconds.")

        combined_answers = []
        for i, generation in enumerate(responses.generations):
            response = generation[0].text  # Get the generated text
            print(f"=== Response for Prompt {i} ===")
            print(response)
            print(f"Response length (chars): {len(response)}")

            try:
                # Extract the 'Answer' and 'Evidence' fields
                answer, evidence_list = self.extract_fields(response, answer_key="Answer", evidence_key="Evidence")
                evidence = '; '.join(evidence_list)
                
                # Determine the parsed_response
                if answer.lower() == "yes" and evidence:
                    parsed_answer = f"Yes: {evidence}"
                elif answer.lower() == "yes":
                    parsed_answer = "Yes"
                elif answer.lower() == "no":
                    parsed_answer = "No"
                else:
                    # If extraction didn't yield 'yes' or 'no', treat as parsing error
                    parsed_answer = "Parsing Error"
            except Exception:
                parsed_answer = "Parsing Error"

            combined_answers.append({
                "parsed_response": parsed_answer,
                "raw_response": response
            })

        return combined_answers


    def analyze_rows_relevance(self, rows, question):
        """
        Analyze a batch of rows with a single question.

        Parameters:
        rows (list of pandas.Series): The list of rows from the DataFrame.
        question (str): The question to ask.

        Returns:
        List[str]: The list of combined answers containing relevance and evidence.
        """
        prompts = [
            self.SINGLE_QUESTION_PROMPT_TEMPLATE.format(
                question=question,
                subsection_title=row['Subsubsection Title'],
                subsection_text=row['Subsubsection Text']
            )
            for row in rows
        ]

        # Run the batch of prompts through the model
        responses = self.llm.generate(prompts)

        combined_answers = []
        for generation in responses.generations:
            response = generation[0].text  # Get the generated text
            try:
                # Extract the Relevance and Evidence fields
                relevance, evidence_list = self.extract_fields(response)
                # Join multiple evidence items into a single string
                evidence = '; '.join(evidence_list)
            except Exception as e:
                relevance = "Parsing Error"
                evidence = ""

            # Combine relevance and evidence
            if relevance in ["Highly Relevant", "Somewhat Relevant"] and evidence:
                combined_answer = f"{relevance}: {evidence}"
            elif relevance in ["Highly Relevant", "Somewhat Relevant"]:
                combined_answer = relevance
            elif relevance == "Not Relevant":
                combined_answer = "Not Relevant"
            else:
                combined_answer = "Parsing Error"

            # For debugging
            if combined_answer == "Parsing Error":
                print("Parsing Error encountered. Response was:")
                print(response)

            combined_answers.append(combined_answer)

        return combined_answers
    
    