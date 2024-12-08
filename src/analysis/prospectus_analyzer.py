import json
import re
import time


class ProspectusAnalyzer:
    """
    A class to analyze bond prospectuses using a language model.
    """ 

    # Define prompt templates as class-level constants
    BASELINE_PROMPT = """
    Question:
    {question}

    Text:
    Title: {subsection_title}
    Text: {subsection_text}

    Answer the question based on the text.
    """

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

    {{
    "Answer": "Yes" or "No",
    "Evidence": "The exact sentences from the document that support your answer; otherwise, leave blank."
    }}

    Note: Only provide the JSON response without any additional text.
    """

    def __init__(self, llm_model):
        """
        Initialize the ProspectusAnalyzer with a language model.

        Parameters:
        llm_model: The language model to use for analysis.
        """
        self.llm = llm_model

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

        # Extract the answer field
        answer_match = re.search(rf'"{answer_key}"\s*:\s*"([^"]+)"', response)
        if answer_match:
            answer = answer_match.group(1).strip()
        else:
            answer = "Parsing Error"

        # Extract the Evidence field(s)
        evidence_match = re.search(rf'"{evidence_key}"\s*:\s*"([^"]*)"', response)
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
        prompts = [
            self.YES_NO_PROMPT_TEMPLATE.format(
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
                # Join multiple evidence items into a single string
                evidence = '; '.join(evidence_list)
                
                # Combine answer and evidence
                if answer.lower() == "yes" and evidence:
                    combined_answer = f"Yes: {evidence}"
                elif answer.lower() == "yes":
                    combined_answer = "Yes"
                elif answer.lower() == "no":
                    combined_answer = "No"
                else:
                    # If extraction didn't yield 'yes' or 'no', treat as parsing error
                    combined_answer = f"Parsing Error: {response}"
            except Exception as e:
                print("Error parsing fields from the response:", e)
                # In case of an exception, include the original response in the error message
                combined_answer = f"Parsing Error: {response}"

            # For debugging (optional)
            if combined_answer.startswith("Parsing Error"):
                pass
                # You can uncomment the following lines to see the error details
                # print("Parsing Error encountered. Full Response was:")
                # print(response)

            combined_answers.append(combined_answer)

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
    
    