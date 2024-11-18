import json
import re

class ProspectusAnalyzer:
    """
    A class to analyze bond prospectuses using a language model.
    """

    def __init__(self, llm_model):
        """
        Initialize the ProspectusAnalyzer with a language model.

        Parameters:
        llm_model: The language model to use for analysis.
        """
        self.llm = llm_model

    def extract_fields(self, response):
        """
        Extract the 'Relevance' and 'Evidence' fields from the model's response.

        Parameters:
        response (str): The response string from the language model.

        Returns:
        Tuple[str, List[str]]: A tuple containing the relevance and a list of evidence strings.
        """
        # Remove any newlines and extra spaces
        response = ' '.join(response.strip().split())

        # Extract the Relevance field
        relevance_match = re.search(r'"Relevance"\s*:\s*"([^"]+)"', response)
        if relevance_match:
            relevance = relevance_match.group(1).strip()
        else:
            relevance = "Parsing Error"

        # Extract the Evidence field(s)
        evidence_match = re.search(r'"Evidence"\s*:\s*(.+?)(?:,?\s*"[^"]+"\s*:|\s*}$)', response)
        if evidence_match:
            evidence_str = evidence_match.group(1).strip()
            # Remove any trailing commas or braces
            evidence_str = evidence_str.rstrip(', }')
            # Split the evidence_str into individual evidence items
            # Evidence items are strings enclosed in double quotes
            evidence_items = re.findall(r'"([^"]+)"', evidence_str)
            evidence = evidence_items
        else:
            evidence = []

        return relevance, evidence

    def analyze_row_single_question(self, row, question):
        """
        Analyze a single row with a given question.

        Parameters:
        row (pandas.Series): The row from the DataFrame.
        question (str): The question to ask.

        Returns:
        str: The combined answer containing relevance and evidence.
        """
        # System and user prompts
        system_prompt = "You are an expert in analyzing bond prospectuses and identifying specific risk factors."

        # Format the user prompt using the row's data
        prompt = f"""
{system_prompt}

For the following question and text, judge whether the text is "Highly Relevant", "Somewhat Relevant", or "Not Relevant".

Question:
{question}

Text:
Subsubsection Title: {row['Subsubsection Title']}
Subsubsection Text: {row['Subsubsection Text']}


Please provide your answer in the following JSON format:

{{
  "Relevance": "Highly Relevant", "Somewhat Relevant", or "Not Relevant",
  "Evidence": "The exact phrases or sentences from the document that support your assessment; otherwise, leave blank."
}}

Note: Only provide the JSON response without any additional text.
"""
        # Run the prompt through the model
        response = self.llm.invoke(input=prompt)

        # Parse the response
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

        return combined_answer

    def analyze_row_single_question_yes_no(self, row, question):
        """
        Analyze a single row with a yes/no question.

        Parameters:
        row (pandas.Series): The row from the DataFrame.
        question (str): The question to ask.

        Returns:
        str: The combined answer containing 'Yes' or 'No' and evidence.
        """
        # System and user prompts
        system_prompt = "You are an expert in analyzing bond prospectuses and identifying specific risk factors."

        # Format the user prompt using the row's data
        prompt = f"""
{system_prompt}

Please answer the following question based on the given text. Provide a clear "Yes" or "No" answer. If "Yes", include the exact phrases or sentences from the text that support your answer.

Text:
Subsubsection Title: {row['Subsubsection Title']}
Subsubsection Text: {row['Subsubsection Text']}

Question:
{question}

Please provide your answer in the following JSON format:

{{
  "Answer": "Yes" or "No",
  "Evidence": "The exact phrases or sentences from the text if 'Yes'; otherwise, leave blank."
}}

Note: Only provide the JSON response without any additional text.
"""
        # Run the prompt through the model
        response = self.llm.invoke(input=prompt)

        # Parse the response
        try:
            # Extract the JSON from the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            json_str = response[start_idx:end_idx]
            result = json.loads(json_str)
            answer = result.get("Answer", "").strip()
            evidence = result.get("Evidence", "").strip()
        except json.JSONDecodeError:
            answer = "Parsing Error"
            evidence = ""

        # Combine answer and evidence
        if answer.lower() == "yes" and evidence:
            combined_answer = f"Yes: {evidence}"
        elif answer.lower() == "yes":
            combined_answer = "Yes"
        elif answer.lower() == "no":
            combined_answer = "No"
        else:
            combined_answer = "Parsing Error"

        # For debugging
        if combined_answer == "Parsing Error":
            print("Parsing Error encountered. Response was:")
            print(response)

        return combined_answer