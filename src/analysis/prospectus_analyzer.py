import json
import re

class ProspectusAnalyzer:
    """
    A class to analyze bond prospectuses using a language model.
    """ 

    # Define prompt templates as class-level constants
    BASELINE_PROMPT = """
    Question:
    {question}

    Text:
    Subsubsection Title: {row['Subsubsection Title']}
    Subsubsection Text: {row['Subsubsection Text']}

    Answer the question based on the text above.
    """

    BASELINE_PROMPT_V2 = """
    Question:
    {question}

    Text:
    Subsubsection Title: {row['Subsubsection Title']}
    Subsubsection Text: {row['Subsubsection Text']}

    Answer the question based on the text above.
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
    Subsubsection Title: {subsection_title}
    Subsubsection Text: {subsection_text}

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
    Subsubsection Title: {subsection_title}
    Subsubsection Text: {subsection_text}

    Provide your response in the following JSON format, without any additional text or commentary:
    {{
        "Relevance": "<Highly Relevant | Somewhat Relevant | Not Relevant>",
        "Evidence": "<Exact phrases or sentences from the text | 'No relevant evidence found'>"
    }}
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
    

    def analyze_rows_batch(self, rows, questions):
        """
        Analyze a batch of rows with corresponding questions.

        Parameters:
        rows (list of pandas.Series): The list of rows from the DataFrame.
        questions (list of str): The list of questions to ask.

        Returns:
        List[str]: The list of combined answers containing relevance and evidence.
        """
        prompts = [
            self.BASELINE_PROMPT_V2.format(
                question=question,
                subsection_title=row['Subsubsection Title'],
                subsection_text=row['Subsubsection Text']
            )
            for row, question in zip(rows, questions)
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
        # system_prompt = "You are an expert in analyzing bond prospectuses and identifying specific risk factors."

        # Format the user prompt using the row's data
        prompt = f"""
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

        prompt = self.SINGLE_QUESTION_PROMPT_TEMPLATE.format(
            question=question,
            subsection_title=row['Subsubsection Title'],
            subsection_text=row['Subsubsection Text']
        )

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

        prompt = self.SINGLE_QUESTION_PROMPT_TEMPLATE.format(
            question=question,
            subsection_title=row['Subsubsection Title'],
            subsection_text=row['Subsubsection Text']
        )
        
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