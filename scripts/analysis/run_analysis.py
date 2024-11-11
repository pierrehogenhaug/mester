from langchain_ollama import OllamaLLM
from tqdm import tqdm
import pandas as pd
import os

from src.analysis.prospectus_analyzer import ProspectusAnalyzer

# Initialize the LLM
llm = OllamaLLM(model="llama3.2")

# Initialize the analyzer
analyzer = ProspectusAnalyzer(llm_model=llm)
print(analyzer)

# # Check if processed file exists
# if os.path.exists('prospectuses_data_processed.csv'):
#     df = pd.read_csv('prospectuses_data_processed.csv')
# else:
#     df = pd.read_csv('prospectuses_data.csv')
#     # Filter out rows that have "failed parsing" in the Section ID column
#     df = df[df['Section ID'] != "failed parsing"]

# # Define the questions
# questions = {
#     "Market Dynamics - a": "Is the company exposed to risks associated with cyclical products?",
#     "Market Dynamics - b": "Does the text mention risks related to demographic or structural trends affecting the market?",
#     "Market Dynamics - c": "Does the text discuss risks due to seasonal volatility in the industry?"
# }

# # Ensure the relevance and evidence columns are created with a compatible data type
# for column_name in questions.keys():
#     if column_name in df.columns:
#         df[column_name] = df[column_name].astype('string')
#     else:
#         df[column_name] = ""

# # Loop over each row in the DataFrame
# for index, row in tqdm(df.iterrows(), total=df.shape[0]):
#     for column_name, question in questions.items():
#         # Check if the answer column is already filled
#         if pd.notnull(df.at[index, column_name]) and df.at[index, column_name] != "":
#             # Skip processing this row for this question
#             continue
#         combined_answer = analyzer.analyze_row_single_question(row, question)
#         df.at[index, column_name] = combined_answer

#     # Save progress every 35 rows
#     if index % 35 == 0 and index != 0:
#         df.to_csv('prospectuses_data_processed.csv', index=False)
#         # Remove the break statement to process all rows
#         # break

# # Save the final DataFrame
# df.to_csv('prospectuses_data_processed.csv', index=False)