import os
import re
import csv
import json
import sys

def main():
    # Increase CSV field size limit to handle very large fields.
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int / 10)

    # Define the project and processed data paths
    project_root = "./"
    processed_root = os.path.join(project_root, "data", "processed")
    
    # Compile a regex to match filenames ending in '_parsed.csv'
    csv_file_pattern = re.compile(r'_parsed\.csv$', re.IGNORECASE)
    
    # Gather all CSV file paths under folders that include "as_expected"
    all_csv_paths = []
    for root, dirs, files in os.walk(processed_root):
        if "as_expected" in root:
            for file in files:
                if csv_file_pattern.search(file):
                    csv_path = os.path.join(root, file)
                    all_csv_paths.append(csv_path)
    
    # Define the risk columns we care about
    risk_columns = [
        "Market Dynamics - a",
        "Intra-Industry Competition - a",
        "Technology Risk - a",
        "Regulatory Framework - a"
    ]
    
    # Create a dictionary to hold counts per Section Title.
    # For each section we store counts for each risk column and a "Total".
    section_counts = {}
    
    # Process each CSV file
    for csv_path in all_csv_paths:
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Get the section title from the row (default to "Unknown" if missing)
                section_title = row.get("Section Title", "Unknown")
                if section_title not in section_counts:
                    # Initialize counts for all risk columns and a total counter
                    section_counts[section_title] = {col: 0 for col in risk_columns}
                    section_counts[section_title]["Total"] = 0
                
                # For each risk column, parse its contents and count if evaluation_answer == "Yes"
                for col in risk_columns:
                    risk_cell = row.get(col, "")
                    try:
                        # Parse the JSON string if present
                        risk_data = json.loads(risk_cell) if risk_cell.strip() else {}
                    except Exception as e:
                        # On error (malformed JSON), treat as no risk detected.
                        risk_data = {}
                    
                    # Check if the risk was detected (evaluation_answer == "Yes")
                    if risk_data.get("evaluation_answer", "") == "Yes":
                        section_counts[section_title][col] += 1
                        section_counts[section_title]["Total"] += 1
    
    # Sort the sections by the total risk count in descending order.
    sorted_sections = sorted(section_counts.items(), key=lambda item: item[1]["Total"], reverse=True)
    
    # Define output CSV file name
    output_file = 'risk_by_title_output.csv'
    
    # Write the aggregated results to the CSV file.
    header = ["Section Title"] + risk_columns + ["Total"]
    with open(output_file, 'w', newline='', encoding='utf-8') as out_csv:
        writer = csv.writer(out_csv)
        writer.writerow(header)
        for section, counts in sorted_sections:
            row_items = [section] + [str(counts[col]) for col in risk_columns] + [str(counts["Total"])]
            writer.writerow(row_items)
    
    print("Output written to", output_file)

if __name__ == "__main__":
    main()