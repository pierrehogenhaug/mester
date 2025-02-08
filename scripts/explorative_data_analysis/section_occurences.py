import os
import glob
import re
import csv
from collections import defaultdict

def main():
    # Regular expression pattern to match section titles
    # Pattern explanation:
    # ^\*\*          : Line starts with '**'
    # \s*            : Optional whitespace
    # ([A-Z\s]+)     : Capture group for one or more uppercase letters and spaces
    # \s*            : Optional whitespace
    # \*\*$          : Ends with '**' and end of line
    title_pattern = re.compile(r'^\*\*\s*([A-Z\s]+)\s*\*\*$')

    # Counter for each dynamic field
    counts = defaultdict(int)
    total_md_files = 0

    # Iterate over folders in ./data/processed
    for company_folder in glob.glob('./data/processed/*'):
        if not os.path.isdir(company_folder):
            continue

        as_expected_folder = os.path.join(company_folder, 'as_expected')
        if not os.path.isdir(as_expected_folder):
            continue

        # For each Markdown file in the as_expected folder
        for md_file in glob.glob(os.path.join(as_expected_folder, '*.md')):
            total_md_files += 1

            # Set to track unique titles found in this specific file
            titles_found_in_file = set()

            # Read the MD file line by line
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        stripped_line = line.strip()
                        match = title_pattern.match(stripped_line)
                        if match:
                            title = match.group(1).strip()
                            titles_found_in_file.add(title)
            except Exception as e:
                print(f"Error reading file {md_file}: {e}")
                continue

            # Update global counts
            for title in titles_found_in_file:
                counts[title] += 1

    # Define the output CSV file path
    output_csv = 'section_titles_counts.csv'

    # Write the results to the CSV file
    try:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)

            # Write the header
            csv_writer.writerow(['field_name', 'occurrences', 'percentage'])

            # Sort the titles by occurrences in descending order, then alphabetically
            for title, occurrence in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
                if total_md_files > 0:
                    percentage = (occurrence / total_md_files) * 100
                else:
                    percentage = 0.0
                # Write the row to the CSV
                csv_writer.writerow([title, occurrence, f"{percentage:.2f}%"])
        
        print(f"CSV file '{output_csv}' has been created successfully.")
    except Exception as e:
        print(f"Error writing to CSV file {output_csv}: {e}")

if __name__ == '__main__':
    main()