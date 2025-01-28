import os
import re
import csv
import glob
import json

def is_bold_all_caps(line: str) -> bool:
    """
    Checks if the line is **SOMETHING** and inside is all uppercase/punctuation.
    Example match: **BUSINESS**, **RISK FACTORS**, **INDUSTRY**.
    """
    line = line.strip()
    # Must start and end with '**'
    if not (line.startswith('**') and line.endswith('**')):
        return False
    
    # Extract text inside the asterisks
    inner_text = line[2:-2].strip()
    if not inner_text:
        return False
    
    # Check if the inner text is uppercase (allow punctuation/numbers/spaces)
    if inner_text.upper() == inner_text:
        return True
    
    return False

def is_purely_numeric_or_punct(s: str) -> bool:
    """
    Returns True if the string is only digits/punctuation/spaces.
    e.g. "2022", "2022:", "2017", "2019 - 2020"
    """
    return bool(re.match(r'^[0-9\.\,\:\-\s]+$', s))

def skip_unwanted_patterns(inner_text: str) -> bool:
    """
    Returns True if the line should be skipped because it matches any unwanted pattern:
      - Table of contents (lots of periods)
      - Disclaimers containing "INVESTING IN THE NOTES INVOLVES A HIGH DEGREE OF RISK"
      - Any line containing "APPENDIX"
    """

    # 1) Table of contents style: detect multiple dots in a row "........."
    if re.search(r'\.{5,}', inner_text):  # e.g. 5+ consecutive periods
        return True

    # 2) Disclaimers
    #    If the text contains "INVESTING IN THE NOTES INVOLVES A HIGH DEGREE OF RISK"
    #    (adjust the substring to your specific case)
    if "INVESTING IN THE NOTES INVOLVES A HIGH DEGREE OF RISK" in inner_text:
        return True

    # 3) Titles containing "APPENDIX"
    if "APPENDIX" in inner_text:
        return True

    return False

def main():
    output_file = 'identified_titles.csv'
    
    # We'll only look for titles that contain RISK FACTORS, BUSINESS, or INDUSTRY
    KEYWORDS = ["RISK FACTORS", "BUSINESS", "INDUSTRY"]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Id','IdentifiedTitles','FileMetadata'])
        writer.writeheader()
        
        # Look through ./data/processed/* for subfolders
        for company_folder in glob.glob('./data_COPY/processed/*'):
            if not os.path.isdir(company_folder):
                continue
            
            company_id = os.path.basename(company_folder)  # e.g. "84"
            
            # The "as_expected" subfolder
            as_expected_folder = os.path.join(company_folder, 'as_expected')
            if not os.path.isdir(as_expected_folder):
                # Skip any folder that doesn't have as_expected
                continue
            
            # Find all .md files within as_expected
            markdown_files = glob.glob(os.path.join(as_expected_folder, '*.md'))
            
            for md_path in markdown_files:
                identified_titles = []
                md_filename = os.path.basename(md_path)
                
                with open(md_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if is_bold_all_caps(line):
                            # Remove "**" from ends
                            inner_text = line[2:-2].strip()

                            # 1) Exclude purely numeric or numeric+punc lines
                            if is_purely_numeric_or_punct(inner_text):
                                continue
                            
                            # 2) Skip unwanted patterns
                            if skip_unwanted_patterns(inner_text):
                                continue
                            
                            # 3) Keep only if it contains one of the KEYWORDS
                            #    (already uppercase, but let's uppercase to be consistent)
                            upper_text = inner_text.upper()
                            if any(keyword in upper_text for keyword in KEYWORDS):
                                identified_titles.append(line.strip())
                
                # Write row for this .md file
                row = {
                    'Id': company_id,
                    'IdentifiedTitles': json.dumps(identified_titles),
                    'FileMetadata': md_filename
                }
                writer.writerow(row)

if __name__ == '__main__':
    main()