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
    return (inner_text.upper() == inner_text)

def is_purely_numeric_or_punct(s: str) -> bool:
    """
    Returns True if the string is only digits/punctuation/spaces.
    e.g. "2022", "2022:", "2017", "2019 - 2020"
    """
    return bool(re.match(r'^[0-9\.\,\:\-\s]+$', s))

def skip_unwanted_patterns(inner_text: str) -> bool:
    """
    Returns True if the line should be skipped because it matches any unwanted pattern:
      - Multiple consecutive periods (Table of Contents style)
      - Disclaimers containing "INVESTING IN THE NOTES INVOLVES A HIGH DEGREE OF RISK"
      - Any line containing "APPENDIX"
    """
    # 1) Table of contents style: detect 5+ consecutive periods
    if re.search(r'\.{5,}', inner_text):
        return True

    # 2) Specific disclaimers
    if "INVESTING IN THE NOTES INVOLVES A HIGH DEGREE OF RISK" in inner_text:
        return True

    # 3) Titles containing "APPENDIX"
    if "APPENDIX" in inner_text:
        return True

    return False

def main():
    # This CSV will contain the final summary
    summary_csv = 'title_summary.csv'

    # Keep track of all titles found across documents
    # - "count" = total times that title has appeared across all docs
    # - "docs" = set of doc_ids that contain this title (to compute presence %)
    all_titles = {}

    # We'll only look for titles containing these KEYWORDS
    KEYWORDS = ["RISK FACTORS", "BUSINESS", "INDUSTRY"]

    # Counter of how many documents we processed
    total_docs = 0

    # Iterate over folders in ./data_COPY/processed
    for company_folder in glob.glob('./data_COPY/processed/*'):
        if not os.path.isdir(company_folder):
            continue

        company_id = os.path.basename(company_folder)  # e.g. "84"
        
        as_expected_folder = os.path.join(company_folder, 'as_expected')
        if not os.path.isdir(as_expected_folder):
            # Skip if there's no as_expected folder
            continue

        markdown_files = glob.glob(os.path.join(as_expected_folder, '*.md'))
        for md_path in markdown_files:
            # We'll treat each .md file as a distinct "document" 
            # for presence-counting purposes
            doc_id = os.path.basename(md_path)
            total_docs += 1

            # Track which titles this document has (to update presence sets once per doc)
            titles_this_doc = set()

            with open(md_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if is_bold_all_caps(line):
                        inner_text = line[2:-2].strip()

                        # 1) Exclude numeric or numeric+punc lines
                        if is_purely_numeric_or_punct(inner_text):
                            continue
                        
                        # 2) Skip unwanted patterns
                        if skip_unwanted_patterns(inner_text):
                            continue
                        
                        # 3) Check if it contains at least one of the KEYWORDS
                        upper_text = inner_text.upper()
                        if any(keyword in upper_text for keyword in KEYWORDS):
                            # Add to our global dictionary
                            titles_this_doc.add(inner_text)

            # Now update global counters
            for t in titles_this_doc:
                if t not in all_titles:
                    all_titles[t] = {
                        'count': 0,
                        'docs': set()
                    }
                # For each occurrence in this doc, 
                # increment count by 1 
                # (if you want exact # of times a title appears in a doc, you might count them separately. 
                #  But if you only want a single presence per doc, you'd increment by 1 once per doc.)
                all_titles[t]['count'] += 1
                all_titles[t]['docs'].add(doc_id)

    # Now we have processed all documents.
    # Write summary CSV: "Title","Count","Percentage Presence"
    with open(summary_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Title', 'Count', 'Percentage Presence']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Sort by descending count
        sorted_titles = sorted(all_titles.items(), key=lambda x: x[1]['count'], reverse=True)

        for title, data in sorted_titles:
            # data['count']: total times found
            # data['docs']: set of doc_ids that have this title
            # presence = (# of docs containing title / total_docs) * 100
            doc_presence_count = len(data['docs'])
            percentage_presence = 0.0
            if total_docs > 0:
                percentage_presence = (doc_presence_count / total_docs) * 100

            writer.writerow({
                'Title': title,
                'Count': data['count'],
                'Percentage Presence': f"{percentage_presence:.2f}"
            })

    print(f"Done! Summary written to {summary_csv} with {total_docs} documents processed.")

if __name__ == '__main__':
    main()