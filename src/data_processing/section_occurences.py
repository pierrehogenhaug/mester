import os
import re
import csv
import glob

def is_bold_all_caps(line: str) -> bool:
    """
    Checks if the line is of the form **SOMETHING** (with optional spaces)
    and that SOMETHING is all uppercase/punctuation.
    Example match: **BUSINESS**, **RISK FACTORS**, **INDUSTRY**.
    """
    line = line.strip()
    if not (line.startswith('**') and line.endswith('**')):
        return False

    inner_text = line[2:-2].strip()
    if not inner_text:
        return False

    # Check if the inner text is effectively uppercase
    return (inner_text.upper() == inner_text)

def is_markdown_header(line: str) -> bool:
    """
    Checks if the line begins with 1-6 '#' characters followed by space
    and then all-uppercase text. Examples:
      # RISK FACTORS
      ## BUSINESS
      ### OUR INDUSTRY
    Returns True if it fits the pattern of a standard Markdown heading
    with all-uppercase text following the hashes.
    """
    line = line.strip()
    # Regex to capture up to 6 '#' plus a space, then some text
    match = re.match(r'^(#{1,3})\s+(.*)$', line)
    if not match:
        return False
    
    # The text after the hashes
    heading_text = match.group(2).strip()
    if not heading_text:
        return False
    
    # Check if heading text is uppercase
    return (heading_text.upper() == heading_text)

def is_purely_numeric_or_punct(s: str) -> bool:
    """
    Returns True if the string is only digits/punctuation/spaces.
    e.g. "2022", "2022:", "2017", "2019 - 2020"
    """
    # This matches only digits, punctuation, spaces, or dashes/commas, etc.
    return bool(re.match(r'^[0-9\.\,\:\-\s]+$', s.strip()))

def skip_unwanted_patterns(line: str) -> bool:
    """
    Returns True if the line should be skipped because it matches any unwanted pattern:
      - Multiple consecutive periods (5+)
      - Disclaimers containing "INVESTING IN THE NOTES INVOLVES A HIGH DEGREE OF RISK"
      - Any line containing "APPENDIX"
    """
    # 1) Table of contents style: detect 5+ consecutive periods
    if re.search(r'\.{5,}', line):
        return True

    # 2) Specific disclaimers
    if "INVESTING IN THE NOTES INVOLVES A HIGH DEGREE OF RISK" in line:
        return True

    # 3) Titles containing "APPENDIX"
    if "APPENDIX" in line:
        return True

    return False

def extract_title(line: str) -> str:
    """
    Checks if the given line qualifies as a 'title' under any of our acceptable formats:
      - **ALL CAPS** (with the entire text inside the asterisks being uppercase)
      - # ALL CAPS, ## ALL CAPS, ..., ###### ALL CAPS
    Returns the line itself (with original formatting) if it qualifies, otherwise None.
    """
    # If it matches **ALL CAPS**:
    if is_bold_all_caps(line):
        return line.strip()

    # If it matches # / ## / ### / etc. followed by ALL CAPS text
    if is_markdown_header(line):
        return line.strip()

    return None


def main():
    # This CSV will contain the final summary
    summary_csv = 'section_summary.csv'

    all_titles = {}  # key: title string, value: {"count": int, "docs": set}
    total_docs = 0

    # Iterate over prospectus folders
    for company_folder in glob.glob('./data_COPY/processed/*'):
        if not os.path.isdir(company_folder):
            continue

        as_expected_folder = os.path.join(company_folder, 'as_expected')
        if not os.path.isdir(as_expected_folder):
            # Skip if there's no as_expected folder
            continue

        # We treat **one folder** == one "document"
        total_docs += 1
        titles_this_prospectus = set()

        markdown_files = glob.glob(os.path.join(as_expected_folder, '*.md'))
        for md_path in markdown_files:
            with open(md_path, 'r', encoding='utf-8') as f:
                for line in f:
                    raw_line = line.strip().upper()
                    if skip_unwanted_patterns(raw_line):
                        continue
                    if is_purely_numeric_or_punct(raw_line):
                        continue

                    potential_title = extract_title(line)
                    if potential_title:
                        titles_this_prospectus.add(potential_title)

        # Now update global counters once for this prospectus
        for t in titles_this_prospectus:
            if t not in all_titles:
                all_titles[t] = {'count': 0, 'docs': set()}
            all_titles[t]['count'] += 1
            all_titles[t]['docs'].add(company_folder)

    # Write summary CSV
    with open(summary_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Title', 'Count', 'Percentage Presence']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        sorted_titles = sorted(all_titles.items(), key=lambda x: x[1]['count'], reverse=True)

        for title, data in sorted_titles:
            doc_presence_count = len(data['docs'])  # number of prospectus folders that have this title
            if total_docs > 0:
                percentage_presence = (doc_presence_count / total_docs) * 100
            else:
                percentage_presence = 0.0

            writer.writerow({
                'Title': title,
                'Count': data['count'],
                'Percentage Presence': f"{percentage_presence:.2f}"
            })

    print(f"Done! Summary written to {summary_csv} with {total_docs} documents (prospectuses) processed.")

if __name__ == '__main__':
    main()