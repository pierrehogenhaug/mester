import os
import re
import csv
import glob

def is_bold_all_caps(line: str) -> bool:
    line = line.strip()
    if not (line.startswith('**') and line.endswith('**')):
        return False

    inner_text = line[2:-2].strip()
    if not inner_text:
        return False

    return (inner_text.upper() == inner_text)

def is_markdown_header(line: str) -> bool:
    line = line.strip()
    # We allow 1-3 '#' for the sake of example
    match = re.match(r'^(#{1,3})\s+(.*)$', line)
    if not match:
        return False
    
    heading_text = match.group(2).strip()
    if not heading_text:
        return False
    
    return (heading_text.upper() == heading_text)

def is_purely_numeric_or_punct(s: str) -> bool:
    return bool(re.match(r'^[0-9\.\,\:\-\s]+$', s.strip()))

def skip_unwanted_patterns(line: str) -> bool:
    # 1) Check for multiple consecutive periods (5+)
    if re.search(r'\.{5,}', line):
        return True
    # 2) Disclaimers
    if "INVESTING IN THE NOTES INVOLVES A HIGH DEGREE OF RISK" in line:
        return True
    # 3) APPENDIX
    if "APPENDIX" in line:
        return True
    return False

def extract_title(line: str) -> str:
    if is_bold_all_caps(line):
        return line.strip()
    if is_markdown_header(line):
        return line.strip()
    return None

def main():
    summary_csv = 'section_summary.csv'

    # Key change: remove "count" entirely – we only track which docs contain each title
    all_titles = {}  # title -> {"docs": set()}

    total_docs = 0

    for company_folder in glob.glob('./data_COPY/processed/*'):
        if not os.path.isdir(company_folder):
            continue

        as_expected_folder = os.path.join(company_folder, 'as_expected')
        if not os.path.isdir(as_expected_folder):
            continue

        markdown_files = glob.glob(os.path.join(as_expected_folder, '*.md'))
        for md_path in markdown_files:
            doc_id = os.path.basename(md_path)
            total_docs += 1

            titles_this_doc = set()

            with open(md_path, 'r', encoding='utf-8') as f:
                for line in f:
                    raw_line = line.strip().upper()
                    if skip_unwanted_patterns(raw_line):
                        continue
                    if is_purely_numeric_or_punct(raw_line):
                        continue

                    potential_title = extract_title(line)
                    if potential_title:
                        titles_this_doc.add(potential_title)

            # Update global tracking
            for t in titles_this_doc:
                if t not in all_titles:
                    all_titles[t] = {
                        'docs': set()
                    }
                all_titles[t]['docs'].add(doc_id)

    # Write CSV
    with open(summary_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Title', 'Count', 'Percentage Presence']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Sort by descending number of docs that contain the title
        sorted_titles = sorted(all_titles.items(), 
                               key=lambda x: len(x[1]['docs']), 
                               reverse=True)

        for title, data in sorted_titles:
            doc_presence_count = len(data['docs'])
            if total_docs > 0:
                percentage_presence = (doc_presence_count / total_docs) * 100
            else:
                percentage_presence = 0.0

            writer.writerow({
                'Title': title,
                # Count = number of distinct documents containing this title
                'Count': doc_presence_count,
                'Percentage Presence': f"{percentage_presence:.2f}"
            })

    print(f"Done! Summary written to {summary_csv} with {total_docs} documents processed.")

if __name__ == '__main__':
    main()