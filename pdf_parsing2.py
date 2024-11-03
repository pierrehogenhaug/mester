import os
import pymupdf  
import pymupdf4llm  
import json
import re
import pandas as pd
from langdetect import detect

# Functions from your original code
def read_markdown(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [line.rstrip('\n') for line in lines]

def merge_same_format_lines(lines):
    merged_lines = []
    prev_format = None
    current_line = ''

    for line in lines:
        line = line.strip()
        if not line:
            continue
        is_bold = line.startswith('**') and line.endswith('**')
        is_italic = line.startswith('_') and line.endswith('_')
        is_bold_italic = line.startswith('**_') and line.endswith('_**')
        if is_bold_italic:
            line_format = 'bold_italic'
        elif is_bold:
            line_format = 'bold'
        elif is_italic:
            line_format = 'italic'
        else:
            line_format = 'normal'

        line_content = line.strip('*_')
        if prev_format == line_format:
            current_line += ' ' + line_content
        else:
            if current_line:
                formatted_line = format_line(prev_format, current_line)
                merged_lines.append(formatted_line)
            current_line = line_content
            prev_format = line_format

    if current_line:
        formatted_line = format_line(prev_format, current_line)
        merged_lines.append(formatted_line)

    return merged_lines

def format_line(line_format, content):
    if line_format == 'bold':
        return f'**{content}**'
    elif line_format == 'bold_italic':
        return f'**_{content}_**'
    elif line_format == 'italic':
        return f'_{content}_'
    else:
        return content

def build_hierarchy(merged_lines):
    hierarchy = []
    current_title = None
    current_subtitle = None
    current_subsubtitle = None
    current_body = []

    for line in merged_lines:
        line = line.strip()
        if not line:
            continue

        is_bold = line.startswith('**') and line.endswith('**')
        is_bold_italic = line.startswith('**_') and line.endswith('_**')
        line_content = line.strip('*_')
        is_upper = line_content.isupper()

        if is_bold_italic:
            # Subsubtitle
            add_body_to_hierarchy(current_body, hierarchy, current_title, current_subtitle, current_subsubtitle)
            current_subsubtitle = line_content
            current_body = []
            # Add subsubtitle to hierarchy
            for section in hierarchy:
                if section['title'] == current_title:
                    for subsection in section['subsections']:
                        if subsection['subtitle'] == current_subtitle:
                            if 'subsubsections' not in subsection:
                                subsection['subsubsections'] = []
                            subsection['subsubsections'].append({'subsubtitle': current_subsubtitle, 'body': ''})
                            break
                    break
        elif is_bold and is_upper:
            # Title
            add_body_to_hierarchy(current_body, hierarchy, current_title, current_subtitle, current_subsubtitle)
            current_title = line_content
            current_subtitle = None
            current_subsubtitle = None
            current_body = []
            hierarchy.append({'title': current_title, 'subsections': []})
        elif is_bold:
            # Subtitle
            add_body_to_hierarchy(current_body, hierarchy, current_title, current_subtitle, current_subsubtitle)
            current_subtitle = line_content
            current_subsubtitle = None
            current_body = []
            # Add subtitle to hierarchy
            for section in hierarchy:
                if section['title'] == current_title:
                    section['subsections'].append({'subtitle': current_subtitle, 'body': '', 'subsubsections': []})
                    break
        else:
            # Body text
            current_body.append(line)

    add_body_to_hierarchy(current_body, hierarchy, current_title, current_subtitle, current_subsubtitle)
    return hierarchy

def add_body_to_hierarchy(current_body, hierarchy, current_title, current_subtitle, current_subsubtitle):
    body_text = ' '.join(current_body).strip()
    if not body_text:
        return
    if current_subsubtitle:
        # Add body to current subsubtitle
        for section in hierarchy:
            if section['title'] == current_title:
                for subsection in section['subsections']:
                    if subsection['subtitle'] == current_subtitle:
                        for subsub in subsection['subsubsections']:
                            if subsub['subsubtitle'] == current_subsubtitle:
                                subsub['body'] += ' ' + body_text
                                return
    elif current_subtitle:
        # Add body to current subtitle
        for section in hierarchy:
            if section['title'] == current_title:
                for subsection in section['subsections']:
                    if subsection['subtitle'] == current_subtitle:
                        subsection['body'] += ' ' + body_text
                        return
    elif current_title:
        # Add body to current title
        for section in hierarchy:
            if section['title'] == current_title:
                if 'body' not in section:
                    section['body'] = ''
                section['body'] += ' ' + body_text
                return
            

def process_prospectus(pdf_file_path, prospectus_id, section_id_map, next_section_id):
    # Convert PDF to markdown
    md_text = pymupdf4llm.to_markdown(pdf_file_path)

    # Read markdown lines
    lines = md_text.split('\n')

    # Process markdown lines
    merged_lines = merge_same_format_lines(lines)
    hierarchy = build_hierarchy(merged_lines)

    # Initialize parsing error as 'N/A' by default
    parsing_error = 'N/A'

    # Check for 'RISK FACTORS' section
    risk_factors_found = any(section['title'] == 'RISK FACTORS' for section in hierarchy)

    if not risk_factors_found:
        # First check: Look for 'RISK FACTORS' in markdown headers
        risk_factors_in_markdown = False
        for line in lines:
            line_lower = line.lower()
            if line.startswith('#') and 'risk factors' in line_lower:
                risk_factors_in_markdown = True
                break
            elif line.startswith('**') and line.endswith('**') and 'risk factors' in line_lower:
                risk_factors_in_markdown = True
                break
            elif 'risk factors' in line_lower:
                risk_factors_in_markdown = True
                break

        if risk_factors_in_markdown:
            parsing_error = 'Risk factors section found but in inconsistent format'
        else:
            # Second check: Check if language is not English
            try:
                language = detect(md_text)
                if language != 'en':
                    parsing_error = 'Language not English'
                else:
                    parsing_error = 'Other reason'
            except:
                parsing_error = 'Other reason'

        # Since there is no 'RISK FACTORS' section, return only the error row
        row = {
            'Prospectus ID': prospectus_id,
            'Section ID': 'failed parsing',
            'Section Title': 'failed parsing',
            'Subsection ID': 'failed parsing',
            'Subsection Title': 'failed parsing',
            'Subsubsection ID': 'failed parsing',
            'Subsubsection Title': 'failed parsing',
            'Subsubsection Text': 'failed parsing',
            'Market Dynamics - a': '',
            'Market Dynamics - b': '',
            'Market Dynamics - c': '',
            'LLM Answer': '',
            'Evidence Text': '',
            'Parsing Error': parsing_error
        }
        data = [row]
        return data, next_section_id

    data = []

    for section in hierarchy:
        section_title = section['title']
        # Get or assign Section ID
        if section_title not in section_id_map:
            section_id_map[section_title] = str(next_section_id)
            next_section_id += 1
        section_id = section_id_map[section_title]

        subsection_counter = 1  # Initialize subsection counter for this section

        # Handle section body if needed (optional)

        for subsection in section['subsections']:
            subsection_title = subsection['subtitle']
            subsection_id = f"{section_id}.{subsection_counter}"
            subsection_counter += 1

            subsubsection_counter = 1  # Initialize subsubsection counter for this subsection

            # Handle subsection body if needed (optional)

            for subsub in subsection.get('subsubsections', []):
                subsubtitle = subsub['subsubtitle']
                subsubsection_body = subsub['body'].strip()

                subsubsection_id = f"{subsection_id}.{subsubsection_counter}"
                subsubsection_counter += 1

                # Collect data into a row
                row = {
                    'Prospectus ID': prospectus_id,
                    'Section ID': section_id,
                    'Section Title': section_title,
                    'Subsection ID': subsection_id,
                    'Subsection Title': subsection_title,
                    'Subsubsection ID': subsubsection_id,
                    'Subsubsection Title': subsubtitle,
                    'Subsubsection Text': subsubsection_body,
                    'Market Dynamics - a': '',
                    'Market Dynamics - b': '',
                    'Market Dynamics - c': '',
                    'LLM Answer': '',
                    'Evidence Text': '',
                    'Parsing Error': 'N/A'
                }
                data.append(row)

            # If subsection has body text directly, collect it as a subsubsection without a title
            if 'body' in subsection and subsection['body'].strip():
                subsubsection_body = subsection['body'].strip()

                subsubsection_id = f"{subsection_id}.{subsubsection_counter}"
                subsubsection_counter += 1

                row = {
                    'Prospectus ID': prospectus_id,
                    'Section ID': section_id,
                    'Section Title': section_title,
                    'Subsection ID': subsection_id,
                    'Subsection Title': subsection_title,
                    'Subsubsection ID': subsubsection_id,
                    'Subsubsection Title': '',  # No subsubtitle
                    'Subsubsection Text': subsubsection_body,
                    'Market Dynamics - a': '',
                    'Market Dynamics - b': '',
                    'Market Dynamics - c': '',
                    'LLM Answer': '',
                    'Evidence Text': '',
                    'Parsing Error': 'N/A'
                }
                data.append(row)

        # If section has body text directly, collect it as a subsection without a title
        if 'body' in section and section['body'].strip():
            subsection_id = f"{section_id}.{subsection_counter}"
            subsection_counter += 1

            subsubsection_id = f"{subsection_id}.1"

            subsubsection_body = section['body'].strip()

            row = {
                'Prospectus ID': prospectus_id,
                'Section ID': section_id,
                'Section Title': section_title,
                'Subsection ID': subsection_id,
                'Subsection Title': '',  # No subtitle
                'Subsubsection ID': subsubsection_id,
                'Subsubsection Title': '',  # No subsubtitle
                'Subsubsection Text': subsubsection_body,
                'Market Dynamics - a': '',
                'Market Dynamics - b': '',
                'Market Dynamics - c': '',
                'LLM Answer': '',
                'Evidence Text': '',
                'Parsing Error': 'N/A'
            }
            data.append(row)

    return data, next_section_id


def main():
    pdf_folder = './'  # Replace with your folder containing the PDFs
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]

    # Initialize mappings and counters for IDs
    section_id_map = {}
    next_section_id = 1

    all_data = []
    prospectus_id = 1  # Start from 1, increment for each prospectus

    for pdf_file in pdf_files:
        pdf_file_path = os.path.join(pdf_folder, pdf_file)
        print(f'Processing {pdf_file_path}')

        data, next_section_id = process_prospectus(
            pdf_file_path, prospectus_id,
            section_id_map, next_section_id)

        all_data.extend(data)
        prospectus_id += 1

    # Create dataframe
    df = pd.DataFrame(all_data, columns=[
        'Prospectus ID',
        'Section ID',
        'Section Title',
        'Subsection ID',
        'Subsection Title',
        'Subsubsection ID',
        'Subsubsection Title',
        'Subsubsection Text',
        'Market Dynamics - a',
        'Market Dynamics - b',
        'Market Dynamics - c',
        'LLM Answer',
        'Evidence Text',
        'Parsing Error'
    ])


    # Save dataframe to CSV
    df.to_csv('prospectuses_data.csv', index=False)
    print('Data saved to prospectuses_data.csv')

if __name__ == '__main__':
    main()