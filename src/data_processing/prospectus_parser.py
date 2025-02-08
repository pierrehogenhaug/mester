"""
Module: prospectus_parser.py

This module provides functions to convert PDF/Markdown prospectus files into a
Markdown representation, build a hierarchical structure of sections and subsections,
and finally export the parsed data to CSV.
"""

import os
import re
import sys
import json
import pandas as pd
from langdetect import detect
import pymupdf4llm  # for converting PDF to markdown
import fitz         # used to open PDF for page count (PyMuPDF)
from tqdm import tqdm


###############################
# Utility functions
###############################
def merge_same_format_lines(lines):
    """
    Merges consecutive lines that share the same markdown "format"
    (bold, bold+italic, italic, or normal).
    """
    merged_lines = []
    prev_format = None
    current_line = ''

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Detect format
        is_bold_italic = line.startswith('**_') and line.endswith('_**')
        is_bold = (line.startswith('**') and line.endswith('**')) and not is_bold_italic
        is_italic = (line.startswith('_') and line.endswith('_')) and not is_bold_italic

        if is_bold_italic:
            line_format = 'bold_italic'
        elif is_bold:
            line_format = 'bold'
        elif is_italic:
            line_format = 'italic'
        else:
            line_format = 'normal'

        line_content = line.strip('*_')

        # If same format, merge into current line
        if prev_format == line_format:
            current_line += ' ' + line_content
        else:
            # If format has changed, push the previous block
            if current_line:
                formatted_line = format_line(prev_format, current_line)
                merged_lines.append(formatted_line)

            current_line = line_content
            prev_format = line_format

    # Append the last block
    if current_line:
        formatted_line = format_line(prev_format, current_line)
        merged_lines.append(formatted_line)

    return merged_lines

def format_line(line_format, content):
    """
    Re-wrap line content with the markdown symbols for the detected format.
    """
    if line_format == 'bold':
        return f'**{content}**'
    elif line_format == 'bold_italic':
        return f'**_{content}_**'
    elif line_format == 'italic':
        return f'_{content}_'
    else:
        return content

def is_valid_section_title(s):
    """
    Checks if a bold+uppercase string is valid as a "section title."
    """
    # Reject if purely numeric/punctuation/spaces
    if re.match(r'^[0-9\.\,\:\-\(\)\s]+$', s):
        return False

    # Reject if has 5 or more consecutive periods
    if re.search(r'\.{5,}', s):
        return False

    # Exclude lines with quarter+year patterns (e.g., "Q4 2020")
    if re.search(r'Q[1-4]\s*20(19|20)', s, re.IGNORECASE):
        return False

    # Exclude lines containing any percentage (e.g. "19%")
    if re.search(r'\d+%', s):
        return False

    # Exclude lines that are just short uppercase country codes
    if re.match(r'^[A-Z]{2,3}$', s):
        return False

    # Exclude lines that match "JANUARY 2020", etc.
    if re.match(r'^(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s+20\d{2}$', s):
        return False

    # Must have at least one alphabetic character
    if not re.search(r'[A-Za-z]', s):
        return False

    # If there's parentheses containing digits, skip (often table data).
    if re.search(r'\(\s*\d', s):
        return False

    # If more than 1 separate numeric chunk appears, skip.
    numeric_chunks = re.findall(r'[0-9]+(?:[.,]\d+)*', s)
    if len(numeric_chunks) > 1:
        return False

    return True

def build_hierarchy(merged_lines):
    """
    Build a hierarchy of sections -> subsections -> subsubsections -> body text.
    """
    hierarchy = []
    current_title = None
    current_subtitle = None
    current_subsubtitle = None
    current_body = []

    for line in merged_lines:
        line = line.strip()
        if not line:
            continue

        is_bold_italic = line.startswith('**_') and line.endswith('_**')
        is_bold = (line.startswith('**') and line.endswith('**')) and not is_bold_italic
        line_content = line.strip('*_')
        is_upper = (line_content == line_content.upper())

        if is_bold_italic:
            # Sub-subsection
            add_body_to_hierarchy(current_body, hierarchy, current_title, current_subtitle, current_subsubtitle)
            current_subsubtitle = line_content
            current_body = []

            # Add a sub-subsection entry
            for section in hierarchy:
                if section['title'] == current_title:
                    for subsection in section['subsections']:
                        if subsection['subtitle'] == current_subtitle:
                            subsection['subsubsections'].append({
                                'subsubtitle': current_subsubtitle,
                                'body': ''
                            })
                            break
                    break

        elif is_bold and is_upper:
            # Potential top-level section
            if is_valid_section_title(line_content):
                add_body_to_hierarchy(current_body, hierarchy, current_title, current_subtitle, current_subsubtitle)
                current_title = line_content
                current_subtitle = None
                current_subsubtitle = None
                current_body = []
                hierarchy.append({
                    'title': current_title,
                    'subsections': []
                })
            else:
                current_body.append(line)

        elif is_bold and not is_upper:
            # Subsection
            add_body_to_hierarchy(current_body, hierarchy, current_title, current_subtitle, current_subsubtitle)
            current_subtitle = line_content
            current_subsubtitle = None
            current_body = []

            for section in hierarchy:
                if section['title'] == current_title:
                    section['subsections'].append({
                        'subtitle': current_subtitle,
                        'body': '',
                        'subsubsections': []
                    })
                    break
        else:
            # Regular body text
            current_body.append(line)

    # Attach any trailing body text
    add_body_to_hierarchy(current_body, hierarchy, current_title, current_subtitle, current_subsubtitle)
    return hierarchy

def add_body_to_hierarchy(current_body, hierarchy, current_title, current_subtitle, current_subsubtitle):
    """
    Append the accumulated body text to the active element (section, subsection, or sub-subsection).
    """
    body_text = ' '.join(current_body).strip()
    if not body_text:
        return

    if current_subsubtitle:
        # Attach body to sub-subsection
        for section in hierarchy:
            if section['title'] == current_title:
                for subsection in section['subsections']:
                    if subsection['subtitle'] == current_subtitle:
                        for subsub in subsection['subsubsections']:
                            if subsub['subsubtitle'] == current_subsubtitle:
                                subsub['body'] += ' ' + body_text
                                return
    elif current_subtitle:
        # Attach body to subsection
        for section in hierarchy:
            if section['title'] == current_title:
                for subsection in section['subsections']:
                    if subsection['subtitle'] == current_subtitle:
                        subsection['body'] += ' ' + body_text
                        return
    elif current_title:
        # Attach body to top-level section
        for section in hierarchy:
            if section['title'] == current_title:
                if 'body' not in section:
                    section['body'] = ''
                section['body'] += ' ' + body_text
                return

def is_section_empty(section):
    """
    Returns True if the section and all its children have no body text.
    """
    if 'body' in section and section['body'].strip():
        return False
    for subsection in section.get('subsections', []):
        if subsection.get('body', '').strip():
            return False
        for subsub in subsection.get('subsubsections', []):
            if subsub.get('body', '').strip():
                return False
    return True


#####################################
# Main processing functions
#####################################
def process_prospectus(input_file_path,
                       original_filename,
                       prospectus_id,
                       section_id_map,
                       next_section_id,
                       from_folder,
                       f_year=None):
    """
    Process a .pdf (or .md) file, convert it to Markdown, build the hierarchy,
    check for a "RISK FACTORS" section, and return a list of rows plus updated state.
    """
    parsing_error = 'N/A'
    md_text = ''

    try:
        # 1) Determine file type and load as md_text
        file_lower = input_file_path.lower()
        if file_lower.endswith('.pdf'):
            # Convert PDF -> Markdown
            md_text = pymupdf4llm.to_markdown(input_file_path)
        elif file_lower.endswith('.md'):
            # Read directly from .md file
            with open(input_file_path, 'r', encoding='utf-8') as f:
                md_text = f.read()
        else:
            raise ValueError("Unsupported file format. Please provide a .pdf or .md file.")

        # 2) Merge lines by format
        lines = md_text.split('\n')
        merged_lines = merge_same_format_lines(lines)

        # 3) Build hierarchy
        hierarchy = build_hierarchy(merged_lines)

        # 4) Risk factors check
        risk_factors_found = any(
            section['title'] == 'RISK FACTORS'
            for section in hierarchy
        )
        risk_factors_empty = False

        if not risk_factors_found:
            # Check if 'risk factors' phrase is in the raw text
            risk_factors_in_markdown = any('risk factors' in line.lower() for line in lines)
            if risk_factors_in_markdown:
                parsing_error = 'Risk factors section found but in inconsistent format'
            else:
                # Possibly detect language
                try:
                    language = detect(md_text)
                    if language != 'en':
                        parsing_error = 'Language not English'
                    else:
                        parsing_error = 'Risk Factors missing'
                except:
                    parsing_error = 'Risk Factors missing'
        else:
            # If found, check if empty
            rf_section = next((sec for sec in hierarchy if sec['title'] == 'RISK FACTORS'), None)
            if rf_section and is_section_empty(rf_section):
                risk_factors_empty = True
                parsing_error = 'Risk factors section is empty'

        # 5) Build the final data rows
        data = []
        for section in hierarchy:
            section_title = section['title']
            # Get or assign Section ID
            if section_title not in section_id_map:
                section_id_map[section_title] = str(next_section_id)
                next_section_id += 1
            section_id = section_id_map[section_title]

            # Counter for subsection numbering
            subsection_counter = 1

            for subsection in section['subsections']:
                subsection_title = subsection['subtitle']
                subsection_id = f"{section_id}.{subsection_counter}"
                subsection_counter += 1

                subsubsection_counter = 1
                for subsub in subsection.get('subsubsections', []):
                    subsubtitle = subsub['subsubtitle']
                    subsub_body = subsub['body'].strip()

                    subsub_id = f"{subsection_id}.{subsubsection_counter}"
                    subsubsection_counter += 1

                    row = {
                        'Prospectus ID': prospectus_id,
                        'Original Filename': original_filename,
                        'Section ID': section_id,
                        'Section Title': section_title,
                        'Subsection ID': subsection_id,
                        'Subsection Title': subsection_title,
                        'Subsubsection ID': subsub_id,
                        'Subsubsection Title': subsubtitle,
                        'Subsubsection Text': subsub_body,
                        'Parsing Error': parsing_error,
                        'From Folder': from_folder,
                        'Prospectus Year': f_year
                    }
                    data.append(row)

                # If there's a body directly under this subsection
                if subsection.get('body', '').strip():
                    subsub_body = subsection['body'].strip()
                    subsub_id = f"{subsection_id}.{subsubsection_counter}"
                    subsubsection_counter += 1

                    row = {
                        'Prospectus ID': prospectus_id,
                        'Original Filename': original_filename,
                        'Section ID': section_id,
                        'Section Title': section_title,
                        'Subsection ID': subsection_id,
                        'Subsection Title': subsection_title,
                        'Subsubsection ID': subsub_id,
                        'Subsubsection Title': '',
                        'Subsubsection Text': subsub_body,
                        'Parsing Error': parsing_error,
                        'From Folder': from_folder,
                        'Prospectus Year': f_year
                    }
                    data.append(row)

            # If there's a body directly under this section
            if section.get('body', '').strip():
                section_body = section['body'].strip()
                subsection_id = f"{section_id}.{subsection_counter}"
                subsubsection_id = f"{subsection_id}.1"

                row = {
                    'Prospectus ID': prospectus_id,
                    'Original Filename': original_filename,
                    'Section ID': section_id,
                    'Section Title': section_title,
                    'Subsection ID': subsection_id,
                    'Subsection Title': '',
                    'Subsubsection ID': subsubsection_id,
                    'Subsubsection Title': '',
                    'Subsubsection Text': section_body,
                    'Parsing Error': parsing_error,
                    'From Folder': from_folder,
                    'Prospectus Year': f_year
                }
                data.append(row)

        # Determine final status
        if not risk_factors_found or risk_factors_empty:
            status = 'not_as_expected'
        else:
            status = 'as_expected'

        return data, next_section_id, status, md_text

    except Exception as e:
        parsing_error = f'Exception occurred: {str(e)}'
        row = {
            'Prospectus ID': prospectus_id,
            'Original Filename': original_filename,
            'Section ID': 'failed parsing',
            'Section Title': 'failed parsing',
            'Subsection ID': 'failed parsing',
            'Subsection Title': 'failed parsing',
            'Subsubsection ID': 'failed parsing',
            'Subsubsection Title': 'failed parsing',
            'Subsubsection Text': 'failed parsing',
            'Parsing Error': parsing_error,
            'From Folder': from_folder,
            'Prospectus Year': f_year
        }
        return [row], next_section_id, 'not_as_expected', ''

def process_prospectus_md(md_file_path, original_filename, prospectus_id, section_id_map, next_section_id, from_folder, f_year=None):
    """
    A wrapper for compatibility with run_script.
    """
    return process_prospectus(md_file_path, original_filename, prospectus_id, section_id_map, next_section_id, from_folder, f_year)


#############################################
# The run_script "full-run" main function
#############################################
def main():
    """
    Full run mode: process multiple prospectus files based on an RMS dataset.
    """
    # Add the project root directory to sys.path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.insert(0, project_root)

    # 1) Read the CSV that has the RMS IDs
    df = pd.read_csv(os.path.join(project_root, 'data', 'rms_with_fundamental_score.csv'))
    print("Loaded RMS dataset with shape:", df.shape)

    # 2) We skip loading/writing a single 'prospectuses_data.csv'.

    # 3) section_id_map + next_section_id
    id_state_file = os.path.join(project_root, 'data', 'id_state.json')
    if os.path.exists(id_state_file):
        try:
            with open(id_state_file, 'r') as f:
                id_state = json.load(f)
            section_id_map = id_state.get('section_id_map', {})
            next_section_id = id_state.get('next_section_id', 1)
        except (json.JSONDecodeError, ValueError):
            print(f"Invalid JSON in {id_state_file}. Starting fresh.")
            section_id_map = {}
            next_section_id = 1
    else:
        section_id_map = {}
        next_section_id = 1

    # 4) prospectus_counter
    prospectus_counter_file = os.path.join(project_root, 'data', 'prospectus_counter.json')
    if os.path.exists(prospectus_counter_file):
        try:
            with open(prospectus_counter_file, 'r') as f:
                prospectus_counter = json.load(f)
        except (json.JSONDecodeError, ValueError):
            print(f"Invalid/empty JSON in {prospectus_counter_file}. Starting fresh.")
            prospectus_counter = {}
    else:
        prospectus_counter = {}

    # 5) pdf_to_prospectus_id (to skip re-parsing)
    pdf_to_prospectus_id_file = os.path.join(project_root, 'data', 'pdf_to_prospectus_id.json')
    if os.path.exists(pdf_to_prospectus_id_file):
        try:
            with open(pdf_to_prospectus_id_file, 'r') as f:
                pdf_to_prospectus_id = json.load(f)
        except (json.JSONDecodeError, ValueError):
            print(f"Invalid/empty JSON in {pdf_to_prospectus_id_file}. Starting fresh.")
            pdf_to_prospectus_id = {}
    else:
        pdf_to_prospectus_id = {}

    def extract_year_from_filename(filename):
        match = re.search(r'\b(19|20)\d{2}\b', filename)
        return int(match.group()) if match else None

    # 6) For each RMS ID, process all .md files in as_expected/
    unique_rms_ids = df['RmsId'].unique()
    for RmsId in tqdm(unique_rms_ids, desc="Processing RMS IDs"):
        rms_id_str = str(RmsId)
        as_expected_folder = os.path.join(project_root, 'data', 'processed', rms_id_str, 'as_expected')
        if not os.path.exists(as_expected_folder):
            continue  # no as_expected folder => skip

        if rms_id_str not in prospectus_counter:
            prospectus_counter[rms_id_str] = 0

        # Gather all .md files in as_expected
        md_files = [f for f in os.listdir(as_expected_folder) if f.lower().endswith('.md')]
        if not md_files:
            continue

        for md_filename in md_files:
            md_file_path = os.path.join(as_expected_folder, md_filename)
            pdf_file_path = md_file_path[:-3] + '.pdf'  # replace .md with .pdf

            md_file_key = os.path.relpath(md_file_path, project_root)

            # If skipping re-parsed files is desired:
            if md_file_key in pdf_to_prospectus_id:
                print(f"Skipping already-processed {md_file_path}")
                continue

            # Build a new prospectus_id
            count_for_this_rms = prospectus_counter[rms_id_str]
            if count_for_this_rms == 0:
                prospectus_id = rms_id_str
            else:
                prospectus_id = f"{rms_id_str}_{count_for_this_rms}"

            # Optionally extract a year from the filename
            f_year = extract_year_from_filename(md_filename)

            # Get PDF page count:
            pdf_page_count = 0
            if os.path.exists(pdf_file_path):
                try:
                    with fitz.open(pdf_file_path) as doc:
                        pdf_page_count = doc.page_count
                except Exception as e:
                    print(f"Could not read PDF {pdf_file_path}: {e}")
                    pdf_page_count = -1  # marker for unreadable

            print(f"Processing {md_file_path} => prospectus_id {prospectus_id}")

            try:
                data, next_section_id, status, md_text = process_prospectus_md(
                    md_file_path=md_file_path,
                    original_filename=md_filename,
                    prospectus_id=prospectus_id,
                    section_id_map=section_id_map,
                    next_section_id=next_section_id,
                    from_folder='as_expected',
                    f_year=f_year
                )

                for row in data:
                    row["PDF Page Count"] = pdf_page_count
                    
                # Save each prospectus's result into a CSV file in as_expected/
                df_out = pd.DataFrame(data)
                out_csv_filename = md_filename.replace('.md', '_parsed_test.csv')
                out_csv_path = os.path.join(as_expected_folder, out_csv_filename)
                df_out.to_csv(out_csv_path, index=False)
                print(f"Saved parsed data to: {out_csv_path}")

                # Mark as processed so we don't re-parse it next time
                pdf_to_prospectus_id[md_file_key] = prospectus_id
                with open(pdf_to_prospectus_id_file, 'w') as f:
                    json.dump(pdf_to_prospectus_id, f)

                # Save updated id_state
                id_state = {
                    'section_id_map': section_id_map,
                    'next_section_id': next_section_id
                }
                with open(id_state_file, 'w') as f:
                    json.dump(id_state, f)

                # Increment the prospectus counter for that RMS
                prospectus_counter[rms_id_str] += 1
                with open(prospectus_counter_file, 'w') as f:
                    json.dump(prospectus_counter, f)

            except Exception as e:
                print(f"Exception parsing {md_file_path}: {e}")
                continue