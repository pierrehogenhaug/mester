import os
import pymupdf  
import pymupdf4llm  
import json
import re
import pandas as pd
from langdetect import detect

def read_markdown(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [line.rstrip('\n') for line in lines]

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

            # Reset current line and format
            current_line = line_content
            prev_format = line_format

    # Append the last block
    if current_line:
        formatted_line = format_line(prev_format, current_line)
        merged_lines.append(formatted_line)

    return merged_lines

def format_line(line_format, content):
    """
    Re-wrap line content with the markdown symbols for
    the detected format.
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
    Must NOT:
      - be purely numeric/punctuation/spaces
      - contain 5+ consecutive periods
      - match certain excluded patterns (quarters, percentages, short country codes)
    """
    # 1) Reject if purely numeric/punctuation/spaces
    if re.match(r'^[0-9\.\,\:\-\s]+$', s):
        return False

    # 2) Reject if has 5 or more consecutive periods
    if re.search(r'\.{5,}', s):
        return False

    # 3) Exclude lines with quarter+year patterns (e.g., "Q4 2020", "Q3 2019", etc.)
    #    If you want to be more general (e.g., Q4 2021, Q4 2022...), you can tweak the regex to match Q[1-4] followed by any 4-digit year.
    #    Example: re.search(r'Q[1-4]\s*20\d{2}', s, re.IGNORECASE)
    if re.search(r'Q[1-4]\s*20(19|20)', s, re.IGNORECASE):
        return False

    # 4) Exclude lines containing any percentage such as "19%", "22%", "4%"
    #    If you strictly want to match lines that ONLY have a digit+%,
    #    you could do: re.match(r'^\d+%$', s). But the user’s examples had mixed bold markers, so a simpler approach:
    if re.search(r'\d+%', s):
        return False

    # 5) Exclude lines that **only** hold a short uppercase country code (e.g. "UK", "US", "FR"…).
    #    If you only want 2- or 3-letter codes, use ^[A-Z]{2,3}$. 
    #    If you want to exclude any single/all caps words with length < N, adjust as needed.
    if re.match(r'^[A-Z]{2,3}$', s):
        return False

    # If it does not match any of the exclusion criteria, accept it
    return True

def build_hierarchy(merged_lines):
    """
    Go through each merged line and build a hierarchy of
    sections -> subsections -> subsubsections -> body text.
    
    - A "Section" is (bold + uppercase) AND passes `is_valid_section_title`.
    - A "Subsection" is a bold line that is NOT all uppercase.
    - A "Sub-subsection" is a bold+italic line.
    - Everything else is body text.
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

        # Identify type
        is_bold_italic = line.startswith('**_') and line.endswith('_**')
        is_bold = (line.startswith('**') and line.endswith('**')) and not is_bold_italic
        line_content = line.strip('*_')

        # Check if the line_content is uppercase (naive check)
        is_upper = (line_content == line_content.upper())

        if is_bold_italic:
            # This is a sub-subsection.
            # First, attach any pending body text to the correct place.
            add_body_to_hierarchy(current_body, hierarchy, current_title, current_subtitle, current_subsubtitle)
            current_subsubtitle = line_content
            current_body = []

            # Add a sub-subsection entry
            for section in hierarchy:
                if section['title'] == current_title:
                    for subsection in section['subsections']:
                        if subsection['subtitle'] == current_subtitle:
                            if 'subsubsections' not in subsection:
                                subsection['subsubsections'] = []
                            subsection['subsubsections'].append({
                                'subsubtitle': current_subsubtitle,
                                'body': ''
                            })
                            break
                    break

        elif is_bold and is_upper:
            # Candidate for top-level section, but only if valid
            if is_valid_section_title(line_content):
                add_body_to_hierarchy(current_body, hierarchy, current_title, current_subtitle, current_subsubtitle)
                current_title = line_content
                current_subtitle = None
                current_subsubtitle = None
                current_body = []

                # Create a new section
                hierarchy.append({
                    'title': current_title,
                    'subsections': []
                })
            else:
                # If it fails the validity check, treat as body text:
                current_body.append(line)

        elif is_bold and not is_upper:
            # This is a subsection
            add_body_to_hierarchy(current_body, hierarchy, current_title, current_subtitle, current_subsubtitle)
            current_subtitle = line_content
            current_subsubtitle = None
            current_body = []

            # Add subsection to the current section
            for section in hierarchy:
                if section['title'] == current_title:
                    section['subsections'].append({
                        'subtitle': current_subtitle,
                        'body': '',
                        'subsubsections': []
                    })
                    break

        else:
            # Regular body text (normal or italic lines).
            current_body.append(line)

    # Attach any trailing body text
    add_body_to_hierarchy(current_body, hierarchy, current_title, current_subtitle, current_subsubtitle)
    return hierarchy

def add_body_to_hierarchy(current_body, hierarchy, current_title, current_subtitle, current_subsubtitle):
    """
    Appends the accumulated body text (in current_body) to whichever
    element is currently 'active' (section, subsection, sub-subsection).
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
        # Attach body to the top-level section
        for section in hierarchy:
            if section['title'] == current_title:
                if 'body' not in section:
                    section['body'] = ''
                section['body'] += ' ' + body_text
                return

def is_section_empty(section):
    """
    Returns True if the section and all its children (subsections, subsubsections)
    have no text in their 'body' fields.
    """
    if 'body' in section and section['body'].strip():
        return False

    for subsection in section.get('subsections', []):
        if 'body' in subsection and subsection['body'].strip():
            return False
        for subsub in subsection.get('subsubsections', []):
            if 'body' in subsub and subsub['body'].strip():
                return False
    return True

def process_prospectus(pdf_file_path,
                       original_filename,
                       prospectus_id,
                       section_id_map,
                       next_section_id,
                       from_folder,
                       f_year=None):
    """
    Main driver function. 
    1) Checks if a .md file already exists with the same base filename.
       - If yes, reads that MD file instead of converting the PDF.
    2) Merges lines, builds a hierarchy, then outputs a list of dict-rows
       for each (section -> subsection -> sub-subsection) plus text.
    3) We still check RISK FACTORS presence for logging or "parsing_error" usage,
       but we do NOT skip extracting other sections if it's missing.
    """
    parsing_error = 'N/A'
    md_text = ''

    try:
        # -----------------------------------------------
        # Check if an existing .md file is present
        # -----------------------------------------------
        base_path, pdf_ext = os.path.splitext(pdf_file_path)
        md_candidate = base_path + ".md"

        if os.path.exists(md_candidate):
            # Use the already-converted markdown file
            with open(md_candidate, 'r', encoding='utf-8') as f:
                md_text = f.read()
        else:
            # Convert PDF to markdown
            md_text = pymupdf4llm.to_markdown(pdf_file_path)

        # Read lines from the MD text
        lines = md_text.split('\n')

        # Merge lines of same format
        merged_lines = merge_same_format_lines(lines)

        # Build the section hierarchy
        hierarchy = build_hierarchy(merged_lines)

        # Detect if "RISK FACTORS" is present at all
        risk_factors_found = any(
            section['title'] == 'RISK FACTORS'
            for section in hierarchy
        )

        # If no RISK FACTORS found, figure out if there's some mention
        if not risk_factors_found:
            risk_factors_in_markdown = False
            for line in lines:
                line_lower = line.lower()
                # Quick check if "risk factors" appears anywhere
                if 'risk factors' in line_lower:
                    risk_factors_in_markdown = True
                    break

            if risk_factors_in_markdown:
                parsing_error = 'Risk factors section found but in inconsistent format'
            else:
                # Possibly check the language
                try:
                    language = detect(md_text)
                    if language != 'en':
                        parsing_error = 'Language not English'
                    else:
                        parsing_error = 'Risk Factors missing'
                except:
                    parsing_error = 'Risk Factors missing'

        # Also check if RISK FACTORS is empty (if we found it).
        risk_factors_empty = False
        if risk_factors_found:
            rf_section = next(
                (sec for sec in hierarchy if sec['title'] == 'RISK FACTORS'),
                None
            )
            if rf_section and is_section_empty(rf_section):
                risk_factors_empty = True
                parsing_error = 'Risk factors section is empty'

        # -----------------------------------------------
        # Build the data rows for ALL sections
        # -----------------------------------------------
        data = []

        for section in hierarchy:
            section_title = section['title']

            # Get or assign Section ID
            if section_title not in section_id_map:
                section_id_map[section_title] = str(next_section_id)
                next_section_id += 1
            section_id = section_id_map[section_title]

            # Keep counters to build e.g. "1.1.1"
            subsection_counter = 1

            # Each subsection
            for subsection in section['subsections']:
                subsection_title = subsection['subtitle']
                subsection_id = f"{section_id}.{subsection_counter}"
                subsection_counter += 1

                subsubsection_counter = 1

                # Each sub-subsection
                for subsub in subsection.get('subsubsections', []):
                    subsubtitle = subsub['subsubtitle']
                    subsubsection_body = subsub['body'].strip()

                    subsubsection_id = f"{subsection_id}.{subsubsection_counter}"
                    subsubsection_counter += 1

                    row = {
                        'Prospectus ID': prospectus_id,
                        'Original Filename': original_filename,
                        'Section ID': section_id,
                        'Section Title': section_title,
                        'Subsection ID': subsection_id,
                        'Subsection Title': subsection_title,
                        'Subsubsection ID': subsubsection_id,
                        'Subsubsection Title': subsubtitle,
                        'Subsubsection Text': subsubsection_body,
                        'Parsing Error': parsing_error,  # The global parsing_error
                        'From Folder': from_folder,
                        'Prospectus Year': f_year
                    }
                    data.append(row)

                # If the subsection itself has direct body text
                if 'body' in subsection and subsection['body'].strip():
                    subsubsection_body = subsection['body'].strip()
                    subsubsection_id = f"{subsection_id}.{subsubsection_counter}"
                    subsubsection_counter += 1

                    row = {
                        'Prospectus ID': prospectus_id,
                        'Original Filename': original_filename,
                        'Section ID': section_id,
                        'Section Title': section_title,
                        'Subsection ID': subsection_id,
                        'Subsection Title': subsection_title,
                        'Subsubsection ID': subsubsection_id,
                        'Subsubsection Title': '',
                        'Subsubsection Text': subsubsection_body,
                        'Parsing Error': parsing_error,
                        'From Folder': from_folder,
                        'Prospectus Year': f_year
                    }
                    data.append(row)

            # If the section itself has direct body text (i.e., no subsections or additional headings)
            if 'body' in section and section['body'].strip():
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
                    'Subsubsection Text': section['body'].strip(),
                    'Parsing Error': parsing_error,
                    'From Folder': from_folder,
                    'Prospectus Year': f_year
                }
                data.append(row)

        # Decide final status: 
        if (not risk_factors_found) or (risk_factors_empty):
            status = 'not_as_expected'
        else:
            status = 'as_expected'

        return data, next_section_id, status, md_text

    except Exception as e:
        # If anything goes wrong, return a single row with the error
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
        data = [row]
        md_text = ''
        return data, next_section_id, 'not_as_expected', md_text