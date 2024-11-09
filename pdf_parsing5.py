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
            

def is_section_empty(section):
    # Check if section has body text
    if 'body' in section and section['body'].strip():
        return False
    # Check if any subsections have text
    for subsection in section.get('subsections', []):
        if 'body' in subsection and subsection['body'].strip():
            return False
        # Check subsubsections
        for subsub in subsection.get('subsubsections', []):
            if 'body' in subsub and subsub['body'].strip():
                return False
    # If no body text found, return True
    return True

def process_prospectus(pdf_file_path, original_filename, prospectus_id, section_id_map, next_section_id):
    parsing_error = 'N/A'  # Initialize parsing error
    md_text = ''  # Initialize md_text
    try:
        # Convert PDF to markdown
        md_text = pymupdf4llm.to_markdown(pdf_file_path)

        # Read markdown lines
        lines = md_text.split('\n')

        # Process markdown lines
        merged_lines = merge_same_format_lines(lines)
        hierarchy = build_hierarchy(merged_lines)

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
                'Original Filename': original_filename,
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
                'Parsing Error': parsing_error
            }
            data = [row]
            return data, next_section_id, 'not_as_expected', md_text

        else:
            # Get the 'RISK FACTORS' section
            risk_factors_section = next((section for section in hierarchy if section['title'] == 'RISK FACTORS'), None)

            # Check if 'RISK FACTORS' section is empty
            if is_section_empty(risk_factors_section):
                parsing_error = 'Risk factors section is empty'
                # Create data row with 'Parsing Error'
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
                    'Market Dynamics - a': '',
                    'Market Dynamics - b': '',
                    'Market Dynamics - c': '',
                    'Parsing Error': parsing_error
                }
                data = [row]
                return data, next_section_id, 'not_as_expected', md_text

        data = []

        for section in hierarchy:
            section_title = section['title']
            if section_title != 'RISK FACTORS':
                continue  # Skip sections other than 'RISK FACTORS'

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
                        'Original Filename': original_filename,
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
                        'Original Filename': original_filename,
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
                    'Original Filename': original_filename,
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
                    'Parsing Error': 'N/A'
                }
                data.append(row)

        return data, next_section_id, 'as_expected', md_text

    except Exception as e:
        # Log exception
        print(f"Exception occurred while processing {pdf_file_path}: {e}")
        # Return data indicating failed parsing
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
            'Market Dynamics - a': '',
            'Market Dynamics - b': '',
            'Market Dynamics - c': '',
            'Parsing Error': parsing_error
        }
        data = [row]
        md_text = ''  # Ensure md_text is defined even if an exception occurs
        return data, next_section_id, 'not_as_expected', md_text


def main():
    pdf_folder = './prospectuses'  # Replace with folder containing the PDFs
    as_expected_folder = os.path.join(pdf_folder, 'as_expected')
    not_as_expected_folder = os.path.join(pdf_folder, 'not_as_expected')

    os.makedirs(as_expected_folder, exist_ok=True)
    os.makedirs(not_as_expected_folder, exist_ok=True)

    # Create markdown_files subfolders
    as_expected_md_folder = os.path.join(as_expected_folder, 'markdown_files')
    not_as_expected_md_folder = os.path.join(not_as_expected_folder, 'markdown_files')

    os.makedirs(as_expected_md_folder, exist_ok=True)
    os.makedirs(not_as_expected_md_folder, exist_ok=True)

    # Load existing data if available
    data_file = 'prospectuses_data.csv'
    if os.path.exists(data_file):
        existing_df = pd.read_csv(data_file)
        all_data = existing_df.to_dict('records')
        # Set the next prospectus ID
        prospectus_id = existing_df['Prospectus ID'].max() + 1
    else:
        all_data = []
        prospectus_id = 1

    # Load section_id_map and next_section_id from JSON if exists
    id_state_file = 'id_state.json'
    if os.path.exists(id_state_file):
        with open(id_state_file, 'r') as f:
            id_state = json.load(f)
        section_id_map = id_state.get('section_id_map', {})
        next_section_id = id_state.get('next_section_id', 1)
    else:
        section_id_map = {}
        next_section_id = 1

    # Get list of unprocessed PDF files
    pdf_files = [f for f in os.listdir(pdf_folder) if os.path.isfile(os.path.join(pdf_folder, f)) and f.lower().endswith('.pdf')]

    for pdf_file in pdf_files:
        pdf_file_path = os.path.join(pdf_folder, pdf_file)
        print(f'Processing {pdf_file_path}')

        try:
            # Process the file
            data, next_section_id, processing_result, md_text = process_prospectus(
                pdf_file_path, pdf_file, prospectus_id,
                section_id_map, next_section_id)
            all_data.extend(data)
        except Exception as e:
            print(f"Exception occurred while processing {pdf_file_path}: {e}")
            processing_result = 'not_as_expected'
            md_text = ''  # Ensure md_text is defined
            # Create a failed parsing entry
            row = {
                'Prospectus ID': prospectus_id,
                'Original Filename': pdf_file,
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
                'Parsing Error': f"Exception occurred: {str(e)}"
            }
            all_data.append(row)

        # Save the updated data after each file
        df = pd.DataFrame(all_data, columns=[
            'Prospectus ID',
            'Original Filename',
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
            'Parsing Error'
        ])
        df.to_csv(data_file, index=False)
        print(f'Data saved to {data_file}')

        # Save the ID state
        id_state = {
            'section_id_map': section_id_map,
            'next_section_id': next_section_id
        }
        with open(id_state_file, 'w') as f:
            json.dump(id_state, f)

        # Determine destination folders
        if processing_result == 'as_expected':
            dest_folder = as_expected_folder
            md_dest_folder = as_expected_md_folder
        else:
            dest_folder = not_as_expected_folder
            md_dest_folder = not_as_expected_md_folder

        # Save the markdown file
        md_file_name = pdf_file[:-4] + '.md'
        md_file_path = os.path.join(md_dest_folder, md_file_name)
        with open(md_file_path, 'w', encoding='utf-8') as f:
            f.write(md_text)

        # Move the PDF file to 'as_expected' or 'not_as_expected' folder
        dest_file_path = os.path.join(dest_folder, pdf_file)
        os.rename(pdf_file_path, dest_file_path)

        prospectus_id += 1

    print('Processing complete.')

if __name__ == '__main__':
    main()