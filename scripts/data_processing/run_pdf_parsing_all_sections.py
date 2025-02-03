import json
import os
import pandas as pd
import shutil
import re
import sys

from tqdm import tqdm

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.data_processing.pdf_parsing_all_sections import process_prospectus

def main():
    # The rest of the code remains the same:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.insert(0, project_root)

    df = pd.read_csv(os.path.join(project_root, 'data', 'rms_with_fundamental_score.csv'))
    print("Found file")
    df['ScoringDate'] = pd.to_datetime(df['ScoringDate'])
    
    # Initialize data structures
    data_file = os.path.join(project_root, 'data', 'prospectuses_data.csv')
    if os.path.exists(data_file):
        try:
            existing_df = pd.read_csv(data_file, dtype={'Prospectus ID': str})
            all_data = existing_df.to_dict('records')
            processed_prospectus_ids = set(existing_df['Prospectus ID'].astype(str).unique())
        except pd.errors.EmptyDataError:
            print(f"{data_file} is empty. Initializing with default values.")
            all_data = []
            processed_prospectus_ids = set()
    else:
        print(f"{data_file} is missing or empty. Initializing with default values.")
        all_data = []
        processed_prospectus_ids = set()

    # Initialize id_state
    id_state_file = os.path.join(project_root, 'data', 'id_state.json')
    if os.path.exists(id_state_file):
        try:
            with open(id_state_file, 'r') as f:
                id_state = json.load(f)
            section_id_map = id_state.get('section_id_map', {})
            next_section_id = id_state.get('next_section_id', 1)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Invalid JSON in {id_state_file}. Reinitializing with default values.")
            section_id_map = {}
            next_section_id = 1
    else:
        section_id_map = {}
        next_section_id = 1
    
    # Initialize prospectus_counter
    prospectus_counter_file = os.path.join(project_root, 'data', 'prospectus_counter.json')
    if os.path.exists(prospectus_counter_file):
        try:
            with open(prospectus_counter_file, 'r') as f:
                prospectus_counter = json.load(f)
        except (json.JSONDecodeError, ValueError):
            print(f"Invalid or empty JSON in {prospectus_counter_file}. Reinitializing with default values.")
            prospectus_counter = {}
    else:
        prospectus_counter = {}
    
    # Initialize pdf_to_prospectus_id
    pdf_to_prospectus_id_file = os.path.join(project_root, 'data', 'pdf_to_prospectus_id.json')
    if os.path.exists(pdf_to_prospectus_id_file):
        try:
            with open(pdf_to_prospectus_id_file, 'r') as f:
                pdf_to_prospectus_id = json.load(f)
        except (json.JSONDecodeError, ValueError):
            print(f"Invalid or empty JSON in {pdf_to_prospectus_id_file}. Reinitializing with default values.")
            pdf_to_prospectus_id = {}
    else:
        pdf_to_prospectus_id = {}

    def extract_year_from_filename(filename):
        match = re.search(r'\b(19|20)\d{2}\b', filename)
        return int(match.group()) if match else None
    
    for RmsId, group_df in df.groupby('RmsId'):
        rms_id_str = str(RmsId)
        rms_folder = os.path.join(project_root, 'data', 'raw', 'sharepoint_reorg_files', rms_id_str)
        rms_folder_manual = os.path.join(project_root, 'data', 'raw', 'raw_manual', rms_id_str)

        # Collect PDF files from both folders
        pdf_files = []
        if os.path.exists(rms_folder):
            pdf_files.extend((os.path.join(rms_folder, f), 'raw') for f in os.listdir(rms_folder) if f.lower().endswith('.pdf'))
        if os.path.exists(rms_folder_manual):
            pdf_files.extend((os.path.join(rms_folder_manual, f), 'raw_manual') for f in os.listdir(rms_folder_manual) if f.lower().endswith('.pdf'))

        if not pdf_files:
            continue  # No files to process

        if rms_id_str not in prospectus_counter:
            prospectus_counter[rms_id_str] = 0

        for scoring_date in group_df['ScoringDate'].unique():
            scoring_year = scoring_date.year

            # Find PDFs with a year in their filename
            files_with_year = []
            for (pdf_path, folder_type) in pdf_files:
                f = os.path.basename(pdf_path)
                f_year = extract_year_from_filename(f)
                if f_year is not None:
                    files_with_year.append((pdf_path, f_year, folder_type))

            # Sort by closest year if found
            if files_with_year:
                files_with_year.sort(key=lambda x: abs(x[1] - scoring_year))
                matched_files = [(fw[0], fw[1], fw[2]) for fw in files_with_year]  # (pdf_path, f_year, folder_type)
            else:
                matched_files = []

            processing_success = False
            for (pdf_file_path, f_year, from_folder) in matched_files:
                pdf_file_key = os.path.relpath(pdf_file_path, project_root)

                # Check if already processed
                if pdf_file_key in pdf_to_prospectus_id:
                    prospectus_id = pdf_to_prospectus_id[pdf_file_key]
                    print(f"PDF file {pdf_file_path} already processed with prospectus_id {prospectus_id}")
                    processing_success = True
                    break
                else:
                    prospectus_id = rms_id_str if prospectus_counter[rms_id_str] == 0 else f"{rms_id_str}_{prospectus_counter[rms_id_str]}"
                    print(f"Current prospectus_id: {prospectus_id}")
                    print(f"Processed prospectus_ids: {processed_prospectus_ids}")

                    try:
                        
                        data, next_section_id, processing_result, md_text = process_prospectus(
                            pdf_file_path, os.path.basename(pdf_file_path), prospectus_id,
                            section_id_map, next_section_id, from_folder=from_folder, f_year=f_year
                        )

                        dest_folder = os.path.join(project_root, 'data', 'processed', rms_id_str)
                        os.makedirs(dest_folder, exist_ok=True)

                        subfolder = 'as_expected' if processing_result == 'as_expected' else 'not_as_expected'
                        dest_subfolder = os.path.join(dest_folder, subfolder)
                        os.makedirs(dest_subfolder, exist_ok=True)
    
                        # Copy the PDF file
                        file_name = os.path.basename(pdf_file_path)
                        dest_pdf_path = os.path.join(dest_subfolder, file_name)
                        shutil.copy2(pdf_file_path, dest_pdf_path)
    
                        # Save the markdown text
                        md_file_name = os.path.splitext(file_name)[0] + '.md'
                        md_file_path = os.path.join(dest_subfolder, md_file_name)
                        with open(md_file_path, 'w', encoding='utf-8') as f:
                            f.write(md_text)

                        # Append data
                        all_data.extend(data)
                        df_data = pd.DataFrame(all_data)
                        df_data.to_csv(data_file, index=False)

                        # Update mappings
                        pdf_to_prospectus_id[pdf_file_key] = prospectus_id
                        with open(pdf_to_prospectus_id_file, 'w') as f:
                            json.dump(pdf_to_prospectus_id, f)

                        processed_prospectus_ids.add(prospectus_id)
                        print(f"Added {prospectus_id} to processed_prospectus_ids")

                        id_state = {
                            'section_id_map': section_id_map,
                            'next_section_id': next_section_id
                        }
                        with open(id_state_file, 'w') as f:
                            json.dump(id_state, f)

                        prospectus_counter[rms_id_str] += 1
                        with open(prospectus_counter_file, 'w') as f:
                            json.dump(prospectus_counter, f)

                        if processing_result == 'as_expected':
                            processing_success = True
                            break
                        else:
                            continue

                    except Exception as e:
                        print(f"Exception occurred while processing {pdf_file_path}: {e}")
                        continue

            if not processing_success:
                print(f"No suitable files processed successfully for RmsId {RmsId} on ScoringDate {scoring_date}")


if __name__ == '__main__':
    main()