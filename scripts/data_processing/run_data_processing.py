import json
import os
import pandas as pd
import shutil
import re
import sys

from tqdm import tqdm

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Add the project root to sys.path
sys.path.insert(0, project_root)

from src.data_processing.pdf_parsing import process_prospectus

def main():
    # Read the dataframe
    # This should be dynamic in case dataframe changes
    df = pd.read_pickle(os.path.join(project_root, 'notebooks', 'df_rms_with_fundamental_score.pkl'))
    print("Found file")
    df['ScoringDate'] = pd.to_datetime(df['ScoringDate'])
    
    # Initialize data structures
    data_file = os.path.join(project_root, 'data', 'prospectuses_data.csv')
    if os.path.exists(data_file):
        existing_df = pd.read_csv(data_file, dtype={'Prospectus ID': str})
        all_data = existing_df.to_dict('records')
        processed_prospectus_ids = set(existing_df['Prospectus ID'].astype(str).unique())
    else:
        all_data = []
        processed_prospectus_ids = set()

    # Initialize id_state
    id_state_file = os.path.join(project_root, 'data', 'id_state.json')
    if os.path.exists(id_state_file):
        with open(id_state_file, 'r') as f:
            id_state = json.load(f)
        section_id_map = id_state.get('section_id_map', {})
        next_section_id = id_state.get('next_section_id', 1)
    else:
        section_id_map = {}
        next_section_id = 1
    
    # For tracking the number of prospectuses per RmsId
    prospectus_counter_file = os.path.join(project_root, 'data', 'prospectus_counter.json')
    if os.path.exists(prospectus_counter_file):
        with open(prospectus_counter_file, 'r') as f:
            prospectus_counter = json.load(f)
    else:
        prospectus_counter = {}
    
    # Initialize mapping from PDF file paths to prospectus_ids
    pdf_to_prospectus_id_file = os.path.join(project_root, 'data', 'pdf_to_prospectus_id.json')
    if os.path.exists(pdf_to_prospectus_id_file):
        with open(pdf_to_prospectus_id_file, 'r') as f:
            pdf_to_prospectus_id = json.load(f)
    else:
        pdf_to_prospectus_id = {}
    
    # Iterate over the dataframe grouped by RmsId
    for RmsId, group_df in df.groupby('RmsId'):
        rms_id_str = str(RmsId)
        rms_folder = os.path.join(project_root, 'data', 'raw', 'sharepoint_reorg_files', rms_id_str)
        if not os.path.exists(rms_folder):
            continue  # Skip to next RmsId

        # Initialize prospectus_counter for this RmsId
        if rms_id_str not in prospectus_counter:
            prospectus_counter[rms_id_str] = 0

        # For each unique ScoringDate
        for scoring_date in group_df['ScoringDate'].unique():
            year = scoring_date.year

            # Get list of PDF files in the folder
            files_in_folder = [f for f in os.listdir(rms_folder) if f.lower().endswith('.pdf')]

            # Function to assign priority to files
            def file_priority(filename):
                priority = 4
                if 'final offerings' in filename.lower():
                    priority = 1
                elif 'offerings' in filename.lower():
                    priority = 2
                elif 'preliminary' in filename.lower():
                    priority = 3
                return priority

            # Function to extract year from filename
            def extract_year_from_filename(filename):
                match = re.search(r'\b(19|20)\d{2}\b', filename)
                return int(match.group()) if match else None

            # Search for files with the year in the name
            matched_files = [f for f in files_in_folder if str(year) in f]
            matched_files.sort(key=lambda x: file_priority(x))

            # If no files matched, check for files with later year
            if not matched_files:
                later_year_files = [f for f in files_in_folder if extract_year_from_filename(f) and extract_year_from_filename(f) > year]
                later_year_files.sort(key=lambda x: file_priority(x))
                matched_files = later_year_files

            # If still no files, check for files with previous year
            if not matched_files:
                previous_year_files = [f for f in files_in_folder if extract_year_from_filename(f) and extract_year_from_filename(f) < year]
                previous_year_files.sort(key=lambda x: file_priority(x))
                matched_files = previous_year_files

            # If still no files, take the first available file
            if not matched_files and files_in_folder:
                matched_files = sorted(files_in_folder, key=lambda x: file_priority(x))

            # Now try processing the files
            processing_success = False
            for file_name in matched_files:
                pdf_file_path = os.path.join(rms_folder, file_name)
                pdf_file_key = os.path.relpath(pdf_file_path, project_root)
                
                # Check if this PDF has already been processed
                if pdf_file_key in pdf_to_prospectus_id:
                    # Already processed this PDF file
                    prospectus_id = pdf_to_prospectus_id[pdf_file_key]
                    print(f"PDF file {pdf_file_path} already processed with prospectus_id {prospectus_id}")
                    processing_success = True  # Assume success since it was processed before
                    break  # Move to the next ScoringDate or RmsId as appropriate
                else:
                    # Generate a new prospectus_id
                    prospectus_id = rms_id_str if prospectus_counter[rms_id_str] == 0 else f"{rms_id_str}_{prospectus_counter[rms_id_str]}"                    
                    print(f"Current prospectus_id: {prospectus_id}")
                    print(f"Processed prospectus_ids: {processed_prospectus_ids}")
                    
                    try:
                        data, next_section_id, processing_result, md_text = process_prospectus(
                            pdf_file_path, file_name, prospectus_id,
                            section_id_map, next_section_id)
                        # Save processed data and files
                        dest_folder = os.path.join(project_root, 'data', 'processed', rms_id_str)
                        os.makedirs(dest_folder, exist_ok=True)

                        # Determine subfolder based on processing_result
                        subfolder = 'as_expected' if processing_result == 'as_expected' else 'not_as_expected'
                        dest_subfolder = os.path.join(dest_folder, subfolder)
                        os.makedirs(dest_subfolder, exist_ok=True)
    
                        # Copy the PDF file
                        dest_pdf_path = os.path.join(dest_subfolder, file_name)
                        shutil.copy2(pdf_file_path, dest_pdf_path)
    
                        # Save the markdown text
                        md_file_name = os.path.splitext(file_name)[0] + '.md'
                        md_file_path = os.path.join(dest_subfolder, md_file_name)
                        with open(md_file_path, 'w', encoding='utf-8') as f:
                            f.write(md_text)

                        # Append data to all_data
                        all_data.extend(data)

                        # Save data after each processed file
                        df_data = pd.DataFrame(all_data)
                        df_data.to_csv(data_file, index=False)

                        # Add to pdf_to_prospectus_id mapping
                        pdf_to_prospectus_id[pdf_file_key] = prospectus_id
                        with open(pdf_to_prospectus_id_file, 'w') as f:
                            json.dump(pdf_to_prospectus_id, f)

                        # Update processed_prospectus_ids
                        processed_prospectus_ids.add(prospectus_id)
                        print(f"Added {prospectus_id} to processed_prospectus_ids")

                        # Save id_state
                        id_state = {
                            'section_id_map': section_id_map,
                            'next_section_id': next_section_id
                        }
                        with open(id_state_file, 'w') as f:
                            json.dump(id_state, f)

                        # Increment prospectus_counter only when processing a new PDF
                        prospectus_counter[rms_id_str] += 1
                        with open(prospectus_counter_file, 'w') as f:
                            json.dump(prospectus_counter, f)

                        if processing_result == 'as_expected':
                            processing_success = True
                            # Break out of the loop since we have a successful processing
                            break
                        else:
                            # Continue to the next file
                            continue

                    except Exception as e:
                        print(f"Exception occurred while processing {pdf_file_path}: {e}")
                        # Do not increment prospectus_counter here
                        continue  # Try the next file
    
            if not processing_success:
                print(f"No suitable files processed successfully for RmsId {RmsId} on ScoringDate {scoring_date}")


if __name__ == '__main__':
    main()