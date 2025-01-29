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
    # 1) Provide the path to your PDF file
    pdf_file_path = r".\data\processed\107\as_expected\Preliminary Offerings 2021.pdf"
    
    # 2) Configure any parameters you need
    original_filename = "Preliminary Offerings 2021.pdf"
    prospectus_id = 107
    
    # If you track unique IDs for each Section Title, store them in a dict.
    # This dict will get updated as new Section Titles are found.
    section_id_map = {}
    
    # The integer ID you want to assign to the next *new* Section Title
    next_section_id = 1
    
    # Folder or other metadata
    from_folder = "as_expected"
    
    # Prospectus year
    f_year = 2021

    # 3) Call the parser
    data, next_section_id, status, md_text = process_prospectus(
        pdf_file_path=pdf_file_path,
        original_filename=original_filename,
        prospectus_id=prospectus_id,
        section_id_map=section_id_map,
        next_section_id=next_section_id,
        from_folder=from_folder,
        f_year=f_year
    )

    # 4) Convert the returned data (list of dicts) into a DataFrame and inspect or save it
    df = pd.DataFrame(data)
    print(df.head(20))      # Print first 20 rows to console
    print("Status:", status)

    # Optionally write to CSV
    df.to_csv("output_sections.csv", index=False)

if __name__ == "__main__":
    main()