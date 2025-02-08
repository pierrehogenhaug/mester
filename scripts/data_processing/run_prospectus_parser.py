"""
Runner: run_prospectus.py

This script provides a command-line interface to process a single PDF/MD file or run the
full processing routine based on the RMS dataset. It uses the functions from the
prospectus_parser module.
"""


import os
import sys
import re
import json
import argparse
import pandas as pd

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Import the processing functions from the module
from src.data_processing.prospectus_parser import process_prospectus, process_prospectus_md, main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Process a single PDF/MD file (with metadata fields empty) "
                    "or run the full processing script."
    )
    parser.add_argument("--file", help="Path to a PDF or MD file to process")
    parser.add_argument("--output", help="Output folder to save the Markdown and CSV files", default=".")
    args = parser.parse_args()

    if args.file:
        # ----------------------------
        # Single-file mode (metadata fields empty)
        # ----------------------------
        input_file_path = args.file
        original_filename = os.path.basename(input_file_path)
        prospectus_id = ""  # empty metadata
        from_folder = ""    # empty metadata
        f_year = None       # empty metadata

        # For a single file, use a dummy section state
        section_id_map = {}
        next_section_id = 1

        data, next_section_id, status, md_text = process_prospectus(
            input_file_path,
            original_filename,
            prospectus_id,
            section_id_map,
            next_section_id,
            from_folder,
            f_year
        )
        print("Status:", status)
        print("Parsed Data:")
        print(json.dumps(data, indent=4))

        # ----------------------------
        # Save output files to the specified output folder
        # ----------------------------
        output_folder = args.output
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Always save the Markdown file.
        # (If the input was a PDF, md_text is the converted Markdown.
        #  If the input was a .md file, md_text contains the file’s original content.)
        out_md_filename = os.path.splitext(original_filename)[0] + ".md"
        out_md_path = os.path.join(output_folder, out_md_filename)
        with open(out_md_path, "w", encoding="utf-8") as f:
            f.write(md_text)
        print(f"Saved Markdown file to: {out_md_path}")

        # Save the parsed data to a CSV file
        out_csv_filename = os.path.splitext(original_filename)[0] + "_parsed.csv"
        out_csv_path = os.path.join(output_folder, out_csv_filename)
        df_out = pd.DataFrame(data)
        df_out.to_csv(out_csv_path, index=False)
        print(f"Saved parsed CSV file to: {out_csv_path}")

    else:
        # ----------------------------
        # Full run mode (using the main() function from prospectus_parser)
        # ----------------------------
        main()