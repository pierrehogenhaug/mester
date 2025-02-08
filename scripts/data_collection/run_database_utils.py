# run_database_extraction.py

import argparse
import os
import sys

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from capfourpy.databases import Database
from src.data_collection.database_utils import (
    get_fundamental_score,
    get_rms_issuer
)
from src.data_processing.data_processing import get_rms_with_fundamental_score

def run_extraction(rms_id=None):
    """
    Connect to the DB, retrieve and process data, optionally filter by RmsId.
    Returns a pandas DataFrame.
    """
    # Initialize the database connection
    db = Database(database="CfRms_prod", azure=True)
    db_c4dw = Database(database="C4DW")

    # Get data from the database
    fundamental_score = get_fundamental_score(db)
    rms_issuer = get_rms_issuer(db)

    # Process and merge data
    rms_with_fundamental_score = get_rms_with_fundamental_score(fundamental_score, rms_issuer)

    # Optional: filter by RmsId if provided
    if rms_id:
        rms_with_fundamental_score = rms_with_fundamental_score[
            rms_with_fundamental_score['RmsId'] == rms_id
        ]

    return rms_with_fundamental_score


def main():
    """
    CLI entry point. When invoked from the command line, this function
    parses arguments, calls run_extraction, and handles CSV printing or saving.
    """
    parser = argparse.ArgumentParser(description="Extract scoring data from database.")
    parser.add_argument('--rms_id', type=str, default=None,
                        help='RmsId to filter data on (optional).')
    parser.add_argument('--no_csv', action='store_true',
                        help='If set, do not output CSV; just print the results.')
    args = parser.parse_args()

    # Fetch the processed dataframe
    df = run_extraction(rms_id=args.rms_id)

    if not args.no_csv:
        # Write data to CSV
        output_dir = os.path.join(project_root, 'data')
        os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
        output_path = os.path.join(output_dir, 'rms_with_fundamental_score.csv')
        df.to_csv(output_path, index=False)

        print(
            "Database extraction and processing complete.\n"
            f"Data saved to '{output_path}'."
        )
    else:
        # Just print the DataFrame to stdout
        print("Database extraction and processing complete. Showing output:")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
    

# isin_rms_link = get_isin_rms_link(db_c4dw)
# findox_mapping_with_rms = get_findox_mapping_with_rms(db_c4dw)
# # Save isin_rms_link to a CSV file
# #isin_rms_link_output_path = os.path.join(output_dir, 'isin_rms_link.csv')
# #isin_rms_link.to_csv(isin_rms_link_output_path, index=False)

# # Save findox_mapping_with_rms to a CSV file
# #findox_mapping_with_rms_output_path = os.path.join(output_dir, 'findox_mapping_with_rms.csv')
# #findox_mapping_with_rms.to_csv(findox_mapping_with_rms_output_path, index=False)