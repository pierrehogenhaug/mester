"""
This script returns a DataFrame with all RmsId, AbbrevName, and a boolean column 'One or Two documents'.
The 'One or Two documents' column is True if the number of files in the Reorg folder of the SharePoint site is 1 or 2. 

Author: Pierre Høgenhaug
Date: Nov 25, 2024
Usage: python scripts/data_collection/no_files_on_reorg.py
"""

import os
import pandas as pd
import sys
from tqdm import tqdm
from urllib.parse import urlparse

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from capfourpy.databases import Database
from src.data_collection.database_utils import get_rms_issuer, get_findox_mapping_with_rms
from src.data_collection.scrape_sharepoint import CustomSharePoint

def get_number_of_files_in_reorg(site_url):
    if pd.isna(site_url):
        return None
    sp = CustomSharePoint(site_url=site_url)
    parsed_url = urlparse(site_url)
    server_relative_url = parsed_url.path.rstrip('/') + '/Reorg/'
    try:
        files = sp.get_files_metadata(server_relative_url)
        num_files = len(files)
        return num_files
    except Exception as e:
        print(f"Failed to get files from {server_relative_url}: {e}")
        return None

def main():
    # Initialize the database connections
    db = Database(database="CfRms_prod", azure=True)
    db_c4dw = Database(database="C4DW")

    # Get data from the database
    rms_issuer = get_rms_issuer(db)  # Contains RmsId and SharePointLink
    findox_mapping_with_rms = get_findox_mapping_with_rms(db_c4dw)  # Contains RmsId and AbbrevName

    # Merge rms_issuer and findox_mapping_with_rms on RmsId
    merged_df = pd.merge(
        rms_issuer[['RmsId', 'SharePointLink']],
        findox_mapping_with_rms[['RmsId', 'AbbrevName', 'ExtIssuerId', 'FinDoxIssuerId']],
        on='RmsId',
        how='inner'
    )

    # Remove duplicates if any
    merged_df = merged_df.drop_duplicates(subset=['RmsId'])

    # Create a list to store the output rows
    output_rows = []

    # Dictionary to store CustomSharePoint instances to avoid redundant initializations
    sp_instances = {}

    for index, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Processing RMS data"):
        RmsId = row['RmsId']
        AbbrevName = row['AbbrevName']
        SharePointLink = row['SharePointLink']
        ExtIssuerId = row['ExtIssuerId']
        FinDoxIssuerId = row['FinDoxIssuerId']

        if pd.isna(SharePointLink):
            continue  # Skip if no SharePointLink

        # Use or create a CustomSharePoint instance for this site URL
        site_url = SharePointLink
        if site_url not in sp_instances:
            sp_instances[site_url] = CustomSharePoint(site_url=site_url)
        sp = sp_instances[site_url]

        num_files = get_number_of_files_in_reorg(site_url)
        if num_files is None:
            continue  # Skip if unable to get number of files

        if num_files == 0:
            one_or_two_documents = False
            output_rows.append({
                'RmsId': RmsId,
                'AbbrevName': AbbrevName,
                'One or Two documents': one_or_two_documents,
                'SharePointLink': SharePointLink,
                'ExtIssuerId': ExtIssuerId,
                'FinDoxIssuerId': FinDoxIssuerId
            })
        elif num_files <= 2:
            one_or_two_documents = True
            output_rows.append({
                'RmsId': RmsId,
                'AbbrevName': AbbrevName,
                'One or Two documents': one_or_two_documents,
                'SharePointLink': SharePointLink,
                'ExtIssuerId': ExtIssuerId,
                'FinDoxIssuerId': FinDoxIssuerId
            })
        else:
            # num_files > 2, do not include this RmsId in the output DataFrame
            continue

    # Create the output DataFrame
    output_df = pd.DataFrame(output_rows)

    # Save the DataFrame to CSV if desired
    output_csv_path = os.path.join(project_root, 'data', 'rmsid_with_few_documents.csv')
    output_df.to_csv(output_csv_path, index=False)

    print(f"DataFrame created with RmsId, AbbrevName, and 'One or Two documents' columns. Saved to {output_csv_path}")

if __name__ == "__main__":
    main()