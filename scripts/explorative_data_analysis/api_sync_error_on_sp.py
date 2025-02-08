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

def file_exists_in_reorg(site_url, filename="Reorg API Sync Issues.txt"):
    """
    Checks if a specific file exists in the /Reorg/ folder of a SharePoint site.

    Parameters
    ----------
    site_url : str
        The SharePoint site URL.
    filename : str
        The name of the file to check for existence.

    Returns
    -------
    bool
        True if the file exists, False otherwise.
    """
    if pd.isna(site_url):
        return False
    sp = CustomSharePoint(site_url=site_url)
    parsed_url = urlparse(site_url)
    server_relative_url = parsed_url.path.rstrip('/') + '/Reorg/'
    try:
        files = sp.get_files_metadata(server_relative_url)
        for file in files:
            if file.name == filename:
                return True
        return False
    except Exception as e:
        print(f"Failed to get files from {server_relative_url}: {e}")
        return False

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

    # Filename to check
    target_filename = "Reorg API Sync Issues.txt"

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

        # Check if the file exists in the /Reorg/ folder
        file_exists = file_exists_in_reorg(site_url, filename=target_filename)

        # Add the result to the output rows
        output_rows.append({
            'RmsId': RmsId,
            'AbbrevName': AbbrevName,
            'File Exists': file_exists,
            'SharePointLink': SharePointLink,
            'ExtIssuerId': ExtIssuerId,
            'FinDoxIssuerId': FinDoxIssuerId
        })

    # Create the output DataFrame
    output_df = pd.DataFrame(output_rows)

    # Save the DataFrame to CSV if desired
    output_csv_path = os.path.join(project_root, 'data', 'rmsid_file_existence.csv')
    output_df.to_csv(output_csv_path, index=False)

    print(f"DataFrame created with RmsId, AbbrevName, and 'File Exists' columns. Saved to {output_csv_path}")

if __name__ == "__main__":
    main()