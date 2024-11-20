import os
import pandas as pd
import sys

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.data_collection.sharepoint_scraper import CustomSharePoint, download_specific_files_for_row
from src.data_processing.data_processing import process_sp_data, merge_data


def main():

    # Read the processed data from the database extraction
    rms_with_fundamental_score = pd.read_csv(os.path.join(project_root, 'data', 'rms_with_fundamental_score.csv'))

    # Function to get SharePoint data
    def get_sp_data():
        ListId = "6ba7678f-2b65-4ad4-8759-21b68035c8c8"
        SiteUrl = "https://c4.sharepoint.com/sites/IMP"
        sp = CustomSharePoint(site_url=SiteUrl)
        sp_data = sp.fetch_list_data(ListId=ListId, SiteUrl=SiteUrl)
        return sp_data

    # Fetch and process SharePoint data
    sp_data = get_sp_data()
    sp_data_unique = process_sp_data(sp_data)

    # Merge the dataframes
    merged_data_unique = merge_data(sp_data_unique, rms_with_fundamental_score)

    # Define the desired metadata values
    desired_document_type = "Legal"
    desired_document_subtype = "Offerings"

    # Create a dictionary to store CustomSharePoint instances
    sp_instances = {}

    # Process each row in the merged dataframe
    for index, row in merged_data_unique.iterrows():
        if pd.isna(row['RmsId']):
            continue  # Skip rows where there is no matching RmsId
        # Download files; the function will create the folder only if files are downloaded
        download_specific_files_for_row(row, desired_document_type, desired_document_subtype)

    # Print the list of RmsId that do not have Prospectus
    sorted_rms_id_df = merged_data_unique[["RmsId"]].sort_values(by="RmsId").reset_index(drop=True)
    rms_id_list = sorted_rms_id_df["RmsId"].to_list()

    directory_path = os.path.join(project_root, 'data', 'raw', 'sharepoint_reorg_files')
    # Get list of RmsId folders that actually exist in the directory
    existing_folders = [int(folder) for folder in os.listdir(directory_path) if folder.isdigit() and int(folder) in rms_id_list]

    # List of RmsId that do not have an associated folder
    rms_id_without_folders = [rms_id for rms_id in rms_id_list if rms_id not in existing_folders]
    print("RmsIds without downloaded files:", rms_id_without_folders)

if __name__ == "__main__":
    main()