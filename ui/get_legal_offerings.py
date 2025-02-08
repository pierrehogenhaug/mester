# file: ./scripts/data_collection/get_legal_offerings.py

import os
import sys
import pandas as pd
from io import BytesIO
from urllib.parse import urlparse

# Adjust Python path to see project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_collection.scrape_sharepoint import (
    CustomSharePoint,
    sp_instances  # shared dictionary of SharePoint connections
)
from src.data_processing.data_processing import process_sp_data, merge_data


def list_legal_offerings_for_rmsid(rms_id: int) -> pd.DataFrame:
    """
    Returns a dataframe of files for a given RmsId that are 
    Document Type = "Legal" and Document SubType = "Offerings".

    Columns returned include:
      - 'FileName'
      - 'ServerRelativeUrl'
      - 'DocumentType'
      - 'DocumentSubType'
      - 'SiteUrl'
    """
    # 1) Read in the processed CSV (the same CSV used by run_sharepoint_download.py)
    #    or your main DB-extraction CSV if that is what you rely on.
    path_csv = os.path.join(project_root, "data", "rms_with_fundamental_score.csv")
    if not os.path.exists(path_csv):
        raise FileNotFoundError(
            f"Could not find the expected CSV file at '{path_csv}'. "
            "Make sure you have run your database extraction first."
        )
    rms_with_fundamental_score = pd.read_csv(path_csv)

    # 2) Fetch the SharePoint list data from the "IMP" site (adjust ListId as needed)
    def _fetch_sp_data():
        ListId = "6ba7678f-2b65-4ad4-8759-21b68035c8c8"
        SiteUrl = "https://c4.sharepoint.com/sites/IMP"
        if SiteUrl not in sp_instances:
            sp_instances[SiteUrl] = CustomSharePoint(site_url=SiteUrl)
        sp = sp_instances[SiteUrl]
        return sp.fetch_list_data(ListId=ListId, SiteUrl=SiteUrl)

    sp_data = _fetch_sp_data()
    sp_data_unique = process_sp_data(sp_data)

    # 3) Merge with DB data so we get EB_SPWebUrl, RmsId, etc.
    merged_data_unique = merge_data(sp_data_unique, rms_with_fundamental_score)

    # 4) Filter for the single RmsId
    df_for_rms = merged_data_unique[merged_data_unique["RmsId"] == rms_id].copy()
    if df_for_rms.empty:
        return pd.DataFrame([])  # If no rows for that RmsId, return empty DataFrame

    # 5) We now check the 'Reorg/' folder in each distinct site
    results = []
    for site_url in df_for_rms["EB_SPWebUrl"].dropna().unique():
        # Get or create an instance of CustomSharePoint
        if site_url not in sp_instances:
            sp_instances[site_url] = CustomSharePoint(site_url=site_url)
        sp = sp_instances[site_url]

        # Construct the server-relative URL for '/Reorg/'
        parsed_url = urlparse(site_url)
        server_relative_url = parsed_url.path.rstrip('/') + '/Reorg/'

        # Attempt to list all files in that folder
        try:
            files = sp.get_files_metadata(server_relative_url)
            for file in files:
                list_item_props = file.listItemAllFields.properties
                doc_type = list_item_props.get("DocumentType", None)
                doc_subtype = list_item_props.get("DocumentSubType", None)
                if doc_type == "Legal" and doc_subtype == "Offerings":
                    # print(f"Adding file: {file.name}, SiteUrl: {site_url}")  # Debug Statement
                    results.append({
                        "FileName": file.name,
                        "ServerRelativeUrl": file.serverRelativeUrl,
                        "DocumentType": doc_type,
                        "DocumentSubType": doc_subtype,
                        "SiteUrl": site_url,  # <- Key addition
                    })
        except Exception as exc:
            print(f"Error listing files from {server_relative_url}: {exc}")

    if not results:
        return pd.DataFrame([])

    # print(f"Total files appended: {len(results)}")  # Debug Statement

    return pd.DataFrame(results)


def download_legal_offering(site_url: str, server_relative_url: str) -> BytesIO:
    """
    Given the exact SiteUrl and ServerRelativeUrl, 
    download the file from SharePoint into memory and return a BytesIO object.
    """
    if site_url not in sp_instances:
        sp_instances[site_url] = CustomSharePoint(site_url=site_url)

    sp = sp_instances[site_url]
    file_stream = sp.download_file(server_relative_url)

    # Read the returned file into a BytesIO so Streamlit can serve it as a download.
    return BytesIO(file_stream.read())