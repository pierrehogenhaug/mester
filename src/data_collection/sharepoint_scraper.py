# src/data_scraping/sharepoint_scraper.py

import os
import pandas as pd
import platform
import sys

from urllib.parse import urlparse
from capfourpy.authentication import get_access_token_interactive, get_azure_db_token_api
from capfourpy.sharepoint import SharePoint

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)


class CustomSharePoint(SharePoint):
    """
    Subclass that inherits from capfourpy SharePoint
    """

    # Class variable to store the token
    idp_token = None

    def _generate_token(self, token: str = "missing idp token"):
        """
        Retrieves an authentication token using different methods based on the environment,
        and caches it as a class-level variable to avoid repeated authentications.

        Parameters
        ----------
        token (str, optional): Default token value, used when deployed. Defaults to "missing idp token".

        Returns
        -------
        str: Authentication token for accessing SharePoint API.
        """
        # Check if the token is already cached
        if CustomSharePoint.idp_token is not None:
            return CustomSharePoint.idp_token

        # Generate the token - different methods for hosted and local
        if platform.system() == "Linux":
            try:
                token = get_azure_db_token_api(scope=self.Scope)
            except:
                token = token  # Should always have a value when deployed, otherwise it will fail
        else:
            print("get_access_token_interactive")
            token = get_access_token_interactive(self.Client_Id, self.Tenant_Id, self.Scope)

        # Cache the token at the class level
        CustomSharePoint.idp_token = token
        return token


    def fetch_list_data(self, ListId: str = None, SiteUrl: str = None) -> pd.DataFrame:
        """
        Retrieves all data from a specified SharePoint list and converts it to a DataFrame.

        Parameters
        ----------
        ListId (str, optional): The unique identifier of the SharePoint list to retrieve.

        Returns
        -------
        pd.DataFrame: DataFrame containing all items from the specified SharePoint list.
        """
        large_list = self.ctx.web.lists.get_by_id(list_id=ListId)

        # items = large_list.items.get().execute_query()
        items = large_list.items.get_all().execute_query()
        # items = large_list.items.get_all().execute_query(500) # adding some number makes it run faster dunno why
        data = [item.properties for item in items]

        return pd.DataFrame(data)


    def get_files_metadata(self, folder_url: str):
        """
        Retrieves files in the specified SharePoint folder along with their metadata.

        Parameters
        ----------
        folder_url : str
            The relative URL of the target SharePoint folder.

        Returns
        -------
        List
            List of files with metadata in the specified folder.
        """
        folder = self.ctx.web.get_folder_by_server_relative_url(folder_url)
        # Expand to include ListItemAllFields to access metadata
        files_metadata = folder.files.expand(["ListItemAllFields"]).get().execute_query()
        return files_metadata
    

# Dictionary to store CustomSharePoint instances for each site URL to avoid redundant initializations
sp_instances = {}

def download_specific_files_for_row(row, desired_document_type, desired_document_subtype):
    """
    Downloads files with specific metadata values from the '/Reorg/' document library for a given site.
    Saves the files into the specified output_folder.

    Parameters
    ----------
    row : pd.Series
        A row from the dataframe containing 'EB_SPWebUrl'.
    desired_document_type : str
        The desired value for the "Document Type" column.
    desired_document_subtype : str
        The desired value for the "Document SubType" column.
    output_folder : str
        The folder path where files will be saved.
    """
    site_url = row['EB_SPWebUrl']
    rms_id = row['RmsId']
    if pd.isna(site_url):
        return

    # Use or create a CustomSharePoint instance for this site URL
    if site_url not in sp_instances:
        sp_instances[site_url] = CustomSharePoint(site_url=site_url)
    sp = sp_instances[site_url]

    # Construct the server-relative URL for the '/Reorg/' folder
    parsed_url = urlparse(site_url)
    server_relative_url = parsed_url.path.rstrip('/') + '/Reorg/'

    # Initialize a flag to check if the folder has been created
    folder_created = False
    files_downloaded = False  # Flag to check if any files were downloaded

    # Get files with metadata in the folder
    try:
        files = sp.get_files_metadata(server_relative_url)
        for file in files:
            list_item_properties = file.listItemAllFields.properties
            document_type = list_item_properties.get("DocumentType", None)
            document_subtype = list_item_properties.get("DocumentSubType", None)
            
            if document_type == desired_document_type and document_subtype == desired_document_subtype:
                # Get the file name
                file_name = file.name

                # Define the output folder path
                output_folder = os.path.join(project_root, 'data', 'raw', 'sharepoint_reorg_files', str(rms_id))

                # Check if the file already exists
                file_path = os.path.join(output_folder, file_name)
                if not os.path.exists(file_path):
                    # Create the folder if it hasn't been created yet
                    if not folder_created:
                        os.makedirs(output_folder, exist_ok=True)
                        folder_created = True

                    # Download the file
                    file_url = file.serverRelativeUrl
                    file_stream = sp.download_file(file_url)
                    # Save the file into the specified output folder
                    with open(file_path, 'wb') as f:
                        f.write(file_stream.read())
                    print(f"Downloaded file {file_name} from {site_url} to {output_folder}")
                    files_downloaded = True
                else:
                    pass
                    #print(f"File {file_name} already exists in {output_folder}, skipping download.")
        if not files_downloaded:
            print(f"No new files to download for RmsId {rms_id} at {server_relative_url}")
    except Exception as e:
        print(f"Failed to download files from {server_relative_url}: {e}")