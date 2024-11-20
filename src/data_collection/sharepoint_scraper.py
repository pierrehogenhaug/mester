from capfourpy.authentication import get_access_token_interactive, get_azure_db_token_api
from capfourpy.sharepoint import SharePoint
from urllib.parse import urlparse
import os
import pandas as pd
import platform

class CustomSharePoint(SharePoint):
    """
    Subclass that inherits from capfourpy SharePoint
    """
    # Class variable to store the token
    idp_token = None

    def _generate_token(self, token: str = "missing idp token"):
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
        large_list = self.ctx.web.lists.get_by_id(list_id=ListId)
        items = large_list.items.get_all().execute_query()
        data = [item.properties for item in items]
        return pd.DataFrame(data)

    def get_files_metadata(self, folder_url: str):
        folder = self.ctx.web.get_folder_by_server_relative_url(folder_url)
        files_metadata = folder.files.expand(["ListItemAllFields"]).get().execute_query()
        return files_metadata

def download_specific_files_for_row(row, desired_document_type, desired_document_subtype, sp_instances):
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
                output_folder = os.path.join('../data/raw/sharepoint_reorg_files/', str(rms_id))

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
                    pass  # File already exists
        if not files_downloaded:
            print(f"No new files to download for RmsId {rms_id} at {server_relative_url}")
    except Exception as e:
        print(f"Failed to download files from {server_relative_url}: {e}")