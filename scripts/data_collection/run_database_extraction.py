from capfourpy.databases import Database
import os
import sys

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.data_collection.database_utils import get_fundamental_score, get_rms_issuer, get_isin_rms_link
from src.data_processing.data_processing import get_rms_with_fundamental_score

def main():
    # Initialize the database connection
    db = Database(database="CfRms_prod", azure=True)
    db_c4dw = Database(database="C4DW")

    # Get data from the database
    fundamental_score = get_fundamental_score(db)
    rms_issuer = get_rms_issuer(db)
    isin_rms_link = get_isin_rms_link(db_c4dw)

    # Process and merge data
    rms_with_fundamental_score = get_rms_with_fundamental_score(fundamental_score, rms_issuer)

    # Save the processed data to a CSV file
    output_dir = os.path.join(project_root, 'data')
    output_path = os.path.join(output_dir, 'rms_with_fundamental_score.csv')
    rms_with_fundamental_score.to_csv(output_path, index=False)

    # Save isin_rms_link to a CSV file
    isin_rms_link_output_path = os.path.join(output_dir, 'isin_rms_link.csv')
    isin_rms_link.to_csv(isin_rms_link_output_path, index=False)

    print("Database extraction and processing complete. Data saved to '../data/rms_with_fundamental_score.csv'.")

if __name__ == "__main__":
    main()