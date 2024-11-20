from capfourpy.databases import Database
from src.data_collection.database_utils import get_fundamental_score, get_rms_issuer
from src.data_processing.data_processing import get_rms_with_fundamental_score

# Initialize the database connection
db = Database(database="CfRms_prod", azure=True)

# Get data from the database
fundamental_score = get_fundamental_score(db)
rms_issuer = get_rms_issuer(db)

# Process and merge data
rms_with_fundamental_score = get_rms_with_fundamental_score(fundamental_score, rms_issuer)

# Save the processed data to a CSV file
rms_with_fundamental_score.to_csv('../data/processed/rms_with_fundamental_score.csv', index=False)
print("Database extraction and processing complete. Data saved to '../data/processed/rms_with_fundamental_score.csv'.")
