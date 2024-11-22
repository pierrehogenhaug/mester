import os
import time
import shutil
from collections import defaultdict
from pathlib import Path
import pandas as pd
import pickle
from selenium import webdriver
from selenium.webdriver.common.by import By

# Initialize paths
DATA_DIR = Path("../data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RMS_FUNDAMENTAL_SCORE_FILE = DATA_DIR / "rms_with_fundamental_score.csv"
RMS_ISIN_LINK_FILE = DATA_DIR / "isin_rms_link.csv"

# Read the grouped_isin list from pickle file
with open(DATA_DIR / "grouped_isins.pkl", "rb") as f:
    grouped_isin = pickle.load(f)

grouped_isin_small = grouped_isin[:5]
# Initialize download directory
download_dir = os.path.abspath("./temp_download")
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

# Set up Selenium WebDriver with download preferences
chrome_options = webdriver.ChromeOptions()
preferences = {
    "download.default_directory": download_dir,
    "directory_upgrade": True,
    "safebrowsing.enabled": True
}
chrome_options.add_experimental_option("prefs", preferences)
browser = webdriver.Chrome(options=chrome_options)

def get_prospectus(browser, isin, output_folder):
    # Clear download directory before starting
    for f in os.listdir(download_dir):
        try:
            os.remove(os.path.join(download_dir, f))
        except Exception as e:
            print(f"Could not delete file {f} in download_dir: {e}")

    url = 'https://registers.esma.europa.eu/publication/searchRegister?core=esma_registers_priii_securities'
    browser.get(url)
    
    time.sleep(1)
    # Find ISIN field
    input_isin = browser.find_element(By.XPATH,'//*[@id="searchFields"]/div[1]/div/input')
    
    # Send ISIN
    input_isin.clear()
    input_isin.send_keys(isin)
    
    time.sleep(2)
    # Search
    search_button = browser.find_element(By.XPATH,'//*[@id="searchSolrButton"]')
    search_button.click()
    
    # Sleep to accommodate loading
    time.sleep(5)
    
    # Check if results are found
    try:
        prosp_button = browser.find_element(By.XPATH,'//*[@id="T01"]/tbody/tr/td[11]/a')
        prosp_button.click()
    except:
        print(f"No results found for ISIN {isin}")
        return False
    
    # Sleep to accommodate loading
    time.sleep(5)
    
    # Expand sections
    for val in range(7):
        try:
            time.sleep(1)
            exp_button = browser.find_element(By.XPATH, f'//*[@id="ui-id-{val}"]')
            exp_button.click()
        except:
            pass
    
    time.sleep(5)
    # Click document
    try:
        doc_button = browser.find_element(By.XPATH, '//*[@id="prospectuses"]/tbody/tr/td[2]/a')
        new_url = doc_button.get_attribute('href')
    except:
        print('Trying other approach')
        try:
            doc_button = browser.find_element(By.XPATH, '//*[@id="related_docs"]/tbody/tr/td[2]/a')
            new_url = doc_button.get_attribute('href')
        except:
            print(f"No documents found for ISIN {isin}")
            return False
    
    print(f"Navigating to document URL: {new_url}")
    browser.get(new_url)
    time.sleep(5)
    for val in range(10):
        try:
            time.sleep(1)
            browser.find_element(By.XPATH, f'//*[@id="detailsParent"]/tbody/tr[{val}]/td[2]/a').click()
        except:
            pass
    
    # Wait for download to finish
    time.sleep(20)
    
    # Attempt to find the downloaded file
    try:
        files = [os.path.join(download_dir, f) for f in os.listdir(download_dir)]
        if len(files) == 0:
            print(f"No files were downloaded for ISIN {isin}")
            return False
        elif len(files) > 1:
            print(f"Multiple files downloaded for ISIN {isin}, cannot determine correct file")
            return False
        else:
            filename = files[0]
            dest_path = os.path.join(output_folder, isin + '.pdf')
            shutil.move(filename, dest_path)
            time.sleep(1)
            return True
    except Exception as e:
        print(f"Failed to move downloaded file for ISIN {isin}: {e}")
        return False

# Load processed RmsIds
processed_rmsids_file = 'processed_rmsids.txt'
if os.path.exists(processed_rmsids_file):
    with open(processed_rmsids_file, 'r') as f:
        processed_rmsids = set(line.strip() for line in f)
else:
    processed_rmsids = set()
    
# Main processing loop
try:
    for rms_entry in grouped_isin_small:
        rms_id = rms_entry[0]
        
        # Skip if RmsId has already been processed
        if str(rms_id) in processed_rmsids:
            print(f"RmsId {rms_id} has already been processed. Skipping.")
            continue
        
        scoring_date_lists = rms_entry[1]
        
        # Define the output folder for this RmsId
        processed_folder = PROCESSED_DIR / str(rms_id) / "as_expected"
        output_folder = RAW_DIR / str(rms_id)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        for scoring_date_entry in scoring_date_lists:
            scoring_date = scoring_date_entry[0]
            isins_list = scoring_date_entry[1]
            
            is_success = False
            
            for isin in isins_list:
                # Define the expected file path for this ISIN
                file_path = output_folder / f"{isin}.pdf"
                if processed_folder.exists():
                    print(f"Prospectus for RmsId {rms_id} already exists at {processed_folder}")
                    is_success = True
                    break  # Move to next ScoringDate
                else:
                    # Try to get prospectus for this ISIN
                    try:
                        success = get_prospectus(browser, isin, output_folder)
                        if success:
                            print(f"Downloaded prospectus for ISIN {isin} to {file_path}")
                            is_success = True
                            break  # Move to next ScoringDate
                        # else:
                        #     print(f"Failed to download prospectus for ISIN {isin}")
                    except Exception as e:
                        print(f"Exception occurred while downloading ISIN {isin}: {e}")
                        # Continue to next ISIN
            if not is_success:
                print(f"Could not obtain prospectus for RmsId {rms_id}, ScoringDate {scoring_date}")
                # Optionally, log this information somewhere
        
        # After processing all ScoringDates for this RmsId, mark it as processed
        with open(processed_rmsids_file, 'a') as f:
            f.write(f"{rms_id}\n")
        processed_rmsids.add(str(rms_id))
        print(f"Finished processing RmsId {rms_id}.")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Close the browser after processing
    browser.quit()