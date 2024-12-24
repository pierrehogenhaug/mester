import sys
import os
import json
import pandas as pd
from typing import Dict

def get_progress_metrics(df: pd.DataFrame) -> Dict[str, int]:
    """
    Given a DataFrame `df`, compute progress metrics and return them as a dict
    suitable for wandb.log(...) or other logging mechanisms.
    """
    specified_columns = [
        "Market Dynamics - a",
        "Intra-Industry Competition - a",
        "Regulatory Framework - a",
        "Technology Risk - a"
    ]

    total_rows = len(df)
    
    metrics = {
        "total_rows": total_rows,
        "fully_parsed_count": 0,  # We will compute this below
    }

    # Initialize counters for each column
    # e.g., processed_count_{col}, parsing_error_count_{col}
    for col in specified_columns:
        col_key_processed = f"processed_count__{col}"
        col_key_parsing_err = f"parsing_error_count__{col}"
        metrics[col_key_processed] = 0
        metrics[col_key_parsing_err] = 0

    # Count how many rows are processed vs. skipped/empty for each column
    for col in specified_columns:
        if col not in df.columns:
            continue
        for val in df[col].fillna("").astype(str):
            val_stripped = val.strip()
            # If empty or skip message, it's definitely not processed
            if (not val_stripped) or ("Skipped processing due to length." in val_stripped):
                continue
            # Attempt to parse JSON
            try:
                data = json.loads(val_stripped)
                metrics[f"processed_count__{col}"] += 1
                parsed_response = str(data.get("parsed_response", "")).lower()
                if parsed_response == "parsing error":
                    metrics[f"parsing_error_count__{col}"] += 1
            except json.JSONDecodeError:
                # JSON parse failed => not counted
                continue

    # Count how many rows are fully parsed across all columns
    for idx, row in df.iterrows():
        row_is_fully_parsed = True

        for col in specified_columns:
            if col not in df.columns:
                row_is_fully_parsed = False
                break
            val = str(row[col]).strip()
            if (not val) or ("Skipped processing due to length." in val):
                row_is_fully_parsed = False
                break
            # Check for "Parsing Error"
            try:
                data = json.loads(val)
                parsed_response = str(data.get("parsed_response", "")).lower()
                if parsed_response == "parsing error":
                    row_is_fully_parsed = False
                    break
            except json.JSONDecodeError:
                row_is_fully_parsed = False
                break

        if row_is_fully_parsed:
            metrics["fully_parsed_count"] += 1

    return metrics


def print_progress(csv_path: str, progress: Dict[str, int]) -> None:
    """
    Prints the progress metrics for a single CSV file in a user-friendly format.
    """
    specified_columns = [
        "Market Dynamics - a",
        "Intra-Industry Competition - a",
        "Regulatory Framework - a",
        "Technology Risk - a"
    ]

    total_rows = progress["total_rows"]
    fully_parsed = progress["fully_parsed_count"]

    print(f"\n=== Progress for: {csv_path} ===")
    print(f"Total rows in file: {total_rows}\n")
    for col in specified_columns:
        key_p = f"processed_count__{col}"
        key_e = f"parsing_error_count__{col}"
        if key_p not in progress:
            # Column missing from the DataFrame
            print(f"Column '{col}': 0 / {total_rows} rows processed.")
            print(f"   --> Among processed rows, 0 had 'Parsing Error'.\n")
        else:
            print(f"Column '{col}': {progress[key_p]} / {total_rows} rows processed.")
            print(f"   --> Among processed rows, {progress[key_e]} had 'Parsing Error'.\n")

    print("Overall fully-parsed rows:")
    print(f"   --> {fully_parsed} / {total_rows} rows have no empty/skipped columns and no 'Parsing Error'.\n")
    print(f"=== End of Progress for: {csv_path} ===\n")


def process_csv_file(csv_path: str) -> None:
    """
    Processes a single CSV file: loads it, computes progress metrics, and prints the progress.
    """
    print(f"Loading data from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File not found: {csv_path}")
        return
    except pd.errors.EmptyDataError:
        print(f"Warning: The file is empty: {csv_path}")
        return
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV file {csv_path}: {e}")
        return

    # Compute the metrics
    progress = get_progress_metrics(df)

    # Print the progress
    print_progress(csv_path, progress)


def traverse_and_process(root_folder: str, filename_prefix: str) -> None:
    """
    Traverses the root_folder and all its subfolders to find and process CSV files
    that start with the specified filename prefix.
    """
    if not os.path.isdir(root_folder):
        print(f"Error: The specified path is not a directory or does not exist: {root_folder}")
        sys.exit(1)

    print(f"Starting to traverse the directory: {root_folder}")

    csv_files_found = False
    for dirpath, dirnames, filenames in os.walk(root_folder):
        # Skip processing files in the root_folder itself
        if os.path.abspath(dirpath) == os.path.abspath(root_folder):
            continue

        for filename in filenames:
            # Check if filename starts with the specified prefix and ends with '.csv' (case-insensitive)
            if filename.lower().startswith(filename_prefix.lower()) and filename.lower().endswith('.csv'):
                csv_files_found = True
                csv_path = os.path.join(dirpath, filename)
                process_csv_file(csv_path)

    if not csv_files_found:
        print(f"No CSV files starting with '{filename_prefix}' found in the subdirectories of: {root_folder}")


def main():
    """
    Entry point of the script. Parses command-line arguments and initiates processing.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Check progress of all CSV files in a directory tree.")
    parser.add_argument(
        'root_folder',
        nargs='?',
        default='./data',
        help='Root folder to search for CSV files. Defaults to "./data".'
    )
    parser.add_argument(
        '--prefix',
        default='prospectuses_data_processed',
        help='Filename prefix to filter CSV files. Defaults to "prospectuses_data_processed".'
    )

    args = parser.parse_args()
    root_folder = args.root_folder
    filename_prefix = args.prefix

    traverse_and_process(root_folder, filename_prefix)


if __name__ == "__main__":
    main()