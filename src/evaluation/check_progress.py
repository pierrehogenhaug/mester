import sys
import json
import pandas as pd

def get_progress_metrics(df: pd.DataFrame) -> dict:
    """
    Given a DataFrame `df`, compute progress metrics and return them as a dict
    suitable for wandb.log(...).
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
    # e.g.: processed_count_{col}, parsing_error_count_{col}
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


def main(csv_path: str):
    """
    The original CLI entry point that prints out the stats.
    We now rely on get_progress_metrics for the logic.
    """
    print(f"Loading data from: {csv_path}\n")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)

    # Reuse the function to get the metrics
    progress = get_progress_metrics(df)

    # Print them in a user-friendly format, as before:
    specified_columns = [
        "Market Dynamics - a",
        "Intra-Industry Competition - a",
        "Regulatory Framework - a",
        "Technology Risk - a"
    ]

    total_rows = progress["total_rows"]
    fully_parsed = progress["fully_parsed_count"]

    print(f"Total rows in file: {total_rows}\n")
    for col in specified_columns:
        key_p = f"processed_count__{col}"
        key_e = f"parsing_error_count__{col}"
        if key_p not in progress:
            # column missing from the DF
            print(f"Column '{col}': 0 / {total_rows} rows processed.")
            print(f"   --> Among processed rows, 0 had 'Parsing Error'.\n")
        else:
            print(f"Column '{col}': {progress[key_p]} / {total_rows} rows processed.")
            print(f"   --> Among processed rows, {progress[key_e]} had 'Parsing Error'.\n")

    print("Overall fully-parsed rows:")
    print(f"   --> {fully_parsed} / {total_rows} rows have no empty/skipped columns and no 'Parsing Error'.\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_progress.py <path_to_processed_csv>")
        sys.exit(1)

    csv_file_path = sys.argv[1]
    main(csv_file_path)