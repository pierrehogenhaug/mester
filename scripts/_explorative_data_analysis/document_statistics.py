sampled_rms_ids = ['367', '999', '1609', '625', '1108', '673', '219', '440', '328', '355', '139', '1629', '1074', '352', '1052', '946', '1897', '317', '653', '642', '1525', '1277', '935', '433', '153', '221', '1261', '199', '130', '252', '377', '84', '518', '201', '989', '1069', '1727', '1739', '258', '1127', '1182', '1096', '311', '1765', '661', '990', '251', '1136', '257', '398']

import os
import pandas as pd
import re
from collections import Counter
from statistics import mean
import sys
import ast
import string
from fuzzywuzzy import fuzz, process

from capfourpy.databases import Database

def main():
    # 1. Identify all CSV files under ./data/processed/{rms_id}/as_expected/*_parsed.csv
    project_root = "./"
    processed_root = os.path.join(project_root, "data", "processed")
    csv_file_pattern = re.compile(r'_parsed\.csv$', re.IGNORECASE)
    
    all_csv_paths = []
    for root, dirs, files in os.walk(processed_root):
        # Only consider paths that contain "as_expected" in them
        if "as_expected" in root:
            for file in files:
                
                if csv_file_pattern.search(file):
                    csv_path = os.path.join(root, file)
                    all_csv_paths.append(csv_path)
    
    # Prepare lists and counters to store aggregated information
    years = []
    risk_factor_counts = []
    total_lines_counts = []
    rms_ids = set()  # to track unique RmsIds (companies)
    
    # New accumulators for combined length info:
    total_combined_length_sum = 0
    total_combined_count = 0
    risk_combined_length_sum = 0
    risk_combined_count = 0
    
    # 2. Gather information from each CSV
    for csv_path in all_csv_paths:
        path_parts = csv_path.split(os.sep)
        if '1074' in path_parts:
            # example: skipping a particular RmsId
            continue

        try:
            processed_idx = path_parts.index("processed")
            # The folder immediately after "processed" should be the RMS id.
            rms_id = path_parts[processed_idx + 1]
        except ValueError:
            print(f"[DEBUG] 'processed' not found in path: {csv_path}")
            continue

        # Process only files that belong to a sampled RMS id
        if rms_id not in sampled_rms_ids:
            continue

        rms_ids.add(rms_id)
        # Read the CSV
        df = pd.read_csv(csv_path, dtype=str)
        if df.empty:
            continue

        # Prospectus Year (from first row, if available)
        year_str = df.iloc[0].get("Prospectus Year", "").strip()
        if year_str:
            try:
                year_int = int(year_str)
                years.append(year_int)
            except ValueError:
                pass  # Could not convert year to integer

        # Count lines with "Section Title" == "RISK FACTORS"
        if "Section Title" in df.columns:
            rf_lines = (df["Section Title"] == "RISK FACTORS").sum()
            risk_factor_counts.append(rf_lines)
        else:
            risk_factor_counts.append(0)

        # Total lines = total rows in CSV
        total_lines_counts.append(len(df))
        
        # --- New: Compute the combined length of "Subsubsection Title" and "Subsubsection Text" ---
        if "Subsubsection Title" in df.columns and "Subsubsection Text" in df.columns:
            # Fill missing values with empty strings, then compute length for each cell
            title_lengths = df["Subsubsection Title"].fillna("").str.len()
            text_lengths = df["Subsubsection Text"].fillna("").str.len()
            df["combined_length"] = title_lengths + text_lengths
            
            # Accumulate totals over all rows
            total_combined_length_sum += df["combined_length"].sum()
            total_combined_count += len(df)
            
            # For rows where Section Title is "RISK FACTORS", accumulate separately
            if "Section Title" in df.columns:
                risk_mask = df["Section Title"] == "RISK FACTORS"
                risk_combined_length_sum += df.loc[risk_mask, "combined_length"].sum()
                risk_combined_count += risk_mask.sum()
    
    # 3. Compute summary statistics
    
    ## 3a. Number of Companies
    number_of_companies = len(rms_ids)
    
    ## 3b. Time span
    if len(years) > 0:
        earliest_year = min(years)
        latest_year = max(years)
        year_counter = Counter(years)
        most_common_year, _ = year_counter.most_common(1)[0]
    else:
        earliest_year = None
        latest_year = None
        most_common_year = None
    
    ## 3c. Risk Factor entries
    total_rf = sum(risk_factor_counts)
    avg_rf = mean(risk_factor_counts) if len(risk_factor_counts) > 0 else 0
    
    ## 3d. Total (Sub)Subsection entries
    total_lines_sum = sum(total_lines_counts)
    avg_lines = mean(total_lines_counts) if len(total_lines_counts) > 0 else 0

    ## 3e. Average combined length of Subsubsection Title and Text per row
    if total_combined_count > 0:
        avg_combined_length = total_combined_length_sum / total_combined_count
    else:
        avg_combined_length = 0

    if risk_combined_count > 0:
        avg_risk_combined_length = risk_combined_length_sum / risk_combined_count
    else:
        avg_risk_combined_length = 0
    
    # 4. Build list of RmsId for DB query
    # Convert to a sorted list; watch out for non-numeric IDs
    list_of_rms_id = sorted(rms_ids, key=lambda x: int(x) if x.isdigit() else x)
    df_rms_ids = pd.DataFrame({"RmsId": list_of_rms_id})
    
    # 5. Query your database for IssuerData, merging on RmsId
    db = Database(database="C4DW")
    issuer_data_query = """
    WITH RankedIssuerData AS (
      SELECT 
        RmsId, 
        AbbrevName, 
        CapFourIndustry, 
        CountryIsoOperating,
        ROW_NUMBER() OVER (PARTITION BY RmsId ORDER BY AbbrevName ASC) AS rn
      FROM [C4DW].[DailyOverview].[IssuerData]
      WHERE RmsId IS NOT NULL 
        AND RmsId > 0
        AND CapFourIndustry IS NOT NULL
        AND CountryIsoOperating IS NOT NULL
    )
    SELECT 
      RmsId, 
      AbbrevName, 
      CapFourIndustry, 
      CountryIsoOperating
    FROM RankedIssuerData
    WHERE rn = 1;
    """
    
    data_database = db.read_sql(issuer_data_query)
    # Ensure the type matches the RmsId set
    data_database["RmsId"] = data_database["RmsId"].astype(str)
    
    # Perform a left merge on RmsId
    merged_df = df_rms_ids.merge(data_database, on="RmsId", how="left")
    
    # 6. Determine distinct industries and countries, plus their most common values
    capfour_industries = merged_df["CapFourIndustry"].dropna()
    country_isos = merged_df["CountryIsoOperating"].dropna()
    
    distinct_industries = capfour_industries.nunique()
    distinct_countries = country_isos.nunique()
    
    most_common_industry = (
        capfour_industries.value_counts().idxmax() 
        if not capfour_industries.empty else None
    )
    most_common_country = (
        country_isos.value_counts().idxmax() 
        if not country_isos.empty else None
    )
    
    # Compute the top three industries and countries with counts
    top_three_industries = capfour_industries.value_counts().head(3)
    top_three_countries = country_isos.value_counts().head(3)
    
    # 7. Print out the consolidated results
    print("=== Consolidated Results ===")
    print(f"Number of Companies (unique RmsId): {number_of_companies}")
    
    if earliest_year is not None and latest_year is not None:
        print(
            f"Time Span: Documents from {earliest_year} to {latest_year} "
            f"(most common year: {most_common_year})"
        )
    else:
        print("Time Span: No valid Prospectus Year data found.")
    
    print(
        f"Risk Factor Entries: On average, each prospectus had {avg_rf:.2f} "
        f"(total across all documents: {total_rf})"
    )
    
    print(
        f"Total (Sub)Subsection Entries (lines): On average, each prospectus had {avg_lines:.2f} lines "
        f"(total across all documents: {total_lines_sum})"
    )
    
    print(
        f"Average combined length (Subsubsection Title + Subsubsection Text) per line (all rows): {avg_combined_length:.2f}"
    )
    print(
        f"Average combined length (Subsubsection Title + Subsubsection Text) per line (where Section Title == 'RISK FACTORS'): {avg_risk_combined_length:.2f}"
    )
    
    print("\nIssuerData Info:")
    print(f"  Distinct CapFourIndustry: {distinct_industries} (most common: {most_common_industry})")
    print("  Top 3 Industries:")
    for industry, count in top_three_industries.items():
        print(f"    {industry}: {count}")
    
    print(f"\n  Distinct CountryIsoOperating: {distinct_countries} (most common: {most_common_country})")
    print("  Top 3 Countries:")
    for country, count in top_three_countries.items():
        print(f"    {country}: {count}")
    
    print("=================================")
    
    ##############################################################################
    # PART B: fundamental-score label reading, merging, and counting
    ##############################################################################
    # We'll do it at the *end* of this script, using the set of RmsIds we found.
    
    print("\n=== Fundamental Score Distribution (using fuzzy matching) ===")

    def clean_text(text: str) -> str:
        """
        Cleans the input text by:
        1) Replacing newline characters with a single space,
        2) Collapsing multiple spaces into one,
        3) Stripping leading/trailing whitespace.
        """
        if pd.isnull(text):
            return ""
        text = text.replace('\r', ' ').replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def parse_tagged_characteristics(s: str) -> list:
        """Safely parses a string representing a list of tagged characteristics."""
        try:
            return ast.literal_eval(s)
        except:
            return []

    def assign_letters(group: pd.DataFrame, positive_letters: list, negative_letters: list, group_name: tuple) -> pd.DataFrame:
        """
        Assigns letters to each row in 'group' based on 'CharacteristicInfluence'.
        group_name is (Category, 'Positive'/'Negative').
        """
        characteristic_influence = group_name[1]  # 'Positive' or 'Negative'
        n = len(group)
        if characteristic_influence == 'Positive':
            letters = positive_letters[:n]
        else:
            letters = negative_letters[:n]
        group = group.copy()
        group['letter'] = letters
        group['Label'] = group['Category'] + '.' + group['letter']
        return group

    # Read the two CSVs:
    fs_path = './data/rms_with_fundamental_score.csv'
    usc_path = './data/unique_score_combinations.csv'

    if not os.path.exists(fs_path) or not os.path.exists(usc_path):
        print(f"[WARNING] Required CSV(s) not found. Check paths:\n  {fs_path}\n  {usc_path}")
        return

    # 1) Analyst ground-truth data
    df_fs = pd.read_csv(fs_path)
    # 2) The reference for how to assign letter-based labels:
    df_usc = pd.read_csv(usc_path)

    # Clean up the reference text
    df_usc['TaggedCharacteristics'] = df_usc['TaggedCharacteristics'].apply(clean_text)

    # Assign letters to each row in df_usc
    positive_letters = list(string.ascii_uppercase)
    negative_letters = list(string.ascii_lowercase)
    grouped = df_usc.groupby(['Category', 'CharacteristicInfluence'])
    processed_groups = []
    for name, group in grouped:
        processed_groups.append(assign_letters(group, positive_letters, negative_letters, name))
    df_usc_labeled = pd.concat(processed_groups).reset_index(drop=True)

    # Process the analyst data
    df_fs['TaggedCharacteristics'] = df_fs['TaggedCharacteristics'].apply(parse_tagged_characteristics)
    df_fs_exploded = df_fs.explode('TaggedCharacteristics')

    # Extract the text and influence from each tagged characteristic
    df_fs_exploded['CharacteristicText'] = df_fs_exploded['TaggedCharacteristics'].apply(
        lambda x: x.get('CharacteristicText', '') if isinstance(x, dict) else ''
    )
    df_fs_exploded['CharacteristicInfluence'] = df_fs_exploded['TaggedCharacteristics'].apply(
        lambda x: x.get('CharacteristicInfluence', '') if isinstance(x, dict) else ''
    )

    # Clean the extracted text so that it roughly matches the reference text
    df_fs_exploded['CharacteristicText'] = df_fs_exploded['CharacteristicText'].apply(clean_text)
    df_fs_exploded['CharacteristicInfluence'] = df_fs_exploded['CharacteristicInfluence'].apply(clean_text)

    # --- Fuzzy Matching ---
    def get_fuzzy_label(row, ref_df, threshold=90):
        """
        For the given row from df_fs_exploded, find the best match from the
        reference DataFrame (ref_df) based on fuzzy matching of CharacteristicText.
        Returns the reference row's 'Label' if a good match is found; otherwise, None.
        """
        category = row['Category']
        influence = row['CharacteristicInfluence']
        text = row['CharacteristicText']
        
        # Limit candidates to those with the same Category and CharacteristicInfluence
        candidates = ref_df[(ref_df['Category'] == category) & (ref_df['CharacteristicInfluence'] == influence)]
        if candidates.empty:
            return None
        
        candidate_texts = candidates['TaggedCharacteristics'].tolist()
        
        # Use fuzzy matching; note that extractOne returns (match, score) in your version.
        best_match = process.extractOne(text, candidate_texts, scorer=fuzz.token_sort_ratio)
        if best_match is None:
            return None
        
        matched_text, score = best_match  # Unpack the two values
        if score < threshold:
            return None
        
        # Retrieve the index of the best match from the candidate list
        match_idx = candidate_texts.index(matched_text)
        label = candidates.iloc[match_idx]['Label']
        return label

    # Apply fuzzy matching to assign labels for each row in df_fs_exploded
    df_fs_exploded['FuzzyLabel'] = df_fs_exploded.apply(lambda row: get_fuzzy_label(row, df_usc_labeled), axis=1)

    # Ensure RmsId is of string type for consistency, then filter to only rows for our companies
    df_fs_exploded['RmsId'] = df_fs_exploded['RmsId'].astype(str)
    relevant_rows = df_fs_exploded[df_fs_exploded['RmsId'].isin(rms_ids)]

    if relevant_rows.empty:
        print("No matching fundamental score labels found for the RmsIds in this dataset.")
        return

    # Count occurrences of each assigned fuzzy label.
    label_counts = relevant_rows['FuzzyLabel'].value_counts(dropna=True)

    # (Optional) Filter to include only labels whose letter (after the dot) is lowercase.
    label_counts = label_counts[label_counts.index.map(lambda lbl: isinstance(lbl, str) and lbl.split('.')[-1].islower())]

    # Create a list of all possible lowercase labels based on the reference CSV
    all_lowercase_labels = sorted(
        df_usc_labeled[df_usc_labeled['Label'].apply(lambda x: x.split('.')[-1].islower())]['Label'].unique()
    )
    # Build a dictionary with counts (using 0 if a label was not found)
    full_label_counts = {label: int(label_counts.get(label, 0)) for label in all_lowercase_labels}

    print("Label distribution for the companies in this dataset:")
    for lbl in all_lowercase_labels:
        print(f"  {lbl}: {full_label_counts[lbl]}")

    print("=== End of Fundamental Score Distribution ===")


if __name__ == "__main__":
    main()