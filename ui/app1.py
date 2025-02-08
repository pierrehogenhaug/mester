# Same as app.py, but it doesn't clear download buttons after one download.
import os
import sys
import streamlit as st

# Initialize session state for offerings_df
if 'offerings_df' not in st.session_state:
    st.session_state.offerings_df = None

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Our script imports
try:
    from scripts.data_collection.run_database_utils import run_extraction
except ModuleNotFoundError as e:
    st.error(f"Could not import run_extraction: {e}")
    st.stop()

try:
    from scripts.data_collection.get_legal_offerings import (
        list_legal_offerings_for_rmsid,
        download_legal_offering
    )
except ModuleNotFoundError as e:
    st.error(f"Could not import the new SharePoint scripts: {e}")
    st.stop()


def main():
    st.title("Fundamental Score Extraction")

    # (A) Prompt user for RmsId
    rms_id_input = st.text_input("Enter RmsId (optional):")

    if rms_id_input:
        try:
            rms_id_int = int(rms_id_input)
        except ValueError:
            st.error("Please enter a valid integer RmsId.")
            st.stop()
    else:
        rms_id_int = None

    # (B) Checkbox: skip CSV output if checked
    skip_csv = st.checkbox("Skip CSV output (just show results in the UI)", value=True)

    # (C) Button to run DB extraction
    if st.button("Extract Scores"):
        with st.spinner("Running database extraction..."):
            try:
                df = run_extraction(rms_id=rms_id_int)
                df = df[['RmsId', 'CompanyName', 'ScoringDate',
                         'CategoryGroup', 'Category', 'Score',
                         'TaggedCharacteristics', 'SharePointLink']]
                st.session_state.extraction_df = df  # Optionally store extraction results
                st.subheader("Results Preview")
                st.dataframe(df)

                if not skip_csv:
                    output_dir = os.path.join(project_root, "data")
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, "rms_with_fundamental_score.csv")
                    df.to_csv(output_path, index=False)
                    st.success(f"Data saved to '{output_path}'")
            except Exception as ex:
                st.error(f"An error occurred during extraction: {ex}")

    st.write("---")
    st.subheader("SharePoint Legal Offerings")
    st.write("Use the button below to list and optionally download Legal Offerings from SharePoint.")

    # Button to retrieve & list legal offerings
    if st.button("Show Legal Offerings for RmsId"):
        if not rms_id_int:
            st.warning("Please enter a valid RmsId above before clicking this button.")
        else:
            with st.spinner("Fetching Legal Offerings from SharePoint..."):
                try:
                    offerings_df = list_legal_offerings_for_rmsid(rms_id_int)
                    st.session_state.offerings_df = offerings_df  # Store in session state
                except Exception as ex:
                    st.error(f"Failed to fetch offerings: {ex}")
                    st.session_state.offerings_df = None

    # Optional: Add a button to clear the offerings_df from session state
    if st.button("Clear Legal Offerings"):
        st.session_state.offerings_df = None
        st.success("Legal Offerings have been cleared.")

    # Check if offerings_df exists in session state and display
    if st.session_state.offerings_df is not None:
        offerings_df = st.session_state.offerings_df
        if offerings_df.empty:
            st.info("No Legal Offerings found for this RmsId.")
        else:
            st.success(f"Found {len(offerings_df)} Legal Offering(s).")
            st.dataframe(offerings_df)

            # Provide download buttons for each file
            for idx, row in offerings_df.iterrows():
                file_name = row["FileName"]
                site_url = row["SiteUrl"]
                server_relative_url = row["ServerRelativeUrl"]

                st.write(f"**File:** {file_name}")
                try:
                    # Download this file in memory
                    file_bytes_io = download_legal_offering(site_url, server_relative_url)
                    # Streamlit's download_button
                    st.download_button(
                        label=f"Download {file_name}",
                        data=file_bytes_io.getvalue(),
                        file_name=file_name,
                        mime="application/octet-stream"
                    )
                except Exception as ex:
                    st.error(f"Unable to download file '{file_name}': {ex}")


if __name__ == "__main__":
    main()