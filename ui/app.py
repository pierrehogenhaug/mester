# file: ./ui/app.py

import os
import sys
import streamlit as st
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Existing code from your question...
try:
    from scripts.data_collection.run_database_extraction import run_extraction
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

# --- IMPORT for PDF parsing ---
try:
    from src.data_processing.pdf_parsing import process_prospectus
except ModuleNotFoundError as e:
    st.error(f"Could not import process_prospectus from pdf_parsing: {e}")
    st.stop()

# --- IMPORT LLM analysis ---
try:
    from scripts.analysis.run_analysis import analyze_prospectus_dataframe

except ModuleNotFoundError as e:
    st.error(f"Could not import analyze_prospectus_dataframe from run_analysis: {e}")
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
                except Exception as ex:
                    st.error(f"Failed to fetch offerings: {ex}")
                    st.stop()

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

    # --- SECTION FOR SINGLE PDF DRAG & DROP ---
    st.write("---")
    st.subheader("Single PDF Parsing")
    st.write("Upload a PDF below to parse its 'RISK FACTORS' section.")

    uploaded_pdf = st.file_uploader("Drag & drop a PDF file here", type=["pdf"])

    if uploaded_pdf is not None:
        # We'll create a unique temporary path to store the uploaded file
        temp_dir = os.path.join(project_root, "data", "temp_uploads")
        os.makedirs(temp_dir, exist_ok=True)

        temp_pdf_path = os.path.join(temp_dir, uploaded_pdf.name)

        # Save the uploaded file to disk
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_pdf.read())

        st.write(f"File uploaded: {uploaded_pdf.name}")
        
        try:
            with st.spinner("Parsing PDF..."):
                data, next_section_id, processing_result, md_text = process_prospectus(
                    pdf_file_path=temp_pdf_path,
                    original_filename=uploaded_pdf.name,
                    prospectus_id="N/A (Manual Upload)",
                    section_id_map={},
                    next_section_id=1,
                    from_folder="Manual Upload",
                    f_year=None
                )

            # Show results
            st.success("Parsing complete!")

            # Show the structured data
            st.write("### Parsed Risk Factors Data")
            if data:
                df_data = pd.DataFrame(data)
                st.dataframe(df_data)

                # Add a button to run the LLM analysis
                if st.button("Run LLM Analysis"):
                    # We can show a progress bar or messages as we process
                    progress_bar = st.progress(0)
                    analysis_status = st.empty()  # a placeholder for text updates

                    def progress_callback(current_index, total_rows):
                        # update the progress bar
                        progress = (current_index + 1) / total_rows
                        progress_bar.progress(progress)
                        analysis_status.write(
                            f"Analyzing row {current_index+1} of {total_rows}..."
                        )

                    with st.spinner("Running LLM Analysis..."):
                        analyzed_df = analyze_prospectus_dataframe(
                            df_data,
                            model_type="openai",        # or "local"
                            local_model_path="path/to/your/model.gguf",
                            sample=False,               # or True for partial testing
                            progress_callback=progress_callback
                        )

                    st.success("LLM Analysis complete!")
                    st.dataframe(analyzed_df)

                    # If you want to see the JSON in detail, you can add an expander
                    with st.expander("Show detailed JSON columns"):
                        st.write(analyzed_df)
    


if __name__ == "__main__":
    main()