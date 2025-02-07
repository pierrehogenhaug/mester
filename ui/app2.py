import os
import sys
import streamlit as st
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.data_collection.run_database_extraction import run_extraction
from scripts.data_collection.get_legal_offerings import (
    list_legal_offerings_for_rmsid,
    download_legal_offering
)
from src.data_processing.pdf_parsing import process_prospectus

# Import the new function
from scripts.analysis.run_analysis import analyze_prospectus_dataframe

def main():
    st.title("Fundamental Score Extraction")

    # (A) RmsId input + DB extraction section
    rms_id_input = st.text_input("Enter RmsId (optional):")
    if rms_id_input:
        try:
            rms_id_int = int(rms_id_input)
        except ValueError:
            st.error("Please enter a valid integer RmsId.")
            st.stop()
    else:
        rms_id_int = None

    skip_csv = st.checkbox("Skip CSV output (just show results in the UI)", value=True)

    if st.button("Extract Scores"):
        with st.spinner("Running database extraction..."):
            try:
                df = run_extraction(rms_id=rms_id_int)
                # Show results in UI...
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
    # [unchanged: code to list & download legal offerings...]

    st.write("---")
    st.subheader("Single PDF Parsing")
    st.write("Upload a PDF below to parse its 'RISK FACTORS' section.")

    uploaded_pdf = st.file_uploader("Drag & drop a PDF file here", type=["pdf"])
    parsed_df = None  # We will store the parsed DataFrame

    if uploaded_pdf is not None:
        # Create a temp folder to store the PDF
        temp_dir = os.path.join(project_root, "data", "temp_uploads")
        os.makedirs(temp_dir, exist_ok=True)
        temp_pdf_path = os.path.join(temp_dir, uploaded_pdf.name)

        # Save the uploaded file
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_pdf.read())

        st.write(f"File uploaded: {uploaded_pdf.name}")

        # Parse the PDF
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

            st.success("Parsing complete!")

            # Convert the result to a DataFrame
            if data:
                parsed_df = pd.DataFrame(data)
                st.write("### Parsed Risk Factors Data")
                st.dataframe(parsed_df)
            else:
                st.warning("No data extracted or data list is empty.")

            # Show the raw Markdown in an expander
            st.write("### Extracted Markdown Text")
            with st.expander("Show/Hide Markdown"):
                st.text(md_text)

            # Show final parsing status
            st.write(f"**Processing result**: `{processing_result}`")
            if processing_result == "as_expected":
                st.success("Looks like the RISK FACTORS section was parsed successfully.")
            else:
                st.warning("Parsing was not as expected; check above for details.")

        except Exception as e:
            st.error(f"Error during parsing: {e}")

    # --- NEW SECTION: LLM Analysis on the Parsed DataFrame ---
    st.write("---")
    st.subheader("Run Risk Factor Analysis with LLM")

    # Some user-selectable parameters for model_type, local path, etc.
    model_type_choice = st.selectbox("Model Type", ["openai", "local"])
    local_model_path = st.text_input("Local Model Path", value="../Llama-3.2-1B-Instruct-Q8_0.gguf")
    enable_streaming = st.checkbox("Enable Streaming (OpenAI only)", value=True)
    run_analysis_button = st.button("Analyze Parsed Data")

    if run_analysis_button:
        if "parsed_df" not in st.session_state or st.session_state["parsed_df"].empty:
            st.warning("No parsed DataFrame available. Please upload and parse a PDF first.")
        else:
            parsed_df = st.session_state["parsed_df"]
            st.info(f"Starting LLM analysis using: {model_type_choice}")

            # Create a progress bar and a placeholder for the streaming text
            progress_bar = st.progress(0)
            status_text = st.empty()

            # This placeholder will show the partial content tokens
            stream_partial_placeholder = st.empty()

            # We'll hold partial text in a buffer for each row
            partial_content_buffer = []

            def progress_callback(idx, total):
                pct = int((idx+1)/total * 100)
                progress_bar.progress(pct)
                status_text.write(f"Analyzing row {idx+1}/{total}")

            def row_stream_callback(token_chunk):
                """
                Whenever a chunk of text arrives from the LLM, append it.
                Then update the UI with the partial text so far.
                """
                partial_content_buffer.append(token_chunk)
                # Display it in a text area or markdown
                stream_partial_placeholder.markdown(
                    "#### Partial Streamed Output (current row)\n" + "".join(partial_content_buffer)
                )

            with st.spinner("Analyzing risk factors..."):
                result_df = analyze_prospectus_dataframe(
                    df=parsed_df.copy(),
                    model_type=model_type_choice,
                    local_model_path=local_model_path,
                    sample=False,
                    progress_callback=progress_callback,
                    streaming=enable_streaming,        # Turn streaming on/off
                    stream_callback=row_stream_callback # Our function above
                )

            # Once done, clear the partial placeholder
            stream_partial_placeholder.empty()
            st.success("Analysis complete!")

            # Show final results
            st.write("### Updated DataFrame with Analysis Results")
            st.dataframe(result_df)

            # Optionally let users download
            csv_download = result_df.to_csv(index=False)
            st.download_button(
                label="Download Analysis CSV",
                data=csv_download.encode("utf-8"),
                file_name="parsed_analysis_results.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()