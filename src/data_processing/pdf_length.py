import os
import csv
import PyPDF2  # pip install PyPDF2

def get_pdf_page_count(pdf_path):
    """
    Given a path to a PDF file, returns the number of pages.
    """
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            return len(reader.pages)
    except Exception as e:
        # You can add logging or print statements here to debug.
        return None

def main():
    base_path = "./data/processed"
    output_csv = "pdf_lengths.csv"

    # Open the CSV file for writing
    with open(output_csv, mode="w", newline="", encoding="utf-8") as csv_file:
        fieldnames = ["company_id", "file_name", "document_length"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        # Walk through each item in the base_path
        for company_id in os.listdir(base_path):
            company_path = os.path.join(base_path, company_id)
            # The item must be a directory to be considered a company folder
            if not os.path.isdir(company_path):
                continue

            as_expected_path = os.path.join(company_path, "as_expected")
            # Check that "as_expected" is actually a directory
            if not os.path.isdir(as_expected_path):
                continue

            # Look for all PDF files inside this "as_expected" folder
            for file_name in os.listdir(as_expected_path):
                if file_name.lower().endswith(".pdf"):
                    pdf_path = os.path.join(as_expected_path, file_name)
                    page_count = get_pdf_page_count(pdf_path)

                    # Only write to CSV if we successfully got a page count
                    if page_count is not None:
                        writer.writerow({
                            "company_id": company_id,
                            "file_name": file_name,
                            "document_length": page_count
                        })

    print(f"CSV file created: {output_csv}")

if __name__ == "__main__":
    main()