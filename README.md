# Prospectus Analysis Project
This project is awesome.

## Project Structure

```plaintext
project_name/
├── data/                     # Data files
│   ├── raw/                  # Original raw data files
│   └── processed/            # Processed data for analysis
├── notebooks/                # Jupyter notebooks for EDA and analysis
├── scripts/                  # Scripts for running analysis and data processing
│   ├── analysis/             # Analysis-related scripts
│   │   └── run_analysis.py   # Script to run the analysis
│   ├── data_processing/      # Data processing scripts
│   │   └── run_data_processing.py  # Script for data processing tasks
│   ├── data_scraping/        # Data scraping scripts
│   └── main.py               # Main script to coordinate the project
├── src/                      # Source code
│   ├── analysis/             # Analysis module
│   │   ├── __init__.py       # Initialization for the analysis module
│   │   └── prospectus_analyzer.py  # Module for prospectus analysis
│   ├── data_processing/      # Data processing module
│   │   ├── __init__.py       # Initialization for data processing module
│   │   └── pdf_parsing.py    # Module for parsing PDF files
│   ├── data_scraping/        # Data collection module
│   │   ├── __init__.py       # Initialization for data collection module
│   │   └── scrape_esma.py    # Module for scraping PDF files from the Esma website
│   └── __init__.py           # Initialization for src package
├── tests/                    # Unit tests
├── .gitignore                # Git ignore file
├── config.yaml               # Project configuration
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/project_name.git
   ```
2. Install dependencies:
   ```bash
    pip install -r requirements.txt
   ```


