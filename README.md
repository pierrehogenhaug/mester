# Prospectus Risk Analysis via LLM
*_Master’s Thesis Project_*

This project automates the collection, processing, and analysis of bond prospectus documents to detect risk factors. It combines data scraping, PDF parsing, large language model (LLM) prompting, and evaluation against analyst ground truth to support risk assessment.

## TOC
1. [Project Overview](#Overview)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Usage](#usage)
   - [Data Collection](#data-collection)
   - [Data Processing](#data-processing)
   - [LLM Analysis](#llm-analysis)
   - [Evaluation](#evaluation)
5. [Requirements](#requirements)

## Overview
Overview
This project supports several key tasks:

**Data Collection**:
- Scrapes prospectus documents from SharePoint and ESMA.
- Extracts data from a database using custom SQL queries.

**Data Processing**:
- Parses PDF files and converts them into structured Markdown.
- Builds a hierarchical representation (sections, subsections, etc.) of prospectus content.

**LLM Analysis**:
- Uses a two-step process (detection and evaluation) to prompt a large language model (LLM) to detect risk factors in subsections of the prospectus.
- Supports both OpenAI and local models (via LlamaCpp).

**Evaluation**:
- Compares LLM predictions with analyst-assigned ground truth.
- Computes performance metrics such as Precision, Recall, F1 Score, and Accuracy.
- Measures intra-annotator agreement using Fleiss’ Kappa.

## Project Structure

```plaintext
project_root/
├── data/
│   ├── processed/                       # Prospectuses (e.g., raw PDF, markdown or parsed output)
│   ├── rms_with_fundamental_score.csv   # CSV generated from database extraction
│   └── unique_score_combinations.csv    # CSV with unique score combinations (for evaluation)
│
├── scripts/                             # Command-line scripts (entry points) for running tasks
│   ├── data_collection/
│   │   ├── run_scrape_sharepoint.py     # Script to run SharePoint scraping
│   │   └── run_database_utils.py        # Script to run database extraction and processing
│   │
│   ├── data_processing/
│   │   └── run_parse_prospectus.py      # Script to parse PDFs/Markdown prospectuses
│   │
│   ├── analysis/
│   │   ├── run_analysis_single_llm.py     # Script to run the one‐step LLM risk detection
│   │   └── run_analysis_two_step_llm.py # Script to run the two‐step LLM risk detection
│   │
│   └── evaluation/
│       ├── run_evaluate_llm.py          # Script to evaluate LLM outputs against ground truth
│       └── run_evaluate_llm_intra_annotator.py  # Script for intra-annotator agreement (Fleiss’ Kappa)
│
├── src/                                 # Source code (modules) organized by functionality
│   ├── data_collection/
│   │   ├── database_utils.py            # Functions for database queries
│   │   ├── scrape_esma.py               # Script to scrape prospectuses from ESMA
│   │   └── scrape_sharepoint.py         # SharePoint scraping module (custom SharePoint class, etc.)
│   │
│   ├── data_processing/
│   │   ├── data_processing.py           # PDF-parsing and data cleaning/merging functions
│   │   └── parse_prospectus.py          # Module to convert PDF/MD files into structured Markdown/CSV
│   │
│   ├── analysis/
│   │   ├── analysis_single_llm.py       # Code to prompt the LLM for risk detection (detection)
│   │   └── analysis_two_step_llm.py     # Code to prompt the LLM for risk detection (detection & evaluation)
│   │
│   └── evaluation/
│       ├── evaluate_llm.py              # Tools for comparing LLM predictions vs. analyst ground truth
│       └── evaluate_llm_intra_annotator.py  # Tools for measuring intra-annotator consistency (Fleiss’ Kappa)
│
├── Dockerfile                           # Multistage 
├── requirements.txt                     # List of required Python packages
└── README.md                            # Project overview and instructions
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
3. Set Up Environment Variables:
For LLM analysis with OpenAI, create a .env file with your API key:

## Usage
Each major component of the project has its own entry-point script. Below are usage examples for each stage.

### Data Collection
- SharePoint Scraping:
Downloads prospectus files from SharePoint and saves them in data/raw/.

   ```bash
   python scripts/data_collection/run_scrape_sharepoint.py
   ```
- Database Extraction:
Extracts data from the underlying database (e.g., fundamental scores and issuer information) and saves CSV files in data/.

   ```bash
   python scripts/data_collection/run_database_utils.py
   ```

### Data Processing
- Prospectus Parsing:
Parses PDF (or Markdown) files to extract structured content (sections, subsections, etc.) and outputs CSV files.

   ```bash
   python scripts/data_processing/run_parse_prospectus.py --file path/to/your/file.pdf --output path/to/output/folder
   ```
- For full-run mode (processing multiple prospectuses), simply run without the --file argument:

   ```bash
   python scripts/data_processing/run_parse_prospectus.py
   ```

### LLM Analysis
- Risk Detection with LLM:
Uses a two-step LLM prompting process to detect risk factors in each subsection of a prospectus.

   ```bash
   python scripts/analysis/run_analysis_two_step_llm.py --model_type openai
   ```
To use a local LLM (e.g., via LlamaCpp):

   ```bash
   python scripts/analysis/run_analysis_two_step_llm.py --model_type local --local_model_path /path/to/local/model
   ```
Additional options such as sampling or processing a specific RMS ID can be provided via command-line flags. See the script’s help for details:

   ```bash
python scripts/analysis/run_analysis_two_step_llm.py --help
   ```

### Evaluation
- LLM Output Evaluation:
Compares LLM predictions against analyst-based ground truth by computing confusion matrices and various metrics.

   ```bash
   python scripts/evaluation/run_evaluate_llm.py path/to/processed_csv1.csv [path/to/processed_csv2.csv ...]
   ```

- Intra-Annotator Agreement:
Computes Fleiss’ Kappa over multiple CSV groups to measure consistency.

   ```bash
   python scripts/evaluation/run_evaluate_llm_intra_annotator.py --processed_root data/processed --question_cols "Market Dynamics - a" "Intra-Industry Competition - a" "Technology Risk - a" "Regulatory Framework - a"
   ```

## Docker Deployment
A multistage Dockerfile is provided to build a lightweight runtime image.
- Build the Docker Image:

   ```bash
   docker build -t prospectus-risk-analysis .
   ```

- Run the Container:
By default, the container runs the SharePoint scraping script. To override the default command, supply a different command (e.g., to run the database extraction):

   ```bash
   docker run --rm prospectus-risk-analysis python scripts/data_collection/run_database_utils.py
   ```

## Requirements
- Python: 3.9 (or later recommended)
- Python Packages: See `requirements.txt`
- LLM Options:
   - For OpenAI models, ensure you have a valid API key in your `.env`.
