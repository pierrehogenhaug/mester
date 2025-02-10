#####################################
# Stage 1: Build Environment (Builder)
#####################################
FROM python:3.9-slim as builder

# Avoid .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install build tools and system dependencies needed for Selenium, PDF processing, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    ca-certificates \
    chromium \
    chromium-driver \
 && rm -rf /var/lib/apt/lists/*

# Set working directory for the build
WORKDIR /app

# Copy requirements and install Python dependencies.
# Using --no-cache-dir to keep the image lean.
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the full project (src/, scripts/, etc.)
COPY . .

#####################################
# Stage 2: Final Runtime Image
#####################################
FROM python:3.9-slim

# Set environment variables:
# - PYTHONDONTWRITEBYTECODE and PYTHONUNBUFFERED for Python behavior
# - CHROME_BIN and PATH so that Chromium can be found by Selenium
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CHROME_BIN=/usr/bin/chromium \
    PATH="/usr/lib/chromium:${PATH}"

# Install only runtime system dependencies (Chromium and its driver)
RUN apt-get update && apt-get install -y --no-install-recommends \
    chromium \
    chromium-driver \
 && rm -rf /var/lib/apt/lists/*

# Set working directory in the runtime container
WORKDIR /app

# Copy the installed Python packages and application code from the builder stage.
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /app /app

# (Optional) Switch to a non-root user for better security:
# RUN adduser --disabled-password appuser && chown -R appuser:appuser /app
# USER appuser

# Default command to run the SharePoint scraping script.
# (Override this CMD when running the container if you wish to run a different script.)
CMD ["python", "scripts/data_collection/run_scrape_sharepoint.py"]