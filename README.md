[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3100/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NPL Semantic

TODO.

# Data

TODO: describe columns (standardize: lowercase, in english, underscore for space)

- num:
- id:
- category:
- concept:
- property:

TODO: describe data (any relevant information) and add references
TODO: if data has additional columns other than standard ones (or if one is missing), mention and describe it here

<!-- dados italian, german data, parkinson paper, swear fluency -->

# Getting Started

Follow these steps to prepare your data, set up the environment, and configure the project for use.

## 1. Add Your Data

Place your files in the `data/raw` folder. If your data is in `.xlsx` format, run the `tocsv.py` script to convert them to `.csv`.

## 2. Set Up the Environment

```bash
# Install uv (if not already installed)
pip install uv

# Create and activate a virtual environment
uv venv -p 3.11 .venv
source .venv/bin/activate  # macOS / Linux
.venv\Scripts\activate  # Windows

# Install dependencies
uv pip install -r requirements.txt -r requirements-dev.txt
pre-commit install
```

## 3. Configure Environment Variables

Create a `.env` file in the project root with:

```
API_KEY=your-api-key
FILENAME=your-filename
````
