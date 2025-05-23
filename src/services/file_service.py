import os
import tempfile
import logging
from typing import Optional

def save_uploaded_file(uploaded_file) -> Optional[str]:
    """Save uploaded file to a temporary location and return the path"""
    try:
        file_suffix = os.path.splitext(uploaded_file.name)[1] if '.' in uploaded_file.name else '.txt'
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp:
            temp.write(uploaded_file.getvalue())
            logging.info(f"Saved uploaded file '{uploaded_file.name}' to temporary path: {temp.name}")
            return temp.name
    except Exception as e:
        logging.error(f"Error saving uploaded file: {e}")
        return None

def get_csv_download_link(df, filename="permutation_results.csv"):
    """Generate a download link for a pandas dataframe as CSV"""
    import base64
    import pandas as pd
    
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Results as CSV</a>'
    return href 