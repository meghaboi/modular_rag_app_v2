import os
import tempfile
import base64
import logging
import pandas as pd
import streamlit as st
from typing import Optional, Union

def save_uploaded_file(uploaded_file) -> Optional[str]:
    """
    Save uploaded file to a temporary location and return the path.
    
    Args:
        uploaded_file: The uploaded file from Streamlit's file_uploader
        
    Returns:
        str: Path to the saved temporary file, or None if save failed
    """
    try:
        file_suffix = os.path.splitext(uploaded_file.name)[1] if '.' in uploaded_file.name else '.txt'
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp:
            temp.write(uploaded_file.getvalue())
            logging.info(f"Saved uploaded file '{uploaded_file.name}' to temporary path: {temp.name}")
            return temp.name
    except Exception as e:
        logging.error(f"Error saving uploaded file: {e}")
        st.error(f"Error saving file: {e}")
        return None

def get_csv_download_link(df: pd.DataFrame, filename: str = "permutation_results.csv") -> str:
    """
    Generate a download link for a pandas dataframe as CSV.
    
    Args:
        df (pd.DataFrame): The dataframe to convert to CSV
        filename (str): The name of the file to download
        
    Returns:
        str: HTML string containing the download link
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Results as CSV</a>'
    return href

def get_file_size(file_path: str) -> Union[int, None]:
    """
    Get the size of a file in bytes.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        int: Size of the file in bytes, or None if file doesn't exist
    """
    try:
        return os.path.getsize(file_path)
    except (OSError, FileNotFoundError) as e:
        logging.error(f"Error getting file size for {file_path}: {e}")
        return None

def is_valid_file(file_path: str) -> bool:
    """
    Check if a file exists and is accessible.
    
    Args:
        file_path (str): Path to the file to check
        
    Returns:
        bool: True if file exists and is accessible, False otherwise
    """
    return os.path.isfile(file_path) and os.access(file_path, os.R_OK)