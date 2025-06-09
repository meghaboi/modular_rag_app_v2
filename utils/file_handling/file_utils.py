import os
import logging
import tempfile
from PyPDF2 import PdfReader

def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary location and return the path"""
    try:
        file_suffix = os.path.splitext(uploaded_file.name)[1] if '.' in uploaded_file.name else '.txt'
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp:
            temp.write(uploaded_file.getvalue())
            temp_path = temp.name
            logging.info(f"Saved uploaded file '{uploaded_file.name}' to temporary path: {temp_path}")
            
            if file_suffix.lower() == '.pdf':
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as text_temp:
                        pdf_reader = PdfReader(temp_path)
                        text_content = ""
                        for page in pdf_reader.pages:
                            text_content += page.extract_text() + "\n"
                        
                        text_temp.write(text_content.encode('utf-8'))
                        text_temp_path = text_temp.name
                    
                    pdf_reader = None
                    try:
                        os.unlink(temp_path)
                        logging.info(f"Converted PDF to text and saved to: {text_temp_path}")
                    except Exception as e:
                        logging.warning(f"Could not delete original PDF file: {e}")
                    
                    return text_temp_path
                except Exception as e:
                    logging.error(f"Error converting PDF to text: {e}")
                    return None
            
            return temp_path
    except Exception as e:
        logging.error(f"Error saving uploaded file: {e}")
        return None 