import os
import re
import logging
import base64
import httpx
from typing import Optional
from openai import OpenAI

def text_to_speech(text: str) -> Optional[bytes]:
    """Generates speech from text using OpenAI TTS and returns audio bytes."""
    if not text or not isinstance(text, str):
        logging.warning("TTS skipped: Input text is empty or not a string.")
        return None

    cleaned_text = re.sub(r'[#*]', '', text) 
    cleaned_text = re.sub(r'http[s]?://\S+', '', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    if not cleaned_text:
        logging.warning("TTS skipped: Text is empty after cleaning.")
        return None

    if not os.getenv("OPENAI_API_KEY"):
        logging.error("OpenAI API Key not found. Cannot generate audio.")
        return None

    try:
        client = OpenAI(timeout=httpx.Timeout(45.0, connect=10.0)) 
        selected_voice = "fable"
        selected_model = "tts-1"

        logging.info(f"Requesting OpenAI TTS: voice='{selected_voice}', model='{selected_model}', text length (cleaned): {len(cleaned_text)}")

        response = client.audio.speech.create(
            model=selected_model,
            voice=selected_voice,
            input=cleaned_text,
            response_format="mp3"
        )

        audio_bytes = response.read()
        logging.info(f"OpenAI TTS audio generated successfully ({len(audio_bytes)} bytes).")
        return audio_bytes

    except ImportError:
        logging.error("OpenAI library not installed. Cannot generate audio.")
        return None
    except Exception as e:
        logging.error(f"Error generating OpenAI TTS audio: {e}", exc_info=True)
        return None

def get_csv_download_link(df, filename="permutation_results.csv"):
    """Generate a download link for a pandas dataframe as CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Results as CSV</a>'
    return href 