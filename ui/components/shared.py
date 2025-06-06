import streamlit as st
import logging
from typing import List, Dict, Any

def display_message_with_audio(message: Dict[str, Any], show_contexts: bool = False):
    """Helper function to display a message with audio and contexts."""
    if message["role"] == "assistant":
        response_text = message.get("content")
        audio_data = message.get("audio") 
        contexts = message.get("contexts", [])
        elapsed_time = message.get("elapsed_time")

        if response_text:
            tab_labels = ["📖 Read Response", "🔊 Hear Response"]
            try:
                tab_text, tab_audio = st.tabs(tab_labels)
            except Exception as e:
                logging.error(f"Error creating tabs: {e}")
                st.write(response_text) 
                if audio_data: st.audio(audio_data, format="audio/mp3")
                tab_text = None 

            if tab_text: 
                with tab_text:
                    st.write(response_text)

                with tab_audio:
                    if audio_data:
                        st.audio(audio_data, format="audio/mp3")
                    else:
                        st.info("Audio playback is not available for this message.")

            if elapsed_time is not None:
                st.write(f"_(JEFF cooked that up in {elapsed_time:.2f} seconds)_")

            if show_contexts and contexts:
                with st.expander("🧠 Check out the textbook bits I used:"):
                    for i, context in enumerate(contexts):
                        st.markdown(f"**Snippet {i+1}:**")
                        st.text(context)
        else: 
            st.write("*Assistant message content missing.*")
    else: 
        st.write(message["content"]) 