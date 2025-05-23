import streamlit as st
import logging
from typing import Optional
from ..core.app import text_to_speech, is_greeting, get_greeting_response

def display_chat_interface():
    """Display the main chat interface"""
    st.title("Ask JEFF - Study Buddy")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("contexts"):
                with st.expander("View Contexts"):
                    for ctx in message["contexts"]:
                        st.markdown(f"**Source:** {ctx['source']}")
                        st.markdown(ctx['text'])
                        st.markdown("---")

    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        if not st.session_state.pipeline:
            st.error("Please upload a document and initialize the pipeline first.")
            return

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Check if it's a greeting
        is_greeting_msg, greeting_type = is_greeting(prompt)
        if is_greeting_msg:
            response = get_greeting_response()
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.write(response)
            return

        # Get response from pipeline
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.pipeline.query(prompt)
                st.write(response["answer"])
                
                # Display contexts if enabled
                if st.session_state.show_contexts and response.get("contexts"):
                    with st.expander("View Contexts"):
                        for ctx in response["contexts"]:
                            st.markdown(f"**Source:** {ctx['source']}")
                            st.markdown(ctx['text'])
                            st.markdown("---")
                
                # Add assistant message to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["answer"],
                    "contexts": response.get("contexts", [])
                })

                # Text-to-speech
                if st.button("🔊 Listen"):
                    audio_bytes = text_to_speech(response["answer"])
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3") 