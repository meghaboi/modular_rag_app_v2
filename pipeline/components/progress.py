from abc import ABC, abstractmethod
import streamlit as st

class ProgressReporter(ABC):
    """Abstract base class for progress reporting"""
    
    @abstractmethod
    def initialize(self, total_steps: int) -> None:
        """Initialize progress reporting with total number of steps"""
        pass
    
    @abstractmethod
    def update(self, current_step: int, message: str) -> None:
        """Update progress with current step and message"""
        pass
    
    @abstractmethod
    def complete(self) -> None:
        """Mark progress as complete"""
        pass

class StreamlitProgressReporter(ProgressReporter):
    """Streamlit implementation of progress reporting"""
    
    def __init__(self):
        self._progress_bar = None
    
    def initialize(self, total_steps: int) -> None:
        """Initialize Streamlit progress bar"""
        self._progress_bar = st.progress(0)
    
    def update(self, current_step: int, message: str) -> None:
        """Update Streamlit progress bar and display message"""
        if self._progress_bar:
            self._progress_bar.progress(current_step)
            st.write(message)
    
    def complete(self) -> None:
        """Complete progress reporting"""
        if self._progress_bar:
            self._progress_bar.progress(1.0) 