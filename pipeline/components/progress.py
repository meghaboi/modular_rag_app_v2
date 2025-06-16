from abc import ABC, abstractmethod
import streamlit as st
import time

class ProgressReporter(ABC):
    """Abstract base class for progress reporting."""
    
    @abstractmethod
    def initialize(self, total_steps: int, initial_message: str = "") -> None:
        """Initialize progress reporting with total number of steps."""
        pass
    
    @abstractmethod
    def update(self, current_step: int, message: str) -> None:
        """Update progress with current step and message."""
        pass
    
    @abstractmethod
    def complete(self, final_message: str = "Completed!") -> None:
        """Mark progress as complete."""
        pass

    @abstractmethod
    def close(self, delay: int = 2) -> None:
        """Close and remove the progress indicator after a delay."""
        pass

class StreamlitProgressReporter(ProgressReporter):
    """Streamlit implementation of progress reporting."""
    
    def __init__(self):
        self._progress_bar = None
        self._total_steps = 1
    
    def initialize(self, total_steps: int, initial_message: str = "Initializing...") -> None:
        """Initialize Streamlit progress bar."""
        self._total_steps = total_steps if total_steps > 0 else 1
        self._progress_bar = st.progress(0, text=initial_message)
    
    def update(self, current_step: int, message: str) -> None:
        """Update Streamlit progress bar and display message."""
        if self._progress_bar:
            progress_percentage = min(1.0, current_step / self._total_steps)
            self._progress_bar.progress(progress_percentage, text=message)
    
    def complete(self, final_message: str = "Completed!") -> None:
        """Set progress to 100% and show final message."""
        if self._progress_bar:
            self._progress_bar.progress(1.0, text=final_message)
    
    def close(self, delay: int = 2) -> None:
        """Close and remove the progress indicator after a delay."""
        if self._progress_bar:
            time.sleep(delay)
            self._progress_bar.empty()
            self._progress_bar = None