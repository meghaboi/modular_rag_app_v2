"""
Core package for the ModularRAG application.
Contains the main application logic and core functionality.
"""

from .session_state import initialize_session_state
from .auto_init import attempt_automatic_initialization

__all__ = ['initialize_session_state', 'attempt_automatic_initialization'] 