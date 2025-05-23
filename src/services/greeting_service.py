import random
from typing import Tuple

def is_greeting(query: str) -> Tuple[bool, str]:
    """Check if the query is a greeting"""
    greetings = {
        "hello": "greeting",
        "hi": "greeting",
        "hey": "greeting",
        "good morning": "greeting",
        "good afternoon": "greeting",
        "good evening": "greeting",
        "how are you": "greeting",
        "what's up": "greeting",
        "how's it going": "greeting"
    }
    
    query_lower = query.lower().strip()
    for greeting, greeting_type in greetings.items():
        if greeting in query_lower:
            return True, greeting_type
    return False, ""

def get_greeting_response() -> str:
    """Get a random greeting response"""
    responses = [
        "Hello! How can I help you today?",
        "Hi there! What would you like to know?",
        "Hey! I'm here to help. What's on your mind?",
        "Greetings! How can I assist you?",
        "Hello! I'm ready to help you learn. What would you like to explore?"
    ]
    return random.choice(responses) 