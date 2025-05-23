from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class SubjectConfig:
    chunk_size: int
    chunk_overlap: int
    similarity_threshold: float
    max_tokens: int
    temperature: float
    system_prompt: str
    top_k: int = 4  # Default value for top_k
    hybrid_alpha: float = 0.5  # Default value for hybrid_alpha

# Default configurations for different subjects
SUBJECT_CONFIGS = {
    "mathematics": SubjectConfig(
        chunk_size=500,  # tokens
        chunk_overlap=100,  # tokens
        similarity_threshold=0.7,
        max_tokens=2000,
        temperature=0.3,
        system_prompt="You are a mathematics tutor. Explain concepts clearly with step-by-step solutions.",
        top_k=4,
        hybrid_alpha=0.6
    ),
    "physics": SubjectConfig(
        chunk_size=600,  # tokens
        chunk_overlap=150,  # tokens
        similarity_threshold=0.75,
        max_tokens=2500,
        temperature=0.4,
        system_prompt="You are a physics tutor. Focus on explaining physical concepts and their real-world applications.",
        top_k=5,
        hybrid_alpha=0.7
    ),
    "chemistry": SubjectConfig(
        chunk_size=500,  # tokens
        chunk_overlap=100,  # tokens
        similarity_threshold=0.7,
        max_tokens=2000,
        temperature=0.3,
        system_prompt="You are a chemistry tutor. Explain chemical concepts with clear examples and safety considerations.",
        top_k=4,
        hybrid_alpha=0.5
    ),
    "biology": SubjectConfig(
        chunk_size=750,  # tokens
        chunk_overlap=150,  # tokens
        similarity_threshold=0.8,
        max_tokens=3000,
        temperature=0.4,
        system_prompt="You are a biology tutor. Focus on explaining biological processes and their significance.",
        top_k=6,
        hybrid_alpha=0.8
    ),
    "computer_science": SubjectConfig(
        chunk_size=400,  # tokens
        chunk_overlap=75,  # tokens
        similarity_threshold=0.65,
        max_tokens=1500,
        temperature=0.2,
        system_prompt="You are a computer science tutor. Explain programming concepts with practical examples.",
        top_k=3,
        hybrid_alpha=0.4
    ),
    "history": SubjectConfig(
        chunk_size=1000,  # tokens
        chunk_overlap=200,  # tokens
        similarity_threshold=0.85,
        max_tokens=4000,
        temperature=0.5,
        system_prompt="You are a history tutor. Focus on explaining historical events and their context.",
        top_k=8,
        hybrid_alpha=0.9
    ),
    "literature": SubjectConfig(
        chunk_size=750,  # tokens
        chunk_overlap=150,  # tokens
        similarity_threshold=0.8,
        max_tokens=3000,
        temperature=0.6,
        system_prompt="You are a literature tutor. Focus on analyzing texts and explaining literary concepts.",
        top_k=7,
        hybrid_alpha=0.85
    ),
    "economics": SubjectConfig(
        chunk_size=600,  # tokens
        chunk_overlap=125,  # tokens
        similarity_threshold=0.75,
        max_tokens=2500,
        temperature=0.4,
        system_prompt="You are an economics tutor. Explain economic concepts with real-world examples.",
        top_k=5,
        hybrid_alpha=0.6
    ),
    "psychology": SubjectConfig(
        chunk_size=750,  # tokens
        chunk_overlap=150,  # tokens
        similarity_threshold=0.8,
        max_tokens=3000,
        temperature=0.5,
        system_prompt="You are a psychology tutor. Focus on explaining psychological concepts and theories.",
        top_k=6,
        hybrid_alpha=0.75
    ),
    "general": SubjectConfig(
        chunk_size=500,  # tokens
        chunk_overlap=100,  # tokens
        similarity_threshold=0.7,
        max_tokens=2000,
        temperature=0.4,
        system_prompt="You are a general tutor. Explain concepts clearly and provide helpful examples.",
        top_k=4,
        hybrid_alpha=0.5
    )
}

def get_subject_config(subject: str) -> SubjectConfig:
    """Get the configuration for a specific subject"""
    return SUBJECT_CONFIGS.get(subject.lower(), SUBJECT_CONFIGS["general"]) 