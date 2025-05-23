from typing import List, Union
import tiktoken

class TokenCounter:
    """Utility class for counting tokens in text"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """Initialize token counter with specified model"""
        try:
            # Try OpenAI's tokenizer first
            self.encoder = tiktoken.encoding_for_model(model_name)
        except:
            # Fallback to GPT-2 tokenizer
            self.encoder = tiktoken.get_encoding("gpt2")
    
    def count_tokens(self, text: Union[str, List[str]]) -> int:
        """Count tokens in text or list of texts"""
        if isinstance(text, list):
            return sum(self.count_tokens(t) for t in text)
        return len(self.encoder.encode(text))
    
    def truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to specified number of tokens"""
        tokens = self.encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.encoder.decode(tokens[:max_tokens])
    
    def split_into_chunks(self, text: str, chunk_size: int, overlap: int = 0) -> List[str]:
        """Split text into chunks of specified token size with overlap"""
        tokens = self.encoder.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            if chunk_tokens:
                chunks.append(self.encoder.decode(chunk_tokens))
            
            if i + chunk_size >= len(tokens):
                break
        
        return chunks 