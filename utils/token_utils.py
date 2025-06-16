from typing import List, Union, Optional
import tiktoken

class TokenCounter:
    """Utility class for counting tokens in text"""
    
    def __init__(self, model_name: str = "gpt-3.5"):
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

class TokenCostManager:
    # Pricing per 1000 tokens (input, output)
    # These are example prices and should be verified and updated with actuals.
    # Model names listed here should correspond to the output of `llm.get_model_name()`
    # after normalization (lower(), replace(' ', '-')).
    PRICING_DATA = {
        # OpenAI - Prices per 1M tokens, converted to per 1K tokens
        "gpt-4": {"input": 0.03, "output": 0.06},  # $30/1M input, $60/1M output
        "gpt-4-32k": {"input": 0.06, "output": 0.12}, # $60/1M input, $120/1M output
        "gpt-3.5": {"input": 0.0005, "output": 0.0015}, # $0.50/1M input, $1.50/1M output (e.g. gpt-3.5-turbo-0125)
        "gpt-3.5-16k": {"input": 0.003, "output": 0.004}, # Older model, may be covered by gpt-3.5-turbo with startswith
        
        # Anthropic - Prices per 1M tokens, converted to per 1K tokens
        "claude-instant-1.2": {"input": 0.0008, "output": 0.0024}, 
        "claude-2": {"input": 0.008, "output": 0.024}, 
        "claude-2.1": {"input": 0.008, "output": 0.024}, 
        "claude-3-opus": {"input": 0.015, "output": 0.075}, 
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        "claude-3-7-sonnet": {"input": 0.003, "output": 0.015},
        "claude-4-sonnet": {"input": 0.003, "output": 0.015},
        
        # Google
        # Gemini 1.5 Flash: $0.35/1M input, $0.70/1M output (for <128K context) -> 0.00035 / 0.00070 per 1K
        "gemini-1.5-flash": {"input": 0.00035, "output": 0.00070}, # Used by factory
        # Gemini 1.5 Pro: $3.50 per 1M tokens input, $10.50 per 1M tokens output (for <=128K context)
        "gemini-1.5-pro-latest": {"input": 0.0035, "output": 0.0105}, 
        # Older "gemini" entry, pricing might be for early gemini-pro or other variant. Keeping for now.
        "gemini": {"input": 0.000125, "output": 0.000375}, 

        # Mistral AI - Platform prices per 1M tokens (EUR converted to USD approx)
        # open-mistral-7b (Mistral Tiny): ~€0.23/1M in, ~€0.23/1M out => ~$0.00025/1k
        # mistral-small-2402 (Mixtral 8x7B): ~€0.69/1M in, ~€2.08/1M out => ~$0.00075/$0.00227 per 1k
        # mistral-medium-2312: ~€2.52/1M in, ~€7.57/1M out => ~$0.00275/$0.00825 per 1k
        # mistral-large-2402: ~€7.39/1M in, ~€22.18/1M out => ~$0.008/$0.024 per 1k
        "open-mistral-7b": {"input": 0.00025, "output": 0.00025},
        "mistral-small-latest": {"input": 0.00075, "output": 0.00227}, # Mapping to mistral-small-2402
        "mistral-medium-latest": {"input": 0.00275, "output": 0.00825},
        "mistral-large-latest": {"input": 0.008, "output": 0.024},

        # LLAMA (often self-hosted, but if using a paid API, e.g. Replicate, Fireworks)
        # Example for Llama-2-70b-chat on Fireworks: $0.90/1M tokens combined.
        # For simplicity, splitting it, but actual pricing might vary.
        "llama-2-70b-chat": {"input": 0.00045, "output": 0.00045}
    }

    @staticmethod
    def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> Optional[float]:
        if not model_name:
            return 0.0

        normalized_model_name = model_name.lower().replace(' ', '-')
        
        model_pricing = TokenCostManager.PRICING_DATA.get(normalized_model_name)
        
        if not model_pricing:
            # Attempt partial matching for versions like 'gpt-3.5-turbo-0125' from 'gpt-3.5-turbo'
            for key_prefix, pricing_info in TokenCostManager.PRICING_DATA.items():
                if normalized_model_name.startswith(key_prefix):
                    model_pricing = pricing_info
                    # print(f"DEBUG: Partial match found: '{normalized_model_name}' matched with '{key_prefix}'")
                    break
            if not model_pricing:
                print(f"Warning: Pricing not found for model '{model_name}' (normalized: '{normalized_model_name}'). Cost will be 0.")
                return 0.0

        input_cost = (input_tokens / 1000.0) * model_pricing["input"]
        output_cost = (output_tokens / 1000.0) * model_pricing["output"]
        return input_cost + output_cost