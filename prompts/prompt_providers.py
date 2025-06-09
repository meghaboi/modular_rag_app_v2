from typing import Dict, Any, Optional
import string
from . import prompt_templates

class PromptError(Exception):
    """Base exception for prompt-related errors."""
    pass

class MissingTemplateError(PromptError):
    """Raised when a requested template is not found."""
    pass

class MissingVariableError(PromptError):
    """Raised when required variables are missing from the template."""
    pass

class BasePromptProvider:
    """Base class for all prompt providers."""
    
    def __init__(self):
        self.templates: Dict[str, str] = {}
        self._load_templates()
    
    def _load_templates(self):
        """Load templates from the templates module. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _load_templates")
    
    def get_prompt(self, template_name: str, **kwargs) -> str:
        """
        Get a prompt by filling in the template with the provided arguments.
        
        Args:
            template_name: Name of the template to use
            **kwargs: Variables to fill in the template
            
        Returns:
            str: The filled prompt
            
        Raises:
            MissingTemplateError: If the template doesn't exist
            MissingVariableError: If required variables are missing
        """
        if template_name not in self.templates:
            raise MissingTemplateError(f"Template '{template_name}' not found")
            
        template = self.templates[template_name]
        required_vars = self._get_required_variables(template)
        missing_vars = [var for var in required_vars if var not in kwargs]
        
        if missing_vars:
            raise MissingVariableError(
                f"Missing required variables for template '{template_name}': {', '.join(missing_vars)}"
            )
            
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise MissingVariableError(f"Missing variable in template: {str(e)}")
    
    def _get_required_variables(self, template: str) -> set:
        """Extract required variables from a template string."""
        formatter = string.Formatter()
        return {field_name for _, field_name, _, _ in formatter.parse(template) if field_name is not None}

class LLMPromptProvider(BasePromptProvider):
    """Provider for LLM-related prompts."""
    
    def _load_templates(self):
        self.templates = {
            'system': prompt_templates.JEFF_SYSTEM_PROMPT,
            'query': prompt_templates.RAG_QUERY_TEMPLATE,
            'chat': prompt_templates.RAG_CHAT_TEMPLATE,
        }

class SummarizerPromptProvider(BasePromptProvider):
    """Provider for summarizer-related prompts."""
    
    def _load_templates(self):
        self.templates = {
            'main_points': prompt_templates.MAIN_POINTS_EXTRACTION_TEMPLATE,
            'point_summary': prompt_templates.POINT_SUMMARIZATION_TEMPLATE,
            'document': prompt_templates.DOCUMENT_SUMMARY_TEMPLATE,
        }

class QueryAnalysisPromptProvider(BasePromptProvider):
    """Provider for query analysis prompts."""
    
    def _load_templates(self):
        self.templates = {
            'analysis': prompt_templates.QUERY_ANALYSIS_TEMPLATE,
            'nature_classification': prompt_templates.PROMPT_NATURE_CLASSIFICATION_TEMPLATE,
        }

class ErrorPromptProvider(BasePromptProvider):
    """Provider for error-related prompts."""
    
    def _load_templates(self):
        self.templates = {
            'error': prompt_templates.ERROR_RESPONSE_TEMPLATE,
        }

class UIPromptProvider(BasePromptProvider):
    """Provider for UI-related prompts."""
    
    def __init__(self):
        """Initialize the UI prompt provider."""
        self._templates = {
            'welcome': prompt_templates.WELCOME_MESSAGE_TEMPLATE,
            'warning': prompt_templates.WARNING_MESSAGE_TEMPLATE
        }

class RerankerPromptProvider(BasePromptProvider):
    """Provider for reranking-related prompts."""
    
    def __init__(self):
        """Initialize the reranker prompt provider."""
        self._templates = {
            'rerank': prompt_templates.RERANKING_TEMPLATE
        }
    
    def get_prompt(self, prompt_type: str, **kwargs) -> str:
        """Get a prompt with the given type and parameters."""
        if prompt_type == 'rerank':
            # Format documents with indices
            documents = []
            for i, doc in enumerate(kwargs.get('documents', [])):
                documents.append(f"Index {i}: {doc}")
            kwargs['documents'] = '\n'.join(documents)
        return super().get_prompt(prompt_type, **kwargs)

# Factory function to get the appropriate provider
def get_provider(provider_type: str) -> BasePromptProvider:
    """
    Factory function to get the appropriate prompt provider.
    
    Args:
        provider_type: Type of provider to get ('llm', 'summarizer', 'query', 'error', 'ui', 'reranker')
        
    Returns:
        BasePromptProvider: The requested provider instance
        
    Raises:
        ValueError: If an invalid provider type is specified
    """
    providers = {
        'llm': LLMPromptProvider,
        'summarizer': SummarizerPromptProvider,
        'query': QueryAnalysisPromptProvider,
        'error': ErrorPromptProvider,
        'ui': UIPromptProvider,
        'reranker': RerankerPromptProvider
    }
    
    if provider_type not in providers:
        raise ValueError(f"Invalid provider type: {provider_type}")
        
    return providers[provider_type]() 