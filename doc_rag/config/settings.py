"""
Configuration settings for the JLR RAG API.
"""
import os
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings."""
   
    # API settings
    API_TITLE: str = "JLR Analysis RAG API"
    API_DESCRIPTION: str = "Retrieval Augmented Generation API for JLR Analysis"
    API_VERSION: str = "1.0.0"
    
    # Change this line:
    # Instead of evaluating the expression immediately, store it as a string
    DEBUG: bool = os.getenv("DEBUG", "True").lower() in ("true", "1", "t")
   
    # Elasticsearch settings
    ELASTIC_HOST: str = os.getenv("ELASTIC_HOST", "localhost:9200")
    ELASTIC_USER: str = os.getenv("ELASTIC_USER", "elastic")
    ELASTIC_PASSWORD: str = os.getenv("ELASTIC_PASSWORD", "")
    ELASTIC_INDEX_NAME: str = os.getenv("ELASTIC_INDEX_NAME", "pdf_docs_rag")
   
    OPENAI_API_KEY: str = ""
    LANGSMITH_TRACING: str = "false"
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "nl2sql"
    ANTHROPIC_API_KEY: str = ""
    VOYAGE_API_KEY: str = ""
    
    # Google Cloud settings
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GCS_BUCKET_NAME: str = os.getenv("GCS_BUCKET_NAME", "")
   
    # Cohere settings
    COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")
   
    # Logging settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
   
    class Config:
        env_file = ".env"
        case_sensitive = True
        
    # Add a model validator to convert string to bool
    @property
    def debug_bool(self) -> bool:
        """Convert DEBUG string to boolean."""
        return self.DEBUG.lower() in ("true", "1", "t")

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()