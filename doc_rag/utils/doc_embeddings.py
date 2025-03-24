"""
Embedding utilities for vector representation of text.
"""
from typing import List
import logging
from vertexai.language_models import TextEmbeddingModel
from google.cloud import aiplatform

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating text embeddings."""
    
    def __init__(self):
        """Initialize the embedding service."""
        self.embedding_model = None
    
    def initialize(self):
        """Initialize the embedding model."""
        try:
            aiplatform.init()
            self.embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")
            logger.info("Embedding model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {str(e)}")
            raise
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embeddings for the text with empty text handling.
        
        Args:
            text: The text to generate embeddings for.
            
        Returns:
            List of embedding values.
        """
        try:
            # Check for empty text
            if not text or not text.strip():
                logger.warning("Empty text provided for embedding")
                return [0.0] * 768
            
            if not self.embedding_model:
                self.initialize()
                
            embeddings = self.embedding_model.get_embeddings([text])
            return embeddings[0].values
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            return [0.0] * 768  # Return zero vector as fallback

# Create a singleton instance
embedding_service = EmbeddingService()