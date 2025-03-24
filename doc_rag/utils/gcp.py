"""
Google Cloud Platform utilities.
"""
import io
import os
import logging
from typing import Optional, Dict, Any, List, Tuple
from google.cloud import storage
from google.auth import default
import google.generativeai as genai
from app.agents.doc_rag.config.settings import get_settings

logger = logging.getLogger(__name__)

class GCPService:
    """Service for Google Cloud Platform interactions."""
    
    def __init__(self):
        """Initialize the GCP service."""
        self.storage_client = None
        self.gemini_model = None
        self._initialized = False
    
    def initialize(self):
        """Initialize GCP clients."""
        if self._initialized:
            return
            
        try:
            settings = get_settings()
            
            # Configure Gemini
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
            
            # Configure Storage
            credentials, project = default()
            self.storage_client = storage.Client(credentials=credentials, project=project)
            
            self._initialized = True
            logger.info("GCP Service initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing GCP service: {str(e)}")
            raise
    
    def generate_context(self, doc: str, chunk: str) -> str:
        """
        Generate contextual description using Gemini.
        
        Args:
            doc: The complete document text.
            chunk: The specific chunk to contextualize.
            
        Returns:
            Generated contextual description.
        """
        if not self._initialized:
            self.initialize()
            
        prompt = f"""
        Please analyze this complete document carefully:
        <document>
        {doc}
        </document>

        Now analyze this specific chunk from the document:
        <chunk>
        {chunk}
        </chunk>

        Create a detailed contextual description of this chunk that captures ALL of the following:
        1. The main topic or subject matter of this chunk
        2. Any key facts, figures, or data points mentioned
        3. How this chunk relates to the sections before and after it
        4. Any important definitions, concepts, or terminology introduced
        5. The role this chunk plays in the overall document's narrative or argument
        6. Any specific examples, case studies, or illustrations mentioned
        7. Technical details or specifications if present
        8. Any conclusions, implications, or recommendations made

        Format your response as a detailed but cohesive paragraph that incorporates all relevant information, but is also not too short.
        Focus on being comprehensive while maintaining clarity.
        Do not include any meta-commentary - just provide the context itself.
        """

        try:
            response = self.gemini_model.generate_content(prompt)
            
            if response and response.text:
                return response.text.strip() or "No context available"
            return "No context available"
        except Exception as e:
            logger.error(f"Error generating context: {str(e)}")
            return "No context available"
    
    def expand_query(self, query: str) -> str:
        """
        Expand the query to improve recall using Gemini.
        
        Args:
            query: The original query.
            
        Returns:
            Expanded query with related terms.
        """
        if not self._initialized:
            self.initialize()
            
        try:
            prompt = f"""
            I need to search for documents about: "{query}"
            
            Please provide 3-5 alternative ways to phrase this query or related terms that might appear in relevant documents.
            Format your response as a comma-separated list of terms or phrases only.
            """
            
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=100
                )
            )
            
            expanded_terms = response.text.strip().split(',')
            expanded_terms = [term.strip() for term in expanded_terms if term.strip()]
            
            expanded_query = query + " " + " ".join(expanded_terms)
            return expanded_query
        except Exception as e:
            logger.error(f"Error expanding query: {str(e)}")
            return query  # Fall back to original query
    
    def get_blob_content(self, bucket_name: str, blob_path: str) -> Optional[bytes]:
        """
        Get content of a blob from GCS.
        
        Args:
            bucket_name: Name of the GCS bucket.
            blob_path: Path to the blob within the bucket.
            
        Returns:
            Blob content as bytes or None if not found.
        """
        if not self._initialized:
            self.initialize()
            
        try:
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            
            if not blob.exists():
                logger.warning(f"Blob {blob_path} does not exist in bucket {bucket_name}")
                return None
                
            return blob.download_as_bytes()
        except Exception as e:
            logger.error(f"Error downloading blob {blob_path}: {str(e)}")
            return None

# Create a singleton instance
gcp_service = GCPService()