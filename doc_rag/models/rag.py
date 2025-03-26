"""
Contextual RAG model implementation with singleton pattern.
"""
import os
import logging
from typing import List, Dict, Any, Tuple, Optional
import google.generativeai as genai
from doc_rag.config.settings import get_settings
from doc_rag.utils.gcp import gcp_service

logger = logging.getLogger(__name__)

# Singleton instance
_instance = None

class ContextualRAG:
    def __new__(cls, vector_db=None):
        """
        Implement singleton pattern to ensure only one RAG instance exists.
        """
        global _instance
        if _instance is None:
            _instance = super(ContextualRAG, cls).__new__(cls)
            _instance._initialized = False
        return _instance
    
    def __init__(self, vector_db=None):
        """
        Initialize the RAG system (only runs once due to singleton pattern).
        
        Args:
            vector_db: Vector database instance.
        """
        # Skip initialization if already done
        if hasattr(self, '_initialized') and self._initialized:
            # If a new vector_db is provided, update it
            if vector_db is not None and self.vector_db != vector_db:
                self.vector_db = vector_db
            return
            
        # Store settings and API keys at class level
        settings = get_settings()
        self.google_api_key = settings.GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY")
        
        if not self.google_api_key:
            logger.error("Google API key not found in settings or environment variables")
            raise ValueError("Google API key not found")
        
        if vector_db is None:
            logger.error("Vector database cannot be None for initial initialization")
            raise ValueError("Vector database cannot be None")
            
        self.vector_db = vector_db
        
        try:
            # Configure genai with the stored API key
            genai.configure(api_key=self.google_api_key)
            
            # Use the standard GenerativeModel interface
            self.gemini = genai.GenerativeModel('gemini-2.0-flash')
            
            # Initialize feedback log
            self.feedback_log = []
            self.query_metrics_log = []
            
            # Mark as initialized
            self._initialized = True
            
            logger.info("ContextualRAG initialized successfully")
        except Exception as e:
            self._initialized = False
            logger.error(f"Error initializing ContextualRAG: {str(e)}")
            raise
    
    def detect_language(self, text: str) -> str:
        """
        Detect language using langdetect library
        Returns ISO 639-1 language code (e.g., 'en', 'es', 'fr', etc.)
        Defaults to 'en' if detection fails
        """
        try:
            from langdetect import detect, LangDetectException
            import re
            
            # Dictionary of language keywords and their corresponding codes
            language_keywords = {
                'german': 'de',
                'deutsch': 'de',
                'spanish': 'es',
                'español': 'es',
                'espanol': 'es',
                'french': 'fr',
                'français': 'fr',
                'francais': 'fr',
                'italian': 'it',
                'italiano': 'it',
                'portuguese': 'pt',
                'português': 'pt',
                'portugues': 'pt',
                'dutch': 'nl',
                'nederlands': 'nl',
                'english': 'en'
            }
            # Check for explicit language requests
            text_lower = text.lower()
            
            # Common patterns for language requests
            patterns = [
                r'(?:reply|respond|answer|write|say|tell|speak|give|provide|write back|communicate|translate).*(?:in|using|with)\s+(\w+)',
                r'(?:in|using|with)\s+(\w+).*(?:reply|respond|answer|write|say|tell|speak|give|provide|write back|communicate|translate)',
                r'(?:translate|convert).*(?:to|into)\s+(\w+)',
                r'(\w+)\s+(?:translation|version|language)',
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                for match in matches:
                    match_lower = match.lower()
                    if match_lower in language_keywords:
                        return language_keywords[match_lower]
            
            # If no explicit request found, use langdetect
            try:
                return detect(text)
            except LangDetectException:
                return 'en'
        except Exception as e:
            logger.error(f"Error detecting language: {str(e)}")
            return 'en'
    
    def get_language_instruction(self, lang_code: str) -> str:
        """
        Returns the language instruction for Gemini based on the detected language code
        """
        language_map = {
            'es': 'You must respond in Spanish. Format your entire response in Spanish.',
            'fr': 'You must respond in French. Format your entire response in French.',
            'de': 'You must respond in German. Format your entire response in German.',
            'it': 'You must respond in Italian. Format your entire response in Italian.',
            'pt': 'You must respond in Portuguese. Format your entire response in Portuguese.',
            'nl': 'You must respond in Dutch. Format your entire response in Dutch.',
            'en': 'You must respond in English. Format your entire response in English.'
        }
        return language_map.get(lang_code, 'You must respond in English. Format your entire response in English.')
    
    def query(self, question: str) -> Tuple[str, List[Dict]]:
        """Query the RAG system with improved analytics"""
        # Ensure we're initialized
        if not self._initialized:
            error_msg = "RAG service not properly initialized"
            logger.error(error_msg)
            return error_msg, []
            
        # Start timing
        import time
        from datetime import datetime
        
        start_time = time.time()
        
        # Track query metrics
        query_metrics = {
            "query": question,
            "timestamp": datetime.now().isoformat(),
            "retrieval_time": 0,
            "llm_time": 0,
            "total_time": 0,
            "num_results": 0,
            "sources": []
        }
        
        # Detect the language of the question
        lang_code = self.detect_language(question)
        language_instruction = self.get_language_instruction(lang_code)
        
        # Get relevant documents with context - use query expansion by default
        retrieval_start = time.time()
        try:
            initial_results = self.vector_db.search(question, k=5, use_expansion=True)
            query_metrics["retrieval_time"] = time.time() - retrieval_start
            query_metrics["num_results"] = len(initial_results)
        except Exception as e:
            logger.error(f"Error retrieving results: {str(e)}")
            return f"Error retrieving search results: {str(e)}", []
        
        # Track sources
        for result in initial_results:
            query_metrics["sources"].append({
                "doc_id": result['metadata']['doc_id'],
                "page": result['metadata']['page_number'],
                "score": result['score']
            })
        
        # Check if the initial results mention specific pages that might contain the answer
        mentioned_pages = []
        for result in initial_results:
            content = result['content'].lower()
            context = result['context'].lower()
            combined_text = content + " " + context
            
            # Look for references to other pages in the text
            import re
            page_references = re.findall(r'page\s+(\d+)', combined_text)
            for page_ref in page_references:
                try:
                    page_num = int(page_ref)
                    doc_id = result['metadata']['doc_id']
                    mentioned_pages.append((doc_id, page_num))
                except ValueError:
                    continue
        
        # If specific pages are mentioned, fetch those pages as well
        additional_results = []
        if mentioned_pages:
            for doc_id, page_num in mentioned_pages:
                try:
                    # Search for content from the specific page
                    page_results = self.vector_db._hybrid_search(f"document:{doc_id} page:{page_num}", k=2)
                    additional_results.extend(page_results)
                    
                    # Add to query metrics
                    query_metrics["sources"].append({
                        "doc_id": doc_id,
                        "page": page_num,
                        "source": "page_reference"
                    })
                except Exception as e:
                    logger.error(f"Error retrieving additional page {page_num} from {doc_id}: {str(e)}")
        
        # Combine initial and additional results, removing duplicates
        all_results = initial_results.copy()
        seen_ids = {(r['metadata']['doc_id'], r['metadata']['page_number'], r['metadata']['paragraph_index']) 
                    for r in initial_results}
        
        for result in additional_results:
            result_id = (result['metadata']['doc_id'], result['metadata']['page_number'], result['metadata']['paragraph_index'])
            if result_id not in seen_ids:
                all_results.append(result)
                seen_ids.add(result_id)
        
        # Update total results count
        query_metrics["num_results_with_references"] = len(all_results)
        
        # Prepare context from retrieved documents
        contexts = []
        for result in all_results:
            contexts.append(f"""Document: {result['metadata']['doc_id']}, Page {result['metadata']['page_number']}
            
Content:
{result['content']}

Context:
{result['context']}""")
        
        full_context = "\n\n---\n\n".join(contexts)
        
        # Modified prompt to be more direct about using found information and emphasizing factual accuracy
        prompt = f"""Based on the following documents, provide a direct and accurate answer.
        If you find specific information in the documents (like names, email addresses, dates, numbers, or other facts), 
        state them explicitly in your response.
        
        IMPORTANT: If information appears in the documents, use it exactly as stated - do not second guess, 
        modify, or deny information that is clearly present in the documents.
        
        If you cannot find the specific answer in the provided context, please say so clearly and provide 
        a recommendation based on what information is available.
        
        Format your response in a clear, readable way using markdown formatting.
        Use bullet points and sections where appropriate, but also be descriptive. 
        Highlight key points and important terms.
        
        {full_context}

        Question: {question}

        Important: 
        1. {language_instruction}
        2. If you do not follow this instruction exactly, you are making an error.
        3. Do NOT use any other language under any circumstances.
        4. If you see specific information in the documents above, include it in your response.
        5. If you're not sure about something, you can say so, but don't deny information that is clearly present in the documents.
        6. Be precise with numbers, dates, and facts exactly as they appear in the documents.
        """
        
        # Generate response
        llm_start = time.time()
        
        try:
            # First, check if our previously initialized model is available
            if not hasattr(self, 'gemini') or self.gemini is None:
                # Re-initialize if needed
                genai.configure(api_key=self.google_api_key)
                self.gemini = genai.GenerativeModel('gemini-2.0-flash')
                
            # Now generate the response
            response = self.gemini.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    top_p=0.95,
                    top_k=40
                )
            )
            answer_text = response.text.strip()
        except Exception as e:
            logger.error(f"Error generating content with Gemini: {str(e)}")
            answer_text = f"I apologize, but I encountered an error processing your request. Error details: {str(e)}"
            
        query_metrics["llm_time"] = time.time() - llm_start
        
        # Calculate total time
        query_metrics["total_time"] = time.time() - start_time
        
        # Log metrics for analysis
        self._log_query_metrics(query_metrics)
        
        return answer_text, all_results
    
    def _log_query_metrics(self, metrics: Dict[str, Any]):
        """Log query metrics for analysis"""
        try:
            # Append to in-memory log
            if not hasattr(self, 'query_metrics_log'):
                self.query_metrics_log = []
            self.query_metrics_log.append(metrics)
            
            # Could also write to a file or database
            logger.info(f"Query processed in {metrics['total_time']:.2f}s (retrieval: {metrics['retrieval_time']:.2f}s, LLM: {metrics['llm_time']:.2f}s)")
        except Exception as e:
            logger.error(f"Error logging metrics: {str(e)}")
    
    def log_feedback(self, query: str, response: str, feedback: str, improvement: str = None):
        """Log user feedback for continuous improvement"""
        from datetime import datetime
        
        feedback_entry = {
            "query": query,
            "response": response,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat()
        }
        
        if improvement:
            feedback_entry["improvement"] = improvement
            
        self.feedback_log.append(feedback_entry)
        
        # Could also write to a file or database
        return len(self.feedback_log)

# Create a function to get the singleton instance
def get_rag_service(vector_db=None) -> ContextualRAG:
    """
    Get the singleton instance of ContextualRAG.
    
    Args:
        vector_db: Optional vector database to use or update
        
    Returns:
        Initialized ContextualRAG instance
    """
    return ContextualRAG(vector_db)