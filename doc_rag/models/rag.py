"""
Retrieval Augmented Generation (RAG) implementation.
"""
import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import google.generativeai as genai
import cohere
from langdetect import detect, LangDetectException

from app.agents.doc_rag.models.vector_db import ContextualElasticVectorDB
from app.agents.doc_rag.utils.documents import detect_language, get_language_instruction
from app.agents.doc_rag.utils.gcp import gcp_service
from app.agents.doc_rag.config.settings import get_settings

logger = logging.getLogger(__name__)

class ContextualRAG:
    """Retrieval Augmented Generation with contextual information."""
    
    def __init__(self, vector_db: Optional[ContextualElasticVectorDB] = None):
        """
        Initialize the RAG system.
        
        Args:
            vector_db: Vector database instance. If None, creates a new one.
        """
        settings = get_settings()
        
        if vector_db is None:
            self.vector_db = ContextualElasticVectorDB()
        else:
            self.vector_db = vector_db
        
        # Initialize Gemini
        gcp_service.initialize()
        self.gemini = gcp_service.gemini_model
        
        # Initialize Cohere for reranking if API key is available
        self.cohere_client = None
        if settings.COHERE_API_KEY:
            try:
                self.cohere_client = cohere.Client(settings.COHERE_API_KEY)
                logger.info("Cohere client initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing Cohere client: {str(e)}")
        
        # Initialize feedback log
        self.feedback_log = []
        self.query_metrics_log = []
    
    def query(self, question: str) -> Tuple[str, List[Dict]]:
        """
        Query the RAG system.
        
        Args:
            question: User's question.
            
        Returns:
            Tuple of (response text, list of sources).
        """
        # Start timing
        start_time = datetime.now()
        
        # Track query metrics
        query_metrics = {
            "query": question,
            "timestamp": start_time.isoformat(),
            "retrieval_time": 0,
            "llm_time": 0,
            "total_time": 0,
            "num_results": 0,
            "sources": []
        }
        
        # Detect the language of the question
        lang_code = detect_language(question)
        language_instruction = get_language_instruction(lang_code)
        
        # Get relevant documents with context - use query expansion by default
        retrieval_start = datetime.now()
        initial_results = self.vector_db.search(question, k=5, use_expansion=True)
        query_metrics["retrieval_time"] = (datetime.now() - retrieval_start).total_seconds()
        query_metrics["num_results"] = len(initial_results)
        
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
                # Search for content from the specific page
                page_results = self.vector_db._hybrid_search(f"document:{doc_id} page:{page_num}", k=2)
                additional_results.extend(page_results)
                
                # Add to query metrics
                query_metrics["sources"].append({
                    "doc_id": doc_id,
                    "page": page_num,
                    "source": "page_reference"
                })
        
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
        llm_start = datetime.now()
        try:
            response = self.gemini.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    top_p=0.95,
                    top_k=40
                )
            )
            answer = response.text.strip()
        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            answer = f"I apologize, but I encountered an error when trying to process your question. Please try again or rephrase your question."
        
        query_metrics["llm_time"] = (datetime.now() - llm_start).total_seconds()
        
        # Calculate total time
        query_metrics["total_time"] = (datetime.now() - start_time).total_seconds()
        
        # Log metrics for analysis
        self._log_query_metrics(query_metrics)
        
        return answer, all_results
    
    def _log_query_metrics(self, metrics: Dict[str, Any]):
        """Log query metrics for analysis."""
        try:
            # Append to in-memory log
            self.query_metrics_log.append(metrics)
            
            # Log performance metrics
            logger.info(f"Query processed in {metrics['total_time']:.2f}s (retrieval: {metrics['retrieval_time']:.2f}s, LLM: {metrics['llm_time']:.2f}s)")
        except Exception as e:
            logger.error(f"Error logging metrics: {str(e)}")
    
    def log_feedback(self, query: str, response: str, feedback: str, improvement: str = None) -> int:
        """
        Log user feedback for continuous improvement.
        
        Args:
            query: User's original query.
            response: System's response.
            feedback: User's feedback ("positive" or "negative").
            improvement: User's suggestion for improvement.
            
        Returns:
            Number of feedback entries.
        """
        feedback_entry = {
            "query": query,
            "response": response,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat()
        }
        
        if improvement:
            feedback_entry["improvement"] = improvement
            
        self.feedback_log.append(feedback_entry)
        logger.info(f"Feedback logged: {feedback} for query: {query[:50]}...")
        
        return len(self.feedback_log)