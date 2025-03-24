"""
API routes for the JLR RAG API.
"""
import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from pydantic import BaseModel

from app.agents.doc_rag.models.rag import ContextualRAG
from app.agents.doc_rag.models.vector_db import ContextualElasticVectorDB
from app.agents.doc_rag.config.settings import get_settings, Settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Singleton instances for services
vector_db = None
rag_service = None

def get_vector_db() -> ContextualElasticVectorDB:
    """Get or initialize the vector database."""
    global vector_db
    if vector_db is None:
        settings = get_settings()
        try:
            vector_db = ContextualElasticVectorDB(settings.ELASTIC_INDEX_NAME)
            logger.info(f"Initialized vector database with index: {settings.ELASTIC_INDEX_NAME}")
        except Exception as e:
            logger.error(f"Error initializing vector database: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error initializing vector database: {str(e)}")
    return vector_db

def get_rag_service() -> ContextualRAG:
    """Get or initialize the RAG service."""
    global rag_service
    if rag_service is None:
        try:
            db = get_vector_db()
            rag_service = ContextualRAG(db)
            logger.info("Initialized RAG service")
        except Exception as e:
            logger.error(f"Error initializing RAG service: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error initializing RAG service: {str(e)}")
    return rag_service

# Request/Response models
class QueryRequest(BaseModel):
    """Query request model."""
    query: str
    search_params: Optional[Dict[str, Any]] = None
    
class QueryResponse(BaseModel):
    """Query response model."""
    answer: str
    sources: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    
class FeedbackRequest(BaseModel):
    """Feedback request model."""
    query: str
    response: str
    feedback_type: str  # positive or negative
    improvement: Optional[str] = None
    
class ProcessDocumentRequest(BaseModel):
    """Document processing request model."""
    bucket_path: str
    parallel_threads: int = 4
    
class DocumentStatsResponse(BaseModel):
    """Document statistics response model."""
    total_chunks: int
    unique_documents: int
    documents: List[Dict[str, Any]]
    folders: List[Dict[str, Any]]

@router.post("/query", response_model=QueryResponse)
async def query_rag(
    request: QueryRequest,
    rag: ContextualRAG = Depends(get_rag_service),
    settings: Settings = Depends(get_settings)
):
    """
    Query the RAG system.
    
    Args:
        request: Query request with search parameters.
        
    Returns:
        Answer and source documents.
    """
    try:
        logger.info(f"Processing query: {request.query}")
        
        # Process the query
        answer, sources = rag.query(request.query)
        
        logger.info(f"Query processed, answer generated with {len(sources)} sources")
        
        # Format sources for response
        formatted_sources = []
        for source in sources:
            formatted_sources.append({
                "content": source["content"],
                "doc_id": source["metadata"]["doc_id"],
                "page_number": source["metadata"]["page_number"],
                "score": source.get("score", 0)
            })
        
        # Create metadata
        metadata = {
            "total_sources": len(sources),
            "sources_used": min(5, len(sources))
        }
        
        response = QueryResponse(
            answer=answer,
            sources=formatted_sources[:5],  # Limit to 5 sources in response
            metadata=metadata
        )
        
        logger.info("Successfully generated query response")
        return response
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@router.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    rag: ContextualRAG = Depends(get_rag_service)
):
    """
    Submit feedback for a query-response pair.
    
    Args:
        request: Feedback information.
        
    Returns:
        Confirmation message.
    """
    try:
        count = rag.log_feedback(
            request.query,
            request.response,
            request.feedback_type,
            request.improvement
        )
        return {"message": f"Feedback recorded successfully. Total feedback entries: {count}"}
    except Exception as e:
        logger.error(f"Error recording feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error recording feedback: {str(e)}")

@router.post("/documents/process")
async def process_documents(
    request: ProcessDocumentRequest,
    background_tasks: BackgroundTasks,
    db: ContextualElasticVectorDB = Depends(get_vector_db)
):
    """
    Process documents from a GCS bucket.
    
    Args:
        request: Document processing request.
        background_tasks: FastAPI background tasks.
        
    Returns:
        Job ID and confirmation message.
    """
    try:
        logger.info(f"Starting document processing for {request.bucket_path}")
        
        # Start processing in the background
        background_tasks.add_task(
            db.process_pdfs,
            request.bucket_path,
            request.parallel_threads
        )
        
        return {
            "message": f"Document processing started for {request.bucket_path}.",
            "status": "processing"
        }
    except Exception as e:
        logger.error(f"Error starting document processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error starting document processing: {str(e)}")

@router.get("/documents/stats", response_model=DocumentStatsResponse)
async def get_document_statistics(
    db: ContextualElasticVectorDB = Depends(get_vector_db)
):
    """
    Get statistics about indexed documents.
    
    Returns:
        Document statistics.
    """
    try:
        logger.info("Fetching document statistics")
        stats = db.get_document_stats()
        logger.info(f"Document statistics fetched: {stats['total_chunks']} chunks, {stats['unique_documents']} documents")
        return stats
    except Exception as e:
        logger.error(f"Error getting document statistics: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting document statistics: {str(e)}")

@router.get("/search/analytics")
async def get_search_analytics(
    limit: int = Query(10, ge=1, le=100),
    db: ContextualElasticVectorDB = Depends(get_vector_db)
):
    """
    Get analytics about search performance.
    
    Args:
        limit: Number of recent queries to include.
        
    Returns:
        Search analytics.
    """
    try:
        analytics = db.get_search_analytics()
        
        # Limit recent queries to the requested amount
        if "recent_queries" in analytics and len(analytics["recent_queries"]) > limit:
            analytics["recent_queries"] = analytics["recent_queries"][-limit:]
            
        return analytics
    except Exception as e:
        logger.error(f"Error getting search analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting search analytics: {str(e)}")

@router.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status.
    """
    return {"status": "healthy", "service": "jlr-rag-api"}

# Add a test endpoint to verify Elasticsearch connection
@router.get("/test_elastic")
async def test_elastic_connection(
    db: ContextualElasticVectorDB = Depends(get_vector_db)
):
    """
    Test Elasticsearch connection.
    
    Returns:
        Connection status.
    """
    try:
        indices = db.es_client.indices.get(index=[db.index_name, db.bm25_index])
        
        return {
            "status": "connected",
            "indices": list(indices.keys()),
            "vector_index": db.index_name,
            "bm25_index": db.bm25_index,
            "ping": db.es_client.ping()
        }
    except Exception as e:
        logger.error(f"Error testing Elasticsearch connection: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error testing Elasticsearch connection: {str(e)}")