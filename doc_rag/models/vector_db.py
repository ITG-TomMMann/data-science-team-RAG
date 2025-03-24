"""
Elasticsearch vector database for document retrieval.
"""
import os
import io
import re
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from elasticsearch import Elasticsearch, helpers
import fitz
from tqdm import tqdm
import sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from app.agents.doc_rag.utils.doc_embeddings import embedding_service
from app.agents.doc_rag.utils.gcp import gcp_service  # Use the class, not the instance
from app.agents.doc_rag.utils.documents import extract_text_from_pdf, split_text_into_paragraphs
from app.agents.doc_rag.config.settings import get_settings

logger = logging.getLogger(__name__)

class ContextualElasticVectorDB:
    """Elasticsearch vector database with contextual retrieval capabilities."""
    
    def __init__(self, index_name: Optional[str] = None):
        """
        Initialize the vector database.
        
        Args:
            index_name: Name of the Elasticsearch index to use.
        """
        settings = get_settings()
        
        if index_name is None:
            index_name = settings.ELASTIC_INDEX_NAME
            
        self.index_name = index_name
        self.bm25_index = f"{index_name}_bm25"
        
        # Initialize Elasticsearch client
        elastic_host = settings.ELASTIC_HOST
        elastic_user = settings.ELASTIC_USER
        elastic_password = settings.ELASTIC_PASSWORD
        
        logger.info(f"Connecting to Elasticsearch at {elastic_host}")
        
        try:
            if not elastic_host.startswith(('http://', 'https://')):
                elastic_host = f"https://{elastic_host}"

            self.es_client = Elasticsearch(
                hosts=[elastic_host],
                basic_auth=(elastic_user, elastic_password),
                retry_on_timeout=True,
                max_retries=3,
                request_timeout=30
            )
            
            # Test connection
            if not self.es_client.ping():
                raise ConnectionError("Failed to connect to Elasticsearch")
            
            logger.info("Successfully connected to Elasticsearch")
        except Exception as e:
            logger.error(f"Error connecting to Elasticsearch: {str(e)}")
            raise
        
        # Create indices if they don't exist
        if not self.es_client.indices.exists(index=self.index_name):
            self._create_vector_index()
        if not self.es_client.indices.exists(index=self.bm25_index):
            self._create_bm25_index()
            
        # Initialize query metrics logging
        self.query_metrics = []
    
    def _create_vector_index(self):
        """Create Elasticsearch index with vector search capabilities."""
        if not self.es_client.indices.exists(index=self.index_name):
            mapping = {
                "mappings": {
                    "properties": {
                        "content": {"type": "text"},
                        "context": {"type": "text"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": 768,  # 768 is the dimension of the embedding model
                            "index": True, 
                            "similarity": "cosine"
                        },
                        "doc_id": {"type": "keyword"},
                        "page_number": {"type": "integer"},
                        "paragraph_index": {"type": "integer"},
                        "folder_path": {"type": "keyword"}
                    }
                }
            }
            self.es_client.indices.create(index=self.index_name, body=mapping)
            logger.info(f"Created vector index: {self.index_name}")

    def _create_bm25_index(self):
        """Create separate BM25 index for text search."""
        if not self.es_client.indices.exists(index=self.bm25_index):
            mapping = {
                "settings": {
                    "analysis": {"analyzer": {"default": {"type": "english"}}},
                    "similarity": {"default": {"type": "BM25"}}
                },
                "mappings": {
                    "properties": {
                        "content": {"type": "text", "analyzer": "english"},
                        "context": {"type": "text", "analyzer": "english"},
                        "doc_id": {"type": "keyword"},
                        "page_number": {"type": "integer"},
                        "paragraph_index": {"type": "integer"},
                        "folder_path": {"type": "keyword"}
                    }
                }
            }
            self.es_client.indices.create(index=self.bm25_index, body=mapping)
            logger.info(f"Created BM25 index: {self.bm25_index}")

    def process_pdfs(self, bucket_path: str, parallel_threads: int = 4) -> Dict[str, Any]:
        """
        Process PDFs from Google Cloud Storage bucket and load into indices.
        
        Args:
            bucket_path: GCS path (gs://bucket-name/path).
            parallel_threads: Number of threads for parallel processing.
            
        Returns:
            Result summary with processed, failed and skipped documents.
        """
        try:
            # Parse bucket path
            if not bucket_path.startswith("gs://"):
                raise ValueError("Bucket path must start with gs://")
            
            logger.info(f"Processing bucket path: {bucket_path}")
            
            parts = bucket_path.split("/")
            if len(parts) < 3:
                raise ValueError("Invalid bucket path format")
            
            bucket_name = parts[2]
            prefix = "/".join(parts[3:]) if len(parts) > 3 else ""
            
            logger.info(f"Bucket name: {bucket_name}")
            logger.info(f"Prefix: {prefix}")
            
            # Use GCP storage client
            gcp_service.initialize()
            bucket = gcp_service.storage_client.bucket(bucket_name)
            if not bucket:
                raise ValueError(f"Could not access bucket: {bucket_name}")
            
            # List all PDF blobs recursively in the bucket with the given prefix
            logger.info("Listing blobs...")
            blobs = list(bucket.list_blobs(prefix=prefix))
            logger.info(f"Found {len(blobs)} total blobs")
            
            pdf_blobs = [blob for blob in blobs if blob.name.lower().endswith('.pdf')]
            logger.info(f"Found {len(pdf_blobs)} PDF files")
            
            existing_docs = set()
            failed_docs = []
            processed_docs = []
            
            # Get existing documents
            results = self.es_client.search(
                index=self.index_name,
                body={
                    "size": 0,
                    "aggs": {
                        "unique_docs": {
                            "terms": {
                                "field": "doc_id",
                                "size": 1000
                            }
                        }
                    }
                }
            )
            for agg_bucket in results['aggregations']['unique_docs']['buckets']:
                existing_docs.add(agg_bucket['key'])

            def process_paragraph(doc_content: str, paragraph: str, metadata: Dict) -> Tuple[Dict, Dict]:
                """Process a single paragraph and prepare for indexing."""
                try:
                    # Extra processing for email addresses
                    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                    emails = re.findall(email_pattern, paragraph)
                    
                    # If emails found, ensure they're prominently featured in the context
                    if emails:
                        email_context = "Contains email addresses: " + ", ".join(emails)
                        context = gcp_service.generate_context(doc_content, f"{paragraph}\n{email_context}")
                    else:
                        context = gcp_service.generate_context(doc_content, paragraph)

                    combined_text = f"{paragraph}\n\n{context}"
                    embedding = embedding_service.get_embedding(combined_text)

                    vector_action = {
                        "_index": self.index_name,
                        "_source": {
                            "content": paragraph,
                            "context": context,
                            "embedding": embedding,
                            **metadata
                        }
                    }
                    
                    bm25_action = {
                        "_index": self.bm25_index,
                        "_source": {
                            "content": paragraph,
                            "context": context,
                            **metadata
                        }
                    }
                    
                    return vector_action, bm25_action
                except Exception as e:
                    logger.error(f"Error processing paragraph in {metadata['doc_id']}: {str(e)}")
                    return None, None

            # Process each PDF file
            for blob in tqdm(pdf_blobs, desc="Processing PDF files"):
                filename = os.path.basename(blob.name)
                folder_path = os.path.dirname(blob.name)
                
                # Skip if document already processed
                if filename in existing_docs:
                    logger.info(f"Skipping {filename} (in {folder_path}) - already processed")
                    continue

                logger.info(f"Processing new document: {filename} from {folder_path}")
                
                try:
                    # Download PDF to memory
                    temp_file = io.BytesIO()
                    blob.download_to_file(temp_file)
                    temp_file.seek(0)
                    
                    # Process the PDF
                    doc_content, pages_content = extract_text_from_pdf(temp_file.read())
                    doc_actions = []
                    
                    # Process paragraphs with full context
                    for page_num, page_text in enumerate(pages_content):
                        paragraphs = split_text_into_paragraphs(page_text)
                        
                        # Process paragraphs in parallel
                        with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
                            futures = []
                            for idx, paragraph in enumerate(paragraphs):
                                metadata = {
                                    "doc_id": filename,
                                    "page_number": page_num + 1,
                                    "paragraph_index": idx,
                                    "folder_path": folder_path
                                }
                                futures.append(
                                    executor.submit(process_paragraph, doc_content, paragraph, metadata)
                                )
                            
                            # Collect results and bulk index
                            for future in tqdm(as_completed(futures), total=len(paragraphs), desc=f"Processing {filename} page {page_num + 1}"):
                                vector_action, bm25_action = future.result()
                                if vector_action and bm25_action:
                                    doc_actions.extend([vector_action, bm25_action])
                    
                    # Bulk index all actions for this document
                    if doc_actions:
                        helpers.bulk(self.es_client, doc_actions)
                        processed_docs.append({"filename": filename, "folder": folder_path})
                        logger.info(f"Successfully processed {filename} from {folder_path}")
                    else:
                        raise Exception("No valid actions generated for document")
                    
                    # Clean up
                    temp_file.close()
                    
                except Exception as e:
                    error_msg = f"Failed to process {filename} from {folder_path}: {str(e)}"
                    logger.error(error_msg)
                    failed_docs.append({"filename": filename, "folder": folder_path, "error": str(e)})
                    continue
                
                # Refresh indices after each document
                self.es_client.indices.refresh(index=self.index_name)
                self.es_client.indices.refresh(index=self.bm25_index)
            
            # Prepare summary message
            if failed_docs:
                failed_msg = "\n".join([f"- {doc['filename']} (in {doc['folder']}): {doc['error']}" for doc in failed_docs])
                logger.warning(f"{len(failed_docs)} document(s) failed to index:\n{failed_msg}")
            
            return {
                "processed": processed_docs,
                "failed": failed_docs,
                "skipped": list(existing_docs)
            }
        except Exception as e:
            logger.error(f"Error processing PDFs: {str(e)}")
            raise

    def search(self, query: str, k: int = 5, use_expansion: bool = True) -> List[Dict[str, Any]]:
        """
        Full search pipeline: hybrid search + reranking with query expansion.
        
        Args:
            query: Search query.
            k: Number of results to return.
            use_expansion: Whether to use query expansion.
            
        Returns:
            List of search results.
        """
        try:
            # Start timing
            start_time = datetime.now()
            
            # Track query metrics
            query_metrics = {
                "query": query,
                "timestamp": start_time.isoformat(),
                "expansion_used": use_expansion,
                "retrieval_time": 0,
                "reranking_time": 0,
                "total_time": 0,
                "num_initial_results": 0,
                "num_final_results": k
            }
            
            # Expand query if enabled
            if use_expansion:
                expanded_query = gcp_service.expand_query(query)
                query_metrics["expanded_query"] = expanded_query
            else:
                expanded_query = query
            
            # Get initial results with wider net
            retrieval_start = datetime.now()
            initial_results = self._hybrid_search(expanded_query, k=30)
            retrieval_time = (datetime.now() - retrieval_start).total_seconds()
            query_metrics["retrieval_time"] = retrieval_time
            query_metrics["num_initial_results"] = len(initial_results)
            
            # Implement reranking here if Cohere client is available
            # For this simplified version, we'll just return the initial results
            rerank_start = datetime.now()
            reranked_results = initial_results[:k]
            rerank_time = (datetime.now() - rerank_start).total_seconds()
            query_metrics["reranking_time"] = rerank_time
            
            # Calculate total time
            query_metrics["total_time"] = (datetime.now() - start_time).total_seconds()
            
            # Log metrics
            self.query_metrics.append(query_metrics)
            
            return reranked_results
        except Exception as e:
            logger.error(f"Error in search pipeline: {str(e)}")
            # Fall back to basic search without reranking
            return self._hybrid_search(query, k=k)

    def _hybrid_search(self, query: str, k: int = 5, semantic_weight: float = 0.7, 
                      filter_recent: bool = False, max_age_days: int = 365) -> List[Dict[str, Any]]:
        """
        Combined vector and BM25 search with dynamic weighting and filtering.
        
        Args:
            query: Search query.
            k: Number of results to return.
            semantic_weight: Weight for semantic search (vs BM25).
            filter_recent: Whether to filter by recency.
            max_age_days: Maximum age in days for documents if filtering by recency.
            
        Returns:
            List of search results.
        """
        try:
            # Analyze query to determine optimal weights
            # Short, keyword queries benefit from higher BM25 weight
            # Longer, natural language queries benefit from higher vector weight
            query_words = len(query.split())
            dynamic_semantic_weight = semantic_weight
            if query_words < 4:
                # For short queries, give more weight to keyword search
                dynamic_semantic_weight = 0.6
            elif query_words > 8:
                # For longer queries, give more weight to semantic search
                dynamic_semantic_weight = 0.8
            
            # Check for document/page specific queries
            doc_filter = None
            page_filter = None
            
            doc_match = re.search(r'document:([^\s]+)', query)
            if doc_match:
                doc_filter = doc_match.group(1)
                query = re.sub(r'document:[^\s]+', '', query).strip()
                
            page_match = re.search(r'page:(\d+)', query)
            if page_match:
                page_filter = int(page_match.group(1))
                query = re.sub(r'page:\d+', '', query).strip()
            
            # Get query embedding
            query_embedding = embedding_service.get_embedding(query)
            
            # Build filter for document/page if specified
            filter_query = []
            if doc_filter:
                filter_query.append({"term": {"doc_id": doc_filter}})
            if page_filter:
                filter_query.append({"term": {"page_number": page_filter}})
            
            # Add filter for document recency if requested
            if filter_recent:
                # Calculate date threshold
                threshold_date = (datetime.now() - timedelta(days=max_age_days)).strftime("%Y-%m-%d")
                filter_query.append({
                    "range": {
                        "created_date": {
                            "gte": threshold_date
                        }
                    }
                })
            
            # Vector search with proper script scoring and filters
            vector_query = {
                "size": 30,
                "_source": ["content", "context", "doc_id", "page_number", "paragraph_index", "folder_path"],
                "query": {
                    "script_score": {
                        "query": {
                            "bool": {
                                "must": {"match_all": {}},
                                "filter": filter_query if filter_query else []
                            }
                        },
                        "script": {
                            "source": """
                            double cosine = cosineSimilarity(params.query_vector, 'embedding');
                            return cosine > -1 ? cosine + 1.0 : 0;
                            """,
                            "params": {"query_vector": query_embedding}
                        }
                    }
                }
            }
            
            vector_results = self.es_client.search(index=self.index_name, body=vector_query)
            
            # BM25 search with improved query handling and filters
            bm25_query = {
                "size": 30,
                "_source": ["content", "context", "doc_id", "page_number", "paragraph_index", "folder_path"],
                "query": {
                    "bool": {
                        "should": [
                            {
                                "match": {
                                    "content": {
                                        "query": query,
                                        "boost": 2.0,
                                        "operator": "or"
                                    }
                                }
                            },
                            {
                                "match": {
                                    "context": {
                                        "query": query,
                                        "operator": "or"
                                    }
                                }
                            },
                            {
                                "match_phrase": {
                                    "content": {
                                        "query": query,
                                        "boost": 3.0
                                    }
                                }
                            }
                        ],
                        "filter": filter_query if filter_query else []
                    }
                }
            }
            
            bm25_results = self.es_client.search(index=self.bm25_index, body=bm25_query)
            
            # Combine results with error handling
            results = []
            seen = set()
            
            # Process vector results
            for hit in vector_results['hits']['hits']:
                doc_id = hit['_source'].get('doc_id')
                if doc_id and hit['_id'] not in seen:
                    results.append({
                        "content": hit['_source'].get('content', ''),
                        "context": hit['_source'].get('context', ''),
                        "metadata": {
                            "doc_id": doc_id,
                            "page_number": hit['_source'].get('page_number', 0),
                            "paragraph_index": hit['_source'].get('paragraph_index', 0),
                            "folder_path": hit['_source'].get('folder_path', '')
                        },
                        "score": max(0, (hit['_score'] - 1.0) * dynamic_semantic_weight),
                        "source": "vector"
                    })
                    seen.add(hit['_id'])
            
            # Process BM25 results
            for hit in bm25_results['hits']['hits']:
                doc_id = hit['_source'].get('doc_id')
                if doc_id and hit['_id'] not in seen:
                    results.append({
                        "content": hit['_source'].get('content', ''),
                        "context": hit['_source'].get('context', ''),
                        "metadata": {
                            "doc_id": doc_id,
                            "page_number": hit['_source'].get('page_number', 0),
                            "paragraph_index": hit['_source'].get('paragraph_index', 0),
                            "folder_path": hit['_source'].get('folder_path', '')
                        },
                        "score": hit['_score'] * (1 - dynamic_semantic_weight),
                        "source": "bm25"
                    })
                    seen.add(hit['_id'])
            
            # Sort and return results
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:k]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            # Return empty results instead of raising error
            return []

    def get_search_analytics(self) -> Dict[str, Any]:
        """
        Get analytics about search performance.
        
        Returns:
            Dictionary with search analytics.
        """
        if not self.query_metrics:
            return {"message": "No search metrics available yet"}
        
        total_queries = len(self.query_metrics)
        avg_retrieval_time = sum(m["retrieval_time"] for m in self.query_metrics) / total_queries
        avg_reranking_time = sum(m.get("reranking_time", 0) for m in self.query_metrics) / total_queries
        avg_total_time = sum(m["total_time"] for m in self.query_metrics) / total_queries
        
        return {
            "total_queries": total_queries,
            "avg_retrieval_time": avg_retrieval_time,
            "avg_reranking_time": avg_reranking_time,
            "avg_total_time": avg_total_time,
            "recent_queries": self.query_metrics[-10:]  # Last 10 queries
        }

    def get_document_stats(self) -> Dict[str, Any]:
        """
        Get statistics about documents in the database.
        
        Returns:
            Dictionary with document statistics.
        """
        try:
            # Get total document count
            doc_count = self.es_client.count(index=self.index_name)['count']
            
            # Get unique document IDs
            doc_agg = self.es_client.search(
                index=self.index_name,
                body={
                    "size": 0,
                    "aggs": {
                        "unique_docs": {
                            "terms": {
                                "field": "doc_id",
                                "size": 1000
                            }
                        }
                    }
                }
            )
            
            # Get folder paths
            folder_agg = self.es_client.search(
                index=self.index_name,
                body={
                    "size": 0,
                    "aggs": {
                        "folders": {
                            "terms": {
                                "field": "folder_path",
                                "size": 100
                            }
                        }
                    }
                }
            )
            
            # Prepare stats
            stats = {
                "total_chunks": doc_count,
                "unique_documents": len(doc_agg['aggregations']['unique_docs']['buckets']),
                "documents": [
                    {"id": bucket['key'], "chunks": bucket['doc_count']} 
                    for bucket in doc_agg['aggregations']['unique_docs']['buckets']
                ],
                "folders": [
                    {"path": bucket['key'], "chunks": bucket['doc_count']} 
                    for bucket in folder_agg['aggregations']['folders']['buckets']
                ]
            }
            
            return stats
        except Exception as e:
            logger.error(f"Error getting document stats: {str(e)}")
            return {"error": str(e)}