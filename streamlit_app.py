import os
from typing import List, Dict, Any, Callable, Tuple
from tqdm import tqdm
from elasticsearch import Elasticsearch, helpers
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.elasticsearch import ElasticsearchStore
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
import numpy as np
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import cohere
import time
import json
import streamlit as st
from pathlib import Path
from langchain_community.vectorstores import ElasticsearchStore
import fitz
from langdetect import detect, LangDetectException
import google.generativeai as genai
from google.cloud import aiplatform, storage
from vertexai.language_models import TextEmbeddingModel
from PIL import Image
import io
from datetime import datetime, timedelta
from google.auth import default

load_dotenv()

class ContextualElasticVectorDB:
    def __init__(self, index_name: str, google_api_key=None, cohere_api_key=None):
        if google_api_key is None:
            google_api_key = os.getenv("GOOGLE_API_KEY")
        if cohere_api_key is None:
            cohere_api_key = os.getenv("COHERE_API_KEY")
            
        genai.configure(api_key=google_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        aiplatform.init()
        credentials, project = default()
        self.storage_client = storage.Client(credentials=credentials, project=project)
        self.embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")

        elastic_host = os.getenv("ELASTIC_HOST")
        elastic_user = os.getenv("ELASTIC_USER")
        elastic_password = os.getenv("ELASTIC_PASSWORD")
        
        print(f"Connecting to Elasticsearch at {elastic_host}")
        
        try:
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
            
            print("Successfully connected to Elasticsearch")
        except Exception as e:
            print(f"Error connecting to Elasticsearch: {str(e)}")
            raise
        
        self.cohere_client = cohere.Client(cohere_api_key)
        
        self.index_name = index_name
        self.bm25_index = f"{index_name}_bm25"
        
        # Create indices if they don't exist
        if not self.es_client.indices.exists(index=self.index_name):
            self._create_vector_index()
        if not self.es_client.indices.exists(index=self.bm25_index):
            self._create_bm25_index()
            
        # Initialize query metrics logging
        self.query_metrics = []
        self.feedback_log = []

    def get_embedding(self, text: str) -> List[float]:
        """Get embeddings using Vertex AI for the user query with empty text handling"""
        try:
            # Check for empty text
            if not text or not text.strip():
                print("Warning: Empty text provided for embedding")
                return [0.0] * 768
            
            embeddings = self.embedding_model.get_embeddings([text])
            return embeddings[0].values
        except Exception as e:
            print(f"Error getting embedding: {str(e)}")
            return [0.0] * 768

    def _create_vector_index(self):
        """Create Elasticsearch index with vector search capabilities"""
        if not self.es_client.indices.exists(index=self.index_name):
            mapping = {
                "mappings": {
                    "properties": {
                        "content": {"type": "text"},
                        "context": {"type": "text"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": 768, # 768 is the dimension of the embedding model
                            "index": True, 
                            "similarity": "cosine" #
                        },
                        "doc_id": {"type": "keyword"},
                        "page_number": {"type": "integer"},
                        "paragraph_index": {"type": "integer"},
                        "folder_path": {"type": "keyword"}
                    }
                }
            }
            self.es_client.indices.create(index=self.index_name, body=mapping)

    def _create_bm25_index(self):
        """Create separate BM25 index for text search"""
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

    def situate_context(self, doc: str, chunk: str) -> str:
        """Generate contextual description using Gemini with prompt caching"""
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

        response = self.gemini_model.generate_content(prompt)
        
        # Add error handling and default return
        if response and response.text:
            return response.text.strip() or "No context available"
        return "No context available"

    def process_pdfs(self, bucket_path: str, parallel_threads: int = 4):
        """Process PDFs from Google Cloud Storage bucket and load into both vector and BM25 indices"""
        try:
            # Parse bucket path
            if not bucket_path.startswith("gs://"):
                raise ValueError("Bucket path must start with gs://")
            
            # Print debug information
            print(f"Processing bucket path: {bucket_path}")
            
            parts = bucket_path.split("/")
            if len(parts) < 3:
                raise ValueError("Invalid bucket path format")
            
            bucket_name = parts[2]
            prefix = "/".join(parts[3:]) if len(parts) > 3 else ""
            
            print(f"Bucket name: {bucket_name}")
            print(f"Prefix: {prefix}")
            
            # Use the instance's storage client
            bucket = self.storage_client.bucket(bucket_name)
            if not bucket:
                raise ValueError(f"Could not access bucket: {bucket_name}")
            
            # List all PDF blobs recursively in the bucket with the given prefix
            print("Listing blobs...")
            blobs = list(bucket.list_blobs(prefix=prefix))
            print(f"Found {len(blobs)} total blobs")
            
            pdf_blobs = [blob for blob in blobs if blob.name.lower().endswith('.pdf')]
            print(f"Found {len(pdf_blobs)} PDF files")
            
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

            def extract_text_with_emails(page) -> str:
                """Extract text while preserving email addresses"""
                # Get raw text
                text = page.get_text()
                
                # Find email addresses using regex
                email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                emails = re.findall(email_pattern, text)
                
                # Ensure emails are preserved in the text
                for email in emails:
                    # Add extra spaces around email to ensure it's preserved as a separate token
                    text = text.replace(email, f" {email} ")
                
                return text

            def process_paragraph(doc_content: str, paragraph: str, metadata: Dict):
                try:
                    # Extra processing for email addresses
                    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                    emails = re.findall(email_pattern, paragraph)
                    
                    # If emails found, ensure they're prominently featured in the context
                    if emails:
                        email_context = "Contains email addresses: " + ", ".join(emails)
                        context = self.situate_context(doc_content, f"{paragraph}\n{email_context}")
                    else:
                        context = self.situate_context(doc_content, paragraph)

                    combined_text = f"{paragraph}\n\n{context}"
                    embedding = self.get_embedding(combined_text)

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
                    print(f"Error processing paragraph in {metadata['doc_id']}: {str(e)}")
                    return None, None

            # Process each PDF file
            for blob in tqdm(pdf_blobs, desc="Processing PDF files"):
                filename = os.path.basename(blob.name)
                folder_path = os.path.dirname(blob.name)
                
                # Skip if document already processed
                if filename in existing_docs:
                    print(f"Skipping {filename} (in {folder_path}) - already processed")
                    continue

                print(f"Processing new document: {filename} from {folder_path}")
                
                try:
                    # Download PDF to memory
                    temp_file = io.BytesIO()
                    blob.download_to_file(temp_file)
                    temp_file.seek(0)
                    
                    # Process the PDF
                    pdf_document = fitz.open(stream=temp_file.read(), filetype="pdf")
                    doc_content = ""
                    pages_content = []
                    
                    # First pass: collect all text
                    for page_num in range(len(pdf_document)):
                        page = pdf_document[page_num]
                        text = extract_text_with_emails(page)
                        doc_content += text + "\n\n"
                        pages_content.append(text)
                    
                    doc_actions = []
                    
                    # Second pass: process paragraphs with full context
                    for page_num, page_text in enumerate(pages_content):
                        paragraphs = [p.strip() for p in page_text.split('\n\n') if p.strip()]
                        
                        # Process paragraphs in parallel
                        with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
                            futures = []
                            for idx, paragraph in enumerate(paragraphs):
                                metadata = {
                                    "doc_id": filename,
                                    "page_number": page_num + 1,
                                    "paragraph_index": idx,
                                    "folder_path": folder_path  # Store the folder path for reference
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
                        print(f"Successfully processed {filename} from {folder_path}")
                    else:
                        raise Exception("No valid actions generated for document")
                    
                    # Clean up
                    pdf_document.close()
                    temp_file.close()
                    
                except Exception as e:
                    error_msg = f"Failed to process {filename} from {folder_path}: {str(e)}"
                    print(error_msg)
                    failed_docs.append({"filename": filename, "folder": folder_path, "error": str(e)})
                    continue
                
                # Refresh indices after each document
                self.es_client.indices.refresh(index=self.index_name)
                self.es_client.indices.refresh(index=self.bm25_index)
            
            # Prepare summary message
            if failed_docs:
                failed_msg = "\n".join([f"- {doc['filename']} (in {doc['folder']}): {doc['error']}" for doc in failed_docs])
                raise Exception(f"{len(failed_docs)} document(s) failed to index:\n{failed_msg}")
            
            return {
                "processed": processed_docs,
                "failed": failed_docs,
                "skipped": list(existing_docs)
            }
        except Exception as e:
            print(f"Error processing PDFs: {str(e)}")
            return None

    def expand_query(self, query: str) -> str:
        """Expand the query to improve recall"""
        try:
            # Use Gemini to expand the query with related terms
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
            
            # Combine original query with expanded terms
            expanded_query = query + " " + " ".join(expanded_terms)
            return expanded_query
        except Exception as e:
            print(f"Error expanding query: {str(e)}")
            return query  # Fall back to original query

    def search(self, query: str, k: int = 5, use_expansion: bool = True) -> List[Dict[str, Any]]:
        """Full search pipeline: hybrid search + reranking with query expansion"""
        try:
            # Start timing
            import time
            start_time = time.time()
            
            # Track query metrics
            query_metrics = {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "expansion_used": use_expansion,
                "retrieval_time": 0,
                "reranking_time": 0,
                "total_time": 0,
                "num_initial_results": 0,
                "num_final_results": k
            }
            
            # Expand query if enabled
            if use_expansion:
                expanded_query = self.expand_query(query)
                query_metrics["expanded_query"] = expanded_query
            else:
                expanded_query = query
            
            # Get initial results with wider net
            retrieval_start = time.time()
            initial_results = self._hybrid_search(expanded_query, k=30)  # Increased from 20
            query_metrics["retrieval_time"] = time.time() - retrieval_start
            query_metrics["num_initial_results"] = len(initial_results)
            
            # Prepare documents for reranking
            documents = [result['content'] for result in initial_results]
            
            # Rerank with Cohere - updated to work with older versions
            rerank_start = time.time()
            try:
                # Try with newer API first
                rerank_response = self.cohere_client.rerank(
                    model="rerank-english-v3.0",
                    query=query,
                    documents=documents,
                    top_n=k
                )
                
                # Process results based on API version
                reranked_results = []
                if hasattr(rerank_response, 'results'):
                    # Newer API
                    for r in rerank_response.results:
                        original_result = initial_results[r.index]
                        reranked_results.append({
                            "content": original_result['content'],
                            "context": original_result['context'],
                            "metadata": original_result['metadata'],
                            "score": r.relevance_score,
                            "original_score": original_result.get('score', 0),
                            "rerank_index": r.index,
                            "source": original_result.get('source', 'unknown')
                        })
                else:
                    # Older API
                    for i, (doc, score) in enumerate(zip(rerank_response.documents, rerank_response.rankings)):
                        # Find the original result that matches this document
                        for j, result in enumerate(initial_results):
                            if result['content'] == doc:
                                reranked_results.append({
                                    "content": result['content'],
                                    "context": result['context'],
                                    "metadata": result['metadata'],
                                    "score": score,
                                    "original_score": result.get('score', 0),
                                    "rerank_index": j,
                                    "source": result.get('source', 'unknown')
                                })
                                break
            
            except Exception as e:
                print(f"Error in reranking: {str(e)}")
                reranked_results = [
                    {
                        "content": result['content'],
                        "context": result['context'],
                        "metadata": result['metadata'],
                        "score": result.get('score', 0),
                        "source": result.get('source', 'unknown')
                    }
                    for result in initial_results[:k]
                ]
            
            query_metrics["reranking_time"] = time.time() - rerank_start
            
            # Calculate total time
            query_metrics["total_time"] = time.time() - start_time
            
            # Log metrics
            self.query_metrics.append(query_metrics)
            
            return reranked_results
        except Exception as e:
            print(f"Error in search pipeline: {str(e)}")
            # Fall back to basic search without reranking
            return self._hybrid_search(query, k=k)

    def _hybrid_search(self, query: str, k: int = 5, semantic_weight: float = 0.7, 
                      filter_recent: bool = False, max_age_days: int = 365) -> List[Dict[str, Any]]:
        """Combined vector and BM25 search with dynamic weighting and filtering"""
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
            
            # Get query embedding using Vertex AI
            query_embedding = self.get_embedding(query)
            
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
                "size": 30,  # Increased from 20
                "_source": ["content", "context", "doc_id", "page_number", "paragraph_index", "folder_path"],
                "query": {
                    "script_score": {
                        "query": {
                            "bool": {
                                "must": {"match_all": {}},  # Search ALL documents
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
                "size": 30,  # Increased from 20
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
            print(f"Error in hybrid search: {str(e)}")
            # Return empty results instead of raising error
            return []

    def get_search_analytics(self) -> Dict[str, Any]:
        """Get analytics about search performance"""
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
        """Get statistics about documents in the database"""
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
            print(f"Error getting document stats: {str(e)}")
            return {"error": str(e)}

    def search_document_content(self, search_term: str) -> List[Dict[str, Any]]:
        """Search for specific content across all documents"""
        try:
            results = self.es_client.search(
                index=self.bm25_index,
                body={
                    "size": 50,
                    "_source": ["content", "doc_id", "page_number", "paragraph_index", "folder_path"],
                    "query": {
                        "multi_match": {
                            "query": search_term,
                            "fields": ["content", "context"],
                            "type": "best_fields"
                        }
                    },
                    "highlight": {
                        "fields": {
                            "content": {},
                            "context": {}
                        },
                        "pre_tags": ["<mark>"],
                        "post_tags": ["</mark>"]
                    }
                }
            )
            
            search_results = []
            for hit in results['hits']['hits']:
                highlight = ""
                if 'highlight' in hit:
                    if 'content' in hit['highlight']:
                        highlight = "... ".join(hit['highlight']['content'])
                    elif 'context' in hit['highlight']:
                        highlight = "... ".join(hit['highlight']['context'])
                
                search_results.append({
                    "doc_id": hit['_source']['doc_id'],
                    "page_number": hit['_source']['page_number'],
                    "paragraph_index": hit['_source']['paragraph_index'],
                    "folder_path": hit['_source'].get('folder_path', ''),
                    "content": hit['_source']['content'],
                    "highlight": highlight,
                    "score": hit['_score']
                })
            
            return search_results
        except Exception as e:
            print(f"Error searching document content: {str(e)}")
            return []

    def reset_indices(self):
        """Clear and recreate the indices"""
        for index in [self.index_name, self.bm25_index]:
            if self.es_client.indices.exists(index=index):
                self.es_client.indices.delete(index=index)
        self._create_vector_index()
        self._create_bm25_index()

    def reset_specific_documents(self, doc_ids: List[str]):
        """Delete specific documents from both indices"""
        for index in [self.index_name, self.bm25_index]:
            self.es_client.delete_by_query(
                index=index,
                body={
                    "query": {
                        "terms": {
                            "doc_id": doc_ids
                        }
                    }
                }
            )
        # Refresh indices
        self.es_client.indices.refresh(index=self.index_name)
        self.es_client.indices.refresh(index=self.bm25_index)

    def get_processed_documents(self) -> List[str]:
        """Get list of all processed documents in the database"""
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
        return [bucket['key'] for bucket in results['aggregations']['unique_docs']['buckets']]

class ContextualRAG:
    def __init__(self, vector_db: ContextualElasticVectorDB):
        self.vector_db = vector_db
        google_api_key = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=google_api_key)
        
        # Use the standard GenerativeModel interface
        self.gemini = genai.GenerativeModel('gemini-2.0-flash')
        
        # Initialize feedback log
        self.feedback_log = []
    
    def detect_language(self, text: str) -> str:
        """
        Detect language using langdetect library
        Returns ISO 639-1 language code (e.g., 'en', 'es', 'fr', etc.)
        Defaults to 'en' if detection fails
        """
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
        # Start timing
        import time
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
        initial_results = self.vector_db.search(question, k=5, use_expansion=True)
        query_metrics["retrieval_time"] = time.time() - retrieval_start
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
        llm_start = time.time()
        response = self.gemini.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
                top_p=0.95,
                top_k=40
            )
        )
        query_metrics["llm_time"] = time.time() - llm_start
        
        # Calculate total time
        query_metrics["total_time"] = time.time() - start_time
        
        # Log metrics for analysis
        self._log_query_metrics(query_metrics)
        
        return response.text.strip(), all_results
    
    def _log_query_metrics(self, metrics: Dict[str, Any]):
        """Log query metrics for analysis"""
        try:
            # Append to in-memory log
            if not hasattr(self, 'query_metrics_log'):
                self.query_metrics_log = []
            self.query_metrics_log.append(metrics)
            
            # Could also write to a file or database
            print(f"Query processed in {metrics['total_time']:.2f}s (retrieval: {metrics['retrieval_time']:.2f}s, LLM: {metrics['llm_time']:.2f}s)")
        except Exception as e:
            print(f"Error logging metrics: {str(e)}")
    
    def log_feedback(self, query: str, response: str, feedback: str, improvement: str = None):
        """Log user feedback for continuous improvement"""
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

class AIChatHistory:
    """Simple class to manage chat history with timestamps"""
    def __init__(self):
        self.messages = []
    
    def add_user_message(self, message: str):
        """Add a user message to the history"""
        self.messages.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    def add_ai_message(self, message: str, sources: List[Dict] = None):
        """Add an AI message to the history with optional sources"""
        self.messages.append({
            "role": "assistant",
            "content": message,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sources": sources if sources else []
        })
    
    def get_messages(self):
        """Get all messages in the history"""
        return self.messages
    
    def clear(self):
        """Clear the chat history"""
        self.messages = []

def display_chat(chat_history: AIChatHistory):
    for msg in chat_history.get_messages():
        if msg.role == "user":
            st.chat_message("user").write(msg.content)
        else:
            with st.chat_message("assistant"):
                st.write(msg.content)
                if msg.sources:
                    st.write("---")
                    st.write("**Sources:**")
                    for source in msg.sources:
                        st.write(f"📄 Document: {source['metadata']['doc_id']}, "
                               f"Page: {source['metadata']['page_number']}")

def initialize_db():
    if 'db' not in st.session_state:
        st.session_state.db = ContextualElasticVectorDB("pdf_docs_rag_04022025")
    if 'storage_client' not in st.session_state:
        credentials, project = default()
        st.session_state.storage_client = storage.Client(credentials=credentials, project=project)

def get_current_time() -> str:
    """Get current time formatted as string"""
    return datetime.now().strftime("%I:%M %p")

def main():
    st.set_page_config(
        page_title="JLR Analysis RAG",
        layout="wide")

    with st.container():
        st.title("Contextual RAG for JLR Analysis")
        st.header("Search and Q&A")
    initialize_db()

    # Initialize RAG system and chat history
    if 'rag' not in st.session_state:
        st.session_state.rag = ContextualRAG(st.session_state.db)
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = AIChatHistory()
    # Add debug mode to session state
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    # Add feedback log to session state
    if 'feedback_log' not in st.session_state:
        st.session_state.feedback_log = []

    # Create a container for the chat
    chat_container = st.container()

    # Chat input at the bottom
    if prompt := st.chat_input("Ask me anything about analytical work that has been done for JLR!", key="unique_chat_input"):
        # Get current timestamp
        current_time = get_current_time()

        # Add user message immediately
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": current_time
        })

        # Check for greeting cases
        greeting_responses = {
            "how are you": "I'm doing well, and you? How can I help you today?",
            "how are you?": "I'm doing well, and you? How can I help you today?",
            "hi": "Hello! How can I help you today?",
            "hi!": "Hello! How can I help you today?",
            "hello": "Hello! How can I help you today?",
            "hello!": "Hello! How can I help you today?",
            "hi, how are you?": "I'm doing well, and you? How can I help you today?",
            "hi how are you": "I'm doing well, and you? How can I help you today?",
            "hello, how are you?": "I'm doing well, and you? How can I help you today?",
            "hello, how are you": "I'm doing well, and you? How can I help you today?",
            "hi, how are you": "I'm doing well, and you? How can I help you today?",
            "hi how are you": "I'm doing well, and you? How can I help you today?",
            "hi how are you?": "I'm doing well, and you? How can I help you today?",
            "hello how are you": "I'm doing well, and you? How can I help you today?",
            "goodbye": "Goodbye! Have a great day!",
            "bye": "Bye! Feel free to come back if you have more questions!",
            "thank you": "You're welcome! Is there anything else I can help you with?",
            "thanks": "You're welcome! Is there anything else I can help you with?",
            "no, that's all": "Thank you for using JLR RAG! Have a great day!",
            "no, that's all!": "Thank you for using JLR RAG! Have a great dWay!",
            "nothing for now": "Thank you for using JLR RAG! Have a great day!",
            "nothing for now!": "Thank you for using JLR RAG! Have a great day!",
            "thank you, goodbye": "Thank you for using JLR RAG! Have a great day!",
            "thank you, goodbye!": "Thank you for using JLR RAG! Have a great day!"
        }
        
        if prompt.lower() in greeting_responses:
            # Add assistant response for greetings
            st.session_state.messages.append({
                "role": "assistant",
                "content": greeting_responses[prompt.lower()],
                "timestamp": current_time
            })
            st.rerun()
            
        try:
            with st.spinner("🤔 Thinking..."):
                answer, results = st.session_state.rag.query(prompt)
                
                # Store the raw results for debugging
                if 'last_results' not in st.session_state:
                    st.session_state.last_results = []
                st.session_state.last_results = results
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": results,
                    "timestamp": current_time
                })
            st.rerun()
        except Exception as e:
            error_message = f"Error during search: {str(e)}"
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message,
                "sources": [],
                "timestamp": current_time
            })
            st.rerun()

    # Display chat history in the container
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "timestamp" in message:
                    st.caption(f":clock2: {message['timestamp']}")
                
                # Show sources only for assistant messages
                if message["role"] == "assistant" and message.get("sources"):
                    displayed_pages = set()
                    
                    # Create an expander for the sources
                    with st.expander("Resources", expanded=False):
                        st.markdown("<div style='text-align: center;'></div>", unsafe_allow_html=True)
                        
                        for result in message["sources"]:
                            page_id = (result['metadata']['doc_id'], result['metadata']['page_number'])
                            if page_id not in displayed_pages:
                                try:
                                    # Get the bucket name from the environment or configuration
                                    bucket_name = os.getenv("GCS_BUCKET_NAME")
                                    bucket = st.session_state.storage_client.bucket(bucket_name)
                                    
                                    # Search for the file recursively in RAG Documents folder
                                    prefix = "JLR_analysis/"
                                    file_found = False
                                    
                                    # List all blobs with the prefix
                                    blobs = bucket.list_blobs(prefix=prefix)
                                    for blob in blobs:
                                        if blob.name.endswith(result['metadata']['doc_id']):
                                            file_found = True
                                            
                                            # Download PDF to memory
                                            temp_file = io.BytesIO()
                                            blob.download_to_file(temp_file)
                                            temp_file.seek(0)
                                            
                                            # Open PDF and get the specific page
                                            with fitz.open(stream=temp_file.read(), filetype="pdf") as pdf_document:
                                                page = pdf_document[result['metadata']['page_number'] - 1]
                                                pix = page.get_pixmap()
                                                
                                                col1, col2, col3 = st.columns([1, 2, 1])
                                                with col2:
                                                    st.image(
                                                        pix.tobytes(),
                                                        caption=f"Source: {result['metadata']['doc_id']}, Page {result['metadata']['page_number']}",
                                                        use_column_width=True
                                                    )
                                            
                                            displayed_pages.add(page_id)
                                            temp_file.close()
                                            break
                                    
                                    if not file_found:
                                        st.error(f"PDF not found in any subfolder: {result['metadata']['doc_id']}")
                                    
                                except Exception as e:
                                    st.error(f"Error loading PDF page: {str(e)}")
                                    st.error(f"Full error details: {type(e).__name__}: {str(e)}")

    # Sidebar for PDF processing from GCS and analytics
    with st.sidebar:
        st.header("Document Processing")
        bucket_path = st.text_input("Enter Google Cloud Storage path (gs://bucket-name/path/to/pdfs/)")
        
        if bucket_path:
            if st.button("Process PDFs"):
                with st.spinner("Processing PDFs from GCS..."):
                    try:
                        st.session_state.db.process_pdfs(bucket_path)
                        st.success("PDFs processed successfully!")
                    except Exception as e:
                        st.error(f"Error processing PDFs: {str(e)}")
        
        if st.button("Reset Database"):
            if st.session_state.db:
                try:
                    st.session_state.db.reset_indices()
                    st.success("Database reset successfully!")
                except Exception as e:
                    st.error(f"Error resetting database: {str(e)}")

        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

        # Get list of processed documents
        processed_docs = st.session_state.db.get_processed_documents()
        
        # Add specific document reprocessing
        specific_docs = st.multiselect(
            "Select documents to reprocess",
            processed_docs
        )
        
        if specific_docs and st.button("Reprocess Selected Documents"):
            # Create temporary directory for PDFs
            temp_dir = Path("temp_pdfs")
            temp_dir.mkdir(exist_ok=True)
            
            with st.spinner("Reprocessing selected documents..."):
                try:
                    # First reset these specific documents
                    st.session_state.db.reset_specific_documents(specific_docs)
                    # Then process them again
                    st.session_state.db.process_pdfs(str(temp_dir))
                    st.success("Documents reprocessed successfully!")
                except Exception as e:
                    st.error(f"Error reprocessing documents: {str(e)}")
        
        # Add document explorer
        st.header("Document Explorer")
        
        # Show document statistics
        if st.button("Show Document Statistics"):
            with st.spinner("Loading document statistics..."):
                stats = st.session_state.db.get_document_stats()
                st.write(f"Total chunks: {stats['total_chunks']}")
                st.write(f"Unique documents: {stats['unique_documents']}")
                
                # Show folders
                with st.expander("Folders"):
                    for folder in stats['folders']:
                        st.write(f"{folder['path']}: {folder['chunks']} chunks")
                
                # Show documents
                with st.expander("Documents"):
                    for doc in stats['documents']:
                        st.write(f"{doc['id']}: {doc['chunks']} chunks")
        
        # Content search
        st.subheader("Search Document Content")
        search_term = st.text_input("Enter search term")
        if search_term:
            with st.spinner("Searching..."):
                search_results = st.session_state.db.search_document_content(search_term)
                
                if search_results:
                    st.write(f"Found {len(search_results)} results")
                    for result in search_results:
                        with st.expander(f"{result['doc_id']} - Page {result['page_number']}"):
                            if result['highlight']:
                                st.markdown(result['highlight'], unsafe_allow_html=True)
                            else:
                                st.write(result['content'][:200] + "...")
                else:
                    st.write("No results found")
        
        # Add debug mode toggle
        st.header("Debug Options")
        debug_mode = st.checkbox("Enable Debug Mode", value=st.session_state.debug_mode)
        if debug_mode != st.session_state.debug_mode:
            st.session_state.debug_mode = debug_mode
            st.rerun()
        
        # Display search analytics
        if st.button("Show Search Analytics"):
            analytics = st.session_state.db.get_search_analytics()
            st.json(analytics)
        
        # Display feedback log
        if st.button("Show Feedback Log"):
            if st.session_state.feedback_log:
                st.write(f"Total feedback entries: {len(st.session_state.feedback_log)}")
                positive = sum(1 for f in st.session_state.feedback_log if f['feedback'] == 'positive')
                negative = sum(1 for f in st.session_state.feedback_log if f['feedback'] == 'negative')
                st.write(f"Positive: {positive}, Negative: {negative}")
                
                with st.expander("View All Feedback"):
                    st.json(st.session_state.feedback_log)
            else:
                st.write("No feedback collected yet")
        
        # Display debug information if enabled
        if st.session_state.debug_mode and 'last_results' in st.session_state and st.session_state.last_results:
            st.header("Search Results Debug")
            
            # Display each result with its score and content
            for i, result in enumerate(st.session_state.last_results):
                with st.expander(f"Result #{i+1} - Score: {result['score']:.4f} - {result['metadata']['doc_id']}"):
                    st.markdown(f"**Source**: {result.get('source', 'unknown')}")
                    if 'original_score' in result:
                        st.markdown(f"**Original Score**: {result['original_score']:.4f}")
                    if 'rerank_index' in result:
                        st.markdown(f"**Original Rank**: {result['rerank_index'] + 1}")
                    
                    st.markdown("### Content")
                    st.markdown(f"```\n{result['content']}\n```")
                    
                    st.markdown("### Context")
                    st.markdown(f"```\n{result['context']}\n```")
                    
                    st.markdown("### Metadata")
                    st.json(result['metadata'])

if __name__ == "__main__":
    main()