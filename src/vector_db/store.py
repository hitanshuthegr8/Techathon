"""
Vector Store - Manages storage and retrieval of failure embeddings
Supports both ChromaDB and Pinecone backends
"""
import numpy as np
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path
import json

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger
from src.vector_db.embedder import FailureEmbedder

logger = get_logger(__name__)


class VectorStore:
    """
    Abstract base class for vector storage backends.
    """
    
    def add(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add embeddings to the store."""
        raise NotImplementedError
    
    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Query similar embeddings."""
        raise NotImplementedError
    
    def delete(self, ids: List[str]) -> bool:
        """Delete embeddings by ID."""
        raise NotImplementedError


class ChromaDBStore(VectorStore):
    """
    ChromaDB implementation for local vector storage.
    """
    
    def __init__(
        self,
        collection_name: str = "failure_patterns",
        persist_directory: str = "./chroma_db"
    ):
        try:
            import chromadb
            from chromadb.config import Settings
            
            self.client = chromadb.Client(Settings(
                persist_directory=persist_directory,
                anonymized_telemetry=False
            ))
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"ChromaDB store initialized: {collection_name}")
            
        except ImportError:
            logger.error("ChromaDB not installed. Install with: pip install chromadb")
            raise
    
    def add(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add embeddings to ChromaDB collection.
        """
        try:
            if ids is None:
                import uuid
                ids = [str(uuid.uuid4()) for _ in range(len(embeddings))]
            
            # Convert metadata to JSON strings (ChromaDB limitation)
            metadata_processed = []
            for meta in metadata:
                processed = {}
                for key, value in meta.items():
                    if isinstance(value, (dict, list)):
                        processed[key] = json.dumps(value)
                    else:
                        processed[key] = str(value)
                metadata_processed.append(processed)
            
            self.collection.add(
                embeddings=embeddings.tolist(),
                metadatas=metadata_processed,
                ids=ids
            )
            
            logger.info(f"Added {len(ids)} embeddings to ChromaDB")
            return ids
            
        except Exception as e:
            logger.error(f"Error adding to ChromaDB: {str(e)}")
            raise
    
    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query similar embeddings from ChromaDB.
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=filter_dict
            )
            
            # Parse results
            parsed_results = []
            if results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    metadata = results['metadatas'][0][i]
                    
                    # Parse JSON fields back to objects
                    for key, value in metadata.items():
                        try:
                            metadata[key] = json.loads(value)
                        except (json.JSONDecodeError, TypeError):
                            pass  # Keep as string
                    
                    parsed_results.append({
                        'id': results['ids'][0][i],
                        'similarity': 1 - results['distances'][0][i],  # Convert distance to similarity
                        'metadata': metadata
                    })
            
            logger.info(f"Retrieved {len(parsed_results)} results from ChromaDB")
            return parsed_results
            
        except Exception as e:
            logger.error(f"Error querying ChromaDB: {str(e)}")
            return []
    
    def delete(self, ids: List[str]) -> bool:
        """
        Delete embeddings from ChromaDB.
        """
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} embeddings from ChromaDB")
            return True
        except Exception as e:
            logger.error(f"Error deleting from ChromaDB: {str(e)}")
            return False


class PineconeStore(VectorStore):
    """
    Pinecone implementation for cloud-based vector storage.
    """
    
    def __init__(
        self,
        index_name: str = "failure-patterns",
        dimension: int = 128,
        api_key: Optional[str] = None,
        environment: Optional[str] = None
    ):
        try:
            import pinecone
            
            # Initialize Pinecone
            if api_key:
                pinecone.init(api_key=api_key, environment=environment)
            else:
                # Try to use environment variables
                pinecone.init()
            
            # Create or connect to index
            if index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric="cosine"
                )
                logger.info(f"Created new Pinecone index: {index_name}")
            
            self.index = pinecone.Index(index_name)
            logger.info(f"Connected to Pinecone index: {index_name}")
            
        except ImportError:
            logger.error("Pinecone not installed. Install with: pip install pinecone-client")
            raise
    
    def add(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add embeddings to Pinecone index.
        """
        try:
            if ids is None:
                import uuid
                ids = [str(uuid.uuid4()) for _ in range(len(embeddings))]
            
            # Prepare vectors for upsert
            vectors = []
            for i, (emb, meta, vec_id) in enumerate(zip(embeddings, metadata, ids)):
                vectors.append({
                    'id': vec_id,
                    'values': emb.tolist(),
                    'metadata': meta
                })
            
            # Upsert in batches of 100
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            logger.info(f"Added {len(ids)} embeddings to Pinecone")
            return ids
            
        except Exception as e:
            logger.error(f"Error adding to Pinecone: {str(e)}")
            raise
    
    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query similar embeddings from Pinecone.
        """
        try:
            results = self.index.query(
                vector=query_embedding.tolist(),
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            parsed_results = []
            for match in results['matches']:
                parsed_results.append({
                    'id': match['id'],
                    'similarity': match['score'],
                    'metadata': match.get('metadata', {})
                })
            
            logger.info(f"Retrieved {len(parsed_results)} results from Pinecone")
            return parsed_results
            
        except Exception as e:
            logger.error(f"Error querying Pinecone: {str(e)}")
            return []
    
    def delete(self, ids: List[str]) -> bool:
        """
        Delete embeddings from Pinecone.
        """
        try:
            self.index.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} embeddings from Pinecone")
            return True
        except Exception as e:
            logger.error(f"Error deleting from Pinecone: {str(e)}")
            return False


def create_vector_store(
    backend: str = "chromadb",
    **kwargs
) -> VectorStore:
    """
    Factory function to create appropriate vector store.
    
    Args:
        backend: "chromadb" or "pinecone"
        **kwargs: Backend-specific configuration
        
    Returns:
        VectorStore instance
    """
    if backend.lower() == "chromadb":
        return ChromaDBStore(**kwargs)
    elif backend.lower() == "pinecone":
        return PineconeStore(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}. Choose 'chromadb' or 'pinecone'")