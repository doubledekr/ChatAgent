import logging
import time
import os
import json
from pinecone import Pinecone, ServerlessSpec

logger = logging.getLogger(__name__)

class PineconeManager:
    """
    Manager class for Pinecone vector database operations
    """
    
    def __init__(self, api_key, environment, index_name="docai", dimension=1536):
        """
        Initialize Pinecone connection and index
        
        Args:
            api_key (str): Pinecone API key
            environment (str): Pinecone environment
            index_name (str): Name of the Pinecone index
            dimension (int): Dimension of the vector embeddings
        """
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.dimension = dimension
        self.index = None
        
        if not api_key:
            logger.error("Pinecone API key not provided")
            raise ValueError("Pinecone API key must be provided")
        
        try:
            # Initialize Pinecone client with the API key
            self.pc = Pinecone(api_key=api_key)
            
            # Get list of indexes
            existing_indexes = [idx.name for idx in self.pc.list_indexes().indexes]
            
            # Check if index exists, if not create it
            if index_name not in existing_indexes:
                logger.info(f"Creating new Pinecone index: {index_name}")
                self.pc.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
                # Wait for index to be ready
                time.sleep(5)  # Wait a bit longer for index creation
            
            # Connect to the index
            self.index = self.pc.Index(index_name)
            logger.info(f"Connected to Pinecone index: {index_name}")
            
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {e}")
            raise
    
    def upsert(self, id, vector, metadata=None):
        """
        Upsert vector and metadata to Pinecone
        
        Args:
            id (str): Unique identifier for the vector
            vector (list): Vector embedding
            metadata (dict): Metadata to store with the vector
        
        Returns:
            bool: Success status
        """
        if not self.index:
            logger.error("Pinecone index not initialized")
            return False
        
        try:
            # Format for the latest Pinecone client
            self.index.upsert(
                vectors=[
                    {
                        "id": id,
                        "values": vector,
                        "metadata": metadata or {}
                    }
                ],
                namespace="default"
            )
            return True
        
        except Exception as e:
            logger.error(f"Error upserting to Pinecone: {e}")
            return False
    
    def query(self, vector, top_k=5, include_metadata=True):
        """
        Query Pinecone for similar vectors
        
        Args:
            vector (list): Query vector
            top_k (int): Number of results to return
            include_metadata (bool): Whether to include metadata in results
        
        Returns:
            dict: Query results
        """
        if not self.index:
            logger.error("Pinecone index not initialized")
            return {"matches": []}
        
        try:
            result = self.index.query(
                namespace="default",
                vector=vector,
                top_k=top_k,
                include_values=True,
                include_metadata=include_metadata
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Error querying Pinecone: {e}")
            return {"matches": []}
    
    def delete(self, ids=None, delete_all=False, namespace="default"):
        """
        Delete vectors from Pinecone
        
        Args:
            ids (list): List of IDs to delete
            delete_all (bool): Whether to delete all vectors
            namespace (str): Namespace to delete from
        
        Returns:
            bool: Success status
        """
        if not self.index:
            logger.error("Pinecone index not initialized")
            return False
        
        try:
            if delete_all:
                # Delete all vectors in the namespace
                self.index.delete(delete_all=True, namespace=namespace)
            elif ids:
                # Delete specific vectors by ID
                self.index.delete(ids=ids, namespace=namespace)
            else:
                logger.warning("No IDs provided and delete_all is False")
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error deleting from Pinecone: {e}")
            return False
            
    def describe_index_stats(self):
        """
        Get statistics about the index
        
        Returns:
            dict: Index statistics
        """
        if not self.index:
            logger.error("Pinecone index not initialized")
            return {"error": "Pinecone index not initialized"}
        
        try:
            # Get statistics about the index
            response = self.index.describe_index_stats()
            return response.to_dict() if hasattr(response, 'to_dict') else response
            
        except Exception as e:
            logger.error(f"Error getting index statistics: {e}")
            return {"error": str(e)}
