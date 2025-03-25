import os
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

# Initialize OpenAI client
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = None

if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    logger.warning("OPENAI_API_KEY not set in environment variables")
    # Client will be initialized later when the key is available

def generate_embeddings(text):
    """
    Generate embeddings for text using OpenAI's text-embedding-ada-002 model
    
    Args:
        text (str): Text to generate embeddings for
        
    Returns:
        list: Vector embeddings
    """
    global client, OPENAI_API_KEY
    
    # Check if client is available
    if client is None:
        # Try to get API key again in case it was added after initialization
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
        if OPENAI_API_KEY:
            client = OpenAI(api_key=OPENAI_API_KEY)
        else:
            logger.error("OpenAI client not initialized - API key is missing")
            raise ValueError("OpenAI API key is required for generating embeddings")
    
    try:
        # Truncate if text is too long (the model has a token limit)
        if len(text) > 8000:
            logger.warning(f"Text too long ({len(text)} chars), truncating to 8000 chars")
            text = text[:8000]
        
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        
        # Extract the embedding vector
        embedding = response.data[0].embedding
        
        return embedding
    
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        # Return a fallback embedding (all zeros) instead of raising the error
        # This allows the app to run in demo mode without API keys
        if "API key" in str(e):
            raise ValueError("OpenAI API key is invalid or missing")
        raise
