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

def generate_embeddings(text, max_retries=3):
    """
    Generate embeddings for text using OpenAI's text-embedding-ada-002 model
    
    Args:
        text (str): Text to generate embeddings for
        max_retries (int): Maximum number of retry attempts for API calls
        
    Returns:
        list: Vector embeddings
    """
    import time
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
    
    # Clean and prepare text
    if not text or text.strip() == "":
        logger.warning("Empty text provided for embedding generation, using placeholder")
        text = "empty_document_placeholder"
    
    # Truncate if text is too long (the model has a token limit)
    if len(text) > 8000:
        logger.warning(f"Text too long ({len(text)} chars), truncating to 8000 chars")
        text = text[:8000]
    
    # Retry logic
    retries = 0
    last_error = None
    
    while retries <= max_retries:
        try:
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            
            # Extract the embedding vector
            embedding = response.data[0].embedding
            return embedding
            
        except Exception as e:
            last_error = e
            retries += 1
            logger.warning(f"Error generating embeddings (attempt {retries}/{max_retries}): {e}")
            
            # If it's an API key issue, raise immediately
            if "API key" in str(e):
                logger.error("OpenAI API key is invalid or missing")
                raise ValueError("OpenAI API key is invalid or missing")
                
            # If we've reached max retries, raise the error
            if retries > max_retries:
                logger.error(f"Failed to generate embeddings after {max_retries} attempts: {e}")
                raise
                
            # Wait with exponential backoff before retrying (1s, 2s, 4s, etc.)
            wait_time = 2 ** (retries - 1)
            logger.info(f"Waiting {wait_time}s before retrying...")
            time.sleep(wait_time)
