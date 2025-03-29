import os
import logging
import json
import re
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

def generate_comprehensive_metadata(text, filename="", file_ext="", max_tags=8, max_retries=3):
    """
    Generate comprehensive metadata and tags for document chunks using OpenAI's GPT-4o model
    
    Args:
        text (str): Text content to analyze
        filename (str): Original filename
        file_ext (str): File extension/type
        max_tags (int): Maximum number of general tags to generate
        max_retries (int): Maximum number of retry attempts for API calls
        
    Returns:
        dict: Comprehensive metadata including various tag categories
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
            return {}
    
    import time
    
    # Define default metadata structure with safe values
    default_metadata = {
        "general_tags": [],
        "subject": "general",
        "subcategory": "document",
        "skill_level": "all levels",
        "chunk_summary": "Document without detailed analysis",
        "key_points": "• Document content",
        "chunk_type": "document",
        "concepts_covered": "",
        "prerequisites": "",
        "next_steps": [],
        "learning_objective": "Understanding document content",
        "application_context": "general knowledge",
        "education_use_case": "reading",
        "question_tags": []
    }
    
    # Clean and prepare text
    if not text or text.strip() == "":
        logger.warning("Empty text provided for metadata generation")
        default_metadata["chunk_summary"] = "Empty document"
        return default_metadata
    
    # Truncate text if it's very long
    if len(text) > 15000:
        logger.warning(f"Text too long ({len(text)} chars), truncating to 15000 chars for metadata generation")
        text = text[:15000]
        
        system_message = f"""
        You are an expert document analyzer and metadata tagger. Analyze the provided text and extract detailed, structured metadata.
        Create appropriate tags for each of the following categories:
        
        1. subject - Core domain/topic (e.g., finance, investing, real estate, psychology)
        2. subcategory - More specific topic (e.g., ETFs, technical analysis, habit formation)
        3. skill_level - Learning level (e.g., beginner, intermediate, advanced)
        4. chunk_summary - Short summary (1-2 sentences) of the content
        5. key_points - List of 3-5 important ideas/facts in the content
        6. chunk_type - Type of content (definition, how-to, case study, example, tip, principle)
        7. concepts_covered - List of key concepts, terms, or formulas mentioned
        8. prerequisites - Topics or knowledge needed to understand this content
        9. next_steps - Logical follow-up topics or learning suggestions
        10. learning_objective - What the user is expected to learn from this content
        11. application_context - Where or how this knowledge is applied (e.g., investing, health, education)
        12. education_use_case - Tag for learning tasks (e.g., quiz_candidate, lesson_topic, glossary)
        13. question_tags - Potential quiz or flashcard prompts (1-3 questions)
        14. general_tags - List of {max_tags} general tags/keywords that describe the content

        Format your response as a JSON object with each category as a key. Use arrays for lists and keep the format clean and consistent.
        If a category is not applicable, use an empty array or appropriate default.
        """
        
        # Prepare conversation for OpenAI
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Filename: {filename}\nFile type: {file_ext}\n\nContent for analysis: {text}"}
        ]
        
        # Retry logic for API calls
        retries = 0
        last_error = None
        
        while retries <= max_retries:
            try:
                # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
                # do not change this unless explicitly requested by the user
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.3,
                    max_tokens=1000,
                    response_format={"type": "json_object"}
                )
                
                metadata_response = response.choices[0].message.content
                
                # Parse JSON response
                try:
                    metadata = json.loads(metadata_response)
                    # Add source filename if provided
                    if filename:
                        metadata["source_filename"] = filename
                    
                    # Convert lists to string for compatibility with existing code where needed
                    if isinstance(metadata.get("key_points"), list):
                        metadata["key_points"] = "\n".join([f"• {point}" for point in metadata["key_points"]])
                        
                    if isinstance(metadata.get("concepts_covered"), list):
                        metadata["concepts_covered"] = ", ".join(metadata["concepts_covered"])
                        
                    if isinstance(metadata.get("prerequisites"), list):
                        metadata["prerequisites"] = ", ".join(metadata["prerequisites"])
                        
                    logger.debug(f"Generated comprehensive metadata with {len(metadata)} categories")
                    return metadata
                    
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON response for metadata: {metadata_response}")
                    retries += 1
                    if retries > max_retries:
                        logger.error("Max retries reached for JSON parsing")
                        default_metadata["error"] = "JSON parse error"
                        return default_metadata
                    else:
                        logger.info(f"Retrying after JSON parse error (attempt {retries}/{max_retries})")
                        time.sleep(1)  # Brief pause before retry
                
            except Exception as e:
                last_error = e
                retries += 1
                logger.warning(f"Error generating metadata (attempt {retries}/{max_retries}): {e}")
                
                # If it's an API key issue, don't retry
                if "API key" in str(e):
                    logger.error("OpenAI API key is invalid or missing")
                    default_metadata["error"] = "API key invalid"
                    return default_metadata
                    
                # If we've reached max retries, return default metadata
                if retries > max_retries:
                    logger.error(f"Failed to generate metadata after {max_retries} attempts: {e}")
                    default_metadata["error"] = str(e)
                    return default_metadata
                    
                # Wait with exponential backoff before retrying
                wait_time = 2 ** (retries - 1)
                logger.info(f"Waiting {wait_time}s before retrying metadata generation...")
                time.sleep(wait_time)

def generate_tags(text, max_tags=8, filename="", file_ext=""):
    """
    Generate relevant tags for a text using OpenAI's GPT-4o model
    
    Args:
        text (str): Text to generate tags for
        max_tags (int): Maximum number of tags to generate
        filename (str): Original filename, if available
        file_ext (str): File extension/type, if available
        
    Returns:
        list: List of tags
    """
    # For backward compatibility, use the comprehensive function but return just the tags
    metadata = generate_comprehensive_metadata(
        text=text,
        filename=filename,
        file_ext=file_ext,
        max_tags=max_tags
    )
    
    # Return general tags if available, otherwise empty list
    return metadata.get("general_tags", [])

def generate_streaming_chat_response(query, search_results, max_retries=2):
    """
    Generate streaming context-aware responses using OpenAI's GPT-4o model
    
    Args:
        query (str): User's question
        search_results (dict): Results from Pinecone query
        max_retries (int): Maximum number of retry attempts for API calls
        
    Yields:
        dict: Streaming chunks of the response
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
            yield {
                "answer": "<p>Error: OpenAI API key missing</p>",
                "status": "error"
            }
            return
    
    import time
    
    try:
        # Extract context from search results
        contexts = []
        for match in search_results.get('matches', []):
            if match.get('metadata') and match['metadata'].get('text'):
                contexts.append(match['metadata']['text'])
        
        # If no contexts found, use a generic response
        if not contexts:
            contexts = ["No specific information found in the documents."]
        
        # Combine contexts
        combined_context = "\n\n---\n\n".join(contexts)
        
        # Create system message with instructions
        system_message = """
        You are a witty, friendly educational assistant that explains complex concepts with clarity and humor.
        Use metaphors and analogies to make concepts more understandable. Be conversational but educational.
        Format your response in HTML. Identify key concepts that could be useful for further learning,
        and format them as clickable buttons using HTML <button class="keyword">Concept</button> tags.
        
        Include 2-3 follow-up questions at the end of your response as buttons with the class "follow-up-question".
        
        For example: <button class="follow-up-question">Tell me more about X?</button>
        
        Format the response as follows:
        1. Main answer with metaphors, analogies, and highlighted <button class="keyword">keywords</button>
        2. A brief summary section (in a <div class="summary"></div>)
        3. Follow-up questions (as buttons with class "follow-up-question")
        """
        
        # Prepare conversation for OpenAI
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Question: {query}\n\nContext from documents: {combined_context}"}
        ]
        
        # Retry logic
        retries = 0
        last_error = None
        full_response = ""
        
        while retries <= max_retries:
            try:
                # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
                # do not change this unless explicitly requested by the user
                stream = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1200,
                    stream=True
                )
                
                # Initial chunk with empty content
                yield {
                    "token": "",
                    "status": "streaming",
                    "full_answer": "",
                    "done": False
                }
                
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        token = chunk.choices[0].delta.content
                        full_response += token
                        
                        # Send the token and the full response so far
                        yield {
                            "token": token,
                            "status": "streaming",
                            "full_answer": full_response,
                            "done": False
                        }
                
                # Stream complete - extract keywords and follow-up questions
                keywords = re.findall(r'<button class="keyword">(.*?)</button>', full_response)
                follow_up_questions = re.findall(r'<button class="follow-up-question">(.*?)</button>', full_response)
                
                # Final chunk with complete data
                yield {
                    "token": "",
                    "status": "complete",
                    "full_answer": full_response,
                    "keywords": keywords,
                    "follow_up_questions": follow_up_questions,
                    "done": True
                }
                
                # Break out of retry loop
                break
                
            except Exception as e:
                last_error = e
                retries += 1
                logger.warning(f"Error in streaming response (attempt {retries}/{max_retries}): {e}")
                
                # If it's an API key issue, don't retry
                if "API key" in str(e):
                    logger.error("OpenAI API key is invalid or missing")
                    yield {
                        "answer": "<p>I apologize, but the OpenAI API key is invalid or missing. Please check your configuration.</p>",
                        "status": "error",
                        "done": True
                    }
                    return
                
                # If we've reached max retries, return error response
                if retries > max_retries:
                    logger.error(f"Failed to generate streaming response after {max_retries} attempts: {e}")
                    yield {
                        "answer": f"<p>I apologize, but I encountered an error after multiple attempts: {str(e)}</p>",
                        "status": "error",
                        "done": True
                    }
                    return
                
                # Wait with exponential backoff before retrying
                wait_time = 2 ** (retries - 1)
                logger.info(f"Waiting {wait_time}s before retrying streaming response generation...")
                time.sleep(wait_time)
                
    except Exception as e:
        logger.error(f"Error in streaming chat response: {e}")
        yield {
            "answer": f"<p>I apologize, but I encountered an error: {str(e)}</p>",
            "status": "error",
            "done": True
        }

def generate_chat_response(query, search_results, max_retries=3):
    """
    Generate context-aware responses using OpenAI's GPT-4o model
    
    Args:
        query (str): User's question
        search_results (dict): Results from Pinecone query
        max_retries (int): Maximum number of retry attempts for API calls
        
    Returns:
        dict: Response with answer and metadata
    """
    try:
        # Extract context from search results
        contexts = []
        for match in search_results.get('matches', []):
            if match.get('metadata') and match['metadata'].get('text'):
                contexts.append(match['metadata']['text'])
        
        # If no contexts found, use a generic response
        if not contexts:
            contexts = ["No specific information found in the documents."]
        
        # Combine contexts
        combined_context = "\n\n---\n\n".join(contexts)
        
        # Create system message with instructions
        system_message = """
        You are a witty, friendly educational assistant that explains complex concepts with clarity and humor.
        Use metaphors and analogies to make concepts more understandable. Be conversational but educational.
        Format your response in HTML. Identify key concepts that could be useful for further learning,
        and format them as clickable buttons using HTML <button class="keyword">Concept</button> tags.
        
        Include 2-3 follow-up questions at the end of your response as buttons with the class "follow-up-question".
        
        For example: <button class="follow-up-question">Tell me more about X?</button>
        
        Format the response as follows:
        1. Main answer with metaphors, analogies, and highlighted <button class="keyword">keywords</button>
        2. A brief summary section (in a <div class="summary"></div>)
        3. Follow-up questions (as buttons with class "follow-up-question")
        """
        
        # Prepare conversation for OpenAI
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Question: {query}\n\nContext from documents: {combined_context}"}
        ]
        
        import time
        
        # Retry logic for API calls
        retries = 0
        last_error = None
        
        while retries <= max_retries:
            try:
                # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
                # do not change this unless explicitly requested by the user
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1200
                )
                
                answer_html = response.choices[0].message.content
                
                # Extract keywords and follow-up questions using regex
                keywords = re.findall(r'<button class="keyword">(.*?)</button>', answer_html)
                follow_up_questions = re.findall(r'<button class="follow-up-question">(.*?)</button>', answer_html)
                
                return {
                    "answer": answer_html,
                    "keywords": keywords,
                    "follow_up_questions": follow_up_questions
                }
                
            except Exception as e:
                last_error = e
                retries += 1
                logger.warning(f"Error generating chat response (attempt {retries}/{max_retries}): {e}")
                
                # If it's an API key issue, don't retry
                if "API key" in str(e):
                    logger.error("OpenAI API key is invalid or missing")
                    return {
                        "answer": "<p>I apologize, but the OpenAI API key is invalid or missing. Please check your configuration.</p>",
                        "keywords": [],
                        "follow_up_questions": []
                    }
                
                # If we've reached max retries, return error response
                if retries > max_retries:
                    logger.error(f"Failed to generate chat response after {max_retries} attempts: {e}")
                    return {
                        "answer": f"<p>I apologize, but I encountered an error after multiple attempts: {str(e)}</p>",
                        "keywords": [],
                        "follow_up_questions": []
                    }
                
                # Wait with exponential backoff before retrying
                wait_time = 2 ** (retries - 1)
                logger.info(f"Waiting {wait_time}s before retrying chat response generation...")
                time.sleep(wait_time)
    
    except Exception as e:
        logger.error(f"Error generating chat response: {e}")
        return {
            "answer": f"<p>I apologize, but I encountered an error: {str(e)}</p>",
            "keywords": [],
            "follow_up_questions": []
        }
