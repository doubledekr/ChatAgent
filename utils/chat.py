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

def generate_chat_response(query, search_results):
    """
    Generate context-aware responses using OpenAI's GPT-4o model
    
    Args:
        query (str): User's question
        search_results (dict): Results from Pinecone query
        
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
        logger.error(f"Error generating chat response: {e}")
        return {
            "answer": f"<p>I apologize, but I encountered an error: {str(e)}</p>",
            "keywords": [],
            "follow_up_questions": []
        }
