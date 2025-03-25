import os
import logging
import tempfile
import uuid
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import json

# Import utility modules
from utils.extract_text import extract_text_from_file, chunk_text
from utils.embedding import generate_embeddings
from utils.pinecone_manager import PineconeManager
from utils.chat import generate_chat_response
import utils.embedding as embedding_module
import utils.chat as chat_module

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev_secret_key")

# Configuration
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'pdf', 'epub', 'txt', 'mobi', 'azw', 'azw3'}
MAX_CONTENT_LENGTH = 20 * 1024 * 1024  # 20MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Check for API keys and initialize services
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV", "")

# API keys status
openai_available = OPENAI_API_KEY is not None
pinecone_available = PINECONE_API_KEY is not None

# Initialize Pinecone if API key is available
pinecone_manager = None
if pinecone_available:
    try:
        pinecone_manager = PineconeManager(
            api_key=PINECONE_API_KEY,
            environment=PINECONE_ENV
        )
        logger.info("Pinecone initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {e}")
        pinecone_available = False

# Initialize OpenAI clients if API key is available
if openai_available and OPENAI_API_KEY:
    embedding_module.client = embedding_module.OpenAI(api_key=OPENAI_API_KEY)
    chat_module.client = chat_module.OpenAI(api_key=OPENAI_API_KEY)
    logger.info("OpenAI clients initialized successfully")

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('upload.html', 
                          openai_available=openai_available,
                          pinecone_available=pinecone_available)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file uploads, extract text, and store in Pinecone"""
    # Check if required API keys are available
    if not openai_available:
        flash("OpenAI API key is required for document processing. Please set the OPENAI_API_KEY environment variable.", "danger")
        return redirect(url_for('index'))
    
    if not pinecone_available:
        flash("Pinecone API key is required for document storage. Please set the PINECONE_API_KEY environment variable.", "danger")
        return redirect(url_for('index'))
    
    if 'file' not in request.files:
        flash('No file part', 'danger')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(request.url)
    
    if not allowed_file(file.filename):
        flash(f'File type not allowed. Supported formats: {", ".join(ALLOWED_EXTENSIONS)}', 'danger')
        return redirect(request.url)
    
    try:
        # Generate a unique filename
        orig_filename = secure_filename(file.filename)
        file_ext = orig_filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}.{file_ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save file to temp directory
        file.save(filepath)
        logger.debug(f"File saved to {filepath}")
        
        # Extract text from file
        extracted_text = extract_text_from_file(filepath, file_ext)
        if not extracted_text:
            flash("Unable to extract text from file. Please check if the file is valid.", "danger")
            return redirect(request.url)
        
        # Get metadata from form
        subject = request.form.get('subject', 'general')
        tags = request.form.get('tags', '')
        tags_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
        
        # Chunk the text
        chunks = chunk_text(extracted_text)
        logger.debug(f"Created {len(chunks)} chunks from the text")
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            try:
                # Generate embeddings
                embedding = generate_embeddings(chunk)
                
                # Store in Pinecone with metadata
                metadata = {
                    'text': chunk,
                    'filename': orig_filename,
                    'filetype': file_ext,
                    'subject': subject,
                    'tags': tags_list,
                    'chunk_id': i,
                    'source': 'user_upload',
                    'user_id': session.get('user_id', 'anonymous')
                }
                
                pinecone_manager.upsert(
                    id=f"{uuid.uuid4()}",
                    vector=embedding,
                    metadata=metadata
                )
            except Exception as chunk_error:
                logger.error(f"Error processing chunk {i}: {chunk_error}")
                # Continue with other chunks even if one fails
                continue
        
        # Clean up temp file
        os.remove(filepath)
        
        flash(f"File '{orig_filename}' successfully processed and stored!", "success")
        return redirect(url_for('index'))
        
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        flash(f"Error processing file: {str(e)}", "danger")
        return redirect(request.url)

@app.route('/chat', methods=['POST'])
def chat():
    """Process chat messages and return AI responses"""
    # Check if required API keys are available
    if not openai_available:
        error_message = "OpenAI API key is required for chat. Please set the OPENAI_API_KEY environment variable."
        return jsonify({
            'answer': f"<p class='text-danger'>{error_message}</p>",
            'keywords': [],
            'follow_up_questions': []
        }), 400
    
    if not pinecone_available:
        error_message = "Pinecone API key is required for document retrieval. Please set the PINECONE_API_KEY environment variable."
        return jsonify({
            'answer': f"<p class='text-danger'>{error_message}</p>",
            'keywords': [],
            'follow_up_questions': []
        }), 400
        
    try:
        data = request.get_json()
        query = data.get('message', '')
        
        if not query:
            return jsonify({'error': 'No message provided'}), 400
        
        # Generate embedding for the query
        query_embedding = generate_embeddings(query)
        
        # Query Pinecone for relevant chunks
        results = pinecone_manager.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )
        
        # Generate response using OpenAI
        response = generate_chat_response(query, results)
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({
            'answer': f"<p class='text-danger'>Error: {str(e)}</p>",
            'keywords': [],
            'follow_up_questions': []
        }), 500

@app.errorhandler(413)
def too_large(e):
    flash(f"File too large. Maximum size is {MAX_CONTENT_LENGTH/(1024*1024)}MB", "danger")
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
