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
from utils.chat import generate_chat_response, generate_tags
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
MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200MB max file size

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
    """Handle file uploads (single or multiple), extract text, and store in Pinecone"""
    # Check if required API keys are available
    if not openai_available:
        flash("OpenAI API key is required for document processing. Please set the OPENAI_API_KEY environment variable.", "danger")
        return redirect(url_for('index'))
    
    if not pinecone_available:
        flash("Pinecone API key is required for document storage. Please set the PINECONE_API_KEY environment variable.", "danger")
        return redirect(url_for('index'))
    
    # Handle single file upload (traditional form submit)
    if 'file' in request.files:
        files = request.files.getlist('file')  # Get all files with 'file' name
        
        if not files or files[0].filename == '':
            flash('No files selected', 'danger')
            return redirect(request.url)
        
        # Check if all files are allowed
        invalid_files = [f.filename for f in files if not allowed_file(f.filename)]
        if invalid_files:
            flash(f'Invalid file types: {", ".join(invalid_files)}. Supported formats: {", ".join(ALLOWED_EXTENSIONS)}', 'danger')
            return redirect(request.url)
            
    # Handle AJAX bulk upload (gets handled in /upload-progress route)
    elif request.form.get('_ajax_upload') == 'true':
        return jsonify({'status': 'error', 'message': 'No files received'}), 400
    else:
        flash('No file part', 'danger')
        return redirect(request.url)
    
    try:
        # Get common metadata for all files
        subject = request.form.get('subject', 'general')
        tags = request.form.get('tags', '')
        manual_tags_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
        
        # Process multiple files
        processed_files = []
        failed_files = []
        all_ai_tags = set()  # To collect all AI tags across files
        
        for file in files:
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
                    failed_files.append(f"{orig_filename} (unable to extract text)")
                    os.remove(filepath)
                    continue
                
                # Generate AI tags if OpenAI is available
                file_ai_tags = []
                if openai_available:
                    try:
                        # Use the first chunk or a portion of text for tag generation
                        sample_text = extracted_text[:20000]  # Use first 20k chars for tag generation
                        file_ai_tags = generate_tags(sample_text, max_tags=10)
                        logger.debug(f"Generated {len(file_ai_tags)} AI tags for {orig_filename}: {file_ai_tags}")
                        all_ai_tags.update(file_ai_tags)  # Add to the set of all tags
                    except Exception as tag_error:
                        logger.error(f"Error generating AI tags for {orig_filename}: {tag_error}")
                        # Continue even if AI tagging fails
                
                # Combine manual and AI tags for this file, removing duplicates
                file_tags_list = manual_tags_list.copy()
                for ai_tag in file_ai_tags:
                    if ai_tag.lower() not in [tag.lower() for tag in file_tags_list]:
                        file_tags_list.append(ai_tag)
                
                # Chunk the text
                chunks = chunk_text(extracted_text)
                logger.debug(f"Created {len(chunks)} chunks from {orig_filename}")
                
                # Process each chunk
                chunk_count = 0
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
                            'tags': file_tags_list,
                            'chunk_id': i,
                            'source': 'user_upload',
                            'user_id': session.get('user_id', 'anonymous')
                        }
                        
                        pinecone_manager.upsert(
                            id=f"{uuid.uuid4()}",
                            vector=embedding,
                            metadata=metadata
                        )
                        chunk_count += 1
                    except Exception as chunk_error:
                        logger.error(f"Error processing chunk {i} of {orig_filename}: {chunk_error}")
                        # Continue with other chunks even if one fails
                        continue
                
                # Clean up temp file
                os.remove(filepath)
                
                # Add to successfully processed files list
                processed_files.append({
                    'filename': orig_filename,
                    'chunks': chunk_count,
                    'tags': file_tags_list
                })
                
            except Exception as file_error:
                logger.error(f"Error processing file {file.filename}: {file_error}")
                failed_files.append(f"{file.filename} ({str(file_error)})")
                # Try to clean up temp file if it exists
                try:
                    if 'filepath' in locals() and os.path.exists(filepath):
                        os.remove(filepath)
                except:
                    pass
        
        # Prepare success/error messages
        if processed_files:
            if len(processed_files) == 1:
                file_info = processed_files[0]
                success_message = f"File '{file_info['filename']}' successfully processed with {file_info['chunks']} chunks!"
                if file_info['tags']:
                    success_message += f" Tags: {', '.join(file_info['tags'])}"
                flash(success_message, "success")
            else:
                flash(f"Successfully processed {len(processed_files)} files with a total of {sum(f['chunks'] for f in processed_files)} chunks!", "success")
                if all_ai_tags:
                    flash(f"AI identified tags across all files: {', '.join(all_ai_tags)}", "info")
        
        if failed_files:
            flash(f"Failed to process {len(failed_files)} files: {', '.join(failed_files)}", "danger")
            
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

@app.route('/upload-progress', methods=['POST'])
def upload_progress():
    """Handle AJAX file uploads with progress tracking"""
    # Check if required API keys are available
    if not openai_available or not pinecone_available:
        return jsonify({
            'status': 'error',
            'message': 'API keys missing. Please check OpenAI and Pinecone API keys.'
        }), 400
    
    # Check if file is in request
    if 'file' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'No file uploaded'
        }), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({
            'status': 'error',
            'message': 'Empty filename'
        }), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            'status': 'error',
            'message': f'Invalid file type. Supported formats: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400
    
    try:
        # Generate a unique filename
        orig_filename = secure_filename(file.filename)
        file_ext = orig_filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}.{file_ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save file
        file.save(filepath)
        
        # Get metadata from request
        subject = request.form.get('subject', 'general')
        tags = request.form.get('tags', '')
        manual_tags_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
        
        # Process the file and return progress updates
        # Extract text
        extracted_text = extract_text_from_file(filepath, file_ext)
        if not extracted_text:
            os.remove(filepath)
            return jsonify({
                'status': 'error',
                'message': 'Unable to extract text from file'
            }), 400
        
        # Generate AI tags
        ai_tags = []
        if openai_available:
            try:
                sample_text = extracted_text[:20000]
                ai_tags = generate_tags(sample_text, max_tags=10)
            except Exception as e:
                logger.error(f"Error generating AI tags: {e}")
                # Continue even if AI tagging fails
        
        # Combine tags
        tags_list = manual_tags_list.copy()
        for ai_tag in ai_tags:
            if ai_tag.lower() not in [tag.lower() for tag in tags_list]:
                tags_list.append(ai_tag)
        
        # Chunk the text
        chunks = chunk_text(extracted_text)
        total_chunks = len(chunks)
        
        # Process each chunk with progress tracking
        processed_chunks = 0
        for i, chunk in enumerate(chunks):
            try:
                # Generate embeddings
                embedding = generate_embeddings(chunk)
                
                # Store in Pinecone
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
                processed_chunks += 1
                
                # For long files, send progress updates every 5 chunks or at the end
                if i == total_chunks - 1 or i % 5 == 0:
                    progress = int((processed_chunks / total_chunks) * 100)
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {e}")
                # Continue with other chunks
        
        # Clean up temp file
        os.remove(filepath)
        
        # Return success response
        return jsonify({
            'status': 'success',
            'message': 'File processed successfully',
            'filename': orig_filename,
            'chunks_processed': processed_chunks,
            'total_chunks': total_chunks,
            'tags': tags_list
        })
        
    except Exception as e:
        logger.error(f"Error in upload-progress: {e}")
        # Try to clean up the temp file if it exists
        try:
            if 'filepath' in locals() and os.path.exists(filepath):
                os.remove(filepath)
        except:
            pass
            
        return jsonify({
            'status': 'error',
            'message': f'Error processing file: {str(e)}'
        }), 500

@app.errorhandler(413)
def too_large(e):
    if request.path == '/upload-progress':
        return jsonify({
            'status': 'error',
            'message': f'File too large. Maximum size is {MAX_CONTENT_LENGTH/(1024*1024)}MB'
        }), 413
    else:
        flash(f"File too large. Maximum size is {MAX_CONTENT_LENGTH/(1024*1024)}MB", "danger")
        return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
