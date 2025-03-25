import os
import logging
import tempfile
import uuid
import shutil
import subprocess
import threading
import time
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import json

# Import utility modules
from utils.extract_text import extract_text_from_file, chunk_text
from utils.embedding import generate_embeddings
from utils.pinecone_manager import PineconeManager
from utils.chat import generate_chat_response, generate_tags, generate_comprehensive_metadata
import utils.embedding as embedding_module
import utils.chat as chat_module

# Import folder processor
import folder_processor

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
                
                # Generate comprehensive metadata if OpenAI is available
                file_ai_tags = []
                file_metadata = {}
                if openai_available:
                    try:
                        # Use the first chunk or a portion of text for metadata generation
                        sample_text = extracted_text[:20000]  # Use first 20k chars for metadata generation
                        file_metadata = generate_comprehensive_metadata(
                            text=sample_text, 
                            filename=orig_filename,
                            file_ext=file_ext,
                            max_tags=10
                        )
                        
                        # Get the general tags for backward compatibility
                        file_ai_tags = file_metadata.get("general_tags", [])
                        logger.debug(f"Generated {len(file_ai_tags)} AI tags for {orig_filename}: {file_ai_tags}")
                        all_ai_tags.update(file_ai_tags)  # Add to the set of all tags
                        
                        # Log summary of generated metadata
                        logger.debug(f"Generated metadata for {orig_filename} with {len(file_metadata)} categories")
                    except Exception as tag_error:
                        logger.error(f"Error generating metadata for {orig_filename}: {tag_error}")
                        # Continue even if metadata generation fails
                
                # Combine manual and AI tags for this file, removing duplicates
                file_tags_list = manual_tags_list.copy()
                for ai_tag in file_ai_tags:
                    if ai_tag.lower() not in [tag.lower() for tag in file_tags_list]:
                        file_tags_list.append(ai_tag)
                
                # Chunk the text
                chunks = chunk_text(extracted_text)
                logger.debug(f"Created {len(chunks)} chunks from {orig_filename}")
                
                # Process each chunk with improved timeout handling
                chunk_count = 0
                batch_size = 3  # Process chunks in smaller batches to avoid timeouts
                
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i+batch_size]
                    
                    for j, chunk in enumerate(batch):
                        chunk_index = i + j
                        try:
                            # Generate embeddings with shorter timeout
                            embedding = generate_embeddings(chunk, max_retries=2)  # Reduced retries for faster feedback
                            
                            # Start with basic metadata
                            metadata = {
                                'text': chunk,
                                'filename': orig_filename,
                                'filetype': file_ext,
                                'subject': file_metadata.get('subject', subject),  # Use AI-generated subject if available
                                'tags': file_tags_list,
                                'chunk_id': chunk_index,
                                'source': 'user_upload',
                                'user_id': session.get('user_id', 'anonymous')
                            }
                            
                            # Add comprehensive AI-generated metadata if available
                            if file_metadata:
                                # Add all detailed metadata fields
                                for key, value in file_metadata.items():
                                    if key != 'general_tags':  # Skip general tags as they're already in the tags field
                                        metadata[key] = value
                            
                            # Proceed only if we have valid embeddings
                            if embedding and len(embedding) > 0:
                                pinecone_manager.upsert(
                                    id=f"{uuid.uuid4()}",
                                    vector=embedding,
                                    metadata=metadata
                                )
                                chunk_count += 1
                            else:
                                logger.warning(f"Empty embedding for chunk {chunk_index} of {orig_filename}")
                                
                        except Exception as chunk_error:
                            # Log the error but continue with the next chunk
                            logger.error(f"Error processing chunk {chunk_index} from {orig_filename}: {chunk_error}")
                            # Consider if not counting failed chunks is appropriate for your UI
                
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
        
        # Generate embedding for the query with retry
        try:
            query_embedding = generate_embeddings(query, max_retries=2)
        except Exception as embed_error:
            logger.error(f"Error generating query embedding: {embed_error}")
            return jsonify({
                'answer': f"<p class='text-danger'>Error generating embedding: {str(embed_error)}</p>",
                'keywords': [],
                'follow_up_questions': []
            }), 500
        
        # Query Pinecone for relevant chunks
        try:
            results = pinecone_manager.query(
                vector=query_embedding,
                top_k=5,
                include_metadata=True
            )
        except Exception as query_error:
            logger.error(f"Error querying Pinecone: {query_error}")
            return jsonify({
                'answer': f"<p class='text-danger'>Error retrieving relevant documents: {str(query_error)}</p>",
                'keywords': [],
                'follow_up_questions': []
            }), 500
        
        # Generate response using OpenAI
        response = generate_chat_response(query, results, max_retries=2)
        
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
        
        # Generate comprehensive metadata
        ai_tags = []
        file_metadata = {}
        if openai_available:
            try:
                sample_text = extracted_text[:20000]  # Use first 20k chars for metadata generation
                file_metadata = generate_comprehensive_metadata(
                    text=sample_text,
                    filename=orig_filename,
                    file_ext=file_ext,
                    max_tags=10
                )
                
                # Get the general tags for backward compatibility
                ai_tags = file_metadata.get("general_tags", [])
                logger.debug(f"Generated {len(ai_tags)} AI tags for {orig_filename}: {ai_tags}")
                
                # Log summary of generated metadata
                logger.debug(f"Generated metadata for {orig_filename} with {len(file_metadata)} categories")
            except Exception as e:
                logger.error(f"Error generating metadata: {e}")
                # Continue even if metadata generation fails
        
        # Combine tags
        tags_list = manual_tags_list.copy()
        for ai_tag in ai_tags:
            if ai_tag.lower() not in [tag.lower() for tag in tags_list]:
                tags_list.append(ai_tag)
        
        # Chunk the text
        chunks = chunk_text(extracted_text)
        total_chunks = len(chunks)
        
        # Process in smaller batches to avoid timeouts
        processed_chunks = 0
        batch_size = 3  # Smaller batches to avoid timeouts
        
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i+batch_size]
            batch_embeddings = []
            batch_metadata = []
            batch_ids = []
            
            # Generate embeddings for the batch
            for j, chunk in enumerate(batch):
                chunk_index = i + j
                try:
                    # Generate embeddings with reduced retries for faster feedback
                    embedding = generate_embeddings(chunk, max_retries=1)
                    
                    if embedding and len(embedding) > 0:
                        # Create metadata
                        metadata = {
                            'text': chunk,
                            'filename': orig_filename,
                            'filetype': file_ext,
                            'subject': file_metadata.get('subject', subject),  # Use AI-generated subject if available
                            'tags': tags_list,
                            'chunk_id': chunk_index,
                            'source': 'user_upload',
                            'user_id': session.get('user_id', 'anonymous')
                        }
                        
                        # Add comprehensive AI-generated metadata if available
                        if file_metadata:
                            # Add all detailed metadata fields
                            for key, value in file_metadata.items():
                                if key != 'general_tags':  # Skip general tags as they're already in the tags field
                                    metadata[key] = value
                                    
                        # Add to batch for upserting
                        batch_embeddings.append(embedding)
                        batch_metadata.append(metadata)
                        batch_ids.append(f"{uuid.uuid4()}")
                
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_index}: {e}")
                    # Continue with other chunks
            
            # Upsert the batch to Pinecone
            if batch_embeddings:
                try:
                    # Format for Pinecone batch upsert
                    vectors = []
                    for k in range(len(batch_embeddings)):
                        vectors.append({
                            "id": batch_ids[k],
                            "values": batch_embeddings[k],
                            "metadata": batch_metadata[k]
                        })
                    
                    # Batch upsert to Pinecone
                    if pinecone_manager.index:
                        pinecone_manager.index.upsert(vectors=vectors, namespace="default")
                        processed_chunks += len(vectors)
                        
                except Exception as batch_error:
                    logger.error(f"Error upserting batch to Pinecone: {batch_error}")
            
            # Update progress
            progress = int((processed_chunks / total_chunks) * 100) if total_chunks > 0 else 100
        
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

@app.route('/check-pinecone', methods=['GET'])
def check_pinecone():
    """Check if Pinecone is properly connected"""
    try:
        # Log pinecone manager status for debugging
        logger.debug(f"Pinecone manager: {pinecone_manager}")
        logger.debug(f"Pinecone manager index: {pinecone_manager.index if pinecone_manager else None}")
        
        # Check for Pinecone availability
        if not pinecone_available or not pinecone_manager:
            return jsonify({
                "status": "error", 
                "message": "Pinecone is not available. Please check your API key and environment.",
                "api_key_set": PINECONE_API_KEY is not None,
                "env_set": PINECONE_ENV != ""
            }), 500
        
        # Get a count of vectors in the index
        if pinecone_manager.index:
            # Use stats method to check connection
            try:
                stats = pinecone_manager.describe_index_stats()
                vector_count = stats.get('total_vector_count', 0)
                return jsonify({
                    "status": "success", 
                    "message": f"Pinecone connected successfully. Index contains {vector_count} vectors.",
                    "stats": stats
                })
            except Exception as inner_e:
                logger.error(f"Error getting index stats: {inner_e}")
                return jsonify({
                    "status": "error", 
                    "message": f"Error getting index stats: {str(inner_e)}",
                    "pinecone_connected": pinecone_manager is not None
                }), 500
        else:
            return jsonify({
                "status": "error", 
                "message": "Pinecone index not initialized",
                "pinecone_connected": pinecone_manager is not None
            }), 500
    except Exception as e:
        logger.error(f"Error checking Pinecone connection: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

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

# Global variable to track the background processor thread
processor_thread = None

@app.route('/folder-monitor')
def folder_monitor():
    """Render the folder monitoring interface"""
    return render_template('folder_monitor.html',
                          openai_available=openai_available,
                          pinecone_available=pinecone_available)

@app.route('/api/upload-status')
def api_upload_status():
    """Get the status of files in upload and processed folders"""
    try:
        # Get files from upload folder
        pending_files = []
        if os.path.exists(folder_processor.UPLOAD_FOLDER):
            for filename in os.listdir(folder_processor.UPLOAD_FOLDER):
                filepath = os.path.join(folder_processor.UPLOAD_FOLDER, filename)
                if os.path.isfile(filepath) and folder_processor.allowed_file(filename):
                    pending_files.append(filename)
        
        # Get files from processed folder
        processed_files = []
        if os.path.exists(folder_processor.PROCESSED_FOLDER):
            for filename in os.listdir(folder_processor.PROCESSED_FOLDER):
                filepath = os.path.join(folder_processor.PROCESSED_FOLDER, filename)
                if os.path.isfile(filepath) and folder_processor.allowed_file(filename):
                    processed_files.append(filename)
        
        # Get processing status
        processing_status = folder_processor.load_status()
        
        return jsonify({
            'status': 'success',
            'pending_files': pending_files,
            'processed_files': processed_files,
            'processing_status': processing_status
        })
    except Exception as e:
        logger.error(f"Error getting upload status: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Error getting status: {str(e)}'
        }), 500

@app.route('/start-processing')
def start_processing():
    """Start the background processor to process files in the upload folder"""
    global processor_thread
    
    try:
        # Check if processor is already running
        if processor_thread and processor_thread.is_alive():
            return jsonify({
                'status': 'success',
                'message': 'Processor is already running'
            })
        
        # Start the processor in a background thread
        processor_thread = threading.Thread(target=folder_processor.run_processor, args=(False,))
        processor_thread.daemon = True
        processor_thread.start()
        
        return jsonify({
            'status': 'success',
            'message': 'Processing started'
        })
    except Exception as e:
        logger.error(f"Error starting processor: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Error starting processor: {str(e)}'
        }), 500

@app.route('/upload-to-folder', methods=['POST'])
def upload_to_folder():
    """Handle file upload to the processing folder"""
    # Check if file is in request
    if 'file' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('folder_monitor'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'danger')
        return redirect(url_for('folder_monitor'))
    
    if not allowed_file(file.filename):
        flash(f'Invalid file type. Supported formats: {", ".join(ALLOWED_EXTENSIONS)}', 'danger')
        return redirect(url_for('folder_monitor'))
    
    try:
        # Save file to upload folder
        filename = secure_filename(file.filename)
        filepath = os.path.join(folder_processor.UPLOAD_FOLDER, filename)
        
        # Create folder if it doesn't exist
        os.makedirs(folder_processor.UPLOAD_FOLDER, exist_ok=True)
        
        # Save the file
        file.save(filepath)
        
        # Initialize the status
        folder_processor.update_file_status(
            filename, 
            'pending', 
            0, 
            f"File uploaded and ready for processing"
        )
        
        flash(f'File "{filename}" uploaded successfully and is ready for processing', 'success')
        return redirect(url_for('folder_monitor'))
    except Exception as e:
        logger.error(f"Error uploading file to folder: {e}")
        flash(f'Error uploading file: {str(e)}', 'danger')
        return redirect(url_for('folder_monitor'))

@app.route('/process-file/<filename>')
def process_single_file(filename):
    """Process a single file from the upload folder"""
    try:
        # Check if file exists
        filepath = os.path.join(folder_processor.UPLOAD_FOLDER, filename)
        if not os.path.exists(filepath):
            return jsonify({
                'status': 'error',
                'message': f'File not found: {filename}'
            }), 404
        
        # Start processing in a background thread
        def process_and_update():
            folder_processor.process_file(filename)
        
        thread = threading.Thread(target=process_and_update)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'success',
            'message': f'Started processing {filename}'
        })
    except Exception as e:
        logger.error(f"Error processing single file: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Error: {str(e)}'
        }), 500

@app.route('/process-all-files')
def process_all_files():
    """Process all files in the upload folder"""
    return start_processing()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
