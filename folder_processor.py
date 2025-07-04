#!/usr/bin/env python3
"""
Folder Processor for AI Document Chat
-----------------------------------
This script monitors an "upload_folder" for document files, processes them
with the AI document chat system, and moves them to a "processed_folder" when done.

Features:
- Watch for new files in upload_folder
- Process files with real-time progress reporting
- Verify processing of each file
- Move successfully processed files to processed_folder
"""

import os
import time
import shutil
import logging
import uuid
import json
from datetime import datetime
from flask import flash

# Import utility modules
from utils.extract_text import extract_text_from_file, chunk_text
from utils.embedding import generate_embeddings
from utils.pinecone_manager import PineconeManager
from utils.chat import generate_comprehensive_metadata

# Initialize logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('folder_processor')

# Configuration
UPLOAD_FOLDER = "upload_folder"
PROCESSED_FOLDER = "processed_folder"
ALLOWED_EXTENSIONS = {'pdf', 'epub', 'txt', 'mobi', 'azw', 'azw3'}
BATCH_SIZE = 5  # Process this many chunks at once

# Status file to track progress
STATUS_FILE = "processing_status.json"

# Initialize API clients
import os
from utils.pinecone_manager import PineconeManager
import utils.embedding as embedding_module
import utils.chat as chat_module
from openai import OpenAI

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


def allowed_file(filename):
    """Check if file type is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_status():
    """Load processing status from the status file"""
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading status file: {e}")
            # If the file exists but is corrupted, create a new empty status file
            logger.warning("Creating a new empty status file due to corruption")
            try:
                # Backup the corrupted file
                if os.path.getsize(STATUS_FILE) > 0:
                    backup_file = f"{STATUS_FILE}.bak.{int(time.time())}"
                    shutil.copy(STATUS_FILE, backup_file)
                    logger.info(f"Backed up corrupted status file to {backup_file}")
                
                # Create a new empty file
                with open(STATUS_FILE, 'w') as f:
                    json.dump({}, f)
                logger.info("Created new empty status file")
            except Exception as backup_error:
                logger.error(f"Error creating backup or new status file: {backup_error}")
    return {}


def save_status(status):
    """Save processing status to the status file"""
    try:
        with open(STATUS_FILE, 'w') as f:
            json.dump(status, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving status file: {e}")


def update_file_status(filename, current_status, progress=0, message="", error=None):
    """Update status for a specific file with robust error handling"""
    try:
        # Load current status with a safety check
        status = load_status()
        if status is None:
            logger.warning("Status file returned None, creating new empty status dictionary")
            status = {}
        
        # Ensure we have a valid dictionary
        if not isinstance(status, dict):
            logger.warning(f"Status file contained invalid data (type: {type(status)}), creating new status dictionary")
            status = {}
        
        # Initialize status entry if it doesn't exist
        if filename not in status or not isinstance(status[filename], dict):
            status[filename] = {
                "start_time": datetime.now().isoformat(),
                "status": current_status,
                "progress": progress,
                "messages": [],
                "errors": [],
                "chunks": 0  # Track processed chunks
            }
        else:
            # Update existing status
            status[filename]["status"] = current_status
            status[filename]["progress"] = progress
            status[filename]["last_updated"] = datetime.now().isoformat()
        
        # Ensure required arrays exist
        if "messages" not in status[filename] or not isinstance(status[filename]["messages"], list):
            status[filename]["messages"] = []
            
        if "errors" not in status[filename] or not isinstance(status[filename]["errors"], list):
            status[filename]["errors"] = []
        
        # Add message if provided
        if message:
            status[filename]["messages"].append({
                "time": datetime.now().isoformat(),
                "message": message
            })
        
        # Add error if provided
        if error:
            status[filename]["errors"].append({
                "time": datetime.now().isoformat(),
                "error": str(error)
            })
        
        # Save updated status
        save_status(status)
        return status
        
    except Exception as e:
        logger.error(f"Error updating file status for {filename}: {e}")
        # Try to recover by creating a minimal status entry
        try:
            minimal_status = {
                filename: {
                    "status": current_status,
                    "progress": progress,
                    "messages": [],
                    "errors": [{"time": datetime.now().isoformat(), "error": f"Status update failed: {str(e)}"}]
                }
            }
            save_status(minimal_status)
            return minimal_status
        except Exception as recovery_error:
            logger.critical(f"Failed to recover from status update error: {recovery_error}")
            return {}


def get_pending_files():
    """Get a list of files pending processing"""
    files = []
    
    # Get all files in the upload folder
    for filename in os.listdir(UPLOAD_FOLDER):
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.isfile(filepath) and allowed_file(filename):
            files.append(filename)
    
    # Load status to check which files are already processed
    status = load_status()
    
    # Filter out files that are already processed
    # Include files with errors or that have been processing for too long (stuck)
    pending_files = []
    for filename in files:
        if filename not in status:
            # New file with no status yet
            pending_files.append(filename)
        elif status[filename]["status"] in ["error", "pending"]:
            # Files with errors or pending status
            pending_files.append(filename)
        elif status[filename]["status"] == "processing":
            # Check if processing is stuck (no progress for a long time)
            last_updated = status[filename].get("last_updated")
            if last_updated:
                try:
                    # Parse ISO format timestamp
                    last_update_time = datetime.fromisoformat(last_updated)
                    # If last update was more than 30 minutes ago, consider it stuck
                    if (datetime.now() - last_update_time).total_seconds() > 1800:  # 30 minutes
                        logger.warning(f"File {filename} seems stuck in processing (last update: {last_updated})")
                        pending_files.append(filename)
                except Exception as e:
                    logger.error(f"Error parsing timestamp for {filename}: {e}")
                    # Include the file to be safe
                    pending_files.append(filename)
    
    return pending_files


def process_file(filename):
    """Process a single file"""
    # Check if API keys are available
    if not openai_available or not pinecone_available:
        update_file_status(
            filename, 
            "error", 
            0, 
            "API keys missing. Please check OpenAI and Pinecone API keys.", 
            "API keys not configured"
        )
        return False
    
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        update_file_status(
            filename, 
            "error", 
            0, 
            f"File {filename} not found in upload folder", 
            "File not found"
        )
        return False
    
    try:
        # Update status to processing
        update_file_status(filename, "processing", 0, f"Started processing {filename}")
        
        # Get file extension
        file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ""
        
        # Extract text from file
        update_file_status(filename, "processing", 5, f"Extracting text from {filename}")
        extracted_text = extract_text_from_file(filepath, file_ext)
        
        if not extracted_text:
            update_file_status(
                filename, 
                "error", 
                0, 
                f"Unable to extract text from {filename}", 
                "Text extraction failed"
            )
            return False
        
        update_file_status(filename, "processing", 15, f"Extracted {len(extracted_text)} characters from {filename}")
        
        # Generate metadata
        update_file_status(filename, "processing", 20, f"Generating comprehensive metadata for {filename}")
        file_metadata = {}
        ai_tags = []
        
        if openai_available:
            try:
                sample_text = extracted_text[:20000]  # Use first 20k chars for metadata generation
                metadata_result = generate_comprehensive_metadata(
                    text=sample_text,
                    filename=filename,
                    file_ext=file_ext,
                    max_tags=10
                )
                
                # Safety check - ensure we have a valid dictionary
                if metadata_result and isinstance(metadata_result, dict):
                    file_metadata = metadata_result
                    
                    # Get the general tags for compatibility
                    if "general_tags" in file_metadata and isinstance(file_metadata["general_tags"], list):
                        ai_tags = file_metadata["general_tags"]
                    else:
                        ai_tags = []  # Use empty list as fallback
                        
                    logger.info(f"Generated {len(ai_tags)} AI tags for {filename}: {ai_tags}")
                    
                    # Log summary of generated metadata
                    logger.info(f"Generated metadata for {filename} with {len(file_metadata)} categories")
                    update_file_status(
                        filename, 
                        "processing", 
                        30, 
                        f"Generated metadata with {len(file_metadata)} categories and {len(ai_tags)} tags"
                    )
                else:
                    # Set safe defaults
                    logger.warning(f"Metadata generation returned invalid result for {filename}")
                    file_metadata = {
                        "subject": "general",
                        "general_tags": [],
                        "chunk_summary": "Auto-generated summary"
                    }
                    ai_tags = []
                    update_file_status(
                        filename, 
                        "processing", 
                        30, 
                        f"Warning: Using default metadata, continuing with processing"
                    )
            except Exception as e:
                logger.error(f"Error generating metadata: {e}")
                # Set safe defaults in case of error
                file_metadata = {
                    "subject": "general",
                    "general_tags": [],
                    "chunk_summary": "Auto-generated summary"
                }
                ai_tags = []
                update_file_status(
                    filename, 
                    "processing", 
                    30, 
                    f"Warning: Metadata generation failed, proceeding with default metadata", 
                    str(e)
                )
                # Continue even if metadata generation fails
        
        # Chunk the text
        update_file_status(filename, "processing", 35, f"Chunking text content")
        chunks = chunk_text(extracted_text)
        total_chunks = len(chunks)
        update_file_status(
            filename, 
            "processing", 
            40, 
            f"Chunked text into {total_chunks} segments for processing"
        )
        
        # Process in batches to avoid timeouts
        processed_chunks = 0
        batch_sizes = []
        
        for i in range(0, total_chunks, BATCH_SIZE):
            batch = chunks[i:i+BATCH_SIZE]
            batch_embeddings = []
            batch_metadata = []
            batch_ids = []
            
            update_file_status(
                filename, 
                "processing", 
                40 + int((i / total_chunks) * 50), 
                f"Processing batch {i//BATCH_SIZE + 1} of {(total_chunks + BATCH_SIZE - 1)//BATCH_SIZE}"
            )
            
            # Generate embeddings for the batch
            for j, chunk in enumerate(batch):
                chunk_index = i + j
                try:
                    # Update status for each chunk
                    update_file_status(
                        filename, 
                        "processing", 
                        40 + int(((i + j) / total_chunks) * 50), 
                        f"Generating embedding for chunk {chunk_index + 1}/{total_chunks}"
                    )
                    
                    # Generate embeddings with retry
                    embedding = generate_embeddings(chunk, max_retries=2)
                    
                    if embedding and len(embedding) > 0:
                        # Create metadata with defaults for safety
                        metadata = {
                            'text': chunk,
                            'filename': filename,
                            'filetype': file_ext,
                            'subject': 'general',  # Default subject
                            'tags': ai_tags if ai_tags else [],  # Ensure tags is a list
                            'chunk_id': chunk_index,
                            'source': 'folder_upload',
                        }
                        
                        # Add subject from metadata if available
                        if file_metadata and isinstance(file_metadata, dict):
                            if 'subject' in file_metadata:
                                metadata['subject'] = file_metadata['subject']
                            
                            # Add comprehensive AI-generated metadata if available
                            for key, value in file_metadata.items():
                                if key != 'general_tags':  # Skip general tags as they're already in the tags field
                                    metadata[key] = value
                                    
                        # Add to batch for upserting
                        batch_embeddings.append(embedding)
                        batch_metadata.append(metadata)
                        batch_ids.append(f"{uuid.uuid4()}")
                
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_index}: {e}")
                    update_file_status(
                        filename, 
                        "processing", 
                        40 + int(((i + j) / total_chunks) * 50), 
                        f"Warning: Failed to process chunk {chunk_index}", 
                        str(e)
                    )
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
                    
                    batch_sizes.append(len(vectors))
                    
                    update_file_status(
                        filename, 
                        "processing", 
                        40 + int(((i + len(batch)) / total_chunks) * 50), 
                        f"Storing batch {i//BATCH_SIZE + 1} ({len(vectors)} vectors) in Pinecone"
                    )
                    
                    # Batch upsert to Pinecone
                    if pinecone_manager and pinecone_manager.index:
                        # Get vector count before upsert
                        try:
                            before_stats = pinecone_manager.describe_index_stats()
                            before_count = before_stats.get('total_vector_count', 0)
                        except Exception as stats_error:
                            logger.warning(f"Could not get before stats: {stats_error}")
                            before_count = 0
                        
                        # Perform the upsert
                        upsert_response = pinecone_manager.index.upsert(vectors=vectors, namespace="default")
                        processed_chunks += len(vectors)
                        
                        # Update chunk count in status file
                        status = load_status()
                        if status and filename in status and isinstance(status[filename], dict):
                            status[filename]["chunks"] = processed_chunks
                            save_status(status)
                        
                        # Get vector count after upsert
                        try:
                            after_stats = pinecone_manager.describe_index_stats()
                            after_count = after_stats.get('total_vector_count', 0)
                            # Log the difference in vector count
                            vectors_added = int(after_count) - int(before_count)
                        except Exception as stats_error:
                            logger.warning(f"Could not get after stats: {stats_error}")
                            after_count = processed_chunks
                            vectors_added = len(vectors)
                        logger.info(f"Added {vectors_added} vectors to Pinecone (upsert batch size: {len(vectors)})")
                        logger.info(f"Pinecone now contains {after_count} total vectors")
                        
                        update_file_status(
                            filename, 
                            "processing", 
                            40 + int(((i + len(batch)) / total_chunks) * 50), 
                            f"Stored batch {i//BATCH_SIZE + 1} in Pinecone (added {vectors_added} vectors), total: {after_count}"
                        )
                    else:
                        raise Exception("Pinecone index not initialized")
                        
                except Exception as batch_error:
                    logger.error(f"Error upserting batch to Pinecone: {batch_error}")
                    update_file_status(
                        filename, 
                        "processing", 
                        40 + int(((i + len(batch)) / total_chunks) * 50), 
                        f"Error: Failed to store batch {i//BATCH_SIZE + 1} in Pinecone", 
                        str(batch_error)
                    )
            
            # Short delay between batches to avoid rate limiting
            time.sleep(0.5)
        
        # Verify processing
        update_file_status(filename, "verifying", 95, "Verifying document processing")
        
        if processed_chunks > 0:
            try:
                # Get current Pinecone stats
                if pinecone_manager:
                    stats = pinecone_manager.describe_index_stats()
                    vector_count = stats.get('total_vector_count', 0)
                    # Update status to include the final chunk count and vector summary
                    # First retrieve the current status to ensure we preserve the chunks field
                    current_status = load_status().get(filename, {})
                    
                    # Now update with the complete information
                    update_file_status(
                        filename, 
                        "verified", 
                        98, 
                        f"Verified storage in Pinecone. Total vectors in index: {vector_count}"
                    )
                    
                    # Make sure chunks count is stored in the status
                    status = load_status()
                    if status and filename in status:
                        if "chunks" not in status[filename] or not status[filename]["chunks"]:
                            status[filename]["chunks"] = processed_chunks
                            save_status(status)
            except Exception as e:
                logger.error(f"Error verifying Pinecone status: {e}")
                update_file_status(
                    filename, 
                    "warning", 
                    98, 
                    "Warning: Could not verify Pinecone status, but chunks were processed", 
                    str(e)
                )
        else:
            update_file_status(
                filename, 
                "error", 
                95, 
                "No chunks were successfully processed", 
                "All chunks failed processing"
            )
            return False
        
        # Move file to processed folder
        try:
            processed_path = os.path.join(PROCESSED_FOLDER, filename)
            shutil.move(filepath, processed_path)
            update_file_status(
                filename, 
                "completed", 
                100, 
                f"Successfully processed {filename} ({processed_chunks}/{total_chunks} chunks). File moved to processed folder."
            )
            logger.info(f"Moved {filename} to processed folder")
            return True
        except Exception as e:
            logger.error(f"Error moving file to processed folder: {e}")
            update_file_status(
                filename, 
                "warning", 
                99, 
                "Warning: File processed successfully but could not be moved to processed folder", 
                str(e)
            )
            return True  # Still consider it successful even if move fails
        
    except Exception as e:
        logger.error(f"Error processing file {filename}: {e}")
        update_file_status(filename, "error", 0, f"Error processing file: {str(e)}", str(e))
        return False


def run_processor(continuous=True):
    """Run the folder processor"""
    logger.info(f"Starting folder processor. Monitoring {UPLOAD_FOLDER} for files...")
    
    while True:
        try:
            # Get pending files
            pending_files = get_pending_files()
            
            if pending_files:
                logger.info(f"Found {len(pending_files)} files to process")
                
                # Process each file
                for filename in pending_files:
                    logger.info(f"Processing {filename}")
                    process_file(filename)
            
            # Check if we should continue monitoring
            if not continuous:
                break
            
            # Wait before checking again
            time.sleep(5)
            
        except KeyboardInterrupt:
            logger.info("Processor stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in processor loop: {e}")
            time.sleep(10)  # Wait a bit longer after an error


def main():
    """Main function"""
    # Ensure directories exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    
    # Check API keys
    if not openai_available:
        logger.error("OpenAI API key is required. Please set the OPENAI_API_KEY environment variable.")
    
    if not pinecone_available:
        logger.error("Pinecone API key is required. Please set the PINECONE_API_KEY environment variable.")
    
    if not openai_available or not pinecone_available:
        return
    
    # Run the processor
    run_processor(continuous=True)


if __name__ == "__main__":
    main()