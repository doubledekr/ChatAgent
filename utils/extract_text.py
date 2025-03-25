import os
import logging
import tempfile
import subprocess
import fitz  # PyMuPDF
import ebooklib
from ebooklib import epub
import chardet
import tiktoken
import shutil
import re

logger = logging.getLogger(__name__)

def extract_text_from_file(filepath, file_ext):
    """
    Extract text content from various file formats
    
    Args:
        filepath (str): Path to the file
        file_ext (str): File extension
        
    Returns:
        str: Extracted text content
    """
    try:
        if file_ext == 'pdf':
            return extract_from_pdf(filepath)
        elif file_ext == 'epub':
            return extract_from_epub(filepath)
        elif file_ext == 'txt':
            return extract_from_txt(filepath)
        elif file_ext in ['mobi', 'azw', 'azw3']:
            return extract_from_kindle(filepath, file_ext)
        else:
            logger.error(f"Unsupported file extension: {file_ext}")
            return None
    except Exception as e:
        logger.error(f"Error extracting text from {filepath}: {e}")
        return None

def extract_from_pdf(filepath):
    """Extract text from PDF using PyMuPDF"""
    text = ""
    try:
        with fitz.open(filepath) as doc:
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        logger.error(f"Error extracting PDF: {e}")
        raise

def extract_from_epub(filepath):
    """Extract text from EPUB using ebooklib"""
    text = ""
    try:
        book = epub.read_epub(filepath)
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                content = item.get_content()
                # Simple HTML tags removal - for production use a proper HTML parser
                content = content.decode('utf-8')
                content = content.replace('<p>', '\n').replace('</p>', '\n')
                content = content.replace('<br>', '\n').replace('<br/>', '\n')
                # Remove other HTML tags
                import re
                content = re.sub(r'<[^>]+>', '', content)
                text += content
        return text
    except Exception as e:
        logger.error(f"Error extracting EPUB: {e}")
        raise

def extract_from_txt(filepath):
    """Extract text from TXT using chardet for encoding detection"""
    try:
        with open(filepath, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
        
        with open(filepath, 'r', encoding=encoding) as file:
            return file.read()
    except Exception as e:
        logger.error(f"Error extracting TXT: {e}")
        raise

def extract_from_kindle(filepath, file_ext):
    """
    Extract text from Kindle formats using Calibre's ebook-convert tool
    or simple text extraction techniques
    """
    try:
        # Try with ebook-convert (if Calibre is installed)
        temp_output = os.path.join(tempfile.gettempdir(), f"output_{os.path.basename(filepath)}.txt")
        
        try:
            # Check if ebook-convert is available
            subprocess.run(['ebook-convert', '--version'], 
                           stdout=subprocess.PIPE, 
                           stderr=subprocess.PIPE, 
                           check=True)
            
            # Use ebook-convert to convert to text
            subprocess.run([
                'ebook-convert', 
                filepath, 
                temp_output
            ], check=True)
            
            # Read the converted text file
            with open(temp_output, 'r', encoding='utf-8') as f:
                text = f.read()
                
            # Clean up temp file
            if os.path.exists(temp_output):
                os.remove(temp_output)
                
            return text
        
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("ebook-convert not available, falling back to simple text extraction")
            
            # Fallback to simple binary extraction and text search
            temp_dir = tempfile.mkdtemp()
            
            try:
                # Binary read the file
                with open(filepath, 'rb') as f:
                    content = f.read()
                
                # Extract ASCII/UTF-8 strings from binary content
                text = ""
                
                # Simple pattern to extract text (ASCII/UTF-8 sequences)
                # Look for sequences of printable ASCII characters
                printable_chars = re.findall(rb'[\x20-\x7E\n\r\t]{4,}', content)
                
                for chars in printable_chars:
                    try:
                        decoded = chars.decode('utf-8', errors='ignore')
                        # Skip strings that are likely not natural language
                        if len(decoded) > 10 and re.search(r'[.!?]', decoded):
                            text += decoded + "\n\n"
                    except UnicodeDecodeError:
                        pass
                        
                if text:
                    return text
                else:
                    logger.warning(f"Could not extract text from {filepath}")
                    return f"Unable to extract text from {os.path.basename(filepath)}. Format not supported without additional tools."
            
            finally:
                # Clean up temp directory
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
    
    except Exception as e:
        logger.error(f"Error extracting Kindle format: {e}")
        raise

def chunk_text(text, max_tokens=500):
    """
    Chunk text into segments of approximately max_tokens each
    
    Args:
        text (str): Text to be chunked
        max_tokens (int): Maximum tokens per chunk
        
    Returns:
        list: List of text chunks
    """
    try:
        # Initialize GPT tokenizer
        enc = tiktoken.encoding_for_model("gpt-4")
        
        # Tokenize text
        tokens = enc.encode(text)
        chunks = []
        
        # Check if text is short enough for a single chunk
        if len(tokens) <= max_tokens:
            return [text]
        
        # Process text into chunks
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = enc.decode(chunk_tokens)
            chunks.append(chunk_text)
        
        return chunks
    
    except Exception as e:
        logger.error(f"Error chunking text: {e}")
        # Fallback to simple character-based chunking if tokenization fails
        return [text[i:i + 2000] for i in range(0, len(text), 2000)]
