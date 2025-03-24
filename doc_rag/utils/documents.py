"""
Document processing utilities.
"""
import io
import re
import logging
from typing import List, Dict, Tuple, Any, Optional
import fitz
from langdetect import detect, LangDetectException

logger = logging.getLogger(__name__)

def extract_text_with_emails(page) -> str:
    """
    Extract text while preserving email addresses.
    
    Args:
        page: PyMuPDF page object.
        
    Returns:
        Extracted text with preserved emails.
    """
    # Get raw text
    text = page.get_text()
    
    # Find email addresses using regex
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    
    # Ensure emails are preserved in the text
    for email in emails:
        # Add extra spaces around email to ensure it's preserved as a separate token
        text = text.replace(email, f" {email} ")
    
    return text

def extract_text_from_pdf(pdf_data: bytes) -> Tuple[str, List[str]]:
    """
    Extract text from PDF data.
    
    Args:
        pdf_data: PDF file data as bytes.
        
    Returns:
        Tuple of (full document text, list of page texts).
    """
    try:
        pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
        doc_content = ""
        pages_content = []
        
        # Collect all text
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            text = extract_text_with_emails(page)
            doc_content += text + "\n\n"
            pages_content.append(text)
        
        pdf_document.close()
        return doc_content, pages_content
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise

def split_text_into_paragraphs(text: str) -> List[str]:
    """
    Split text into paragraphs.
    
    Args:
        text: Text to split.
        
    Returns:
        List of paragraphs.
    """
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    return paragraphs

def detect_language(text: str) -> str:
    """
    Detect language of the text.
    
    Args:
        text: Text to analyze.
        
    Returns:
        ISO 639-1 language code (e.g., 'en', 'es', 'fr').
    """
    # Dictionary of language keywords and their corresponding codes
    language_keywords = {
        'german': 'de',
        'deutsch': 'de',
        'spanish': 'es',
        'español': 'es',
        'espanol': 'es',
        'french': 'fr',
        'français': 'fr',
        'francais': 'fr',
        'italian': 'it',
        'italiano': 'it',
        'portuguese': 'pt',
        'português': 'pt',
        'portugues': 'pt',
        'dutch': 'nl',
        'nederlands': 'nl',
        'english': 'en'
    }
    
    # Check for explicit language requests
    text_lower = text.lower()
    
    # Common patterns for language requests
    patterns = [
        r'(?:reply|respond|answer|write|say|tell|speak|give|provide|write back|communicate|translate).*(?:in|using|with)\s+(\w+)',
        r'(?:in|using|with)\s+(\w+).*(?:reply|respond|answer|write|say|tell|speak|give|provide|write back|communicate|translate)',
        r'(?:translate|convert).*(?:to|into)\s+(\w+)',
        r'(\w+)\s+(?:translation|version|language)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            match_lower = match.lower()
            if match_lower in language_keywords:
                return language_keywords[match_lower]
    
    # If no explicit request found, use langdetect
    try:
        return detect(text)
    except LangDetectException:
        return 'en'  # Default to English

def get_language_instruction(lang_code: str) -> str:
    """
    Get language instruction for the LLM.
    
    Args:
        lang_code: ISO 639-1 language code.
        
    Returns:
        Language instruction string.
    """
    language_map = {
        'es': 'You must respond in Spanish. Format your entire response in Spanish.',
        'fr': 'You must respond in French. Format your entire response in French.',
        'de': 'You must respond in German. Format your entire response in German.',
        'it': 'You must respond in Italian. Format your entire response in Italian.',
        'pt': 'You must respond in Portuguese. Format your entire response in Portuguese.',
        'nl': 'You must respond in Dutch. Format your entire response in Dutch.',
        'en': 'You must respond in English. Format your entire response in English.'
    }
    return language_map.get(lang_code, 'You must respond in English. Format your entire response in English.')