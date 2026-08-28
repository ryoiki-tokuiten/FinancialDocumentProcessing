"""
Utility functions for file handling and document loading.

Handles all Gemini-supported MIME types:
- Images: PNG, JPEG, WEBP, HEIC, HEIF
- Documents: PDF, plain text
- Video: MP4, WEBM, MOV, MPEG, FLV, WMV, 3GPP
- Audio: MP3, WAV, FLAC, AAC, OGG
"""

import mimetypes
import base64
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

# Initialize mimetypes with additional mappings
mimetypes.add_type("image/heic", ".heic")
mimetypes.add_type("image/heif", ".heif")
mimetypes.add_type("video/3gpp", ".3gp")
mimetypes.add_type("video/3gpp", ".3gpp")
mimetypes.add_type("audio/flac", ".flac")
mimetypes.add_type("audio/aac", ".aac")


# Gemini-supported MIME types organized by category
SUPPORTED_IMAGE_TYPES = {
    "image/png", "image/jpeg", "image/webp", "image/heic", "image/heif"
}

SUPPORTED_DOCUMENT_TYPES = {
    "application/pdf", "text/plain"
}

SUPPORTED_VIDEO_TYPES = {
    "video/mp4", "video/webm", "video/quicktime", "video/mpeg",
    "video/x-flv", "video/x-ms-wmv", "video/3gpp"
}

SUPPORTED_AUDIO_TYPES = {
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav",
    "audio/flac", "audio/aac", "audio/ogg"
}

# Maximum inline data size (20MB)
MAX_INLINE_SIZE = 20 * 1024 * 1024


def get_mime_type(file_path: str) -> Optional[str]:
    """
    Detect MIME type of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        MIME type string or None if unknown
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type


def is_supported_format(mime_type: str) -> bool:
    """
    Check if a MIME type is supported by Gemini.
    
    Args:
        mime_type: The MIME type to check
        
    Returns:
        True if supported, False otherwise
    """
    all_supported = (
        SUPPORTED_IMAGE_TYPES | 
        SUPPORTED_DOCUMENT_TYPES | 
        SUPPORTED_VIDEO_TYPES | 
        SUPPORTED_AUDIO_TYPES
    )
    return mime_type in all_supported


def encode_file_base64(file_path: str) -> str:
    """
    Encode a file to base64 string.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Base64 encoded string
    """
    with open(file_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def get_file_size(file_path: str) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in bytes
    """
    return os.path.getsize(file_path)


def load_document(path: str) -> List[Dict[str, Any]]:
    """
    Load document for Gemini. Handles all supported MIME types.
    
    For files > 20MB, uses the Gemini Files API for upload.
    For smaller files, uses inline base64 encoding.
    
    Args:
        path: Path to the document file
        
    Returns:
        List of content parts suitable for Gemini API
        
    Raises:
        ValueError: If file type is not supported
        FileNotFoundError: If file does not exist
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Document not found: {path}")
    
    mime_type = get_mime_type(path)
    if mime_type is None:
        raise ValueError(f"Could not determine MIME type for: {path}")
    
    if not is_supported_format(mime_type):
        raise ValueError(f"Unsupported file type: {mime_type}")
    
    file_size = get_file_size(path)
    
    # For large files, use Gemini Files API
    if file_size > MAX_INLINE_SIZE:
        return load_large_file(path, mime_type)
    
    # For plain text - read and return as text
    if mime_type == "text/plain":
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return [{"type": "text", "text": content}]

    # All other supported types are sent as inline binary content.
    if mime_type == "application/pdf" or mime_type in (
        SUPPORTED_IMAGE_TYPES | SUPPORTED_VIDEO_TYPES | SUPPORTED_AUDIO_TYPES
    ):
        encoded = encode_file_base64(path)
        return [{
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{encoded}"}
        }]
    
    raise ValueError(f"Unhandled MIME type: {mime_type}")


def load_large_file(path: str, mime_type: str) -> List[Dict[str, Any]]:
    """
    Load a large file (>20MB) using the Gemini Files API.
    
    Args:
        path: Path to the file
        mime_type: MIME type of the file
        
    Returns:
        List containing file URI reference
    """
    try:
        import google.generativeai as genai
        
        # Upload file to Gemini Files API
        uploaded_file = genai.upload_file(path, mime_type=mime_type)
        
        return [{
            "type": "file_data",
            "file_data": {
                "file_uri": uploaded_file.uri,
                "mime_type": mime_type
            }
        }]
    except ImportError:
        raise ImportError(
            "google-generativeai package required for large file uploads. "
            "Install with: pip install google-generativeai"
        )


def validate_document_path(path: str) -> tuple[bool, str]:
    """
    Validate a document path for processing.
    
    Args:
        path: Path to the document
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not path:
        return False, "Document path is empty"
    
    if not os.path.exists(path):
        return False, f"File not found: {path}"
    
    if not os.path.isfile(path):
        return False, f"Path is not a file: {path}"
    
    mime_type = get_mime_type(path)
    if mime_type is None:
        return False, f"Could not determine file type: {path}"
    
    if not is_supported_format(mime_type):
        return False, f"Unsupported file type: {mime_type}"
    
    return True, ""


def get_document_info(path: str) -> Dict[str, Any]:
    """
    Get information about a document file.
    
    Args:
        path: Path to the document
        
    Returns:
        Dictionary with file information
    """
    size_bytes = get_file_size(path) if os.path.exists(path) else 0

    return {
        "path": path,
        "filename": os.path.basename(path),
        "extension": Path(path).suffix,
        "mime_type": get_mime_type(path),
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 2) if size_bytes else 0,
        "requires_upload": size_bytes > MAX_INLINE_SIZE
    }
