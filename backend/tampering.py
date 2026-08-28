"""
Document Metadata and EXIF Inspector.
Extracts raw creation/modification metadata, PDF producer/creator tags, and image EXIF
to provide the agent with document provenance context.
"""

import logging
from pathlib import Path
from typing import Dict, Any
from PIL import Image
from PIL.ExifTags import TAGS

logger = logging.getLogger(__name__)

def inspect_document_tampering(document_path: str) -> Dict[str, Any]:
    """
    Extracts raw metadata and EXIF data from PDF or image documents.
    Provides complete provenance data to the LLM agent.
    """
    path = Path(document_path)
    if not path.exists():
        return {"error": "File not found"}
        
    ext = path.suffix.lower()
    meta_info: Dict[str, Any] = {
        "file_format": ext.lstrip("."),
        "file_size_bytes": path.stat().st_size
    }
    
    # 1. PDF Metadata Extraction
    if ext == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            meta = reader.metadata or {}
            raw_pdf_meta = {}
            for k, v in meta.items():
                clean_k = str(k).lstrip("/")
                raw_pdf_meta[clean_k] = str(v)
            meta_info["pdf_metadata"] = raw_pdf_meta
            meta_info["num_pages"] = len(reader.pages)
        except Exception as e:
            logger.debug(f"Failed to extract PDF metadata: {e}")
            
    # 2. Image EXIF & Metadata Extraction (JPEG, PNG, TIFF, WebP)
    elif ext in [".jpg", ".jpeg", ".png", ".tiff", ".webp"]:
        try:
            with Image.open(str(path)) as img:
                meta_info["dimensions"] = {"width": img.width, "height": img.height}
                meta_info["color_mode"] = img.mode
                
                exif_data = img.getexif()
                if exif_data:
                    exif_dict = {}
                    for tag_id, value in exif_data.items():
                        tag_name = TAGS.get(tag_id, str(tag_id))
                        exif_dict[tag_name] = str(value)
                    if exif_dict:
                        meta_info["exif"] = exif_dict
                        
                # Capture PNG text info chunks if present
                if ext == ".png" and hasattr(img, "text") and img.text:
                    meta_info["png_text_chunks"] = {str(k): str(v) for k, v in img.text.items()}
        except Exception as e:
            logger.debug(f"Failed to extract image EXIF: {e}")
            
    return meta_info
