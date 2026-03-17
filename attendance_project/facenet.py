import logging
from pathlib import Path
from keras_facenet import FaceNet
import pickle
import os

logger = logging.getLogger(__name__)

# Cache directory for models
CACHE_DIR = Path.home() / ".keras" / "models"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Global FaceNet model instance (loaded once)
_facenet_instance = None

def get_facenet_model():
    """
    Get FaceNet model instance with caching.
    Downloads model only on first call, then reuses cached version.
    
    Returns:
        FaceNet: Cached FaceNet model instance
    """
    global _facenet_instance
    
    if _facenet_instance is None:
        try:
            logger.info("Loading FaceNet model from cache...")
            _facenet_instance = FaceNet()
            logger.info("FaceNet model loaded successfully (cached)")
        except Exception as e:
            logger.error(f"Error loading FaceNet model: {str(e)}")
            raise
    
    return _facenet_instance

def get_embedding(face_img):
    """
    Get face embedding from FaceNet model (cached).
    
    Args:
        face_img: Face image array (160x160)
        
    Returns:
        embedding: Face embedding vector
    """
    try:
        model = get_facenet_model()
        embedding = model.embeddings([face_img])[0]
        return embedding
    except Exception as e:
        logger.error(f"Error getting embedding: {str(e)}")
        raise

def clear_cache():
    """Clear the cached model instance (optional)"""
    global _facenet_instance
    _facenet_instance = None
    logger.info("FaceNet model cache cleared")