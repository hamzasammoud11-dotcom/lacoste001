"""
Image Encoder - BiomedCLIP Integration
=======================================

Encodes biological images (microscopy, gels, spectra) into 512-dim vectors
using BiomedCLIP (CLIP trained on biomedical image-text pairs).

Model: microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
Source: https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
Paper: https://arxiv.org/abs/2303.00915

Output: 512-dimensional embeddings (ViT-B/16 architecture)
Note: OBMEncoder projects these to 768-dim for unified cross-modal search

This encoder enables:
- Image similarity search
- Cross-modal search (image Γåö text/molecule/protein)
- Image-based experimental result retrieval
"""

import logging
import numpy as np
from typing import List, Union, Optional
from PIL import Image
import io
import base64
import os

from bioflow.core import BioEncoder, Modality, EmbeddingResult

logger = logging.getLogger(__name__)

try:
    import open_clip
    import torch
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logger.warning("open_clip not available. Install with: pip install open_clip_torch")


class ImageEncoder(BioEncoder):
    """
    BiomedCLIP image encoder for biological images.
    
    Supports:
    - Microscopy images (cells, tissues)
    - Gel electrophoresis images
    - Spectroscopy images
    - X-ray crystallography
    
    Output: 512-dimensional embeddings aligned with biomedical text
    Note: When used via OBMEncoder, automatically projected to 768-dim
    
    Example:
        >>> from PIL import Image
        >>> encoder = ImageEncoder()
        >>> img = Image.open("gel_image.jpg")
        >>> result = encoder.encode(img, Modality.IMAGE)
        >>> print(len(result.vector))  # 512
    """
    
    def __init__(
        self,
        model_name: str = "biomedclip",
        device: str = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize BiomedCLIP encoder.
        
        Args:
            model_name: Model identifier (default: biomedclip)
            device: torch device (auto-detected if None)
            cache_dir: Cache directory for model weights
        """
        if not CLIP_AVAILABLE:
            raise ImportError(
                "open_clip_torch required for image encoding. "
                "Install: pip install open_clip_torch"
            )
        
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        
        logger.info(f"ImageEncoder initialized (device={self.device})")
    
    def _lazy_load(self):
        """Load model on first use."""
        if self._model is not None:
            return
        
        logger.info("Loading BiomedCLIP model...")
        
        try:
            # Load BiomedCLIP from HuggingFace Hub
            # Model: microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
            model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
                'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
                cache_dir=self.cache_dir
            )
            
            tokenizer = open_clip.get_tokenizer(
                'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
            )
            
            model.to(self.device)
            model.eval()
            
            self._model = model
            self._preprocess = preprocess_val  # Use validation preprocessing
            self._tokenizer = tokenizer
            
            logger.info("Γ£à BiomedCLIP model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load BiomedCLIP: {e}")
            raise RuntimeError(
                f"Could not load BiomedCLIP model: {e}\n"
                "Make sure open_clip_torch is installed: pip install open_clip_torch"
            )
    
    def encode(
        self,
        images: Union[str, bytes, Image.Image, List[Union[str, bytes, Image.Image]]],
        modality: Modality = Modality.IMAGE
    ) -> EmbeddingResult:
        """
        Encode images to embeddings.
        
        Args:
            images: Single or list of:
                - File path (str)
                - Raw bytes
                - PIL Image
                - Base64 string (with 'data:image/' prefix)
            modality: Should be Modality.IMAGE
        
        Returns:
            EmbeddingResult with 768-dim vector
        """
        self._lazy_load()
        
        # Normalize to list for batch processing
        if not isinstance(images, list):
            images = [images]
            single_input = True
        else:
            single_input = False
        
        # Encode all images
        results = []
        
        with torch.no_grad():
            for img_input in images:
                try:
                    # Convert to PIL Image
                    pil_image = self._to_pil_image(img_input)
                    
                    # Preprocess
                    image_tensor = self._preprocess(pil_image).unsqueeze(0).to(self.device)
                    
                    # Encode
                    image_features = self._model.encode_image(image_tensor)
                    
                    # Normalize (L2 normalization)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    
                    # Convert to numpy
                    embedding = image_features.cpu().numpy()[0]
                    
                    results.append(EmbeddingResult(
                        vector=embedding.tolist(),
                        modality=Modality.IMAGE,
                        dimension=len(embedding),
                        metadata={
                            "model": "BiomedCLIP",
                            "image_size": pil_image.size,
                            "device": str(self.device)
                        }
                    ))
                    
                except Exception as e:
                    logger.error(f"Failed to encode image: {e}")
                    # Return zero vector on error
                    results.append(EmbeddingResult(
                        vector=[0.0] * 768,
                        modality=Modality.IMAGE,
                        dimension=768,
                        metadata={"error": str(e)}
                    ))
        
        # Return single result or first result for single input
        return results[0] if single_input else results
    
    def batch_encode(
        self,
        images: List[Union[str, bytes, Image.Image]],
        modality: Modality = Modality.IMAGE
    ) -> List[EmbeddingResult]:
        """
        Batch encode multiple images efficiently.
        
        Args:
            images: List of images (paths, bytes, or PIL Images)
            modality: Should be Modality.IMAGE
            
        Returns:
            List of EmbeddingResults
        """
        return [self.encode(img, modality) for img in images]
    
    def _to_pil_image(self, img_input: Union[str, bytes, Image.Image]) -> Image.Image:
        """Convert various input formats to PIL Image."""
        
        # Already PIL Image
        if isinstance(img_input, Image.Image):
            return img_input.convert('RGB')
        
        # Base64 string
        if isinstance(img_input, str) and img_input.startswith('data:image'):
            # Extract base64 data after comma
            base64_data = img_input.split(',')[1]
            img_bytes = base64.b64decode(base64_data)
            return Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # File path
        if isinstance(img_input, str) and os.path.exists(img_input):
            return Image.open(img_input).convert('RGB')
        
        # URL - download using requests
        if isinstance(img_input, str) and (img_input.startswith('http://') or img_input.startswith('https://')):
            import requests
            try:
                response = requests.get(img_input, timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
                response.raise_for_status()
                return Image.open(io.BytesIO(response.content)).convert('RGB')
            except Exception as e:
                raise ValueError(f"Failed to download image from URL: {e}")
        
        # Raw bytes
        if isinstance(img_input, bytes):
            return Image.open(io.BytesIO(img_input)).convert('RGB')
        
        raise ValueError(f"Unsupported image input type: {type(img_input)}")
    
    @property
    def dimension(self) -> int:
        """Output dimension (512 for BiomedCLIP ViT-B/16)."""
        return 512
    
    @property
    def supported_modalities(self) -> List[Modality]:
        """Supported modalities."""
        return [Modality.IMAGE]
