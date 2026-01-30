"""Test ImageEncoder dimension output"""
from bioflow.plugins.encoders.image_encoder import ImageEncoder
from bioflow.core.base import Modality
import base64
from PIL import Image
import io

enc = ImageEncoder()

# Test with a simple base64 image
img = Image.new('RGB', (224, 224), color='red')
buf = io.BytesIO()
img.save(buf, format='PNG')
img_b64 = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()

result = enc.encode(img_b64, Modality.IMAGE)
print(f'Embedding dimension: {len(result.vector)}')
print(f'Result dimension attr: {result.dimension}')
