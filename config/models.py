# config/models.py
from transformers import pipeline
import torch
import coqui_tts

# Detect device
device = 0 if torch.cuda.is_available() else -1

# ----------------------------
# Load Object Detection Model
# ----------------------------
object_detector = pipeline(
    "object-detection",
    model="facebook/detr-resnet-50",
    device=device
)

# ----------------------------
# Load Caption Model
# ----------------------------
caption_model = pipeline(
    "image-to-text",
    model="Salesforce/blip-image-captioning-base",
    device=device
)

# ----------------------------
# Load Text-to-Speech Model
# ----------------------------
tts_model = coqui_tts.TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
