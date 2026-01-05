# config/models.py
from transformers import pipeline
import torch

# Detect device
device = 0 if torch.cuda.is_available() else -1

# ----------------------------
# Object Detection
# ----------------------------
object_detector = pipeline(
    "object-detection",
    model="facebook/detr-resnet-50",
    device=device
)

# ----------------------------
# Image Captioning
# ----------------------------
caption_model = pipeline(
    "image-to-text",
    model="Salesforce/blip-image-captioning-base",
    device=device
)

# ----------------------------
# Text-to-Speech (Python 3.12 compatible)
# Using Hugging Face TTS
# ----------------------------
tts_model = pipeline(
    "text-to-speech",
    model="espnet/kan-bayashi_ljspeech_tts_train_parallel_wavegan.v1"
)
