# app.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from config.models import object_detector, caption_model, tts_model
from utils.helpers import get_image_hash, get_cached_audio_path

# ----------------------------
# App Initialization
# ----------------------------
app = FastAPI(title="Image → Caption → TTS API (Python 3.12)")

# Allow CORS for frontend testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# API Endpoint
# ----------------------------
@app.post("/generate-audio/")
async def generate_audio(file: UploadFile = File(...)):
    # Read image bytes
    image_bytes = await file.read()
    
    # Check cache
    image_hash = get_image_hash(image_bytes)
    cached_audio_path = get_cached_audio_path(image_hash)
    if os.path.exists(cached_audio_path):
        return FileResponse(cached_audio_path, media_type="audio/wav")

    # Open image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Step 1: Object Detection
    objects = object_detector(image)
    labels = [obj['label'] for obj in objects if obj['score'] > 0.5]
    
    # Step 2: Caption Generation
    sentence = caption_model(image)[0]['generated_text']
    
    # Optional: Append detected objects to sentence for clarity
    if labels:
        sentence += f" Detected objects: {', '.join(labels)}."
    
    # Step 3: Text-to-Speech using Hugging Face TTS
    tts_output = tts_model(sentence)
    audio_bytes = tts_output["wav"]
    
    # Save audio to cache
    with open(cached_audio_path, "wb") as f:
        f.write(audio_bytes)
    
    return FileResponse(cached_audio_path, media_type="audio/wav")
