# app.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from config.models import object_detector, caption_model, tts_model
from utils.helpers import get_image_hash, get_cached_audio_path

# ----------------------------
# App Initialization
# ----------------------------
app = FastAPI(title="Image → Caption → TTS API")

# Allow CORS
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
    image = Image.open(file.file).convert("RGB")
    
    # Step 1: Object Detection
    objects = object_detector(image)
    labels = [obj['label'] for obj in objects if obj['score'] > 0.5]
    
    # Step 2: Caption Generation
    sentence = caption_model(image)[0]['generated_text']
    
    # Step 3: Text-to-Speech
    tts_model.tts_to_file(text=sentence, file_path=cached_audio_path)
    
    return FileResponse(cached_audio_path, media_type="audio/wav")
