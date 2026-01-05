# schemas/request.py
from pydantic import BaseModel

class AudioRequest(BaseModel):
    # For now we are using UploadFile, but we can extend here later
    pass
