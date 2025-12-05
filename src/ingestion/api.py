from fastapi import FastAPI, File, UploadFile, Form
from pathlib import Path
import uuid
from datetime import datetime
import json

app = FastAPI(title="Audio Ingestion API")

UPLOAD_DIR = Path("data/raw/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.post("/upload")
async def upload_audio(
    file: UploadFile = File(...),
    label: str = Form(...)
):
    # Generate unique identifier
    file_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    # Save audio file
    file_ext = Path(file.filename).suffix
    audio_path = UPLOAD_DIR / f"{file_id}{file_ext}"
    
    with audio_path.open("wb") as f:
        content = await file.read()
        f.write(content)
    
    # Save metadata
    metadata = {
        "file_id": file_id,
        "filename": file.filename,
        "label": label,
        "timestamp": timestamp,
        "path": str(audio_path)
    }
    
    metadata_path = UPLOAD_DIR / f"{file_id}.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)
    
    return {"status": "success", "file_id": file_id, "metadata": metadata}

@app.get("/health")
async def health():
    return {"status": "healthy"}
