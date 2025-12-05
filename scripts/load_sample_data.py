import requests
from pathlib import Path
import random

def load_samples(num_samples=100, classes=["yes", "no", "up", "down", "left"]):
    base_path = Path("data/raw/speech_commands")
    api_url = "http://localhost:8000/upload"
    
    for class_name in classes:
        class_dir = base_path / class_name
        if not class_dir.exists():
            continue
            
        audio_files = list(class_dir.glob("*.wav"))
        samples = random.sample(audio_files, min(num_samples // len(classes), len(audio_files)))
        
        for audio_file in samples:
            with open(audio_file, "rb") as f:
                files = {"file": (audio_file.name, f, "audio/wav")}
                data = {"label": class_name}
                response = requests.post(api_url, files=files, data=data)
                print(f"Uploaded {audio_file.name}: {response.json()}")

if __name__ == "__main__":
    load_samples()
