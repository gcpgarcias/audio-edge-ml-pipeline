import requests
from pathlib import Path
import random
import time

def load_samples(num_samples=1000, classes=["yes", "no", "up", "down", "left"]):
    base_path = Path("data/raw/speech_commands")
    api_url = "http://localhost:8000/upload"
    
    total_uploaded = 0
    
    for class_name in classes:
        class_dir = base_path / class_name
        if not class_dir.exists():
            print(f"❌ Warning: {class_dir} not found")
            continue
            
        audio_files = list(class_dir.glob("*.wav"))
        print(f"\n📁 Found {len(audio_files)} files for class '{class_name}'")
        
        samples_per_class = num_samples // len(classes)
        samples = random.sample(audio_files, min(samples_per_class, len(audio_files)))
        
        print(f"⬆️  Uploading {len(samples)} samples for '{class_name}'...")
        
        successful = 0
        for i, audio_file in enumerate(samples):
            try:
                with open(audio_file, "rb") as f:
                    files = {"file": (audio_file.name, f, "audio/wav")}
                    data = {"label": class_name}
                    response = requests.post(api_url, files=files, data=data, timeout=10)
                    
                    if response.status_code == 200:
                        successful += 1
                        if (i + 1) % 50 == 0:
                            print(f"  ✓ {i + 1}/{len(samples)} uploaded")
                    else:
                        print(f"  ✗ Failed: {audio_file.name} - {response.status_code}")
                        
            except Exception as e:
                print(f"  ✗ Error uploading {audio_file.name}: {e}")
        
        print(f"✅ Uploaded {successful}/{len(samples)} for '{class_name}'")
        total_uploaded += successful
    
    print(f"\n🎉 Total uploaded: {total_uploaded}/{num_samples}")
    return total_uploaded

if __name__ == "__main__":
    print("🚀 Starting data upload...")
    print("⏰ This will take 5-10 minutes...\n")
    
    # Check API is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            print("✓ API is running\n")
        else:
            print("⚠️  API health check failed")
    except Exception as e:
        print(f"❌ Cannot reach API: {e}")
        print("Start API with: uvicorn src.ingestion.api:app --reload --port 8000")
        exit(1)
    
    start_time = time.time()
    total = load_samples(num_samples=1000)
    elapsed = time.time() - start_time
    
    print(f"\n⏱️  Completed in {elapsed:.1f} seconds")
    print(f"📊 Average: {elapsed/total:.2f}s per file")
