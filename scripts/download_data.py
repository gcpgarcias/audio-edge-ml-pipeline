import os
import urllib.request
import tarfile
from pathlib import Path

def download_speech_commands():
    data_dir = Path("data/raw/speech_commands")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    url = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
    tar_path = data_dir / "speech_commands.tar.gz"
    
    if not tar_path.exists():
        print(f"Downloading dataset from {url}...")
        urllib.request.urlretrieve(url, tar_path)
        print("Download complete.")
    
    print("Extracting dataset...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(data_dir)
    
    print(f"Dataset ready at {data_dir}")

if __name__ == "__main__":
    download_speech_commands()
