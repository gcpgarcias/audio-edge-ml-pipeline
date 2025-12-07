import tensorflow as tf
import numpy as np
from pathlib import Path
import json
import time
import random
import requests
from datetime import datetime
import threading
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EdgeDeviceSimulator:
    def __init__(
        self,
        device_id: str,
        model_path: Path,
        data_dir: Path,
        api_url: str = "http://localhost:8000",
        confidence_threshold: float = 0.7,
        inference_interval: float = 2.0
    ):
        self.device_id = device_id
        self.model_path = model_path
        self.data_dir = Path(data_dir)
        self.api_url = api_url
        self.confidence_threshold = confidence_threshold
        self.inference_interval = inference_interval
        
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Check quantization
        input_scale, input_zero_point = self.input_details[0]['quantization']
        self.is_quantized = input_scale != 0.0
        self.input_scale = input_scale
        self.input_zero_point = input_zero_point
        
        # Load class names
        class_names_path = Path("data/models/class_names.json")
        with open(class_names_path) as f:
            self.class_names = json.load(f)
        
        # Statistics
        self.stats = {
            "total_inferences": 0,
            "positive_detections": 0,
            "data_transmissions": 0,
            "avg_confidence": 0.0,
            "avg_inference_time_ms": 0.0
        }
        
        self.inference_times = []
        self.confidences = []
        
        logger.info(f"Device {device_id} initialized")
        logger.info(f"Model: {model_path.name}")
        logger.info(f"Quantized: {self.is_quantized}")
        logger.info(f"Classes: {self.class_names}")
    
    def load_random_sample(self):
        """Load a random audio sample for inference."""
        metadata_files = list(self.data_dir.glob("*.json"))
        
        if not metadata_files:
            return None, None, None
        
        metadata_path = random.choice(metadata_files)
        
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        processed_path = Path(metadata.get("processed_path"))
        if not processed_path.exists():
            return None, None, None
        
        # Load spectrogram
        spectrogram = np.load(processed_path)
        label = metadata["label"]
        audio_path = metadata.get("path")
        
        return spectrogram, label, audio_path
    
    def run_inference(self, spectrogram):
        """Run inference on spectrogram."""
        # Prepare input
        input_data = np.expand_dims(spectrogram, axis=0)
        input_data = np.expand_dims(input_data, axis=-1).astype(np.float32)
        
        if self.is_quantized:
            # Quantize input
            input_data = input_data / self.input_scale + self.input_zero_point
            input_data = input_data.astype(self.input_details[0]['dtype'])
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        start_time = time.time()
        self.interpreter.invoke()
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Get output
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        if self.is_quantized:
            # Dequantize output
            output_scale, output_zero_point = self.output_details[0]['quantization']
            output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
        
        # Get prediction
        probabilities = tf.nn.softmax(output_data[0]).numpy()
        predicted_idx = np.argmax(probabilities)
        confidence = probabilities[predicted_idx]
        predicted_class = self.class_names[predicted_idx]
        
        return predicted_class, confidence, inference_time, probabilities
    
    def send_telemetry(self, prediction, confidence, inference_time, true_label):
        """Send inference telemetry to server."""
        telemetry = {
            "device_id": self.device_id,
            "timestamp": datetime.now().isoformat(),
            "prediction": prediction,
            "confidence": float(confidence),
            "inference_time_ms": float(inference_time),
            "true_label": true_label,
            "correct": prediction == true_label
        }
        
        try:
            # In a real system, this would go to MQTT or a telemetry endpoint
            # For now, we'll just save to a local file
            telemetry_dir = Path("data/telemetry")
            telemetry_dir.mkdir(parents=True, exist_ok=True)
            
            telemetry_file = telemetry_dir / f"{self.device_id}_telemetry.jsonl"
            with open(telemetry_file, "a") as f:
                f.write(json.dumps(telemetry) + "\n")
                
        except Exception as e:
            logger.error(f"Failed to send telemetry: {e}")
    
    def send_audio_to_server(self, audio_path, prediction, confidence):
        """Simulate sending audio file to server when positive detection occurs."""
        try:
            if not Path(audio_path).exists():
                logger.warning(f"Audio file not found: {audio_path}")
                return False
            
            with open(audio_path, 'rb') as f:
                files = {'file': (Path(audio_path).name, f, 'audio/wav')}
                data = {
                    'label': prediction,
                    'device_id': self.device_id,
                    'confidence': confidence
                }
                
                # Note: In real deployment, this would handle network errors,
                # retries, and local buffering for offline scenarios
                response = requests.post(
                    f"{self.api_url}/upload",
                    files=files,
                    data=data,
                    timeout=5
                )
                
                if response.status_code == 200:
                    logger.info(f"[{self.device_id}] Audio uploaded successfully")
                    self.stats["data_transmissions"] += 1
                    return True
                else:
                    logger.error(f"Upload failed: {response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to upload audio: {e}")
            return False
    
    def run_continuous_inference(self, duration_seconds: int = 300, save_stats: bool = True):
        """Run continuous inference simulation for specified duration."""
        logger.info(f"[{self.device_id}] Starting continuous inference for {duration_seconds}s")
        
        start_time = time.time()
        
        while (time.time() - start_time) < duration_seconds:
            # Load random sample
            spectrogram, true_label, audio_path = self.load_random_sample()
            
            if spectrogram is None:
                logger.warning(f"[{self.device_id}] No samples available")
                time.sleep(self.inference_interval)
                continue
            
            # Run inference
            prediction, confidence, inference_time, probabilities = self.run_inference(spectrogram)
            
            # Update statistics
            self.stats["total_inferences"] += 1
            self.inference_times.append(inference_time)
            self.confidences.append(confidence)
            self.stats["avg_inference_time_ms"] = np.mean(self.inference_times)
            self.stats["avg_confidence"] = np.mean(self.confidences)
            
            # Send telemetry
            self.send_telemetry(prediction, confidence, inference_time, true_label)

            if self.stats["total_inferences"] % 10 == 0:
                logger.info(f"[{self.device_id}] Processed {self.stats['total_inferences']} inferences")
            
            # Check if positive detection
            is_positive = confidence >= self.confidence_threshold
            
            if is_positive:
                self.stats["positive_detections"] += 1
                logger.info(
                    f"[{self.device_id}] POSITIVE: {prediction} "
                    f"(confidence: {confidence:.3f}, true: {true_label})"
                )
                
                # Simulate sending audio to server
                if audio_path:
                    self.send_audio_to_server(audio_path, prediction, confidence)
            else:
                logger.debug(
                    f"[{self.device_id}] inference: {prediction} "
                    f"(confidence: {confidence:.3f}, true: {true_label})"
                )
            
            # Wait before next inference
            time.sleep(self.inference_interval)
        
        # Save final statistics
        if save_stats:
            self.save_statistics()
        
        logger.info(f"[{self.device_id}] Simulation complete")
        return self.stats
    
    def save_statistics(self):
        """Save device statistics to file."""
        stats_dir = Path("data/device_stats")
        stats_dir.mkdir(parents=True, exist_ok=True)
        
        stats_file = stats_dir / f"{self.device_id}_stats.json"
        
        detailed_stats = {
            **self.stats,
            "device_id": self.device_id,
            "model_path": str(self.model_path),
            "confidence_threshold": self.confidence_threshold,
            "timestamp": datetime.now().isoformat(),
            "positive_rate": float(self.stats["positive_detections"] / max(self.stats["total_inferences"], 1)),
            "inference_times_p50": float(np.percentile(self.inference_times, 50)) if self.inference_times else 0.0,
            "inference_times_p95": float(np.percentile(self.inference_times, 95)) if self.inference_times else 0.0,
            "inference_times_p99": float(np.percentile(self.inference_times, 99)) if self.inference_times else 0.0,
        }
        
        # Convert all numpy types to native Python types
        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Apply conversion to all values in stats
        detailed_stats = {k: convert_to_native(v) for k, v in detailed_stats.items()}
        
        with open(stats_file, "w") as f:
            json.dump(detailed_stats, f, indent=2)
        
        logger.info(f"Statistics saved to {stats_file}")

def simulate_device_fleet(
    num_devices: int = 3,
    model_path: str = "data/models/audio_classifier_int8.tflite",
    data_dir: str = "data/processed/spectrograms",
    duration_seconds: int = 60,
    api_url: str = "http://localhost:8000"
):
    """Simulate multiple edge devices running in parallel."""
    logger.info(f"Simulating fleet of {num_devices} devices for {duration_seconds}s")
    
    devices = []
    threads = []
    
    # Create devices
    for i in range(num_devices):
        device_id = f"device_{i+1:03d}"
        # Vary confidence thresholds to simulate different deployment scenarios
        confidence_threshold = 0.25 + (i * 0.02)
        # Vary inference intervals
        inference_interval = 1.5 + (i * 0.5)
        
        device = EdgeDeviceSimulator(
            device_id=device_id,
            model_path=Path(model_path),
            data_dir=Path(data_dir),
            api_url=api_url,
            confidence_threshold=confidence_threshold,
            inference_interval=inference_interval
        )
        devices.append(device)
    
    # Run devices in parallel threads
    for device in devices:
        thread = threading.Thread(
            target=device.run_continuous_inference,
            args=(duration_seconds,),
            daemon=True
        )
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Aggregate statistics
    logger.info("\n" + "="*60)
    logger.info("FLEET STATISTICS")
    logger.info("="*60)
    
    total_inferences = sum(d.stats["total_inferences"] for d in devices)
    total_detections = sum(d.stats["positive_detections"] for d in devices)
    total_transmissions = sum(d.stats["data_transmissions"] for d in devices)
    avg_inference_time = np.mean([d.stats["avg_inference_time_ms"] for d in devices])
    
    logger.info(f"Total devices: {num_devices}")
    logger.info(f"Total inferences: {total_inferences}")
    logger.info(f"Total positive detections: {total_detections}")
    logger.info(f"Total data transmissions: {total_transmissions}")
    logger.info(f"Average inference time: {avg_inference_time:.2f} ms")
    logger.info(f"Detection rate: {total_detections/max(total_inferences, 1):.2%}")
    
    for device in devices:
        logger.info(f"\n{device.device_id}:")
        logger.info(f"  Inferences: {device.stats['total_inferences']}")
        logger.info(f"  Detections: {device.stats['positive_detections']}")
        logger.info(f"  Avg confidence: {device.stats['avg_confidence']:.3f}")
        logger.info(f"  Avg inference: {device.stats['avg_inference_time_ms']:.2f} ms")
    
    logger.info("="*60)
    
    return devices

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Simulate edge device deployment")
    parser.add_argument("--num-devices", type=int, default=3, help="Number of devices to simulate")
    parser.add_argument("--model-path", type=str, default="data/models/audio_classifier_int8.tflite")
    parser.add_argument("--data-dir", type=str, default="data/processed/spectrograms")
    parser.add_argument("--duration", type=int, default=60, help="Simulation duration in seconds")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000")
    
    args = parser.parse_args()
    
    # Ensure API is running
    try:
        response = requests.get(f"{args.api_url}/health", timeout=2)
        if response.status_code != 200:
            logger.warning("API health check failed, continuing anyway...")
    except Exception as e:
        logger.warning(f"Could not reach API at {args.api_url}: {e}")
        logger.info("Make sure to start the API: uvicorn src.ingestion.api:app --reload")
    
    # Run simulation
    devices = simulate_device_fleet(
        num_devices=args.num_devices,
        model_path=args.model_path,
        data_dir=args.data_dir,
        duration_seconds=args.duration,
        api_url=args.api_url
    )

if __name__ == "__main__":
    main()
