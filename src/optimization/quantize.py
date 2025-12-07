import tensorflow as tf
import numpy as np
from pathlib import Path
import json
import mlflow
import mlflow.tensorflow
from datetime import datetime
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelOptimizer:
    def __init__(self, model_path: str, test_data_dir: str):
        self.model_path = Path(model_path)
        self.test_data_dir = Path(test_data_dir)
        self.model = None
        self.test_data = None
        
    def load_model(self):
        """Load the trained Keras model."""
        logger.info(f"Loading model from {self.model_path}")
        self.model = tf.keras.models.load_model(self.model_path)
        return self.model
    
    def load_test_data(self, num_samples: int = 100):
        """Load test spectrograms for calibration and evaluation."""
        logger.info(f"Loading test data from {self.test_data_dir}")
        
        metadata_files = list(self.test_data_dir.glob("*.json"))[:num_samples]
        spectrograms = []
        labels = []
        
        for metadata_path in metadata_files:
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            processed_path = Path(metadata.get("processed_path"))
            if not processed_path.exists():
                continue
            
            spectrogram = np.load(processed_path)
            spectrograms.append(spectrogram)
            labels.append(metadata["label"])
        
        # Add channel dimension
        self.test_data = np.array(spectrograms)[..., np.newaxis].astype(np.float32)
        self.test_labels = labels
        
        logger.info(f"Loaded {len(self.test_data)} test samples")
        return self.test_data
    
    def representative_dataset_gen(self):
        """Generator for representative dataset used in quantization calibration."""
        for sample in self.test_data:
            yield [np.expand_dims(sample, axis=0)]
    
    def convert_to_tflite(
        self,
        output_path: Path,
        quantize: bool = True,
        optimization_mode: str = "full"
    ):
        """
        Convert Keras model to TFLite format with optional quantization.
        
        Args:
            output_path: Path to save the TFLite model
            quantize: Whether to apply quantization
            optimization_mode: 'none', 'dynamic', 'float16', or 'full' (int8)
        """
        logger.info(f"Converting model to TFLite (mode: {optimization_mode})")
        start_time = time.time()
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        if quantize:
            if optimization_mode == "dynamic":
                # Dynamic range quantization (weights only)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                
            elif optimization_mode == "float16":
                # Float16 quantization
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
                
            elif optimization_mode == "full":
                # Full integer quantization (weights and activations)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.representative_dataset = self.representative_dataset_gen
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
        
        # Convert model
        tflite_model = converter.convert()
        conversion_time = time.time() - start_time
        
        # Save TFLite model
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        model_size = output_path.stat().st_size
        logger.info(f"TFLite model saved to {output_path}")
        logger.info(f"Model size: {model_size / 1024:.2f} KB")
        logger.info(f"Conversion time: {conversion_time:.2f} seconds")
        
        return tflite_model, model_size, conversion_time
    
    def evaluate_tflite_model(self, tflite_model_path: Path, num_samples: int = None):
        """Evaluate TFLite model accuracy and inference speed."""
        logger.info(f"Evaluating TFLite model: {tflite_model_path}")
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Check if model uses int8 quantization
        input_scale, input_zero_point = input_details[0]['quantization']
        output_scale, output_zero_point = output_details[0]['quantization']
        is_quantized = input_scale != 0.0
        
        logger.info(f"Model quantization: {'INT8' if is_quantized else 'FLOAT32'}")
        
        # Evaluate on test data
        if num_samples is None:
            num_samples = len(self.test_data)
        
        correct = 0
        inference_times = []
        
        # Load class names for comparison
        class_names_path = Path("data/models/class_names.json")
        with open(class_names_path) as f:
            class_names = json.load(f)
        
        label_to_idx = {label: idx for idx, label in enumerate(class_names)}
        
        for i in range(min(num_samples, len(self.test_data))):
            # Prepare input
            input_data = np.expand_dims(self.test_data[i], axis=0)
            
            if is_quantized:
                # Quantize input
                input_data = input_data / input_scale + input_zero_point
                input_data = input_data.astype(input_details[0]['dtype'])
            
            # Run inference
            interpreter.set_tensor(input_details[0]['index'], input_data)
            
            start_time = time.time()
            interpreter.invoke()
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Get output
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            if is_quantized:
                # Dequantize output
                output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
            
            # Get prediction
            predicted_idx = np.argmax(output_data[0])
            true_idx = label_to_idx[self.test_labels[i]]
            
            if predicted_idx == true_idx:
                correct += 1
        
        accuracy = correct / num_samples
        avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms
        std_inference_time = np.std(inference_times) * 1000
        
        logger.info(f"Accuracy: {accuracy:.4f} ({correct}/{num_samples})")
        logger.info(f"Avg inference time: {avg_inference_time:.2f} ms (±{std_inference_time:.2f} ms)")
        
        return {
            "accuracy": accuracy,
            "avg_inference_time_ms": avg_inference_time,
            "std_inference_time_ms": std_inference_time,
            "is_quantized": is_quantized
        }
    
    def compare_models(self, original_model_path: Path, tflite_model_path: Path):
        """Compare original and optimized models."""
        logger.info("\n" + "="*60)
        logger.info("MODEL COMPARISON")
        logger.info("="*60)
        
        # Original model evaluation
        logger.info("\nEvaluating original Keras model...")
        original_model = tf.keras.models.load_model(original_model_path)
        
        # Prepare test data for Keras evaluation
        test_data = self.test_data
        
        # Load class names
        class_names_path = Path("data/models/class_names.json")
        with open(class_names_path) as f:
            class_names = json.load(f)
        label_to_idx = {label: idx for idx, label in enumerate(class_names)}
        test_labels_idx = np.array([label_to_idx[label] for label in self.test_labels])
        
        # Original model inference
        original_times = []
        for sample in test_data[:100]:
            start = time.time()
            _ = original_model.predict(np.expand_dims(sample, axis=0), verbose=0)
            original_times.append(time.time() - start)
        
        original_predictions = original_model.predict(test_data, verbose=0)
        original_accuracy = np.mean(np.argmax(original_predictions, axis=1) == test_labels_idx)
        original_inference_time = np.mean(original_times) * 1000
        
        # TFLite model evaluation
        logger.info("\nEvaluating TFLite model...")
        tflite_results = self.evaluate_tflite_model(tflite_model_path, num_samples=100)
        
        # Size comparison
        original_size = original_model_path.stat().st_size / 1024
        tflite_size = tflite_model_path.stat().st_size / 1024
        size_reduction = ((original_size - tflite_size) / original_size) * 100
        
        comparison = {
            "original_model": {
                "size_kb": original_size,
                "accuracy": float(original_accuracy),
                "avg_inference_time_ms": original_inference_time
            },
            "tflite_model": {
                "size_kb": tflite_size,
                "accuracy": tflite_results["accuracy"],
                "avg_inference_time_ms": tflite_results["avg_inference_time_ms"],
                "is_quantized": tflite_results["is_quantized"]
            },
            "improvements": {
                "size_reduction_percent": size_reduction,
                "size_reduction_ratio": f"{original_size/tflite_size:.1f}x",
                "accuracy_loss_percent": (original_accuracy - tflite_results["accuracy"]) * 100,
                "speedup_ratio": f"{original_inference_time/tflite_results['avg_inference_time_ms']:.1f}x"
            }
        }
        
        # Print comparison
        logger.info("\nORIGINAL KERAS MODEL:")
        logger.info(f"  Size: {original_size:.2f} KB")
        logger.info(f"  Accuracy: {original_accuracy:.4f}")
        logger.info(f"  Inference time: {original_inference_time:.2f} ms")
        
        logger.info("\nOPTIMIZED TFLITE MODEL:")
        logger.info(f"  Size: {tflite_size:.2f} KB")
        logger.info(f"  Accuracy: {tflite_results['accuracy']:.4f}")
        logger.info(f"  Inference time: {tflite_results['avg_inference_time_ms']:.2f} ms")
        logger.info(f"  Quantization: {'INT8' if tflite_results['is_quantized'] else 'FLOAT32'}")
        
        logger.info("\nIMPROVEMENTS:")
        logger.info(f"  Size reduction: {size_reduction:.1f}% ({comparison['improvements']['size_reduction_ratio']})")
        logger.info(f"  Accuracy loss: {comparison['improvements']['accuracy_loss_percent']:.2f}%")
        logger.info(f"  Speed improvement: {comparison['improvements']['speedup_ratio']}")
        logger.info("="*60 + "\n")
        
        return comparison

def optimize_model_with_mlflow(
    model_path: str = "data/models/audio_classifier.h5",
    data_dir: str = "data/processed/spectrograms",
    experiment_name: str = "model-optimization"
):
    """Main optimization function with MLflow tracking."""
    
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Initialize optimizer
        optimizer = ModelOptimizer(model_path, data_dir)
        
        # Load model and test data
        model = optimizer.load_model()
        test_data = optimizer.load_test_data(num_samples=100)
        
        # Log original model info
        original_size = Path(model_path).stat().st_size
        mlflow.log_param("original_model_size_bytes", original_size)
        mlflow.log_param("original_model_size_kb", original_size / 1024)
        
        # Test different optimization strategies
        optimization_modes = [
            ("float32", False, "none"),
            ("dynamic", True, "dynamic"),
            ("float16", True, "float16"),
            ("int8", True, "full")
        ]
        
        results = {}
        
        for mode_name, quantize, opt_mode in optimization_modes:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing optimization mode: {mode_name}")
            logger.info(f"{'='*60}")
            
            output_path = Path(f"data/models/audio_classifier_{mode_name}.tflite")
            
            # Convert model
            tflite_model, model_size, conversion_time = optimizer.convert_to_tflite(
                output_path,
                quantize=quantize,
                optimization_mode=opt_mode
            )
            
            # Evaluate model
            eval_results = optimizer.evaluate_tflite_model(output_path, num_samples=100)
            
            # Calculate metrics
            size_reduction = ((original_size - model_size) / original_size) * 100
            compression_ratio = original_size / model_size
            
            # Log to MLflow
            mlflow.log_metric(f"{mode_name}_model_size_kb", model_size / 1024)
            mlflow.log_metric(f"{mode_name}_size_reduction_percent", size_reduction)
            mlflow.log_metric(f"{mode_name}_compression_ratio", compression_ratio)
            mlflow.log_metric(f"{mode_name}_accuracy", eval_results["accuracy"])
            mlflow.log_metric(f"{mode_name}_inference_time_ms", eval_results["avg_inference_time_ms"])
            mlflow.log_metric(f"{mode_name}_conversion_time_seconds", conversion_time)
            
            # Log model artifact
            mlflow.log_artifact(str(output_path))
            
            results[mode_name] = {
                "model_path": str(output_path),
                "size_kb": model_size / 1024,
                "size_reduction_percent": size_reduction,
                "compression_ratio": compression_ratio,
                "accuracy": eval_results["accuracy"],
                "inference_time_ms": eval_results["avg_inference_time_ms"],
                "conversion_time_seconds": conversion_time,
                "is_quantized": eval_results["is_quantized"]
            }
        
        # Save results summary
        results_path = Path("data/models/optimization_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        mlflow.log_artifact(str(results_path))
        
        # Compare best quantized model with original
        logger.info("\n" + "="*60)
        logger.info("FINAL COMPARISON: Original vs INT8 Quantized")
        logger.info("="*60)
        
        comparison = optimizer.compare_models(
            Path(model_path),
            Path(f"data/models/audio_classifier_int8.tflite")
        )
        
        comparison_path = Path("data/models/model_comparison.json")
        with open(comparison_path, "w") as f:
            json.dump(comparison, f, indent=2)
        
        mlflow.log_artifact(str(comparison_path))
        
        # Log best model metrics
        mlflow.log_metric("best_size_reduction_percent", 
                         comparison["improvements"]["size_reduction_percent"])
        mlflow.log_metric("best_accuracy_loss_percent",
                         comparison["improvements"]["accuracy_loss_percent"])
        
        logger.info("\nOptimization complete! Check MLflow UI for detailed metrics.")
        
        return results, comparison

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize trained model for edge deployment")
    parser.add_argument("--model-path", type=str, default="data/models/audio_classifier.h5")
    parser.add_argument("--data-dir", type=str, default="data/processed/spectrograms")
    parser.add_argument("--experiment-name", type=str, default="model-optimization")
    
    args = parser.parse_args()
    
    results, comparison = optimize_model_with_mlflow(
        model_path=args.model_path,
        data_dir=args.data_dir,
        experiment_name=args.experiment_name
    )
    
    print("\n" + "="*60)
    print("OPTIMIZATION SUMMARY")
    print("="*60)
    for mode, metrics in results.items():
        print(f"\n{mode.upper()}:")
        print(f"  Size: {metrics['size_kb']:.2f} KB ({metrics['size_reduction_percent']:.1f}% reduction)")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Inference: {metrics['inference_time_ms']:.2f} ms")

if __name__ == "__main__":
    main()
