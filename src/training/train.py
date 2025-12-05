def train_model(
    data_dir: str,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    filters: list = [16, 32, 64],
    dropout_rate: float = 0.3,
    experiment_name: str = "audio-classification"
):
    """Main training function with MLflow tracking."""
    
    # Start overall timing
    start_time = time.time()
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name)
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"cnn_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Log parameters
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("filters", filters)
        mlflow.log_param("dropout_rate", dropout_rate)
        mlflow.log_param("optimizer", "adam")
        
        # Load dataset
        logger.info("Loading dataset...")
        dataset_start = time.time()
        dataset_loader = SpectrogramDataset(data_dir)
        train_ds, val_ds, class_names = dataset_loader.load_dataset()
        dataset_load_time = time.time() - dataset_start
        
        mlflow.log_metric("dataset_load_time_seconds", dataset_load_time)
        logger.info(f"Dataset loaded in {dataset_load_time:.2f} seconds")
        
        mlflow.log_param("num_classes", len(class_names))
        mlflow.log_param("train_samples", len(list(train_ds)))
        
        # Prepare datasets
        train_ds = dataset_loader.prepare_for_training(train_ds, batch_size=batch_size, shuffle=True)
        val_ds = dataset_loader.prepare_for_training(val_ds, batch_size=batch_size, shuffle=False)
        
        # Get input shape from first batch
        for x, y in train_ds.take(1):
            input_shape = x.shape[1:]
        
        logger.info(f"Input shape: {input_shape}")
        logger.info(f"Classes: {class_names}")
        
        # Create model
        logger.info("Creating model...")
        model_creation_start = time.time()
        model = create_cnn_model(input_shape, len(class_names), filters, dropout_rate)
        model_creation_time = time.time() - model_creation_start
        
        mlflow.log_metric("model_creation_time_seconds", model_creation_time)
        
        # Compile model
        compilation_start = time.time()
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        compilation_time = time.time() - compilation_start
        
        mlflow.log_metric("model_compilation_time_seconds", compilation_time)
        
        # Log model summary
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        mlflow.log_text('\n'.join(model_summary), "model_summary.txt")
        
        # Count parameters
        trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
        mlflow.log_param("trainable_parameters", int(trainable_params))
        
        # Custom callback to track epoch timing
        class TimingCallback(keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.epoch_times = []
                self.epoch_start_time = None
            
            def on_epoch_begin(self, epoch, logs=None):
                self.epoch_start_time = time.time()
            
            def on_epoch_end(self, epoch, logs=None):
                epoch_time = time.time() - self.epoch_start_time
                self.epoch_times.append(epoch_time)
                mlflow.log_metric("epoch_time_seconds", epoch_time, step=epoch)
                logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f} seconds")
        
        timing_callback = TimingCallback()
        
        # Callbacks
        callbacks = [
            timing_callback,
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            ),
            keras.callbacks.TensorBoard(
                log_dir=f"logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        ]
        
        # Train model
        logger.info("Starting training...")
        training_start = time.time()
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        training_time = time.time() - training_start
        
        mlflow.log_metric("total_training_time_seconds", training_time)
        mlflow.log_metric("total_training_time_minutes", training_time / 60)
        
        # Calculate average epoch time
        avg_epoch_time = sum(timing_callback.epoch_times) / len(timing_callback.epoch_times)
        mlflow.log_metric("avg_epoch_time_seconds", avg_epoch_time)
        
        logger.info(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        logger.info(f"Average epoch time: {avg_epoch_time:.2f} seconds")
        
        # Log metrics
        for epoch in range(len(history.history['loss'])):
            mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
            mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
            mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
            mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
        
        # Final metrics
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        
        logger.info(f"Final training accuracy: {final_train_acc:.4f}")
        logger.info(f"Final validation accuracy: {final_val_acc:.4f}")
        
        # Save class names
        logger.info("Saving artifacts...")
        save_start = time.time()
        
        class_names_path = Path("data/models/class_names.json")
        class_names_path.parent.mkdir(parents=True, exist_ok=True)
        with open(class_names_path, "w") as f:
            json.dump(class_names, f)
        
        mlflow.log_artifact(str(class_names_path))
        
        # Log model
        model_logging_start = time.time()
        mlflow.tensorflow.log_model(
            model,
            artifact_path="model",
            registered_model_name="audio-classifier"
        )
        model_logging_time = time.time() - model_logging_start
        mlflow.log_metric("model_logging_time_seconds", model_logging_time)
        
        # Save model locally
        model_path = Path("data/models/audio_classifier.h5")
        model.save(model_path)
        
        save_time = time.time() - save_start
        mlflow.log_metric("artifact_save_time_seconds", save_time)
        
        logger.info(f"Model saved to {model_path}")
        
        # Total pipeline time
        total_time = time.time() - start_time
        mlflow.log_metric("total_pipeline_time_seconds", total_time)
        mlflow.log_metric("total_pipeline_time_minutes", total_time / 60)
        
        # Create timing summary
        timing_summary = {
            "dataset_load_time": f"{dataset_load_time:.2f}s",
            "model_creation_time": f"{model_creation_time:.2f}s",
            "model_compilation_time": f"{compilation_time:.2f}s",
            "total_training_time": f"{training_time:.2f}s ({training_time/60:.2f}m)",
            "avg_epoch_time": f"{avg_epoch_time:.2f}s",
            "model_logging_time": f"{model_logging_time:.2f}s",
            "artifact_save_time": f"{save_time:.2f}s",
            "total_pipeline_time": f"{total_time:.2f}s ({total_time/60:.2f}m)"
        }
        
        mlflow.log_dict(timing_summary, "timing_summary.json")
        
        logger.info("\n" + "="*50)
        logger.info("TIMING SUMMARY")
        logger.info("="*50)
        for key, value in timing_summary.items():
            logger.info(f"{key}: {value}")
        logger.info("="*50)
        
        return model, history, class_names
