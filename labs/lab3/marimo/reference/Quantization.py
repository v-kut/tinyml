import marimo

__generated_with = "0.23.14"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Quantization Techniques for TinyML
    ## Integer, Dynamic Range, Float16, and Quantization-Aware Training

    After opening this notebook, run:

    > **Kernel → Restart Kernel → Run All Cells**

    This notebook presents a clean, end-to-end workflow for model quantization using **TensorFlow Lite** and **TensorFlow Model Optimization Toolkit (TF-MOT)**. It is written for a **local macOS environment** that matches your pinned setup:

    - Python 3.11
    - TensorFlow 2.14.1
    - Keras 2.14.0
    - TensorFlow Model Optimization 0.8.0
    - NumPy < 2

    The notebook is organized into two major parts:

    1. **Post-Training Quantization (PTQ)**
       We train a baseline CNN on MNIST and convert it into:
       - a standard float32 TFLite model
       - a fully integer-quantized INT8 model
       - a dynamic range quantized model
       - a float16-quantized model

    2. **Quantization-Aware Training (QAT)**
       We prepare a quantization-aware model, fine-tune it with a small learning rate, and then compare:
       - the QAT TFLite baseline
       - the QAT + INT8 model
       - the QAT + dynamic range model

    ## Why this notebook matters

    Quantization is one of the most important model-compression methods in TinyML. A correctly converted **float32 TFLite baseline** should be very close to the Keras baseline. If the float32 TFLite baseline is much lower, the issue is usually notebook state, preprocessing, or evaluation mismatch rather than quantization itself.
    """)
    return


@app.cell
def _():
    # Environment validation and version check
    import os
    os.environ["KERAS_BACKEND"] = "tensorflow"

    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    import tensorflow as tf

    # Mac-friendly setting for this small TinyML notebook:
    # disable the Apple Metal GPU path and run TensorFlow on CPU only.
    # Restart the kernel before running this notebook so this setting takes effect.
    try:
        tf.config.set_visible_devices([], "GPU")
        print("GPU disabled. Running TensorFlow on CPU only.")
    except RuntimeError as e:
        print("Could not disable GPU because TensorFlow devices were already initialized:", e)

    import tensorflow_model_optimization as tfmot

    try:
        import keras
        keras_version = keras.__version__
    except Exception:
        keras_version = "Not available from standalone keras package"

    print("Python version      :", sys.version.split()[0])
    print("TensorFlow version  :", tf.__version__)
    print("Keras version       :", keras_version)
    print("TF-MOT version      :", tfmot.__version__)
    print("NumPy version       :", np.__version__)
    print("Visible devices     :", tf.config.get_visible_devices())

    # Optional safety checks for the pinned environment used on your laptop.
    assert tf.__version__.startswith("2.14"), "This notebook expects TensorFlow 2.14.x."
    assert tfmot.__version__ == "0.8.0", "This notebook expects tensorflow-model-optimization==0.8.0."

    OUTPUT_DIR = Path("pruning_outputs")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    SEED = 42
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    return OUTPUT_DIR, Path, np, pd, plt, sys, tf, tfmot


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Imports, reproducibility, and experiment settings

    We keep the experiment small enough to run comfortably on a laptop, while still producing meaningful accuracy numbers for comparison.

    A few implementation choices are deliberate:

    - We use **sparse labels** instead of one-hot labels to keep the training pipeline simple.
    - We use a compact CNN that is realistic for a TinyML teaching example.
    - We use conservative training settings and early stopping so the baseline remains numerically stable before conversion.
    - The baseline training is allowed to run for more than 5 epochs, but early stopping is not allowed to terminate before 5 full epochs are completed.
    - We use a larger representative calibration set for INT8 PTQ than the broken version of the notebook.
    """)
    return


@app.cell
def _(pd):
    # Standard experiment settings
    # These settings are intentionally conservative for a stable teaching demo.
    # The earlier low TFLite/quantized accuracy was caused by stale notebook state
    # and overly long baseline/QAT training without safeguards.
    #
    # For the baseline model, EPOCHS_BASELINE is the maximum number of epochs.
    # Early stopping is allowed to stop training only after MIN_EPOCHS_BASELINE
    # full epochs have completed.
    MIN_EPOCHS_BASELINE = 5
    EPOCHS_BASELINE = 15
    EPOCHS_QAT = 3
    BATCH_SIZE = 32
    REP_DATASET_SAMPLES = 200

    pd.set_option("display.precision", 4)
    return (
        BATCH_SIZE,
        EPOCHS_BASELINE,
        EPOCHS_QAT,
        MIN_EPOCHS_BASELINE,
        REP_DATASET_SAMPLES,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Dataset preparation

    We use the **MNIST** handwritten digit dataset. It is simple enough for quick experimentation, yet expressive enough to show the effect of quantization clearly.

    ### Preprocessing steps

    1. Load MNIST from `tensorflow.keras.datasets`.
    2. Normalize pixel values to the range `[0, 1]`.
    3. Reshape images into `(28, 28, 1)` for convolution.
    4. Keep labels in integer form for sparse categorical training.
    """)
    return


@app.cell
def _(np):
    from tensorflow.keras.datasets import mnist

    # Load dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize to [0, 1] and add the channel dimension
    x_train = (x_train.astype("float32") / 255.0)[..., np.newaxis]
    x_test = (x_test.astype("float32") / 255.0)[..., np.newaxis]

    # Keep labels as integer class indices
    y_train = y_train.astype("int32")
    y_test = y_test.astype("int32")

    print("Training set:", x_train.shape, y_train.shape)
    print("Test set    :", x_test.shape, y_test.shape)
    print("Input dtype :", x_train.dtype)
    print("Label dtype :", y_train.dtype)
    return x_test, x_train, y_test, y_train


@app.cell
def _(plt, x_train, y_train):
    # Visual sanity check: show a few sample digits
    (_fig, _axes) = plt.subplots(2, 5, figsize=(10, 4))
    for (idx, _ax) in enumerate(_axes.flat):
        _ax.imshow(x_train[idx].squeeze(), cmap='gray')
        _ax.set_title(f'Label: {y_train[idx]}')
        _ax.axis('off')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Build and train the baseline CNN

    The baseline model is intentionally modest:

    - one convolution layer for feature extraction
    - one max-pooling layer for spatial downsampling
    - one hidden dense layer
    - one softmax output layer for 10-way classification

    This is a good teaching model because it is simple, fast to train, and easy to convert to TensorFlow Lite.
    """)
    return


@app.cell
def _(tf):
    from tensorflow.keras import Model, Input
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

    def create_cnn_model() -> tf.keras.Model:
        inputs = Input(shape=(28, 28, 1), name="image")
        x = Conv2D(32, (3, 3), activation="relu", name="conv_1")(inputs)
        x = MaxPooling2D((2, 2), name="pool_1")(x)
        x = Flatten(name="flatten")(x)
        x = Dense(128, activation="relu", name="dense_1")(x)
        outputs = Dense(10, activation="softmax", name="classifier")(x)
        return Model(inputs, outputs, name="mnist_cnn")

    baseline_model = create_cnn_model()
    baseline_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    baseline_model.summary()
    return (baseline_model,)


@app.cell
def _(
    BATCH_SIZE,
    EPOCHS_BASELINE,
    MIN_EPOCHS_BASELINE,
    baseline_model,
    np,
    tf,
    x_test,
    x_train,
    y_test,
    y_train,
):
    # Train the baseline model
    # We want the baseline to train for at least MIN_EPOCHS_BASELINE epochs,
    # but still stop later if validation loss starts getting worse.
    #
    # Important detail:
    # Keras EarlyStopping(start_from_epoch=...) delays both stopping AND best-weight
    # tracking. Here we use a small custom callback so the best weights are tracked
    # from epoch 1, while stopping is delayed until after the minimum epoch count.

    class EarlyStoppingAfterMinEpochs(tf.keras.callbacks.Callback):
        """Early stopping that tracks best weights from epoch 1,
        but does not allow stopping before min_epochs have completed.
        """

        def __init__(
            self,
            monitor="val_loss",
            patience=2,
            min_epochs=5,
            min_delta=0.0,
            mode="min",
            restore_best_weights=True,
        ):
            super().__init__()
            self.monitor = monitor
            self.patience = patience
            self.min_epochs = min_epochs
            self.min_delta = min_delta
            self.mode = mode
            self.restore_best_weights = restore_best_weights

            if mode == "min":
                self.best = np.inf
                self.is_improvement = lambda current, best: current < best - self.min_delta
            elif mode == "max":
                self.best = -np.inf
                self.is_improvement = lambda current, best: current > best + self.min_delta
            else:
                raise ValueError("mode must be 'min' or 'max'")

            self.wait = 0
            self.best_weights = None
            self.best_epoch = 0

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            current = logs.get(self.monitor)

            if current is None:
                return

            if self.is_improvement(current, self.best):
                self.best = current
                self.wait = 0
                self.best_epoch = epoch + 1
                if self.restore_best_weights:
                    self.best_weights = self.model.get_weights()
            else:
                self.wait += 1

            # Do not stop until the minimum number of full epochs is completed.
            if epoch + 1 < self.min_epochs:
                return

            if self.wait >= self.patience:
                self.model.stop_training = True
                if self.restore_best_weights and self.best_weights is not None:
                    self.model.set_weights(self.best_weights)
                print(
                    f"\nEarly stopping after epoch {epoch + 1}. "
                    f"Restored best weights from epoch {self.best_epoch}."
                )


    early_stop_baseline = EarlyStoppingAfterMinEpochs(
        monitor="val_loss",
        patience=2,
        min_epochs=MIN_EPOCHS_BASELINE,
        restore_best_weights=True,
    )

    history_baseline = baseline_model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=EPOCHS_BASELINE,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop_baseline],
        verbose=1,
    )

    baseline_loss, baseline_accuracy = baseline_model.evaluate(x_test, y_test, verbose=0)
    print(f"Baseline Keras test accuracy: {baseline_accuracy:.4f}")
    return baseline_accuracy, history_baseline


@app.cell
def _(history_baseline, pd, plt):
    # Plot baseline training history
    history_df = pd.DataFrame(history_baseline.history)
    (_fig, _axes) = plt.subplots(1, 2, figsize=(12, 4))
    history_df[['loss', 'val_loss']].plot(ax=_axes[0], title='Baseline Training Loss')
    _axes[0].set_xlabel('Epoch')
    _axes[0].set_ylabel('Loss')
    _axes[0].grid(True, alpha=0.3)
    history_df[['accuracy', 'val_accuracy']].plot(ax=_axes[1], title='Baseline Training Accuracy')
    _axes[1].set_xlabel('Epoch')
    _axes[1].set_ylabel('Accuracy')
    _axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Helper functions

    Before running the quantization experiments, we define a few reusable utilities:

    - a **representative dataset generator** for full INT8 calibration
    - a **TFLite conversion helper** to keep conversion code clean
    - a **TFLite evaluation helper** that works for float, float16, and int8 models
    - a simple **results table builder** for compact comparison
    """)
    return


@app.cell
def _(Path, REP_DATASET_SAMPLES, np, pd, tf, x_train):
    def representative_data_gen():
        """Representative dataset used for full integer calibration.

        The generator must yield samples with the same shape and dtype expected
        by the original float model before quantization.
        """
        for sample in x_train[:REP_DATASET_SAMPLES]:
            yield [sample[np.newaxis, ...].astype(np.float32)]

    def convert_to_tflite(model: tf.keras.Model, output_path: Path, optimizations=None, representative_dataset=None, supported_ops=None, supported_types=None, inference_input_type=None, inference_output_type=None) -> bytes:
        """Convert a Keras model to TFLite and save it to disk."""
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        if optimizations is not None:
            converter.optimizations = optimizations
        if representative_dataset is not None:
            converter.representative_dataset = representative_dataset
        if supported_ops is not None:
            converter.target_spec.supported_ops = supported_ops
        if supported_types is not None:
            converter.target_spec.supported_types = supported_types
        if inference_input_type is not None:
            converter.inference_input_type = inference_input_type
        if inference_output_type is not None:
            converter.inference_output_type = inference_output_type
        tflite_model = converter.convert()
        output_path.write_bytes(tflite_model)
        return tflite_model

    def get_model_size_kb(model_path: Path) -> float:
        return model_path.stat().st_size / 1024.0

    def _prepare_tflite_input(sample: np.ndarray, input_details: dict) -> np.ndarray:
        """Prepare one input sample for a TFLite interpreter."""
        input_dtype = input_details['dtype']
        (input_scale, input_zero_point) = input_details['quantization']
        sample = sample.astype(np.float32)
        if input_dtype in (np.int8, np.uint8):
            if input_scale == 0:
                raise ValueError('Quantized model has invalid input quantization scale 0.')
            sample = np.round(sample / input_scale + input_zero_point)
            info = np.iinfo(input_dtype)
            sample = np.clip(sample, info.min, info.max).astype(input_dtype)
        elif input_dtype == np.float16:
            sample = sample.astype(np.float16)
        else:
            sample = sample.astype(input_dtype)
        return sample

    def _dequantize_tflite_output(output: np.ndarray, output_details: dict) -> np.ndarray:
        """Dequantize output if needed. For argmax this is not strictly required,
        but keeping it explicit makes the evaluation easier to teach/debug."""
        output_dtype = output_details['dtype']
        (output_scale, output_zero_point) = output_details['quantization']
        if output_dtype in (np.int8, np.uint8) and output_scale != 0:
            output = (output.astype(np.float32) - output_zero_point) * output_scale
        return output

    def tflite_predict(model_path: Path, x_data: np.ndarray, max_samples=None) -> np.ndarray:
        """Return TFLite model outputs for x_data."""
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        input_index = input_details['index']
        output_index = output_details['index']
        n = len(x_data) if max_samples is None else min(max_samples, len(x_data))
        outputs = []
        for i in range(n):
            sample = _prepare_tflite_input(x_data[i:i + 1], input_details)
            interpreter.set_tensor(input_index, sample)
            interpreter.invoke()
            output = interpreter.get_tensor(output_index)
            output = _dequantize_tflite_output(output, output_details)
            outputs.append(output[0])
        return np.asarray(outputs)

    def evaluate_tflite_model(model_path: Path, x_data: np.ndarray, y_data: np.ndarray) -> float:
        """Evaluate a TFLite model on a classification dataset.

        This helper handles float32, float16, int8, and uint8 inputs using the
        exact dtype/scale expected by the converted TFLite model.
        """
        outputs = tflite_predict(model_path, x_data)
        predictions = np.argmax(outputs, axis=1)
        return float(np.mean(predictions == y_data[:len(predictions)]))

    def compare_keras_and_tflite(keras_model: tf.keras.Model, tflite_path: Path, x_data: np.ndarray, y_data: np.ndarray, n: int=1000) -> pd.DataFrame:
        """Sanity check: float32 TFLite should closely match Keras.

        If this comparison fails, do not trust the quantization table yet. Restart
        the kernel and run all cells in order.
        """
        n = min(n, len(x_data))
        keras_outputs = keras_model.predict(x_data[:n], verbose=0)
        tflite_outputs = tflite_predict(tflite_path, x_data, max_samples=n)
        keras_preds = np.argmax(keras_outputs, axis=1)
        tflite_preds = np.argmax(tflite_outputs, axis=1)
        summary = {'subset_samples': n, 'keras_subset_accuracy': float(np.mean(keras_preds == y_data[:n])), 'tflite_subset_accuracy': float(np.mean(tflite_preds == y_data[:n])), 'keras_tflite_prediction_agreement': float(np.mean(keras_preds == tflite_preds)), 'max_abs_probability_difference': float(np.max(np.abs(keras_outputs - tflite_outputs))), 'mean_abs_probability_difference': float(np.mean(np.abs(keras_outputs - tflite_outputs)))}
        return pd.DataFrame([summary]).round(6)

    def build_results_table(rows):
        df = pd.DataFrame(rows)
        df['size_kb'] = df['size_kb'].round(2)
        df['accuracy'] = df['accuracy'].round(4)
        df['accuracy_drop_vs_keras_baseline'] = df['accuracy_drop_vs_keras_baseline'].round(4)
        return df

    return (
        build_results_table,
        compare_keras_and_tflite,
        convert_to_tflite,
        evaluate_tflite_model,
        get_model_size_kb,
        representative_data_gen,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part I. Post-Training Quantization (PTQ)

    In PTQ, we first train a standard floating-point model and then convert it into one or more lower-precision deployment formats.

    We compare four deployment targets:

    1. **Baseline TFLite**
    2. **Full Integer Quantization (INT8)**
    3. **Dynamic Range Quantization**
    4. **Float16 Quantization**

    ### Practical interpretation

    - **Baseline TFLite** is the reference conversion with no compression-focused quantization.
    - **INT8 PTQ** is usually the most relevant choice for **microcontroller deployment**.
    - **Dynamic range quantization** is easy to apply and reduces size, but it does not fully quantize the runtime path.
    - **Float16 quantization** is often more useful on hardware that benefits from half precision, but it is generally **not the primary choice for MCUs**.
    """)
    return


@app.cell
def _(
    OUTPUT_DIR,
    baseline_model,
    convert_to_tflite,
    representative_data_gen,
    tf,
):
    # Convert the trained baseline model into several TFLite variants
    baseline_tflite_path = OUTPUT_DIR / 'model_baseline.tflite'
    int8_tflite_path = OUTPUT_DIR / 'model_integer_quant.tflite'
    dynamic_tflite_path = OUTPUT_DIR / 'model_dynamic_quant.tflite'
    float16_tflite_path = OUTPUT_DIR / 'model_float16_quant.tflite'
    _ = convert_to_tflite(model=baseline_model, output_path=baseline_tflite_path)
    # 1) Baseline TFLite
    _ = convert_to_tflite(model=baseline_model, output_path=int8_tflite_path, optimizations=[tf.lite.Optimize.DEFAULT], representative_dataset=representative_data_gen, supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8], inference_input_type=tf.int8, inference_output_type=tf.int8)
    _ = convert_to_tflite(model=baseline_model, output_path=dynamic_tflite_path, optimizations=[tf.lite.Optimize.DEFAULT])
    _ = convert_to_tflite(model=baseline_model, output_path=float16_tflite_path, optimizations=[tf.lite.Optimize.DEFAULT], supported_types=[tf.float16])
    print('Saved PTQ models:')
    for _path in [baseline_tflite_path, int8_tflite_path, dynamic_tflite_path, float16_tflite_path]:
    # 2) Full INT8 PTQ
    # 3) Dynamic range quantization
    # 4) Float16 quantization
        print(' -', _path)
    return (
        baseline_tflite_path,
        dynamic_tflite_path,
        float16_tflite_path,
        int8_tflite_path,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Sanity check: Keras baseline vs float32 TFLite baseline

    Before looking at the quantized models, confirm that the plain float32 TFLite baseline matches the Keras baseline. This is the key diagnostic check. If this table shows a large mismatch, the issue is not quantization; it is notebook state, preprocessing, or evaluation mismatch.
    """)
    return


@app.cell
def _(
    baseline_model,
    baseline_tflite_path,
    compare_keras_and_tflite,
    x_test,
    y_test,
):
    # Sanity check: the float32 TFLite model should closely match the Keras model
    keras_tflite_sanity = compare_keras_and_tflite(
        baseline_model,
        baseline_tflite_path,
        x_test,
        y_test,
        n=1000,
    )

    keras_tflite_sanity
    return


@app.cell
def _(
    baseline_accuracy,
    baseline_tflite_path,
    build_results_table,
    dynamic_tflite_path,
    evaluate_tflite_model,
    float16_tflite_path,
    get_model_size_kb,
    int8_tflite_path,
    x_test,
    y_test,
):
    # Evaluate PTQ models
    ptq_rows = []
    for (_model_name, _model_path) in [('Baseline TFLite', baseline_tflite_path), ('PTQ INT8', int8_tflite_path), ('PTQ Dynamic Range', dynamic_tflite_path), ('PTQ Float16', float16_tflite_path)]:
        _acc = evaluate_tflite_model(_model_path, x_test, y_test)
        ptq_rows.append({'model': _model_name, 'file': _model_path.name, 'size_kb': get_model_size_kb(_model_path), 'accuracy': _acc, 'accuracy_drop_vs_keras_baseline': baseline_accuracy - _acc})
    ptq_results = build_results_table(ptq_rows)
    ptq_results
    return (ptq_results,)


@app.cell
def _(baseline_accuracy, np, pd, ptq_results):
    # Add the original Keras baseline as a reference row (not a file-size comparison row)
    keras_reference = pd.DataFrame(
        [
            {
                "model": "Baseline Keras",
                "file": "-",
                "size_kb": np.nan,
                "accuracy": round(float(baseline_accuracy), 4),
                "accuracy_drop_vs_keras_baseline": 0.0,
            }
        ]
    )

    pd.concat([keras_reference, ptq_results], ignore_index=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### PTQ interpretation

    The most important first check is that **Baseline TFLite** is close to **Baseline Keras**. If those two differ substantially, the problem is not caused by quantization yet.

    When you inspect the table above, the key questions are:

    - Which quantized model is the **smallest**?
    - Which quantized model keeps the **highest accuracy**?
    - Which one is the most realistic choice for a **microcontroller deployment**?

    For TinyML on MCUs, the most relevant answer is usually the **full INT8 model**, because it reduces memory footprint and enables an integer-only inference path. Dynamic range and float16 models are useful comparison points, but they typically serve different deployment trade-offs.
    """)
    return


@app.cell
def _(plt, ptq_results):
    # Visual comparison of PTQ model sizes
    ptq_plot_df = ptq_results.copy()

    plt.figure(figsize=(8, 4))
    plt.bar(ptq_plot_df["model"], ptq_plot_df["size_kb"])
    plt.ylabel("Model size (KB)")
    plt.title("PTQ model size comparison")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.bar(ptq_plot_df["model"], ptq_plot_df["accuracy"])
    plt.ylabel("Accuracy")
    plt.ylim(max(0.0, ptq_plot_df["accuracy"].min() - 0.02), min(1.0, ptq_plot_df["accuracy"].max() + 0.02))
    plt.title("PTQ accuracy comparison")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part II. Quantization-Aware Training (QAT)

    Unlike PTQ, QAT simulates quantization during training. This lets the model adapt to low-precision behavior while learning.

    In practice, QAT is often preferred when:

    - PTQ causes too much accuracy loss
    - the task is sensitive to quantization noise
    - you want a stronger deployment-ready INT8 model

    Here we take the already trained baseline model, transform it into a quantization-aware model, and fine-tune it.
    """)
    return


@app.cell
def _(baseline_model, tf, tfmot):
    # Build the QAT model from the trained baseline model
    qat_model = tfmot.quantization.keras.quantize_model(baseline_model)

    # Use a small learning rate for QAT fine-tuning. QAT starts from the trained
    # baseline weights, so we want gentle adaptation rather than aggressive retraining.
    qat_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    qat_model.summary()
    return (qat_model,)


@app.cell
def _(BATCH_SIZE, EPOCHS_QAT, qat_model, tf, x_test, x_train, y_test, y_train):
    # Fine-tune the quantization-aware model
    early_stop_qat = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=1,
        restore_best_weights=True,
    )

    history_qat = qat_model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=EPOCHS_QAT,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop_qat],
        verbose=1,
    )

    qat_loss, qat_accuracy = qat_model.evaluate(x_test, y_test, verbose=0)
    print(f"QAT Keras test accuracy: {qat_accuracy:.4f}")
    return history_qat, qat_accuracy


@app.cell
def _(history_qat, pd, plt):
    # Plot QAT training history
    history_qat_df = pd.DataFrame(history_qat.history)
    _ax = history_qat_df[['loss', 'val_loss']].plot(figsize=(8, 4), title='QAT Training Loss')
    _ax.set_xlabel('Epoch')
    _ax.set_ylabel('Loss')
    _ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    _ax = history_qat_df[['accuracy', 'val_accuracy']].plot(figsize=(8, 4), title='QAT Training Accuracy')
    _ax.set_xlabel('Epoch')
    _ax.set_ylabel('Accuracy')
    _ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Convert the QAT model to TFLite

    We create three deployment variants from the QAT-trained model:

    1. **QAT baseline TFLite**
    2. **QAT + full INT8**
    3. **QAT + dynamic range**

    A float16 version is not included here because the main purpose of QAT in this notebook is to improve low-bit deployment behavior, especially for **INT8-oriented TinyML pipelines**.
    """)
    return


@app.cell
def _(OUTPUT_DIR, convert_to_tflite, qat_model, representative_data_gen, tf):
    qat_baseline_tflite_path = OUTPUT_DIR / 'qat_model_baseline.tflite'
    qat_int8_tflite_path = OUTPUT_DIR / 'qat_model_integer_quant.tflite'
    qat_dynamic_tflite_path = OUTPUT_DIR / 'qat_model_dynamic_quant.tflite'
    _ = convert_to_tflite(model=qat_model, output_path=qat_baseline_tflite_path)
    # 1) QAT baseline TFLite
    _ = convert_to_tflite(model=qat_model, output_path=qat_int8_tflite_path, optimizations=[tf.lite.Optimize.DEFAULT], representative_dataset=representative_data_gen, supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8], inference_input_type=tf.int8, inference_output_type=tf.int8)
    _ = convert_to_tflite(model=qat_model, output_path=qat_dynamic_tflite_path, optimizations=[tf.lite.Optimize.DEFAULT])
    print('Saved QAT models:')
    for _path in [qat_baseline_tflite_path, qat_int8_tflite_path, qat_dynamic_tflite_path]:
    # 2) QAT + full INT8
    # 3) QAT + dynamic range
        print(' -', _path)
    return (
        qat_baseline_tflite_path,
        qat_dynamic_tflite_path,
        qat_int8_tflite_path,
    )


@app.cell
def _(
    baseline_accuracy,
    build_results_table,
    evaluate_tflite_model,
    get_model_size_kb,
    qat_baseline_tflite_path,
    qat_dynamic_tflite_path,
    qat_int8_tflite_path,
    x_test,
    y_test,
):
    # Evaluate QAT-based TFLite models
    qat_rows = []
    for (_model_name, model_path) in [('QAT Baseline TFLite', qat_baseline_tflite_path), ('QAT INT8', qat_int8_tflite_path), ('QAT Dynamic Range', qat_dynamic_tflite_path)]:
        _acc = evaluate_tflite_model(model_path, x_test, y_test)
        qat_rows.append({'model': _model_name, 'file': model_path.name, 'size_kb': get_model_size_kb(model_path), 'accuracy': _acc, 'accuracy_drop_vs_keras_baseline': baseline_accuracy - _acc})
    qat_results = build_results_table(qat_rows)
    qat_results
    return (qat_results,)


@app.cell
def _(baseline_accuracy, np, pd, qat_accuracy, qat_results):
    # Add both Keras references for context
    qat_reference = pd.DataFrame(
        [
            {
                "model": "Baseline Keras",
                "file": "-",
                "size_kb": np.nan,
                "accuracy": round(float(baseline_accuracy), 4),
                "accuracy_drop_vs_keras_baseline": 0.0,
            },
            {
                "model": "QAT Keras",
                "file": "-",
                "size_kb": np.nan,
                "accuracy": round(float(qat_accuracy), 4),
                "accuracy_drop_vs_keras_baseline": round(float(baseline_accuracy - qat_accuracy), 4),
            },
        ]
    )

    pd.concat([qat_reference, qat_results], ignore_index=True)
    return


@app.cell
def _(plt, qat_results):
    # Visual comparison of QAT model sizes and accuracy
    plt.figure(figsize=(8, 4))
    plt.bar(qat_results["model"], qat_results["size_kb"])
    plt.ylabel("Model size (KB)")
    plt.title("QAT model size comparison")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.bar(qat_results["model"], qat_results["accuracy"])
    plt.ylabel("Accuracy")
    plt.ylim(max(0.0, qat_results["accuracy"].min() - 0.02), min(1.0, qat_results["accuracy"].max() + 0.02))
    plt.title("QAT model accuracy comparison")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 5. Side-by-side summary

    The final table below puts the PTQ and QAT deployment results together. This makes it easier to answer the main practical question:

    > **If I want a model that is small enough for embedded deployment but still accurate enough to trust, which conversion path should I choose?**
    """)
    return


@app.cell
def _(pd, ptq_results, qat_results):
    combined_results = pd.concat(
        [
            ptq_results.assign(pipeline="PTQ"),
            qat_results.assign(pipeline="QAT"),
        ],
        ignore_index=True,
    )

    combined_results = combined_results[
        ["pipeline", "model", "file", "size_kb", "accuracy", "accuracy_drop_vs_keras_baseline"]
    ]

    combined_results
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 6. Key takeaways

    ### Post-Training Quantization (PTQ)
    PTQ is the easiest path to deployment. It requires no retraining and is often the first technique to try. Full INT8 PTQ is especially relevant for microcontrollers.

    ### Quantization-Aware Training (QAT)
    QAT generally becomes valuable when the PTQ model loses too much accuracy. By simulating quantization during training, the model can learn to behave better after conversion.

    ### Which model is usually best for TinyML?
    For a **microcontroller-oriented TinyML pipeline**, the most important final candidate is usually the **full INT8 model**, because it is the most deployment-friendly. The PTQ INT8 model is simpler to obtain, while the QAT INT8 model is often the better choice if you need stronger accuracy retention.

    ### Final practical recommendation
    A sensible workflow is:

    1. Train a float baseline model.
    2. Try **PTQ INT8** first.
    3. If accuracy loss is too large, move to **QAT INT8**.
    4. Export the final `.tflite` model for embedded deployment.

    That pattern aligns well with how real TinyML model compression is typically done in practice.
    """)
    return


@app.cell
def _(sys, tf):
    import platform
    print('Python executable:', sys.executable)
    print('Python version:', platform.python_version())
    print('TensorFlow version:', tf.__version__)
    print('Devices:', tf.config.list_physical_devices())
    return


if __name__ == "__main__":
    app.run()
