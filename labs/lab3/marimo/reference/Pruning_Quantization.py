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
    # Model Pruning with Quantization for TinyML

    This is designed for a pinned local setup such as:

    - Python 3.11
    - TensorFlow 2.14.1
    - Keras 2.14.0
    - TensorFlow Model Optimization 0.8.0
    - NumPy < 2

    The workflow is organized into two parts:

    1. **Model pruning with sparsity**
       We train a compact CNN on MNIST, apply pruning, and compare:
       - a pruned model saved **with** the pruning mask
       - a pruned model that has been **stripped** and converted with sparsity-aware optimization
       - a baseline TFLite reference model

    2. **Pruning + float16 quantization**
       We then convert the pruned models into float16 TFLite form and compare size and accuracy again.

    ## Why this notebook matters

    In TinyML, pruning and quantization are often discussed together, but they do **not** help in the same way:

    - **Pruning** introduces sparsity by zeroing out less important weights.
    - **Quantization** reduces numeric precision to shrink model storage and improve deployment feasibility.
    - A pruned model does **not** automatically become smaller unless the pruning mask is removed and the export path actually uses sparsity-aware conversion.

    ## Notes before you run this notebook

    - Use the Jupyter kernel created by your local setup script.
    - This notebook is for your **local environment**. It does **not** reinstall packages.
    - All generated models are saved into a local `pruning_outputs/` folder.
    - This is an **answer key** version, so all student TODOs have been completed.
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

    # Force CPU-only execution for this small TinyML/pruning demo.
    # On Apple Silicon, the tensorflow-metal GPU path can be slower for small models
    # and for TensorFlow Model Optimization pruning wrappers.
    try:
        tf.config.set_visible_devices([], "GPU")
        print("GPU disabled. Running TensorFlow on CPU only.")
    except RuntimeError as e:
        print("GPU could not be disabled because TensorFlow was already initialized:", e)

    print("Visible TensorFlow devices:", tf.config.get_visible_devices())

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

    # Optional safety checks for the pinned environment used on your laptop.
    assert tf.__version__.startswith("2.14"), "This notebook expects TensorFlow 2.14.x."
    assert tfmot.__version__ == "0.8.0", "This notebook expects tensorflow-model-optimization==0.8.0."

    OUTPUT_DIR = Path("pruning_outputs")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    SEED = 42
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    return OUTPUT_DIR, Path, np, pd, plt, tf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Imports, reproducibility, and experiment settings

    We keep the example intentionally compact so that it stays practical for a local laptop workflow.

    A few implementation choices are deliberate:

    - We use **sparse integer labels** instead of one-hot labels.
    - We keep the CNN small enough to train quickly while still being meaningful.
    - We train for only a small number of epochs because this notebook is meant to be a **demo answer key**, not a long training run.
    """)
    return


@app.cell
def _():
    # Standard experiment settings
    EPOCHS_BASELINE = 5
    EPOCHS_PRUNING = 2
    BATCH_SIZE = 32

    # Early stopping settings.
    # Baseline training should run at least 5 epochs before early stopping is allowed.
    MIN_EPOCHS_BEFORE_EARLY_STOPPING = 5
    EARLY_STOPPING_PATIENCE = 2
    EARLY_STOPPING_MIN_DELTA = 1e-4


    from tensorflow.keras import Model, Input
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
    from tensorflow.keras.datasets import mnist

    from tensorflow_model_optimization.sparsity.keras import (
        prune_low_magnitude,
        PolynomialDecay,
        UpdatePruningStep,
        strip_pruning,
    )

    return (
        BATCH_SIZE,
        Conv2D,
        Dense,
        EARLY_STOPPING_MIN_DELTA,
        EARLY_STOPPING_PATIENCE,
        EPOCHS_BASELINE,
        EPOCHS_PRUNING,
        Flatten,
        Input,
        MIN_EPOCHS_BEFORE_EARLY_STOPPING,
        MaxPooling2D,
        Model,
        PolynomialDecay,
        UpdatePruningStep,
        mnist,
        prune_low_magnitude,
        strip_pruning,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Dataset preparation

    We use the **MNIST** handwritten digit dataset.

    ### Preprocessing steps

    1. Load MNIST from `tensorflow.keras.datasets`.
    2. Normalize pixel values into the range `[0, 1]`.
    3. Reshape images into `(28, 28, 1)` for convolution.
    4. Keep labels as integers for sparse categorical training.
    """)
    return


@app.cell
def _(mnist, np):
    # Load and preprocess MNIST
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    print("Training data shape:", x_train.shape)
    print("Training labels shape:", y_train.shape)
    print("Test data shape:", x_test.shape)
    print("Test labels shape:", y_test.shape)
    return x_test, x_train, y_test, y_train


@app.cell
def _(plt, x_train, y_train):
    # Visual sanity check
    (_fig, _axes) = plt.subplots(2, 5, figsize=(10, 4))
    for (idx, ax) in enumerate(_axes.flat):
        ax.imshow(x_train[idx].squeeze(), cmap='gray')
        ax.set_title(f'Label: {y_train[idx]}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Build and train the baseline CNN

    The baseline model is intentionally simple:

    - one convolution layer
    - one max-pooling layer
    - one hidden dense layer
    - one softmax output layer

    This makes the notebook fast to run and easy to convert to TensorFlow Lite.
    """)
    return


@app.cell
def _(
    BATCH_SIZE,
    Conv2D,
    Dense,
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_PATIENCE,
    EPOCHS_BASELINE,
    Flatten,
    Input,
    MIN_EPOCHS_BEFORE_EARLY_STOPPING,
    MaxPooling2D,
    Model,
    tf,
    x_test,
    x_train,
    y_test,
    y_train,
):
    def make_adam(learning_rate=0.001):
        """Use legacy Adam when available because it is faster on many M1/M2/M3 Mac setups."""
        try:
            return tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
        except AttributeError:
            return tf.keras.optimizers.Adam(learning_rate=learning_rate)


    def create_cnn_model() -> tf.keras.Model:
        inputs = Input(shape=(28, 28, 1), name="image")
        x = Conv2D(32, (3, 3), activation="relu", name="conv_1")(inputs)
        x = MaxPooling2D((2, 2), name="pool_1")(x)
        x = Flatten(name="flatten")(x)
        x = Dense(128, activation="relu", name="dense_1")(x)
        outputs = Dense(10, activation="softmax", name="classifier")(x)
        return Model(inputs, outputs, name="mnist_cnn")

    baseline_early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=EARLY_STOPPING_MIN_DELTA,
        restore_best_weights=True,
        start_from_epoch=MIN_EPOCHS_BEFORE_EARLY_STOPPING,
        verbose=1,
    )

    baseline_model = create_cnn_model()
    baseline_model.compile(
        optimizer=make_adam(0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history_baseline = baseline_model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=EPOCHS_BASELINE,
        batch_size=BATCH_SIZE,
        callbacks=[baseline_early_stopping],
        verbose=1,
    )

    baseline_loss, baseline_accuracy = baseline_model.evaluate(x_test, y_test, verbose=0)
    print(f"Baseline Keras accuracy: {baseline_accuracy:.4f}")
    return (
        baseline_accuracy,
        baseline_model,
        create_cnn_model,
        history_baseline,
        make_adam,
    )


@app.cell
def _(history_baseline, pd, plt):
    # Plot baseline training history
    history_baseline_df = pd.DataFrame(history_baseline.history)
    (_fig, _axes) = plt.subplots(1, 2, figsize=(12, 4))
    history_baseline_df[['loss', 'val_loss']].plot(ax=_axes[0], title='Baseline Training Loss')
    _axes[0].set_xlabel('Epoch')
    _axes[0].set_ylabel('Loss')
    history_baseline_df[['accuracy', 'val_accuracy']].plot(ax=_axes[1], title='Baseline Training Accuracy')
    _axes[1].set_xlabel('Epoch')
    _axes[1].set_ylabel('Accuracy')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Helper functions

    Before running pruning and quantization experiments, we define reusable utilities for:

    - TFLite conversion
    - TFLite model evaluation
    - model file size measurement
    - compact result table construction
    """)
    return


@app.cell
def _(Path, np, tf):
    def get_size_kb(path: Path) -> float:
        return path.stat().st_size / 1024.0


    def convert_to_tflite(
        keras_model: tf.keras.Model,
        output_path: Path,
        optimizations=None,
        supported_types=None,
    ):
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

        if optimizations is not None:
            converter.optimizations = optimizations

        if supported_types is not None:
            converter.target_spec.supported_types = supported_types

        tflite_model = converter.convert()
        output_path.write_bytes(tflite_model)
        return output_path


    def evaluate_tflite_model(model_path: Path, x_eval: np.ndarray, y_eval: np.ndarray) -> float:
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_index = input_details[0]["index"]
        output_index = output_details[0]["index"]

        input_dtype = input_details[0]["dtype"]
        input_scale, input_zero_point = input_details[0]["quantization"]

        correct = 0

        for i in range(len(x_eval)):
            sample = x_eval[i:i+1].astype(np.float32)

            if input_dtype == np.int8:
                sample = sample / input_scale + input_zero_point
                sample = np.round(sample).astype(np.int8)
            elif input_dtype == np.uint8:
                sample = sample / input_scale + input_zero_point
                sample = np.round(sample).astype(np.uint8)
            elif input_dtype == np.float16:
                sample = sample.astype(np.float16)
            else:
                sample = sample.astype(input_dtype)

            interpreter.set_tensor(input_index, sample)
            interpreter.invoke()

            prediction = interpreter.get_tensor(output_index)
            predicted_label = int(np.argmax(prediction, axis=1)[0])

            if predicted_label == int(y_eval[i]):
                correct += 1

        return correct / len(y_eval)


    def make_result_row(model_name: str, path: Path, accuracy: float) -> dict:
        return {
            "model": model_name,
            "file": path.name,
            "size_kb": round(get_size_kb(path), 2),
            "accuracy": round(float(accuracy), 4),
        }



    def print_kernel_sparsity_percent(model: tf.keras.Model, label: str) -> float:
        """Print percent of zero kernel weights.

        For a TF-MOT pruning wrapper, this reports the effective sparse kernel
        by multiplying the stored kernel by the pruning mask. For a stripped model,
        it reports the zeros stored directly in the kernel tensors.
        """
        total_weights = 0
        zero_weights = 0

        for layer in model.layers:
            for weight in layer.weights:
                weight_name = weight.name.lower()

                # Only count trainable kernel tensors, not biases, masks, thresholds, or pruning_step.
                if "kernel" not in weight_name:
                    continue

                kernel = weight.numpy()

                # If this layer has a pruning mask with the same shape, count effective kernel * mask.
                masks = [w.numpy() for w in layer.weights if "mask" in w.name.lower() and tuple(w.shape) == tuple(weight.shape)]
                if masks:
                    kernel = kernel * masks[0]

                total_weights += kernel.size
                zero_weights += int(np.sum(kernel == 0))

        sparsity = 100.0 * zero_weights / total_weights if total_weights else 0.0
        print(f"{label} kernel sparsity: {sparsity:.2f}% ({zero_weights} / {total_weights} kernel weights are zero)")
        return sparsity

    return (
        convert_to_tflite,
        evaluate_tflite_model,
        make_result_row,
        print_kernel_sparsity_percent,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part I. Model Pruning with Sparsity

    Pruning gradually pushes selected weights to zero during training. However, a key practical detail is easy to miss:

    > A pruned model does **not automatically** become much smaller just because many weights are zero.

    To get an actually compact deployment artifact, we usually need to:

    1. **strip the pruning wrappers**
    2. export the model again
    3. use **sparsity-aware** conversion where appropriate

    In this section we compare:

    1. a baseline TFLite model
    2. a pruned model exported **with** the pruning mask
    3. a stripped sparse model exported with sparsity-aware optimization
    """)
    return


@app.cell
def _(
    BATCH_SIZE,
    EARLY_STOPPING_MIN_DELTA,
    EPOCHS_PRUNING,
    PolynomialDecay,
    UpdatePruningStep,
    create_cnn_model,
    make_adam,
    np,
    print_kernel_sparsity_percent,
    prune_low_magnitude,
    tf,
    x_test,
    x_train,
    y_test,
    y_train,
):
    def apply_pruning(model: tf.keras.Model) -> tf.keras.Model:
        steps_per_epoch = int(np.ceil(len(x_train) / BATCH_SIZE))

        pruning_params = {
            "pruning_schedule": PolynomialDecay(
                initial_sparsity=0.50,
                final_sparsity=0.90,
                begin_step=0,
                end_step=steps_per_epoch * EPOCHS_PRUNING,
                frequency=1,
            )
        }

        return prune_low_magnitude(model, **pruning_params)


    pruning_early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=1,
        min_delta=EARLY_STOPPING_MIN_DELTA,
        restore_best_weights=False,
        start_from_epoch=1,
        verbose=1,
    )

    pruned_model = apply_pruning(create_cnn_model())
    pruned_model.compile(
        optimizer=make_adam(0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history_pruned = pruned_model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=EPOCHS_PRUNING,
        batch_size=BATCH_SIZE,
        callbacks=[UpdatePruningStep(), pruning_early_stopping],
        verbose=1,
    )

    pruned_loss, pruned_accuracy = pruned_model.evaluate(x_test, y_test, verbose=0)
    print(f"Pruned Keras accuracy (with wrappers): {pruned_accuracy:.4f}")


    # Simple sparsity check after pruning, before stripping.
    print_kernel_sparsity_percent(pruned_model, "Pruned model with pruning mask")
    return history_pruned, pruned_accuracy, pruned_model


@app.cell
def _(history_pruned, pd, plt):
    # Plot pruning training history
    history_pruned_df = pd.DataFrame(history_pruned.history)
    (_fig, _axes) = plt.subplots(1, 2, figsize=(12, 4))
    history_pruned_df[['loss', 'val_loss']].plot(ax=_axes[0], title='Pruning Training Loss')
    _axes[0].set_xlabel('Epoch')
    _axes[0].set_ylabel('Loss')
    history_pruned_df[['accuracy', 'val_accuracy']].plot(ax=_axes[1], title='Pruning Training Accuracy')
    _axes[1].set_xlabel('Epoch')
    _axes[1].set_ylabel('Accuracy')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(
    OUTPUT_DIR,
    baseline_model,
    convert_to_tflite,
    make_adam,
    print_kernel_sparsity_percent,
    pruned_model,
    strip_pruning,
    tf,
    x_test,
    y_test,
):
    # Export three deployment artifacts:
    # 1) baseline TFLite
    # 2) pruned TFLite with pruning mask still present
    # 3) stripped sparse TFLite with sparsity-aware optimization

    baseline_tflite_path = OUTPUT_DIR / "baseline_model.tflite"
    pruned_with_mask_path = OUTPUT_DIR / "pruned_model_with_mask.tflite"
    pruned_sparse_path = OUTPUT_DIR / "pruned_model_sparse.tflite"

    # Baseline TFLite
    _ = convert_to_tflite(
        keras_model=baseline_model,
        output_path=baseline_tflite_path,
    )

    # Pruned model with mask
    _ = convert_to_tflite(
        keras_model=pruned_model,
        output_path=pruned_with_mask_path,
    )

    # Stripped sparse model
    stripped_model = strip_pruning(pruned_model)

    # Simple sparsity check after removing the pruning wrappers/masks.
    print_kernel_sparsity_percent(stripped_model, "Pruned model after stripping")

    # strip_pruning returns a new Keras model, so compile it before evaluate().
    # This does not retrain or change the weights; it only attaches the loss/metrics.
    stripped_model.compile(
        optimizer=make_adam(0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    stripped_loss, stripped_accuracy = stripped_model.evaluate(x_test, y_test, verbose=0)
    print(f"Pruned Keras accuracy (after stripping): {stripped_accuracy:.4f}")

    _ = convert_to_tflite(
        keras_model=stripped_model,
        output_path=pruned_sparse_path,
        optimizations=[tf.lite.Optimize.EXPERIMENTAL_SPARSITY],
    )

    print("Saved:")
    print(" -", baseline_tflite_path)
    print(" -", pruned_with_mask_path)
    print(" -", pruned_sparse_path)
    return (
        baseline_tflite_path,
        pruned_sparse_path,
        pruned_with_mask_path,
        stripped_accuracy,
        stripped_model,
    )


@app.cell
def _(
    baseline_accuracy,
    baseline_tflite_path,
    evaluate_tflite_model,
    make_result_row,
    np,
    pd,
    pruned_accuracy,
    pruned_sparse_path,
    pruned_with_mask_path,
    stripped_accuracy,
    x_test,
    y_test,
):
    # Evaluate and compare Part I models
    part1_rows = []

    baseline_tflite_acc = evaluate_tflite_model(baseline_tflite_path, x_test, y_test)
    pruned_with_mask_acc = evaluate_tflite_model(pruned_with_mask_path, x_test, y_test)
    pruned_sparse_acc = evaluate_tflite_model(pruned_sparse_path, x_test, y_test)

    part1_rows.append(make_result_row("Baseline TFLite", baseline_tflite_path, baseline_tflite_acc))
    part1_rows.append(make_result_row("Pruned TFLite (with mask)", pruned_with_mask_path, pruned_with_mask_acc))
    part1_rows.append(make_result_row("Pruned Sparse TFLite", pruned_sparse_path, pruned_sparse_acc))

    part1_results = pd.DataFrame(part1_rows)

    keras_reference = pd.DataFrame(
        [
            {
                "model": "Baseline Keras",
                "file": "-",
                "size_kb": np.nan,
                "accuracy": round(float(baseline_accuracy), 4),
            },
            {
                "model": "Pruned Keras (before stripping)",
                "file": "-",
                "size_kb": np.nan,
                "accuracy": round(float(pruned_accuracy), 4),
            },
            {
                "model": "Pruned Keras (after stripping)",
                "file": "-",
                "size_kb": np.nan,
                "accuracy": round(float(stripped_accuracy), 4),
            },
        ]
    )

    part1_display = pd.concat([keras_reference, part1_results], ignore_index=True)
    part1_display
    return (part1_results,)


@app.cell
def _(part1_results, plt):
    # Visual comparison for Part I
    part1_plot_df = part1_results.copy()

    plt.figure(figsize=(8, 4))
    plt.bar(part1_plot_df["model"], part1_plot_df["size_kb"])
    plt.ylabel("Model size (KB)")
    plt.title("Part I: Pruning and sparsity model size comparison")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Part I interpretation

    This is the key lesson from the pruning section:

    - The **pruned model with the mask still attached** usually does **not** give the best deployment size.
    - The **stripped sparse model** is the more meaningful deployment artifact.
    - Pruning only becomes practically useful for compression when the export pipeline reflects the sparsity structure.

    That is why pruning is often described as a **two-step** compression workflow: sparsify first, then export properly.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part II. Pruning + Float16 Quantization

    Now we convert the pruned models into **float16-quantized TFLite** form.

    This part corresponds to the student TODO block in the original notebook. The answer key below fills in the missing conversion commands.

    We compare two cases:

    1. **Pruned model with mask + float16 quantization**
    2. **Stripped sparse model + float16 quantization**

    The main question is whether float16 conversion alone is enough, or whether removing the pruning mask still matters.
    """)
    return


@app.cell
def _(OUTPUT_DIR, pruned_model, stripped_model, tf):
    # Float16 quantization for pruned model with mask
    pruned_with_mask_float16_path = OUTPUT_DIR / "pruned_model_with_mask_float16.tflite"

    converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model_with_mask_float16 = converter.convert()
    pruned_with_mask_float16_path.write_bytes(tflite_model_with_mask_float16)

    print("Saved Float16 quantized model (with mask).")

    # Float16 quantization for stripped sparse model
    pruned_sparse_float16_path = OUTPUT_DIR / "pruned_model_sparse_float16.tflite"

    converter = tf.lite.TFLiteConverter.from_keras_model(stripped_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT, tf.lite.Optimize.EXPERIMENTAL_SPARSITY]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model_sparse_float16 = converter.convert()
    pruned_sparse_float16_path.write_bytes(tflite_model_sparse_float16)

    print("Saved Float16 quantized model (stripped sparse).")
    return pruned_sparse_float16_path, pruned_with_mask_float16_path


@app.cell
def _(
    evaluate_tflite_model,
    make_result_row,
    pd,
    pruned_sparse_float16_path,
    pruned_with_mask_float16_path,
    x_test,
    y_test,
):
    # Evaluate and compare the float16 models
    part2_rows = []

    with_mask_float16_acc = evaluate_tflite_model(pruned_with_mask_float16_path, x_test, y_test)
    sparse_float16_acc = evaluate_tflite_model(pruned_sparse_float16_path, x_test, y_test)

    part2_rows.append(make_result_row("Pruned + Float16 (with mask)", pruned_with_mask_float16_path, with_mask_float16_acc))
    part2_rows.append(make_result_row("Pruned Sparse + Float16", pruned_sparse_float16_path, sparse_float16_acc))

    part2_results = pd.DataFrame(part2_rows)
    part2_results
    return (part2_results,)


@app.cell
def _(part2_results, plt):
    # Visual comparison for Part II
    plt.figure(figsize=(8, 4))
    plt.bar(part2_results["model"], part2_results["size_kb"])
    plt.ylabel("Model size (KB)")
    plt.title("Part II: Float16 quantized pruned model size comparison")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.show()

    print("Part II accuracies:")
    for _, row in part2_results.iterrows():
        print(f"{row['model']}: {row['accuracy']:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 5. Side-by-side summary

    The final table combines the main deployment artifacts from both parts so it is easier to compare what each step is really doing.
    """)
    return


@app.cell
def _(part1_results, part2_results, pd):
    summary_results = pd.concat(
        [
            part1_results.assign(pipeline="Pruning"),
            part2_results.assign(pipeline="Pruning + Float16"),
        ],
        ignore_index=True,
    )

    summary_results = summary_results[["pipeline", "model", "file", "size_kb", "accuracy"]]
    summary_results
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 6. Key takeaways

    ### What pruning alone does
    Pruning pushes many weights toward zero, but that by itself does **not** guarantee a compact deployment file.

    ### Why stripping matters
    The pruning wrappers and masks are useful during training, but they are not the ideal final deployment representation. Stripping them is what turns the model into a cleaner sparse artifact.

    ### What float16 adds
    Float16 quantization reduces storage precision, so it can further reduce model size. However, if the pruning mask is still present, the export may still be less efficient than the stripped sparse version.

    ### Practical TinyML lesson
    A sensible workflow is:

    1. train a baseline model
    2. apply pruning if sparsity is desired
    3. strip the pruning wrappers before final export
    4. then apply the deployment-oriented conversion path you want, such as sparsity-aware export or quantization

    That sequence is much closer to how pruning is used in real TinyML compression pipelines.
    """)
    return


if __name__ == "__main__":
    app.run()
