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
    # EE 446 TinyML — Model Pruning with Quantization
    ## Student TODO Version: Pruning and Quantization of a DNN Using the UCI Human Activity Recognition Dataset

    ### Overview
    In this notebook, you will:
    - train a baseline DNN on the **UCI HAR** dataset,
    - apply **magnitude-based pruning**,
    - compare the pruned model before and after `strip_pruning(...)`, and
    - combine pruning with **float16 quantization**.

    Use the **`Python (tinyml-arduino)`** Jupyter kernel for this notebook.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Environment Setup

    This notebook is designed to run with the **`Python (tinyml-arduino)`** Jupyter kernel that you already created.

    This notebook assumes the environment already contains:
    - `tensorflow==2.14.1`
    - `tensorflow-model-optimization==0.8.0`
    - `numpy`, `pandas`, `matplotlib`, and `scikit-learn`

    Do **not** reinstall TensorFlow packages inside the notebook if you are already using the working TinyML environment.
    """)
    return


@app.cell
def _():
    import os
    import math
    import zipfile
    import random
    import urllib.request
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import tensorflow_model_optimization as tfmot

    import gc

    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
    from tensorflow import keras
    from tensorflow.keras import layers

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    tf.keras.backend.clear_session()
    gc.collect()

    print("Python executable:", os.sys.executable)
    print("TensorFlow version:", tf.__version__)
    print("TF-MOT version:", tfmot.__version__)
    return (
        ConfusionMatrixDisplay,
        accuracy_score,
        classification_report,
        confusion_matrix,
        gc,
        keras,
        layers,
        np,
        pd,
        plt,
        tf,
        zipfile,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Download and Extract the UCI HAR Dataset

    The original dataset contains:
    - **561 numerical features** extracted from smartphone sensor signals,
    - **6 activity classes**, and
    - predefined **training** and **test** splits.

    The code below downloads and extracts the dataset if it is not already present in the working directory.
    """)
    return


@app.cell
def _(zipfile):
    from pathlib import Path
    from urllib.request import urlretrieve

    BASE_DIR = Path("./assets")

    data_dir = BASE_DIR / "data"
    dataset_dir = data_dir / "UCI HAR Dataset"
    zip_path = data_dir / "uci_har_dataset.zip"

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"

    if not dataset_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(data_dir)
        print("Dataset downloaded.")
    else:
        print("Dataset already exists.")
    return BASE_DIR, Path, dataset_dir


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Load the Data
    """)
    return


@app.cell
def _(dataset_dir, pd):
    def load_har_data():
        x_train = pd.read_csv(
            f"{dataset_dir}/train/X_train.txt", sep='\s+', header=None
        ).values.astype("float32")
        y_train = pd.read_csv(
            f"{dataset_dir}/train/y_train.txt", sep='\s+', header=None
        ).values.astype("float32")
        x_test = pd.read_csv(
            f"{dataset_dir}/test/X_test.txt", sep='\s+', header=None
        ).values.astype("float32")
        y_test = pd.read_csv(
            f"{dataset_dir}/test/y_test.txt", sep='\s+', header=None
        ).values.astype("float32")

        y_train = y_train - 1
        y_test = y_test - 1
        y_train = y_train.flatten()
        y_test = y_test.flatten()

        print("Training data shape:", x_train.shape)
        print("Training labels shape:", y_train.shape)
        print("Test data shape:", x_test.shape)
        print("Test labels shape:", y_test.shape)

        return (x_train, y_train, x_test, y_test)

    x_train, y_train, x_test, y_test = load_har_data()

    class_names = [
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING",
    ]

    num_features = x_train.shape[1]   # 561
    num_classes = len(class_names)    # 6
    return (
        class_names,
        num_classes,
        num_features,
        x_test,
        x_train,
        y_test,
        y_train,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Quick Inspection
    """)
    return


@app.cell
def _(class_names, num_classes, pd, y_train):
    # TODO:
    # Create a small summary table showing:
    # - class index,
    # - class name, and
    # - number of training samples in each class.

    train_counts = pd.Series(y_train).value_counts().sort_index()

    class_summary = pd.DataFrame({
        "Class Index": list(range(num_classes)),
        "Class Name": class_names,
        "Train Samples": [int(train_counts.get(i, 0)) for i in range(num_classes)],
    })
    class_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Train a Baseline DNN

    We will use a compact dense neural network that is appropriate for a numerical-feature TinyML workflow.

    ### Architecture
    - Input: 561 features
    - Dense(256, ReLU)
    - Dense(128, ReLU)
    - Dense(64, ReLU)
    - Dense(6, Softmax)
    """)
    return


@app.cell
def _(keras, layers, num_classes, num_features):
    def build_baseline_model(input_dim, num_classes):
        # TODO:
        # Build and compile a DNN with the following architecture:
        # Input -> Dense(256, relu) -> Dense(128, relu) -> Dense(64, relu) -> Dense(num_classes, softmax)
        # Use Adam with learning rate 1e-3.
        # Use sparse_categorical_crossentropy as the loss.
        # Track accuracy as a metric.

        model = keras.Sequential([
            keras.Input(shape=(input_dim,)),
            layers.Dense(256, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    baseline_model = build_baseline_model(num_features, num_classes)
    baseline_model.summary()
    return (baseline_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Train the Baseline Model
    """)
    return


@app.cell
def _(baseline_model, keras, x_train, y_train):
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True
        )
    ]

    # TODO:
    # Train the baseline model using:
    # - validation_split=0.2
    # - epochs=40
    # - batch_size=64
    # - callbacks=callbacks

    history = baseline_model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=40,
        batch_size=64,
        callbacks=callbacks,
    )
    return (history,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Training Curves
    """)
    return


@app.cell
def _(history, plt):
    # TODO:
    # Plot:
    # 1. training accuracy vs validation accuracy
    # 2. training loss vs validation loss

    fig_curves, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(12, 4))

    ax_acc.plot(history.history["accuracy"], label="train")
    ax_acc.plot(history.history["val_accuracy"], label="val")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.legend()

    ax_loss.plot(history.history["loss"], label="train")
    ax_loss.plot(history.history["val_loss"], label="val")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.legend()

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Evaluate the Baseline Keras Model
    """)
    return


@app.cell
def _(
    ConfusionMatrixDisplay,
    accuracy_score,
    baseline_model,
    class_names,
    classification_report,
    confusion_matrix,
    np,
    plt,
    x_test,
    y_test,
):
    # TODO:
    # 1. Predict class probabilities on X_test
    # 2. Convert probabilities to class labels using argmax
    # 3. Compute the test accuracy
    # 4. Print the classification report
    # 5. Plot the confusion matrix

    baseline_probs = baseline_model.predict(x_test)
    baseline_preds = np.argmax(baseline_probs, axis=1)

    baseline_test_acc = accuracy_score(y_test, baseline_preds)
    print("Baseline Keras test accuracy: {:.4f}".format(baseline_test_acc))
    print(classification_report(y_test, baseline_preds, target_names=class_names))

    baseline_cm = confusion_matrix(y_test, baseline_preds)
    ConfusionMatrixDisplay(baseline_cm, display_labels=class_names).plot(
        xticks_rotation=45, cmap="Blues"
    )
    plt.title("Baseline Keras Model Confusion Matrix")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part I: Model Pruning with Sparsity

    In this part, we apply **magnitude-based pruning** to the DNN. The key idea is to gradually set small-magnitude weights to zero during training.

    We will compare:
    1. the baseline TensorFlow Lite model,
    2. the pruned model converted **without** stripping the pruning wrappers, and
    3. the stripped sparse model converted with **experimental sparsity-aware optimization**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. TensorFlow Lite Utilities

    The following helper functions are used to:
    - convert Keras models to TensorFlow Lite,
    - evaluate TensorFlow Lite models on the test set, and
    - measure model size.
    """)
    return


@app.cell
def _(BASE_DIR, Path, accuracy_score, np, tf, x_train):
    def save_binary_model(model_content, filename):
        model_dir = Path(f"./{BASE_DIR}/pruning_quantization/models/")
        model_dir.mkdir(parents=True, exist_ok=True)

        filepath = model_dir / filename
        filepath.write_bytes(model_content)

        return filepath.stat().st_size / 1024  # KB

    def representative_dataset_gen():
        # TODO:
        # Yield 300 representative samples from X_train as float32 tensors.
        # Each yielded item should be in the form: [sample]

        for i in range(300):
            sample = x_train[i:i + 1].astype(np.float32)
            yield [sample]

    def evaluate_tflite_model(tflite_model, X, y_true):
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        input_scale, input_zero_point = input_details["quantization"]
        output_scale, output_zero_point = output_details["quantization"]

        y_pred = []

        for i in range(len(X)):
            x = X[i:i+1].astype(np.float32)

            # TODO:
            # Quantize the input only when the input dtype is int8 or uint8.
            # Otherwise keep the input in the required floating-point dtype.

            if input_details["dtype"] in (np.int8, np.uint8):
                x = x / input_scale + input_zero_point
                x = x.astype(input_details["dtype"])
            else:
                x = x.astype(input_details["dtype"])

            interpreter.set_tensor(input_details["index"], x)
            interpreter.invoke()

            output = interpreter.get_tensor(output_details["index"])

            # TODO:
            # If the output is quantized, dequantize it back to float32.

            if output_details["dtype"] in (np.int8, np.uint8):
                output = (output.astype(np.float32) - output_zero_point) * output_scale

            y_pred.append(np.argmax(output, axis=1)[0])

        y_pred = np.array(y_pred)
        acc = accuracy_score(y_true, y_pred)
        return acc, y_pred

    def convert_to_tflite_fp32(model):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        # TODO: return the converted FP32 TensorFlow Lite model

        tflite_model = converter.convert()
        return tflite_model

    def convert_to_tflite_dynamic_range(model):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        # TODO:
        # Apply default optimization and return the converted model.

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        return tflite_model

    def convert_to_tflite_float16(model):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        # TODO:
        # Apply default optimization
        # Set supported_types to [tf.float16]
        # Return the converted model

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        return tflite_model

    def convert_to_tflite_int8(model):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        # TODO:
        # Apply default optimization
        # Attach representative_dataset_gen
        # Restrict to TFLITE_BUILTINS_INT8
        # Set inference_input_type and inference_output_type to tf.int8
        # Return the converted model

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        return tflite_model

    return convert_to_tflite_fp32, evaluate_tflite_model, save_binary_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Convert the Baseline Model to TensorFlow Lite
    """)
    return


@app.cell
def _(
    baseline_model,
    convert_to_tflite_fp32,
    evaluate_tflite_model,
    save_binary_model,
    x_test,
    y_test,
):
    # TODO:
    # Convert the baseline model to FP32 TensorFlow Lite.
    # Save the .tflite file, compute its size in KB, and evaluate it on X_test.

    # TODO:
    # Convert the baseline model into:
    # - FP32 TFLite
    # - dynamic range TFLite
    # - float16 TFLite
    # - int8 TFLite

    # Save each model to disk and record its size in KB.
    # Evaluate each TFLite model on the test set.

    tflite_fp32 = convert_to_tflite_fp32(baseline_model)
    size_fp32_kb = save_binary_model(tflite_fp32, "model_fp32.tflite")

    acc_fp32, preds_fp32 = evaluate_tflite_model(tflite_fp32, x_test, y_test)

    print(f"FP32:acc={acc_fp32:.4f} | size={size_fp32_kb:.1f} KB")
    return acc_fp32, size_fp32_kb


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Apply Magnitude-Based Pruning

    We will prune the DNN using a **polynomial decay schedule**:
    - start from low sparsity,
    - gradually increase sparsity during training, and
    - finish with a highly sparse model.

    After training, we will compare:
    - the pruned model **with** the pruning wrappers still present, and
    - the final sparse model after applying `strip_pruning(...)`.
    """)
    return


@app.cell
def _(keras, layers, np, num_classes, num_features, x_train, y_train):
    from tensorflow_model_optimization.sparsity.keras import (
        prune_low_magnitude,
        PolynomialDecay,
        UpdatePruningStep,
        strip_pruning
    )

    pruning_epochs = 12
    batch_size = 64

    # TODO:
    # Compute steps_per_epoch using 80% of the training set and the selected batch size.
    # Define pruning_params using PolynomialDecay with:
    # - initial_sparsity=0.20
    # - final_sparsity=0.85
    # - begin_step=0
    # - end_step=steps_per_epoch * pruning_epochs

    steps_per_epoch = int(np.ceil(0.8 * x_train.shape[0] / batch_size))

    pruning_params = {
        "pruning_schedule": PolynomialDecay(
            initial_sparsity=0.20,
            final_sparsity=0.85,
            begin_step=0,
            end_step=steps_per_epoch * pruning_epochs,
        )
    }

    # TODO:
    # Create the pruned model by wrapping a fresh baseline DNN with prune_low_magnitude.
    # Compile it with Adam(1e-3), sparse_categorical_crossentropy, and accuracy.

    def build_fresh_baseline():
        return keras.Sequential([
            keras.Input(shape=(num_features,)),
            layers.Dense(256, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ])

    pruned_model = prune_low_magnitude(build_fresh_baseline(), **pruning_params)
    pruned_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    pruned_model.summary()

    pruning_callbacks = [
        UpdatePruningStep(),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True
        )
    ]

    # TODO:
    # Train the pruned model using:
    # - validation_split=0.2
    # - epochs=pruning_epochs
    # - batch_size=batch_size
    # - callbacks=pruning_callbacks

    pruned_history = pruned_model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=pruning_epochs,
        batch_size=batch_size,
        callbacks=pruning_callbacks,
    )
    return pruned_model, strip_pruning


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Convert the Pruned Model Before and After Stripping the Pruning Wrappers

    First, we convert the pruned model **with** the pruning wrappers still attached.

    Next, we strip the pruning wrappers and convert the resulting sparse model with:
    - `tf.lite.Optimize.EXPERIMENTAL_SPARSITY`

    This is the proper way to preserve sparsity in the exported TensorFlow Lite model.
    """)
    return


@app.cell
def _(
    convert_to_tflite_fp32,
    evaluate_tflite_model,
    pruned_model,
    save_binary_model,
    strip_pruning,
    tf,
    x_test,
    y_test,
):
    # TODO:
    # 1. Convert the pruned model WITH the pruning wrappers still attached to FP32 TensorFlow Lite.
    # 2. Save the model and evaluate it on X_test.
    # 3. Strip the pruning wrappers using strip_pruning(...).
    # 4. Convert the stripped model with tf.lite.Optimize.EXPERIMENTAL_SPARSITY.
    # 5. Save the stripped sparse model and evaluate it on X_test.

    tflite_pruned_with_mask = convert_to_tflite_fp32(pruned_model)
    size_pruned_with_mask_kb = save_binary_model(tflite_pruned_with_mask, "model_pruned_with_mask.tflite")
    acc_pruned_with_mask, preds_pruned_with_mask = evaluate_tflite_model(
        tflite_pruned_with_mask, x_test, y_test
    )

    stripped_model = strip_pruning(pruned_model)

    def convert_to_tflite_sparse(model):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.EXPERIMENTAL_SPARSITY]
        tflite_model = converter.convert()
        return tflite_model

    tflite_stripped_sparse = convert_to_tflite_sparse(stripped_model)
    size_stripped_sparse_kb = save_binary_model(tflite_stripped_sparse, "model_stripped_sparse.tflite")
    acc_stripped_sparse, preds_stripped_sparse = evaluate_tflite_model(
        tflite_stripped_sparse, x_test, y_test
    )

    print(f"Pruned (with mask): acc={acc_pruned_with_mask:.4f} | size={size_pruned_with_mask_kb:.1f} KB")
    print(f"Stripped Sparse: acc={acc_stripped_sparse:.4f} | size={size_stripped_sparse_kb:.1f} KB")
    return (
        acc_pruned_with_mask,
        acc_stripped_sparse,
        preds_stripped_sparse,
        size_pruned_with_mask_kb,
        size_stripped_sparse_kb,
        stripped_model,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. Part I Comparison: Accuracy and Model Size
    """)
    return


@app.cell
def _(
    acc_fp32,
    acc_pruned_with_mask,
    acc_stripped_sparse,
    pd,
    size_fp32_kb,
    size_pruned_with_mask_kb,
    size_stripped_sparse_kb,
):
    # TODO:
    # Create a comparison DataFrame for Part I with the columns:
    # Model, Format, Test Accuracy, Model Size (KB)

    # Include:
    # - baseline FP32 TFLite
    # - pruned FP32 TFLite with mask
    # - stripped sparse FP32 TFLite

    part1_results = pd.DataFrame({
        "Model": ["Baseline", "Pruned (with mask)", "Stripped Sparse"],
        "Format": ["FP32", "FP32", "FP32 (sparse)"],
        "Test Accuracy": [acc_fp32, acc_pruned_with_mask, acc_stripped_sparse],
        "Model Size (KB)": [size_fp32_kb, size_pruned_with_mask_kb, size_stripped_sparse_kb],
    })
    part1_results
    return (part1_results,)


@app.cell
def _(part1_results, plt):
    # TODO:
    # Plot:
    # 1. a bar chart of the Part I model sizes
    # 2. a bar chart of the Part I test accuracies

    fig_p1, (ax_size1, ax_acc1) = plt.subplots(1, 2, figsize=(12, 4))

    ax_size1.bar(part1_results["Model"], part1_results["Model Size (KB)"], color="steelblue")
    ax_size1.set_title("Part I Model Size (KB)")
    ax_size1.tick_params(axis="x", rotation=20)

    ax_acc1.bar(part1_results["Model"], part1_results["Test Accuracy"], color="darkorange")
    ax_acc1.set_title("Part I Test Accuracy")
    ax_acc1.set_ylim(0, 1.0)
    ax_acc1.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Confusion Matrix for the Stripped Sparse Model
    """)
    return


@app.cell
def _(
    ConfusionMatrixDisplay,
    class_names,
    classification_report,
    confusion_matrix,
    plt,
    preds_stripped_sparse,
    y_test,
):
    # TODO:
    # Plot the confusion matrix for the stripped sparse TFLite model.
    # Print the classification report for the stripped sparse TFLite model.

    print(classification_report(y_test, preds_stripped_sparse, target_names=class_names))

    stripped_sparse_cm = confusion_matrix(y_test, preds_stripped_sparse)
    ConfusionMatrixDisplay(stripped_sparse_cm, display_labels=class_names).plot(
        xticks_rotation=45, cmap="Purples"
    )
    plt.title("Stripped Sparse Model Confusion Matrix")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part II: Model Pruning + Float16 Quantization

    In this part, we combine **pruning** and **float16 quantization**.

    We will compare:
    1. the pruned TensorFlow Lite model **with** the pruning wrappers still attached, after float16 quantization, and
    2. the stripped sparse TensorFlow Lite model after **both** sparsity-aware optimization and float16 quantization.

    This lets us observe whether properly finalizing the pruned model leads to a more compact deployable representation.
    """)
    return


@app.cell
def _(
    evaluate_tflite_model,
    pruned_model,
    save_binary_model,
    stripped_model,
    tf,
    x_test,
    y_test,
):
    # TODO:
    # Part II: combine pruning and float16 quantization.
    #
    # 1. Convert the pruned model with mask using:
    #    - optimizations = [tf.lite.Optimize.DEFAULT]
    #    - supported_types = [tf.float16]
    # 2. Save and evaluate the float16 model with mask.
    # 3. Convert the stripped sparse model using:
    #    - optimizations = [tf.lite.Optimize.DEFAULT, tf.lite.Optimize.EXPERIMENTAL_SPARSITY]
    #    - supported_types = [tf.float16]
    # 4. Save and evaluate the stripped sparse + float16 model.

    def convert_to_tflite_float16_pruned(model, sparsity_aware=False):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        optimizations = [tf.lite.Optimize.DEFAULT]
        if sparsity_aware:
            optimizations.append(tf.lite.Optimize.EXPERIMENTAL_SPARSITY)
        converter.optimizations = optimizations
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        return tflite_model

    tflite_pruned_mask_fp16 = convert_to_tflite_float16_pruned(pruned_model, sparsity_aware=False)
    size_pruned_mask_fp16_kb = save_binary_model(tflite_pruned_mask_fp16, "model_pruned_mask_fp16.tflite")
    acc_pruned_mask_fp16, preds_pruned_mask_fp16 = evaluate_tflite_model(
        tflite_pruned_mask_fp16, x_test, y_test
    )

    tflite_stripped_sparse_fp16 = convert_to_tflite_float16_pruned(stripped_model, sparsity_aware=True)
    size_stripped_sparse_fp16_kb = save_binary_model(tflite_stripped_sparse_fp16, "model_stripped_sparse_fp16.tflite")
    acc_stripped_sparse_fp16, preds_stripped_sparse_fp16 = evaluate_tflite_model(
        tflite_stripped_sparse_fp16, x_test, y_test
    )

    print(f"Pruned + Float16 (with mask): acc={acc_pruned_mask_fp16:.4f} | size={size_pruned_mask_fp16_kb:.1f} KB")
    print(f"Stripped Sparse + Float16: acc={acc_stripped_sparse_fp16:.4f} | size={size_stripped_sparse_fp16_kb:.1f} KB")
    return (
        acc_pruned_mask_fp16,
        acc_stripped_sparse_fp16,
        preds_stripped_sparse_fp16,
        size_pruned_mask_fp16_kb,
        size_stripped_sparse_fp16_kb,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 12. Part II Comparison: Accuracy and Model Size
    """)
    return


@app.cell
def _(
    acc_pruned_mask_fp16,
    acc_pruned_with_mask,
    acc_stripped_sparse,
    acc_stripped_sparse_fp16,
    pd,
    size_pruned_mask_fp16_kb,
    size_pruned_with_mask_kb,
    size_stripped_sparse_fp16_kb,
    size_stripped_sparse_kb,
):
    # TODO:
    # Create a Part II comparison DataFrame with the columns:
    # Model, Format, Test Accuracy, Model Size (KB)
    #
    # Include:
    # - pruned FP32 with mask
    # - stripped sparse FP32
    # - pruned float16 with mask
    # - stripped sparse float16

    part2_results = pd.DataFrame({
        "Model": [
            "Pruned (with mask)",
            "Stripped Sparse",
            "Pruned (with mask)",
            "Stripped Sparse",
        ],
        "Format": ["FP32", "FP32 (sparse)", "Float16", "Float16 (sparse)"],
        "Test Accuracy": [
            acc_pruned_with_mask,
            acc_stripped_sparse,
            acc_pruned_mask_fp16,
            acc_stripped_sparse_fp16,
        ],
        "Model Size (KB)": [
            size_pruned_with_mask_kb,
            size_stripped_sparse_kb,
            size_pruned_mask_fp16_kb,
            size_stripped_sparse_fp16_kb,
        ],
    })
    part2_results
    return (part2_results,)


@app.cell
def _(part2_results, plt):
    # TODO:
    # Plot:
    # 1. a bar chart of Part II model sizes
    # 2. a bar chart of Part II test accuracies

    fig_p2, (ax_size2, ax_acc2) = plt.subplots(1, 2, figsize=(12, 4))

    labels_p2 = part2_results["Model"] + " (" + part2_results["Format"] + ")"

    ax_size2.bar(labels_p2, part2_results["Model Size (KB)"], color="steelblue")
    ax_size2.set_title("Part II Model Size (KB)")
    ax_size2.tick_params(axis="x", rotation=25)

    ax_acc2.bar(labels_p2, part2_results["Test Accuracy"], color="darkorange")
    ax_acc2.set_title("Part II Test Accuracy")
    ax_acc2.set_ylim(0, 1.0)
    ax_acc2.tick_params(axis="x", rotation=25)

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Confusion Matrix for the Stripped Sparse + Float16 Model
    """)
    return


@app.cell
def _(
    ConfusionMatrixDisplay,
    class_names,
    classification_report,
    confusion_matrix,
    plt,
    preds_stripped_sparse_fp16,
    y_test,
):
    # TODO:
    # Plot the confusion matrix for the stripped sparse + float16 TFLite model.
    # Print the classification report for the stripped sparse + float16 TFLite model.

    print(classification_report(y_test, preds_stripped_sparse_fp16, target_names=class_names))

    stripped_sparse_fp16_cm = confusion_matrix(y_test, preds_stripped_sparse_fp16)
    ConfusionMatrixDisplay(stripped_sparse_fp16_cm, display_labels=class_names).plot(
        xticks_rotation=45, cmap="Greens"
    )
    plt.title("Stripped Sparse + Float16 Model Confusion Matrix")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 13. Summary Questions

    Write short answers to the following:
    1. Did pruning alone reduce the TensorFlow Lite file size when the pruning wrappers were still attached?
       > No, it actually doubled it.
    3. Why does `strip_pruning(...)` matter before export?
       > it removes the mask and allows to ben
    5. Which model had the smallest file size in this notebook?
       > Stripped sparse float16.
    7. Did float16 quantization noticeably change the test accuracy?
       > No.
    9. If you were deploying this model on a resource-constrained device, which version would you choose and why?
       > Stripped sparse float32 since thats what devices usually have hw acceleration for.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 14. Submission Requirements

    Submit the following:
    - your completed notebook,
    - the generated `.tflite` files,
    - output cells or screenshots showing the comparison tables,
    - confusion matrices for the baseline model and your final highlighted compressed model,
    - and short written observations answering the summary questions.

    Make sure your notebook runs from top to bottom without errors using the **`Python (tinyml-arduino)`** kernel.
    """)
    return


@app.cell
def _(gc, tf):
    # gimme my vram back

    tf.keras.backend.clear_session()
    gc.collect()
    return


if __name__ == "__main__":
    app.run()
