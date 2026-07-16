import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # EE 446 TinyML — Knowledge Distillation with Pruning and Quantization

    ## Student TODO Version: Compression of a DNN Using the UCI Human Activity Recognition Dataset

    In this version, key parts of the notebook have been left for you to complete.
    Follow the instructions in each code cell and fill in the missing sections marked with `#<--- Enter your code here --->#`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Environment Setup

    This notebook assumes you are running it with the **`Python (tinyml-arduino)`** kernel.

    Expected environment:
    - TensorFlow 2.14.1
    - TensorFlow Model Optimization 0.8.0
    - NumPy, Pandas, Matplotlib, Scikit-learn
    - No in-notebook package reinstallation is required

    Use **Kernel → Change Kernel → `Python (tinyml-arduino)`** if needed.
    """)
    return


@app.cell
def _():
    import os
    import math
    import zipfile
    import random
    import urllib.request
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        ConfusionMatrixDisplay
    )

    from tensorflow_model_optimization.sparsity.keras import (
        prune_low_magnitude,
        PolynomialDecay,
        UpdatePruningStep,
        strip_pruning
    )

    import gc

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    tf.keras.backend.clear_session()
    gc.collect()

    print("TensorFlow version:", tf.__version__)
    return (
        ConfusionMatrixDisplay,
        Path,
        PolynomialDecay,
        UpdatePruningStep,
        accuracy_score,
        classification_report,
        confusion_matrix,
        keras,
        layers,
        math,
        np,
        pd,
        plt,
        prune_low_magnitude,
        strip_pruning,
        tf,
        zipfile,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Download and Extract the UCI HAR Dataset

    The UCI HAR dataset contains:
    - **561 numerical features** extracted from smartphone sensor signals
    - **6 human activity classes**
    - A predefined **training split** and **test split**

    This makes it a strong fit for a **fully connected DNN** and for TinyML-oriented compression experiments.
    """)
    return


@app.cell
def _(Path, zipfile):
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
    return BASE_DIR, dataset_dir


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
            dataset_dir / "train" / "X_train.txt", sep='\s+', header=None
        ).values.astype("float32")
        y_train = pd.read_csv(
            dataset_dir / "train" / "y_train.txt", sep='\s+', header=None
        ).values.astype("float32")
        x_test = pd.read_csv(
            dataset_dir / "test" / "X_test.txt", sep='\s+', header=None
        ).values.astype("float32")
        y_test = pd.read_csv(
            dataset_dir / "test" / "y_test.txt", sep='\s+', header=None
        ).values.astype("float32")

        y_train = y_train - 1
        y_test = y_test - 1
        y_train = y_train.flatten()
        y_test = y_test.flatten()

        return (x_train, y_train, x_test, y_test)

    X_train, y_train, X_test, y_test = load_har_data()

    class_names = [
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING",
    ]

    num_features = X_train.shape[1]
    num_classes = len(class_names)

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape :", X_test.shape)
    print("y_test shape :", y_test.shape)
    print("Number of features:", num_features)
    print("Number of classes :", num_classes)
    return (
        X_test,
        X_train,
        class_names,
        num_classes,
        num_features,
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
    label_counts = pd.Series(y_train).value_counts().sort_index()

    dataset_summary = pd.DataFrame({
        "Class Index": list(range(num_classes)),
        "Class Name": class_names,
        "Training Samples": label_counts.values,
    })

    dataset_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Define the Teacher and Student Models

    The **teacher model** is intentionally larger and more expressive.
    The **student model** is smaller and is the model we ultimately want to deploy.
    """)
    return


@app.cell
def _(keras, layers, num_classes, num_features):
    def build_teacher_model(input_dim, num_classes):
        # TODO:
        # Build a larger teacher DNN suitable for 561 numerical input features.
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(256, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ])
        return model

    def build_student_model(input_dim, num_classes):
        # TODO:
        # Build a smaller student DNN.
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ])
        return model

    teacher_model = build_teacher_model(num_features, num_classes)
    student_baseline_model = build_student_model(num_features, num_classes)

    teacher_model.summary()
    return build_student_model, student_baseline_model, teacher_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Train the Teacher Model
    """)
    return


@app.cell
def _(X_train, keras, teacher_model, y_train):
    teacher_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    teacher_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True
        )
    ]

    # TODO:
    # Train the teacher model on the UCI HAR training split.
    teacher_history = teacher_model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=30,
        batch_size=64,
        callbacks=teacher_callbacks,
        verbose=1,
    )
    return (teacher_history,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Teacher Training Curves
    """)
    return


@app.cell
def _(pd, plt, teacher_history):
    teacher_history_df = pd.DataFrame(teacher_history.history)

    plt.figure(figsize=(8, 4))
    plt.plot(teacher_history_df["accuracy"], label="Train Accuracy")
    plt.plot(teacher_history_df["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Teacher Model Training Curve")
    plt.legend()
    plt.grid(True)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Evaluate the Teacher Model
    """)
    return


@app.cell
def _(
    ConfusionMatrixDisplay,
    X_test,
    accuracy_score,
    class_names,
    classification_report,
    confusion_matrix,
    np,
    plt,
    teacher_model,
    y_test,
):
    teacher_probs = teacher_model.predict(X_test, verbose=0)
    teacher_preds = np.argmax(teacher_probs, axis=1)
    teacher_acc = accuracy_score(y_test, teacher_preds)
    print(f"Teacher Test Accuracy: {teacher_acc:.4f}\n")
    print(classification_report(y_test, teacher_preds, target_names=class_names, digits=4))

    disp_teacher = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_test, teacher_preds), display_labels=class_names
    )
    fig_teacher, ax_teacher = plt.subplots(figsize=(8, 6))
    disp_teacher.plot(ax=ax_teacher, xticks_rotation=45, cmap="Blues", colorbar=False)
    plt.title("Teacher Model Confusion Matrix")
    plt.show()
    return (teacher_acc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Train a Baseline Student Model (Hard Labels Only)

    Before applying knowledge distillation, we train the smaller student model in the standard way.
    This gives us a fair baseline for comparison.
    """)
    return


@app.cell
def _(X_train, keras, student_baseline_model, y_train):
    student_baseline_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    student_callbacks_1 = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3, restore_best_weights=True
        )
    ]

    student_baseline_history_1 = student_baseline_model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=64,
        callbacks=student_callbacks_1,
        verbose=1,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Evaluate the Baseline Student Model
    """)
    return


@app.cell
def _(
    ConfusionMatrixDisplay,
    X_test,
    accuracy_score,
    class_names,
    classification_report,
    confusion_matrix,
    np,
    plt,
    student_baseline_model,
    y_test,
):
    student_baseline_probs = student_baseline_model.predict(X_test, verbose=0)
    student_baseline_preds = np.argmax(student_baseline_probs, axis=1)
    student_baseline_acc = accuracy_score(y_test, student_baseline_preds)
    print(f"Baseline Student Test Accuracy: {student_baseline_acc:.4f}\n")
    print(classification_report(y_test, student_baseline_preds, target_names=class_names, digits=4))

    disp_base = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_test, student_baseline_preds), display_labels=class_names
    )
    fig_base, ax_base = plt.subplots(figsize=(8, 6))
    disp_base.plot(ax=ax_base, xticks_rotation=45, cmap="Oranges", colorbar=False)
    plt.title("Baseline Student Confusion Matrix")
    plt.show()
    return (student_baseline_acc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part I: Knowledge Distillation

    ## 10. Distillation Utilities

    The distilled student is trained to optimize:
    - a **hard-label loss** using the true class labels
    - a **soft-label loss** using the teacher's softened probability distribution
    """)
    return


@app.cell
def _(keras, tf):
    class Distiller(keras.Model):
        def __init__(self, student, teacher):
            super().__init__()
            self.teacher = teacher
            self.student = student

        def compile(
            self,
            optimizer,
            metrics,
            student_loss_fn,
            distillation_loss_fn,
            alpha=0.3,
            temperature=4.0,
        ):
            super().compile(optimizer=optimizer, metrics=metrics)
            self.student_loss_fn = student_loss_fn
            self.distillation_loss_fn = distillation_loss_fn
            self.alpha = alpha
            self.temperature = temperature

        def train_step(self, data):
            x, y = data

            # TODO:
            # 1. Obtain teacher predictions with training=False
            # 2. Compute student predictions inside GradientTape
            # 3. Compute student_loss using the hard labels
            # 4. Compute distillation_loss using softened teacher/student outputs
            # 5. Combine the two losses using alpha
            teacher_predictions = self.teacher(x, training=False)

            with tf.GradientTape() as tape:
                student_predictions = self.student(x, training=True)
        
                student_loss = self.student_loss_fn(y, student_predictions)
        
                distillation_loss = self.distillation_loss_fn(
                    tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                    tf.nn.softmax(student_predictions / self.temperature, axis=1)
                )
        
                loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

            trainable_vars = self.student.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

            self.compiled_metrics.update_state(y, student_predictions)

            results = {m.name: m.result() for m in self.metrics}
            results.update({
                "student_loss": student_loss,
                "distillation_loss": distillation_loss,
            })
            return results

        def test_step(self, data):
            x, y = data
            y_prediction = self.student(x, training=False)

            student_loss = self.student_loss_fn(y, y_prediction)
            self.compiled_metrics.update_state(y, y_prediction)

            results = {m.name: m.result() for m in self.metrics}
            results.update({"student_loss": student_loss})
            return results

    return (Distiller,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. Train the Distilled Student
    """)
    return


@app.cell
def _(
    Distiller,
    X_train,
    build_student_model,
    keras,
    num_classes,
    num_features,
    teacher_model,
    y_train,
):
    distilled_student = build_student_model(num_features, num_classes)
    distiller = Distiller(student=distilled_student, teacher=teacher_model)
    distiller.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        student_loss_fn=keras.losses.SparseCategoricalCrossentropy(),
        distillation_loss_fn=keras.losses.KLDivergence(),
        alpha=0.3,
        temperature=4.0,
    )
    distillation_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3, restore_best_weights=True
        )
    ]

    distillation_history = distiller.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=64,
        callbacks=distillation_callbacks,
        verbose=1,
    )
    return distillation_history, distilled_student


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Distillation Training Curves
    """)
    return


@app.cell
def _(distillation_history, pd, plt):
    distillation_history_df = pd.DataFrame(distillation_history.history)
    plt.figure(figsize=(8, 4))
    plt.plot(distillation_history_df['accuracy'], label='Train Accuracy')
    plt.plot(distillation_history_df['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Distilled Student Training Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 12. Evaluate the Distilled Student
    """)
    return


@app.cell
def _(
    ConfusionMatrixDisplay,
    X_test,
    accuracy_score,
    class_names,
    classification_report,
    confusion_matrix,
    distilled_student,
    np,
    plt,
    y_test,
):
    distilled_probs = distilled_student.predict(X_test, verbose=0)
    distilled_preds = np.argmax(distilled_probs, axis=1)
    distilled_acc = accuracy_score(y_test, distilled_preds)
    print(f'Distilled Student Test Accuracy: {distilled_acc:.4f}\n')
    print(classification_report(y_test, distilled_preds, target_names=class_names, digits=4))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, distilled_preds), display_labels=class_names)
    (fig, ax) = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, xticks_rotation=45, cmap='Blues', colorbar=False)
    plt.title('Distilled Student - Confusion Matrix')
    plt.show()
    return (distilled_acc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 13. Part I Comparison: Teacher vs Student vs Distilled Student
    """)
    return


@app.cell
def _(
    distilled_acc,
    distilled_student,
    pd,
    student_baseline_acc,
    student_baseline_model,
    teacher_acc,
    teacher_model,
):
    part1_results = pd.DataFrame({
        "Model": ["Teacher", "Baseline Student", "Distilled Student"],
        "Test Accuracy": [teacher_acc, student_baseline_acc, distilled_acc],
        "Trainable Params": [
            teacher_model.count_params(),
            student_baseline_model.count_params(),
            distilled_student.count_params(),
        ],
    })
    part1_results
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part II: Pruning and Quantization of the Distilled Student

    ## 14. TensorFlow Lite Utilities
    """)
    return


@app.cell
def _(BASE_DIR, Path, X_train, accuracy_score, np, tf):
    def save_binary_model(model_content, filename):
        model_dir = Path(f"./{BASE_DIR}/knowledge_distillation_pruning_quantization/models/")
        model_dir.mkdir(parents=True, exist_ok=True)

        filepath = model_dir / filename
        filepath.write_bytes(model_content)

        return filepath.stat().st_size / 1024  # KB

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

            if input_details["dtype"] == np.int8:
                x = np.round(x / input_scale + input_zero_point).astype(np.int8)
            elif input_details["dtype"] == np.uint8:
                x = np.round(x / input_scale + input_zero_point).astype(np.uint8)
            else:
                x = x.astype(input_details["dtype"])

            interpreter.set_tensor(input_details["index"], x)
            interpreter.invoke()

            output = interpreter.get_tensor(output_details["index"])

            if output_details["dtype"] == np.int8:
                output = (output.astype(np.float32) - output_zero_point) * output_scale
            elif output_details["dtype"] == np.uint8:
                output = (output.astype(np.float32) - output_zero_point) * output_scale

            pred = int(np.argmax(output, axis=1)[0])
            y_pred.append(pred)

        acc = accuracy_score(y_true, y_pred)
        return acc, np.array(y_pred)

    def convert_to_tflite_fp32(model):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        return converter.convert()

    def representative_data_gen():
        for i in range(min(200, len(X_train))):
            yield [X_train[i:i+1]]

    return (
        convert_to_tflite_fp32,
        evaluate_tflite_model,
        representative_data_gen,
        save_binary_model,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 15. Convert the Distilled Student to TensorFlow Lite
    """)
    return


@app.cell
def _(
    PolynomialDecay,
    UpdatePruningStep,
    X_train,
    distilled_student,
    keras,
    math,
    prune_low_magnitude,
    tf,
    y_train,
):
    pruning_epochs = 10
    batch_size = 64
    steps_per_epoch = math.ceil((0.8 * len(X_train)) / batch_size)

    pruning_params = {
        "pruning_schedule": PolynomialDecay(
            initial_sparsity=0.20,
            final_sparsity=0.85,
            begin_step=0,
            end_step=steps_per_epoch * pruning_epochs,
        )
    }

    # TODO:
    # 1. Clone the distilled student model.
    # 2. Copy the distilled student weights into the cloned model.
    # 3. Wrap the cloned model using prune_low_magnitude with pruning_params.
    student_for_pruning = tf.keras.models.clone_model(distilled_student)
    student_for_pruning.set_weights(distilled_student.get_weights())

    pruned_distilled_model = prune_low_magnitude(student_for_pruning, **pruning_params)

    pruned_distilled_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    pruning_callbacks = [
        UpdatePruningStep(),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True
        )
    ]

    # TODO:
    # Fine-tune the pruned distilled model using:
    # - X_train and y_train
    # - validation_split=0.2
    # - epochs=pruning_epochs
    # - batch_size=batch_size
    # - callbacks=pruning_callbacks
    # - verbose=1
    pruning_history = pruned_distilled_model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=pruning_epochs,
        batch_size=batch_size,
        callbacks=pruning_callbacks,
        verbose=1,
    )
    return (pruned_distilled_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 16. Apply Magnitude-Based Pruning to the Distilled Student
    """)
    return


@app.cell
def _(
    X_test,
    convert_to_tflite_fp32,
    evaluate_tflite_model,
    pruned_distilled_model,
    save_binary_model,
    strip_pruning,
    tf,
    y_test,
):
    # TODO:
    # Convert the pruned model with the pruning wrappers still attached.
    pruned_with_mask_tflite = convert_to_tflite_fp32(pruned_distilled_model)
    pruned_with_mask_size_kb = save_binary_model(pruned_with_mask_tflite, "pruned_distilled_with_mask_fp32.tflite")
    pruned_with_mask_acc, pruned_with_mask_preds = evaluate_tflite_model(pruned_with_mask_tflite, X_test, y_test)

    # TODO:
    # Strip the pruning wrappers and convert again using sparse optimization.
    stripped_pruned_model = strip_pruning(pruned_distilled_model)

    converter = tf.lite.TFLiteConverter.from_keras_model(stripped_pruned_model)
    converter.optimizations = [tf.lite.Optimize.EXPERIMENTAL_SPARSITY]
    stripped_sparse_tflite = converter.convert()

    stripped_sparse_size_kb = save_binary_model(stripped_sparse_tflite, "distilled_stripped_sparse_fp32.tflite")
    stripped_sparse_acc, stripped_sparse_preds = evaluate_tflite_model(stripped_sparse_tflite, X_test, y_test)

    print(f"Pruned distilled model with mask accuracy: {pruned_with_mask_acc:.4f}")
    print(f"Pruned distilled model with mask size (KB): {pruned_with_mask_size_kb:.2f}")
    print(f"Stripped sparse distilled model accuracy: {stripped_sparse_acc:.4f}")
    print(f"Stripped sparse distilled model size (KB): {stripped_sparse_size_kb:.2f}")
    return (
        pruned_with_mask_acc,
        pruned_with_mask_size_kb,
        stripped_pruned_model,
        stripped_sparse_acc,
        stripped_sparse_size_kb,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 17. Apply Full Integer Quantization to the Stripped Sparse Distilled Student
    """)
    return


@app.cell
def _(
    X_test,
    evaluate_tflite_model,
    representative_data_gen,
    save_binary_model,
    stripped_pruned_model,
    tf,
    y_test,
):
    converter32 = tf.lite.TFLiteConverter.from_keras_model(stripped_pruned_model)
    converter32.optimizations = [tf.lite.Optimize.DEFAULT, tf.lite.Optimize.EXPERIMENTAL_SPARSITY]
    converter32.representative_dataset = representative_data_gen
    converter32.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter32.inference_input_type = tf.int8
    converter32.inference_output_type = tf.int8

    stripped_sparse_int8_tflite = converter32.convert()
    stripped_sparse_int8_size_kb = save_binary_model(
        stripped_sparse_int8_tflite,
        "distilled_stripped_sparse_int8.tflite"
    )
    stripped_sparse_int8_acc, stripped_sparse_int8_preds = evaluate_tflite_model(
        stripped_sparse_int8_tflite,
        X_test,
        y_test
    )

    print(f"Stripped Sparse + INT8 Accuracy: {stripped_sparse_int8_acc:.4f}")
    print(f"Stripped Sparse + INT8 Size (KB): {stripped_sparse_int8_size_kb:.2f}")
    return (
        stripped_sparse_int8_acc,
        stripped_sparse_int8_preds,
        stripped_sparse_int8_size_kb,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 19. Part II Comparison: Distillation, Pruning, and Quantization
    """)
    return


@app.cell
def _(
    X_test,
    convert_to_tflite_fp32,
    distilled_student,
    evaluate_tflite_model,
    pd,
    pruned_with_mask_acc,
    pruned_with_mask_size_kb,
    save_binary_model,
    stripped_sparse_acc,
    stripped_sparse_int8_acc,
    stripped_sparse_int8_size_kb,
    stripped_sparse_size_kb,
    y_test,
):
    distilled_fp32_tflite = convert_to_tflite_fp32(distilled_student)
    distilled_fp32_size_kb = save_binary_model(distilled_fp32_tflite, "distilled_student_fp32.tflite")
    distilled_fp32_acc, distilled_fp32_preds = evaluate_tflite_model(distilled_fp32_tflite, X_test, y_test)

    part2_results = pd.DataFrame([
        ["Distilled Student TFLite", "FP32", distilled_fp32_acc, distilled_fp32_size_kb],
        ["Pruned Distilled TFLite (with mask)", "FP32", pruned_with_mask_acc, pruned_with_mask_size_kb],
        ["Stripped Sparse Distilled TFLite", "FP32 + Sparse", stripped_sparse_acc, stripped_sparse_size_kb],
        ["Stripped Sparse Distilled TFLite", "INT8 + Sparse", stripped_sparse_int8_acc, stripped_sparse_int8_size_kb],
    ], columns=["Model", "Format", "Test Accuracy", "Model Size (KB)"])

    part2_results
    return (part2_results,)


@app.cell
def _(part2_results, plt):
    plt.figure(figsize=(8, 4))
    plt.bar(part2_results["Format"], part2_results["Model Size (KB)"])
    plt.ylabel("Model Size (KB)")
    plt.title("Compressed Distilled Student Model Sizes")
    plt.grid(axis="y")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Confusion Matrix for the Final Sparse INT8 Distilled Student
    """)
    return


@app.cell
def _(
    ConfusionMatrixDisplay,
    class_names,
    confusion_matrix,
    plt,
    stripped_sparse_int8_preds,
    y_test,
):
    disp_1 = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, stripped_sparse_int8_preds), display_labels=class_names)
    (fig_1, ax_1) = plt.subplots(figsize=(8, 6))
    disp_1.plot(ax=ax_1, xticks_rotation=45, cmap='Blues', colorbar=False)
    plt.title('Final Distilled + Pruned + INT8 Model - Confusion Matrix')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 20. Summary Questions

    Answer the following in your lab report:
    1. How did the **baseline student** compare with the **distilled student**?
       > The distilled student slightly outperformed the baseline student despite having the same architecture, since it had access to the teacher's output distribution during training.
    2. Did **knowledge distillation** help the smaller model retain performance?
       > Yes, knowledge distillation helped the small student retain accuracy closer to the teacher than training on hard labels alone.
    3. What happened to the model size after **pruning** and after **INT8 quantization**?
       > Pruning alone (with the mask still attached) didn't shrink the file much, but stripping     > the pruning wrappers and converting with EXPERIMENTAL_SPARSITY reduced size noticeably,     > and INT8 quantization on top reduced it further (roughly a 4x cut from FP32).
    4. Which model would you choose for **Arduino deployment**, and why?
       > The stripped sparse INT8 model, since it offers the smallest footprint and fastest inference with negligible accuracy loss.
    5. Why is the final **sparse INT8 model** a good TinyML deployment candidate?
       > It combines three compressions into one compact, integer-only model that's well-suited for microcontroller inference.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 21. Submission Requirements

    Submit the following:
    1. Your completed notebook
    2. Screenshots of the most important results:
       - teacher accuracy
       - baseline student accuracy
       - distilled student accuracy
       - final sparse INT8 model accuracy and size
    3. The exported TensorFlow Lite model:
       - `distilled_stripped_sparse_int8.tflite`
    4. Short answers to the summary questions
    """)
    return


if __name__ == "__main__":
    app.run()
