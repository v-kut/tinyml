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
    # Knowledge Distillation with Pruning and Quantization for TinyML

    This is designed for a pinned local setup such as:

    - Python 3.11
    - TensorFlow 2.14.1
    - Keras 2.14.0
    - TensorFlow Model Optimization 0.8.0
    - NumPy < 2

    The workflow is organized into two parts:

    1. **Teacher and student training with knowledge distillation**
       We train a teacher CNN on MNIST, then train a smaller student using a standard distillation objective.

    2. **Student pruning + full INT8 quantization**
       We transfer the trained student weights into a prunable model, fine-tune with sparsity, strip the pruning wrappers, and export a **fully integer TensorFlow Lite model** using a representative dataset.

    ## Why this notebook matters

    In TinyML, these compression methods solve different deployment problems:

    - **Knowledge distillation** improves the performance of a smaller model by letting it learn from a stronger teacher.
    - **Pruning** introduces sparsity, which can reduce memory and computation when exported correctly.
    - **INT8 quantization** reduces storage precision and makes deployment more hardware-friendly.

    The key practical lesson is that these steps are often **combined**, not used in isolation.
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
    return OUTPUT_DIR, Path, np, pd, plt, tf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Imports, reproducibility, and experiment settings

    We keep the demo intentionally compact so that it stays practical for a local laptop workflow.

    A few implementation choices are deliberate:

    - We use **sparse integer labels** instead of one-hot labels.
    - The teacher and student output **logits**, which makes the distillation loss cleaner.
    - We train on a **subset** of MNIST to keep runtime short for demo purposes.
    - The final deployment artifact is a **fully integer** TFLite model for the pruned student.
    """)
    return


@app.cell
def _(tf):
    # Standard experiment settings
    TRAIN_SAMPLES = 12000
    TEST_SAMPLES = 3000

    EPOCHS_TEACHER = 15
    EPOCHS_DISTILL = 15
    EPOCHS_PRUNING = 5

    BATCH_SIZE = 64
    TEMPERATURE = 5.0
    ALPHA = 0.5

    MIN_EPOCHS_TEACHER = 5
    MIN_EPOCHS_DISTILL = 5
    MIN_EPOCHS_PRUNING = 2


    def make_adam(learning_rate=0.001):
        """Use the legacy Adam optimizer when available because it is faster on many Mac TensorFlow setups."""
        try:
            return tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
        except AttributeError:
            return tf.keras.optimizers.Adam(learning_rate=learning_rate)


    class MinimumEpochEarlyStopping(tf.keras.callbacks.Callback):
        """Early stopping that always allows a minimum number of epochs first."""

        def __init__(
            self,
            monitor="val_loss",
            mode="min",
            patience=2,
            min_epochs=5,
            restore_best_weights=True,
        ):
            super().__init__()
            self.monitor = monitor
            self.mode = mode
            self.patience = patience
            self.min_epochs = min_epochs
            self.restore_best_weights = restore_best_weights
            self.best_value = None
            self.best_weights = None
            self.wait = 0

        def _improved(self, current_value):
            if self.best_value is None:
                return True
            if self.mode == "max":
                return current_value > self.best_value
            return current_value < self.best_value

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            current_value = logs.get(self.monitor)

            if current_value is None:
                return

            epoch_number = epoch + 1

            if self._improved(current_value):
                self.best_value = current_value
                self.wait = 0
                if self.restore_best_weights:
                    self.best_weights = self.model.get_weights()
            elif epoch_number >= self.min_epochs:
                self.wait += 1

            if epoch_number < self.min_epochs:
                return

            if self.wait >= self.patience:
                self.model.stop_training = True
                if self.restore_best_weights and self.best_weights is not None:
                    self.model.set_weights(self.best_weights)
                print(
                    f"Early stopping after epoch {epoch_number}. "
                    f"Best {self.monitor}: {self.best_value:.4f}"
                )

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
        ALPHA,
        BATCH_SIZE,
        Conv2D,
        Dense,
        EPOCHS_DISTILL,
        EPOCHS_PRUNING,
        EPOCHS_TEACHER,
        Flatten,
        Input,
        MIN_EPOCHS_DISTILL,
        MIN_EPOCHS_PRUNING,
        MIN_EPOCHS_TEACHER,
        MaxPooling2D,
        MinimumEpochEarlyStopping,
        Model,
        PolynomialDecay,
        TEMPERATURE,
        TEST_SAMPLES,
        TRAIN_SAMPLES,
        UpdatePruningStep,
        make_adam,
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
    4. Keep labels as integer class indices for sparse categorical training.
    5. Use a moderate subset so the notebook remains lightweight for a demo.
    """)
    return


@app.cell
def _(TEST_SAMPLES, TRAIN_SAMPLES, mnist, np):
    # Load and preprocess MNIST
    (x_train_full, y_train_full), (x_test_full, y_test_full) = mnist.load_data()

    x_train = x_train_full[:TRAIN_SAMPLES].astype("float32") / 255.0
    y_train = y_train_full[:TRAIN_SAMPLES].astype("int32")

    x_test = x_test_full[:TEST_SAMPLES].astype("float32") / 255.0
    y_test = y_test_full[:TEST_SAMPLES].astype("int32")

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
    ## 3. Build and train the teacher CNN

    The teacher model is intentionally moderate in size:

    - one convolution layer
    - one max-pooling layer
    - one hidden dense layer
    - one logits output layer

    That is large enough to act as a useful teacher, while still being fast to train in a demo notebook.
    """)
    return


@app.cell
def _(
    BATCH_SIZE,
    Conv2D,
    Dense,
    EPOCHS_TEACHER,
    Flatten,
    Input,
    MIN_EPOCHS_TEACHER,
    MaxPooling2D,
    MinimumEpochEarlyStopping,
    Model,
    make_adam,
    tf,
    x_test,
    x_train,
    y_test,
    y_train,
):
    def create_teacher_model() -> tf.keras.Model:
        inputs = Input(shape=(28, 28, 1), name="image")
        x = Conv2D(32, (3, 3), activation="relu", name="conv_1")(inputs)
        x = MaxPooling2D((2, 2), name="pool_1")(x)
        x = Flatten(name="flatten")(x)
        x = Dense(128, activation="relu", name="dense_1")(x)
        outputs = Dense(10, name="logits")(x)
        return Model(inputs, outputs, name="teacher_cnn")


    teacher_model = create_teacher_model()
    teacher_model.compile(
        optimizer=make_adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    early_stop_teacher = MinimumEpochEarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=2,
        min_epochs=MIN_EPOCHS_TEACHER,
        restore_best_weights=True,
    )

    history_teacher = teacher_model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=EPOCHS_TEACHER,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop_teacher],
        verbose=1,
    )

    teacher_loss, teacher_accuracy = teacher_model.evaluate(x_test, y_test, verbose=0)
    print(f"Teacher Keras accuracy: {teacher_accuracy:.4f}")
    return history_teacher, teacher_accuracy, teacher_model


@app.cell
def _(history_teacher, pd, plt):
    # Plot teacher training history
    history_teacher_df = pd.DataFrame(history_teacher.history)
    (_fig, _axes) = plt.subplots(1, 2, figsize=(12, 4))
    history_teacher_df[['loss', 'val_loss']].plot(ax=_axes[0], title='Teacher Training Loss')
    _axes[0].set_xlabel('Epoch')
    _axes[0].set_ylabel('Loss')
    history_teacher_df[['accuracy', 'val_accuracy']].plot(ax=_axes[1], title='Teacher Training Accuracy')
    _axes[1].set_xlabel('Epoch')
    _axes[1].set_ylabel('Accuracy')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Build the student model and train it with knowledge distillation

    The student is smaller than the teacher:

    - fewer convolution filters
    - smaller hidden dense layer
    - same 10-class logits output

    We implement distillation using a custom `Distiller` class. The total student loss is:

    \[
    L = \alpha L_{\text{student}} + (1 - \alpha) L_{\text{distill}}
    \]

    where the distillation term compares teacher and student softened outputs using temperature scaling.
    """)
    return


@app.cell
def _(
    ALPHA,
    BATCH_SIZE,
    Conv2D,
    Dense,
    EPOCHS_DISTILL,
    Flatten,
    Input,
    MIN_EPOCHS_DISTILL,
    MaxPooling2D,
    MinimumEpochEarlyStopping,
    Model,
    TEMPERATURE,
    make_adam,
    teacher_model,
    tf,
    x_test,
    x_train,
    y_test,
    y_train,
):
    def create_student_model() -> tf.keras.Model:
        inputs = Input(shape=(28, 28, 1), name="image")
        x = Conv2D(16, (3, 3), activation="relu", name="conv_1")(inputs)
        x = MaxPooling2D((2, 2), name="pool_1")(x)
        x = Flatten(name="flatten")(x)
        x = Dense(64, activation="relu", name="dense_1")(x)
        outputs = Dense(10, name="logits")(x)
        return Model(inputs, outputs, name="student_cnn")


    class Distiller(tf.keras.Model):
        def __init__(self, student, teacher):
            super().__init__()
            self.student = student
            self.teacher = teacher

        def compile(
            self,
            optimizer,
            metrics,
            student_loss_fn,
            distillation_loss_fn,
            alpha=0.1,
            temperature=3.0,
        ):
            super().compile(optimizer=optimizer, metrics=metrics)
            self.student_loss_fn = student_loss_fn
            self.distillation_loss_fn = distillation_loss_fn
            self.alpha = alpha
            self.temperature = temperature

        def train_step(self, data):
            x_batch, y_batch = data

            teacher_predictions = self.teacher(x_batch, training=False)

            with tf.GradientTape() as tape:
                student_predictions = self.student(x_batch, training=True)

                student_loss = self.student_loss_fn(y_batch, student_predictions)
                distillation_loss = self.distillation_loss_fn(
                    tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                    tf.nn.softmax(student_predictions / self.temperature, axis=1),
                ) * (self.temperature ** 2)

                loss = self.alpha * student_loss + (1.0 - self.alpha) * distillation_loss

            gradients = tape.gradient(loss, self.student.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))

            self.compiled_metrics.update_state(y_batch, student_predictions)

            results = {metric.name: metric.result() for metric in self.metrics}
            results.update(
                {
                    "student_loss": student_loss,
                    "distillation_loss": distillation_loss,
                }
            )
            return results

        def test_step(self, data):
            x_batch, y_batch = data
            student_predictions = self.student(x_batch, training=False)

            student_loss = self.student_loss_fn(y_batch, student_predictions)
            self.compiled_metrics.update_state(y_batch, student_predictions)

            results = {metric.name: metric.result() for metric in self.metrics}
            results.update({"student_loss": student_loss})
            return results


    student_model = create_student_model()

    distiller = Distiller(student=student_model, teacher=teacher_model)
    distiller.compile(
        optimizer=make_adam(0.001),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        distillation_loss_fn=tf.keras.losses.KLDivergence(),
        alpha=ALPHA,
        temperature=TEMPERATURE,
    )

    early_stop_distill = MinimumEpochEarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=2,
        min_epochs=MIN_EPOCHS_DISTILL,
        restore_best_weights=True,
    )

    history_student = distiller.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=EPOCHS_DISTILL,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop_distill],
        verbose=1,
    )

    student_model.compile(
        optimizer=make_adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    student_loss, student_accuracy = student_model.evaluate(x_test, y_test, verbose=0)
    print(f"Distilled student Keras accuracy: {student_accuracy:.4f}")
    return history_student, student_accuracy, student_model


@app.cell
def _(history_student, pd, plt):
    # Plot student distillation history
    history_student_df = pd.DataFrame(history_student.history)
    plot_columns_loss = [c for c in ['student_loss', 'val_student_loss'] if c in history_student_df.columns]
    plot_columns_acc = [c for c in ['accuracy', 'val_accuracy'] if c in history_student_df.columns]
    (_fig, _axes) = plt.subplots(1, 2, figsize=(12, 4))
    if plot_columns_loss:
        history_student_df[plot_columns_loss].plot(ax=_axes[0], title='Student Distillation Loss')
    _axes[0].set_xlabel('Epoch')
    _axes[0].set_ylabel('Loss')
    if plot_columns_acc:
        history_student_df[plot_columns_acc].plot(ax=_axes[1], title='Student Distillation Accuracy')
    _axes[1].set_xlabel('Epoch')
    _axes[1].set_ylabel('Accuracy')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Helper functions

    Before running pruning and INT8 conversion, we define reusable utilities for:

    - standard TFLite conversion
    - model file size measurement
    - float and quantized TFLite evaluation
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
    ):
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        tflite_model = converter.convert()
        output_path.write_bytes(tflite_model)
        return output_path


    def evaluate_tflite_model(model_path: Path, x_eval: np.ndarray, y_eval: np.ndarray) -> float:
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        input_scale, input_zero_point = input_details["quantization"]
        output_scale, output_zero_point = output_details["quantization"]

        correct = 0

        for x_sample, y_true in zip(x_eval, y_eval):
            x_input = np.expand_dims(x_sample, axis=0).astype(np.float32)

            if input_details["dtype"] != np.float32 and input_scale > 0:
                x_input = np.round(x_input / input_scale + input_zero_point).astype(input_details["dtype"])

            interpreter.set_tensor(input_details["index"], x_input)
            interpreter.invoke()

            y_pred = interpreter.get_tensor(output_details["index"])

            if output_details["dtype"] != np.float32 and output_scale > 0:
                y_pred = (y_pred.astype(np.float32) - output_zero_point) * output_scale

            pred_label = int(np.argmax(y_pred, axis=1)[0])
            correct += int(pred_label == int(y_true))

        return correct / len(y_eval)


    def make_result_row(model_name: str, model_path: Path, accuracy: float) -> dict:
        return {
            "model": model_name,
            "file": model_path.name,
            "size_kb": round(get_size_kb(model_path), 2),
            "accuracy": round(float(accuracy), 4),
        }


    def print_kernel_sparsity_percent(model: tf.keras.Model, label: str) -> float:
        """Print simple kernel sparsity percentage for regular or TF-MOT pruned models."""
        total_weights = 0
        zero_weights = 0

        for layer in model.layers:
            # TF-MOT pruning wrapper case: use kernel * mask as the effective deployed weights.
            if hasattr(layer, "layer") and layer.__class__.__name__.lower().startswith("prune"):
                kernel = None
                candidate_masks = []

                for weight in layer.weights:
                    weight_name = weight.name.lower()
                    weight_value = weight.numpy()

                    if "kernel" in weight_name and "mask" not in weight_name:
                        kernel = weight_value
                    elif "mask" in weight_name:
                        candidate_masks.append(weight_value)

                if kernel is not None:
                    mask = None
                    for candidate_mask in candidate_masks:
                        if candidate_mask.shape == kernel.shape:
                            mask = candidate_mask
                            break

                    effective_kernel = kernel * mask if mask is not None else kernel
                    total_weights += effective_kernel.size
                    zero_weights += int(np.sum(np.isclose(effective_kernel, 0.0)))

            # Standard Keras layer case: count only kernel-like arrays, not bias vectors.
            else:
                for weight_value in layer.get_weights():
                    if weight_value.ndim > 1:
                        total_weights += weight_value.size
                        zero_weights += int(np.sum(np.isclose(weight_value, 0.0)))

        sparsity_percent = 100.0 * zero_weights / total_weights if total_weights > 0 else 0.0
        print(
            f"{label} kernel sparsity: {sparsity_percent:.2f}% "
            f"({zero_weights:,} / {total_weights:,} zero kernel weights)"
        )
        return sparsity_percent

    return (
        convert_to_tflite,
        evaluate_tflite_model,
        make_result_row,
        print_kernel_sparsity_percent,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part I. Teacher and student deployment references

    Before pruning the student, we export standard TFLite versions of the teacher and the distilled student.

    This gives us two useful deployment references:

    1. a stronger but larger **teacher**
    2. a smaller distilled **student**

    That way, when we later prune and quantize the student, we can compare the final artifact against both of these baselines.
    """)
    return


@app.cell
def _(
    OUTPUT_DIR,
    convert_to_tflite,
    evaluate_tflite_model,
    make_result_row,
    np,
    pd,
    student_accuracy,
    student_model,
    teacher_accuracy,
    teacher_model,
    x_test,
    y_test,
):
    teacher_tflite_path = OUTPUT_DIR / "teacher_model.tflite"
    student_tflite_path = OUTPUT_DIR / "student_model.tflite"

    _ = convert_to_tflite(teacher_model, teacher_tflite_path)
    _ = convert_to_tflite(student_model, student_tflite_path)

    teacher_tflite_accuracy = evaluate_tflite_model(teacher_tflite_path, x_test, y_test)
    student_tflite_accuracy = evaluate_tflite_model(student_tflite_path, x_test, y_test)

    reference_results = pd.DataFrame(
        [
            {
                "model": "Teacher Keras",
                "file": "-",
                "size_kb": np.nan,
                "accuracy": round(float(teacher_accuracy), 4),
            },
            {
                "model": "Student Keras",
                "file": "-",
                "size_kb": np.nan,
                "accuracy": round(float(student_accuracy), 4),
            },
            make_result_row("Teacher TFLite", teacher_tflite_path, teacher_tflite_accuracy),
            make_result_row("Student TFLite", student_tflite_path, student_tflite_accuracy),
        ]
    )

    reference_results
    return (reference_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part II. Student pruning + full INT8 quantization

    This section corresponds to the student TODO block in the original notebook.

    The intended answer key workflow is:

    1. clone the trained student model
    2. apply pruning with a gradual sparsity schedule
    3. fine-tune the pruned student for a few epochs
    4. strip the pruning wrappers
    5. export a **fully integer** TFLite model using:
       - a representative dataset
       - INT8 built-in ops
       - INT8 input and output tensors

    This produces the final deployment artifact:
    `pruned_quantized_student_model.tflite`
    """)
    return


@app.cell
def _(
    BATCH_SIZE,
    EPOCHS_PRUNING,
    MIN_EPOCHS_PRUNING,
    MinimumEpochEarlyStopping,
    PolynomialDecay,
    UpdatePruningStep,
    make_adam,
    np,
    print_kernel_sparsity_percent,
    prune_low_magnitude,
    student_model,
    tf,
    x_test,
    x_train,
    y_test,
    y_train,
):
    def apply_pruning_to_trained_student(
        trained_student: tf.keras.Model,
        batch_size: int,
        epochs_pruning: int,
    ) -> tf.keras.Model:
        student_clone = tf.keras.models.clone_model(trained_student)
        student_clone.set_weights(trained_student.get_weights())

        steps_per_epoch = int(np.ceil(len(x_train) / batch_size))

        pruning_params = {
            "pruning_schedule": PolynomialDecay(
                initial_sparsity=0.50,
                final_sparsity=0.90,
                begin_step=0,
                end_step=steps_per_epoch * epochs_pruning,
            )
        }

        pruned_student = prune_low_magnitude(student_clone, **pruning_params)
        return pruned_student


    pruned_student_model = apply_pruning_to_trained_student(
        trained_student=student_model,
        batch_size=BATCH_SIZE,
        epochs_pruning=EPOCHS_PRUNING,
    )

    pruned_student_model.compile(
        optimizer=make_adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    early_stop_pruning = MinimumEpochEarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=1,
        min_epochs=MIN_EPOCHS_PRUNING,
        restore_best_weights=False,
    )

    history_pruned = pruned_student_model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=EPOCHS_PRUNING,
        batch_size=BATCH_SIZE,
        callbacks=[UpdatePruningStep(), early_stop_pruning],
        verbose=1,
    )

    # Simple sparsity check after pruning, before removing pruning wrappers.
    print_kernel_sparsity_percent(pruned_student_model, "Pruned student with pruning mask")

    pruned_student_loss, pruned_student_accuracy = pruned_student_model.evaluate(x_test, y_test, verbose=0)
    print(f"Pruned student Keras accuracy: {pruned_student_accuracy:.4f}")
    return history_pruned, pruned_student_model


@app.cell
def _(history_pruned, pd, plt):
    # Plot pruning training history
    history_pruned_df = pd.DataFrame(history_pruned.history)
    (_fig, _axes) = plt.subplots(1, 2, figsize=(12, 4))
    history_pruned_df[['loss', 'val_loss']].plot(ax=_axes[0], title='Pruned Student Training Loss')
    _axes[0].set_xlabel('Epoch')
    _axes[0].set_ylabel('Loss')
    history_pruned_df[['accuracy', 'val_accuracy']].plot(ax=_axes[1], title='Pruned Student Training Accuracy')
    _axes[1].set_xlabel('Epoch')
    _axes[1].set_ylabel('Accuracy')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Completed answer-key version of the student TODO block

    The next cell directly fills in the missing student code:

    - pruning the trained student
    - setting up the TFLite converter for **full INT8 quantization**
    - defining a representative dataset
    - saving the final deployment artifact
    """)
    return


@app.cell
def _(
    OUTPUT_DIR,
    np,
    print_kernel_sparsity_percent,
    pruned_student_model,
    strip_pruning,
    tf,
    x_train,
):
    # Apply pruning to the trained student model
    stripped_pruned_student_model = strip_pruning(pruned_student_model)

    # Simple sparsity check after removing pruning wrappers.
    print_kernel_sparsity_percent(stripped_pruned_student_model, "Pruned student after stripping")

    # Convert the pruned model to TFLite using full integer quantization
    def representative_dataset():
        for i in range(min(200, len(x_train))):
            yield [x_train[i:i+1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(stripped_pruned_student_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT, tf.lite.Optimize.EXPERIMENTAL_SPARSITY]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    pruned_quantized_student_model = converter.convert()

    # Save the pruned and quantized student model
    pruned_quantized_student_path = OUTPUT_DIR / "pruned_quantized_student_model.tflite"
    with open(pruned_quantized_student_path, "wb") as f:
        f.write(pruned_quantized_student_model)

    print("Pruned + Quantized Student Model saved to:", pruned_quantized_student_path)
    return (pruned_quantized_student_path,)


@app.cell
def _(
    evaluate_tflite_model,
    make_result_row,
    pd,
    pruned_quantized_student_path,
    x_test,
    y_test,
):
    # Evaluate the final INT8 TFLite model and compare with earlier references
    pruned_quantized_student_accuracy = evaluate_tflite_model(
        pruned_quantized_student_path,
        x_test,
        y_test,
    )

    final_results = pd.DataFrame(
        [
            make_result_row(
                "Pruned + INT8 Student TFLite",
                pruned_quantized_student_path,
                pruned_quantized_student_accuracy,
            )
        ]
    )

    final_results
    return (final_results,)


@app.cell
def _(final_results, pd, reference_results):
    # Side-by-side comparison
    summary_results = pd.concat(
        [
            reference_results.assign(pipeline="Teacher / Student Reference"),
            final_results.assign(pipeline="Pruned + INT8 Deployment"),
        ],
        ignore_index=True,
    )

    summary_results = summary_results[["pipeline", "model", "file", "size_kb", "accuracy"]]
    summary_results
    return (summary_results,)


@app.cell
def _(plt, summary_results):
    # Visual comparison of deployment artifact sizes
    plot_df = summary_results.dropna(subset=["size_kb"]).copy()

    plt.figure(figsize=(9, 4))
    plt.bar(plot_df["model"], plot_df["size_kb"])
    plt.ylabel("Model size (KB)")
    plt.title("Knowledge distillation, pruning, and INT8 deployment size comparison")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.show()

    print("Deployment accuracies:")
    for _, row in plot_df.iterrows():
        matching_accuracy = summary_results.loc[summary_results["model"] == row["model"], "accuracy"].iloc[0]
        print(f"{row['model']}: {matching_accuracy:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 6. Key takeaways

    ### What knowledge distillation does
    Knowledge distillation improves the quality of a compact student by transferring information from a stronger teacher.

    ### What pruning adds
    Pruning introduces sparsity into the student, but the model should be **stripped** before final export so the deployment artifact is cleaner.

    ### Why full INT8 quantization matters
    Full integer quantization reduces storage precision and produces a model that is much more suitable for resource-constrained deployment.

    ### Practical TinyML lesson
    A sensible combined pipeline is:

    1. train a stronger teacher
    2. distill a smaller student
    3. prune the student if sparsity is desired
    4. strip pruning wrappers
    5. export the final model using full INT8 TFLite conversion

    That sequence is a realistic compression workflow for TinyML deployment.
    """)
    return


if __name__ == "__main__":
    app.run()
