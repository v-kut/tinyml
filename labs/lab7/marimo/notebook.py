import marimo

__generated_with = "0.23.15"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # EE 446 TinyML - Lab 7 Part II: Tiny Ensemble Learning

    This notebook demonstrates a complete TinyML ensemble-learning pipeline for IMU-based human activity classification. The code is already implemented. Your task is to run the notebook, understand the pipeline, and answer the discussion questions at the end.

    The pipeline uses the `mHealth_subject6.log` dataset and builds three model branches using different input representations:

    - Raw IMU windows
    - Standard-scaled IMU windows
    - Min-max-scaled IMU windows

    Each branch trains an autoencoder, uses the encoder output as a latent feature vector, and trains a classifier. The three classifier softmax outputs are then concatenated and passed into a small stacked meta-classifier. The notebook then compresses selected models using pruning, quantization-aware training (QAT), and full integer TFLite conversion for TinyML deployment.

    This notebook is designed for the local `tinyml-arduino` environment.

    Recommended way to start JupyterLab:

    ```bash
    source ~/ai/projects/tinyml-arduino/bin/activate
    jupyter lab
    ```

    Required files in the same folder as this notebook:

    - `mHealth_subject6.log`
    - Optional for the final validation section: `imu_500_rows.csv`

    You do not need to fill in missing code. Run the cells in order and use the results to answer the discussion questions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Discussion Questions

    Submit your answers for Part II in the same PDF used for Lab 8 Part I.

    ### Question 1: Model architecture and ensemble flow

    Draw a complete diagram of the full ensemble pipeline. Your diagram should include enough detail for someone to reconstruct the model architecture from your drawing.

    Include the following details:

    - The input window size and number of input neurons. In this notebook, each input window has 100 time steps and 6 IMU features, so the flattened input dimension is 600.
    - The three input branches: raw input, standard-scaled input, and min-max-scaled input.
    - The encoder architecture in each branch, including the number of hidden layers, number of neurons in each layer, and activation function used in each layer.
    - The classifier architecture in each branch, including the number of input neurons, hidden-layer neurons, output neurons, and activation functions.
    - The number of models used in the ensemble and how they are connected.
    - The size of each branch output and how the three outputs are concatenated before being passed into the stacked meta-classifier.
    - The stacked meta-classifier architecture, including input size, hidden layer size, output size, and activation functions.

    ### Question 2: Pruning and QAT methodology

    Explain how this notebook compresses the models for TinyML deployment.

    In your answer, describe:

    - How pruning is applied and what sparsity schedule is used.
    - Why the pruning wrappers are stripped before QAT.
    - How QAT is applied after pruning.
    - Why the notebook uses an additional mask-enforcement step during and after QAT instead of using only a simple pruning-then-QAT pipeline.
    - How the final model is converted to an int8 TFLite model and why representative data is needed.
    - How pruning and int8 quantization help when deploying to a resource-constrained device such as the Arduino Nano 33 BLE Sense.

    ### Question 3: Arduino deployment and real-world behavior

    Deploy the quantized model using the provided Arduino sketch for this lab. Place the Arduino Nano 33 BLE Sense flat on a stable surface, open the serial monitor, and observe the predicted activity labels.

    In your answer, include:

    - A screenshot of the serial monitor output.
    - A short explanation of whether the output is acceptable for the resting board condition.
    - Any issues you observe, such as unstable predictions, unexpected labels, sensor noise, scaling mismatch, or mismatch between the training data and the real board orientation.
    - One or two practical changes that could improve deployment reliability.

    ### Question 4: Deployment Behavior on the Arduino

    After deploying the ensemble model on the Arduino Nano 33 BLE Sense / Sense Rev2, did you observe any unexpected prediction behavior? For example, did the model repeatedly predict the same class even when the board was moved differently?

    Briefly explain why this may happen. In your answer, discuss possible causes such as differences between the original training dataset and the live Arduino sensor data, sensor placement, axis orientation, unit conversion, feature scaling, or mismatch between the motion patterns used during training and testing.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Environment check

    This cell verifies that the notebook is running inside the intended Python environment and that the required libraries are available. It does not install or uninstall packages.
    """)
    return


@app.cell
def _():
    import os
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")

    import sys
    import platform
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import tensorflow as tf

    try:
        import tensorflow_model_optimization as tfmot
    except ImportError as exc:
        raise ImportError(
            "tensorflow_model_optimization is not installed in this environment. "
            "Activate tinyml-arduino and install tensorflow-model-optimization before running this notebook."
        ) from exc

    print("Python executable:", sys.executable)
    print("Python version:", platform.python_version())
    print("TensorFlow version:", tf.__version__)
    print("TF-MOT version:", tfmot.__version__)
    return Path, np, pd, plt, tf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Load and prepare the mHealth dataset

    The mHealth subject 6 log file contains multiple sensor channels and an activity label. This lab uses six IMU features: three accelerometer channels and three gyroscope channels. The label column is converted from 1-based labels to 0-based labels for TensorFlow training.
    """)
    return


@app.cell
def _(Path, mo, pd):
    DATA_FILE = Path("mHealth_subject6.log")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "mHealth_subject6.log was not found. Place the file in the same folder as this notebook and rerun this cell."
        )

    sensor_features = ["Accel_X", "Accel_Y", "Accel_Z", "Gyro_X", "Gyro_Y", "Gyro_Z"]
    sensor_cols = {
        5: "Accel_X", 6: "Accel_Y", 7: "Accel_Z",
        8: "Gyro_X", 9: "Gyro_Y", 10: "Gyro_Z"
    }

    label_map = {
        0: "Standing still",
        1: "Sitting and relaxing",
        2: "Lying down",
        3: "Walking",
        4: "Climbing stairs",
        5: "Waist bends forward",
        6: "Frontal elevation of arms",
        7: "Knees bending (crouching)",
        8: "Cycling",
        9: "Jogging",
        10: "Running",
        11: "Jump front and back",
    }

    raw_df = pd.read_csv(DATA_FILE, sep="\t", header=None)
    raw_df = raw_df[raw_df[23] > 0].copy()
    raw_df.rename(columns=sensor_cols, inplace=True)

    sensor_df = raw_df[sensor_features].copy()
    sensor_df["Label"] = raw_df[23].to_numpy(dtype=int) - 1

    print("Dataset shape:", sensor_df.shape)
    print("Class labels:", sorted(sensor_df["Label"].unique()))
    print("\nOriginal feature statistics:")
    mo.output.append(sensor_df[sensor_features].describe())
    return label_map, sensor_df, sensor_features


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Create raw, standard-scaled, and min-max-scaled datasets

    The ensemble uses three versions of the same IMU signal. This creates diversity across the branches without changing the underlying activity labels.
    """)
    return


@app.cell
def _(mo, np, pd, sensor_df, sensor_features):
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    X_features = sensor_df[sensor_features].to_numpy(dtype=np.float32)
    y_labels = sensor_df["Label"].to_numpy(dtype=int)

    std_scaler = StandardScaler()
    X_std = std_scaler.fit_transform(X_features).astype(np.float32)

    minmax_scaler = MinMaxScaler()
    X_minmax = minmax_scaler.fit_transform(X_features).astype(np.float32)

    sensor_df_raw = pd.DataFrame(X_features, columns=sensor_features)
    sensor_df_raw["Label"] = y_labels

    sensor_df_std = pd.DataFrame(X_std, columns=sensor_features)
    sensor_df_std["Label"] = y_labels

    sensor_df_minmax = pd.DataFrame(X_minmax, columns=sensor_features)
    sensor_df_minmax["Label"] = y_labels

    print("Standard-scaled feature statistics:")
    mo.output.append(sensor_df_std[sensor_features].describe())

    print("Min-max-scaled feature statistics:")
    mo.output.append(sensor_df_minmax[sensor_features].describe())
    return (
        minmax_scaler,
        sensor_df_minmax,
        sensor_df_raw,
        sensor_df_std,
        std_scaler,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Convert the time series into flattened windows

    Each sample is a 100-time-step window with 6 sensor features. The model input dimension is therefore:

    `100 time steps x 6 features = 600 input neurons`

    The split is done within each class to preserve the temporal structure of each activity segment.
    """)
    return


@app.cell
def _(np, sensor_df_minmax, sensor_df_raw, sensor_df_std, sensor_features):
    WINDOW_SIZE = 100
    STRIDE = 1
    NUM_CLASSES = 12

    def create_windows(X_seq, label, window_size=WINDOW_SIZE, stride=STRIDE):
        windows = []
        _labels = []
        for start in range(0, len(X_seq) - window_size + 1, stride):
            window = X_seq[start:start + window_size].reshape(-1)
            windows.append(window)
            _labels.append(label)
        return (windows, _labels)

    def window_and_split(df_scaled, train_fraction=0.7):
        (X_train_windows, X_test_windows) = ([], [])
        (y_train_labels, y_test_labels) = ([], [])
        for label in sorted(df_scaled['Label'].unique()):
            df_class = df_scaled[df_scaled['Label'] == label]
            X_class = df_class[sensor_features].to_numpy(dtype=np.float32)
            split_idx = int(train_fraction * len(X_class))
            (train_windows, train_labels) = create_windows(X_class[:split_idx], label)
            (test_windows, test_labels) = create_windows(X_class[split_idx:], label)
            X_train_windows.extend(train_windows)
            y_train_labels.extend(train_labels)
            X_test_windows.extend(test_windows)
            y_test_labels.extend(test_labels)
        return (np.asarray(X_train_windows, dtype=np.float32), np.asarray(X_test_windows, dtype=np.float32), np.asarray(y_train_labels, dtype=int), np.asarray(y_test_labels, dtype=int))
    (X_train_raw, X_test_raw, y_train_raw, y_test_raw) = window_and_split(sensor_df_raw)
    (X_train_std, X_test_std, y_train_std, y_test_std) = window_and_split(sensor_df_std)
    (X_train_minmax, X_test_minmax, y_train_minmax, y_test_minmax) = window_and_split(sensor_df_minmax)
    print('Raw input:', X_train_raw.shape, X_test_raw.shape)
    print('Standard-scaled input:', X_train_std.shape, X_test_std.shape)
    print('Min-max-scaled input:', X_train_minmax.shape, X_test_minmax.shape)
    return (
        NUM_CLASSES,
        STRIDE,
        WINDOW_SIZE,
        X_test_minmax,
        X_test_raw,
        X_test_std,
        X_train_minmax,
        X_train_raw,
        X_train_std,
        y_test_minmax,
        y_test_raw,
        y_test_std,
        y_train_minmax,
        y_train_raw,
        y_train_std,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Visualize the three input representations with t-SNE

    This visualization helps compare how the raw, standard-scaled, and min-max-scaled windows are distributed before training the autoencoders. To keep the notebook practical on a local machine, the plot uses a stratified subset of test windows.
    """)
    return


@app.cell
def _(
    X_test_minmax,
    X_test_raw,
    X_test_std,
    label_map,
    np,
    plt,
    y_test_minmax,
    y_test_raw,
    y_test_std,
):
    from sklearn.manifold import TSNE
    TSNE_MAX_SAMPLES = 2000

    def stratified_subset(X, y, max_samples=TSNE_MAX_SAMPLES, random_state=42):
        rng = np.random.default_rng(random_state)
        _labels = np.unique(y)
        per_class = max(1, max_samples // len(_labels))
        chosen = []
        for label in _labels:
            idx = np.where(y == label)[0]
            sample_size = min(per_class, len(idx))
            chosen.extend(rng.choice(idx, size=sample_size, replace=False))
        chosen = np.asarray(chosen)
        rng.shuffle(chosen)
        return (X[chosen], y[chosen])

    def compute_tsne(X, random_state=42):
        kwargs = dict(n_components=2, perplexity=30, learning_rate=300, random_state=random_state)
        try:
            return TSNE(max_iter=1000, **kwargs).fit_transform(X)
        except TypeError:
            return TSNE(n_iter=1000, **kwargs).fit_transform(X)

    def run_tsne_and_plot(X_test, y_test, title, ax):
        (X_sub, y_sub) = stratified_subset(X_test, y_test)
        X_tsne = compute_tsne(X_sub)
        for label in sorted(np.unique(y_sub)):
            idx = y_sub == label
            ax.scatter(X_tsne[idx, 0], X_tsne[idx, 1], s=5, label=label_map[label])
        ax.set_title(title)
        ax.set_xlabel('t-SNE component 1')
        ax.set_ylabel('t-SNE component 2')
        ax.grid(True, alpha=0.3)
    (_fig, _axes) = plt.subplots(1, 3, figsize=(20, 6))
    run_tsne_and_plot(X_test_raw, y_test_raw, 'Raw windows', _axes[0])
    run_tsne_and_plot(X_test_std, y_test_std, 'Standard-scaled windows', _axes[1])
    run_tsne_and_plot(X_test_minmax, y_test_minmax, 'Min-max-scaled windows', _axes[2])
    (_handles, _labels) = _axes[0].get_legend_handles_labels()
    _fig.legend(_handles, _labels, title='Activity', loc='center right', bbox_to_anchor=(1.02, 0.5), fontsize=8)
    plt.tight_layout()
    plt.show()
    return compute_tsne, stratified_subset


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Train one autoencoder for each input representation

    Each autoencoder reconstructs its own input representation. The encoder part is then used as a compact feature extractor.

    Encoder architecture for each branch:

    `600 input neurons -> Dense(64, ReLU) -> Dense(32, linear)`

    Decoder architecture for each branch:

    `32 latent neurons -> Dense(64, ReLU) -> Dense(600, linear)`
    """)
    return


@app.cell
def _(X_train_minmax, X_train_raw, X_train_std):
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.optimizers import Adam

    AUTOENCODER_EPOCHS = 50
    BATCH_SIZE = 256
    LATENT_DIM = 32


    def build_autoencoder(input_dim, latent_dim=LATENT_DIM):
        input_layer = Input(shape=(input_dim,), name="input_window")
        encoded = Dense(64, activation="relu", name="encoder_dense_64")(input_layer)
        latent = Dense(latent_dim, activation="linear", name="latent_32")(encoded)
        decoded = Dense(64, activation="relu", name="decoder_dense_64")(latent)
        reconstructed = Dense(input_dim, activation="linear", name="reconstruction")(decoded)

        autoencoder = Model(inputs=input_layer, outputs=reconstructed, name="autoencoder")
        encoder = Model(inputs=input_layer, outputs=latent, name="encoder")
        autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
        return autoencoder, encoder


    def train_autoencoder_branch(X_train, branch_name):
        input_dim = X_train.shape[1]
        autoencoder, encoder = build_autoencoder(input_dim)
        print(f"Training autoencoder for {branch_name} input")
        autoencoder.fit(
            X_train,
            X_train,
            epochs=AUTOENCODER_EPOCHS,
            batch_size=BATCH_SIZE,
            shuffle=True,
            verbose=1,
        )
        return autoencoder, encoder

    models = {
        "raw": train_autoencoder_branch(X_train_raw, "raw"),
        "std": train_autoencoder_branch(X_train_std, "standard-scaled"),
        "minmax": train_autoencoder_branch(X_train_minmax, "min-max-scaled"),
    }
    return Adam, BATCH_SIZE, Dense, Input, Sequential, models


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Visualize encoder latent spaces

    The encoder maps each 600-dimensional input window to a 32-dimensional latent representation. The t-SNE plots below visualize these latent spaces.
    """)
    return


@app.cell
def _(
    X_test_minmax,
    X_test_raw,
    X_test_std,
    compute_tsne,
    label_map,
    models,
    np,
    plt,
    stratified_subset,
    y_test_minmax,
    y_test_raw,
    y_test_std,
):
    def plot_latent_tsne(encoder, X_test, y_test, title, ax):
        (X_sub, y_sub) = stratified_subset(X_test, y_test)
        latent = encoder.predict(X_sub, verbose=0)
        latent_tsne = compute_tsne(latent)
        for label in sorted(np.unique(y_sub)):
            idx = y_sub == label
            ax.scatter(latent_tsne[idx, 0], latent_tsne[idx, 1], s=5, label=label_map[label])
        ax.set_title(title)
        ax.set_xlabel('t-SNE component 1')
        ax.set_ylabel('t-SNE component 2')
        ax.grid(True, alpha=0.3)
    (_fig, _axes) = plt.subplots(1, 3, figsize=(20, 6))
    plot_latent_tsne(models['raw'][1], X_test_raw, y_test_raw, 'Raw encoder latent space', _axes[0])
    plot_latent_tsne(models['std'][1], X_test_std, y_test_std, 'Standard-scaled encoder latent space', _axes[1])
    plot_latent_tsne(models['minmax'][1], X_test_minmax, y_test_minmax, 'Min-max-scaled encoder latent space', _axes[2])
    (_handles, _labels) = _axes[0].get_legend_handles_labels()
    _fig.legend(_handles, _labels, title='Activity', loc='center right', bbox_to_anchor=(1.02, 0.5), fontsize=8)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Train one classifier on each latent representation

    Each classifier receives a 32-dimensional encoder output and predicts one of 12 activity classes.

    Classifier architecture for each branch:

    `32 latent input neurons -> Dense(20, ReLU) -> Dense(12, Softmax)`
    """)
    return


@app.cell
def _(
    Adam,
    BATCH_SIZE,
    Dense,
    Input,
    NUM_CLASSES,
    Sequential,
    X_test_minmax,
    X_test_raw,
    X_test_std,
    X_train_minmax,
    X_train_raw,
    X_train_std,
    models,
    y_test_minmax,
    y_test_raw,
    y_test_std,
    y_train_minmax,
    y_train_raw,
    y_train_std,
):
    from tensorflow.keras.utils import to_categorical

    CLASSIFIER_EPOCHS = 50

    y_train_raw_cat = to_categorical(y_train_raw, num_classes=NUM_CLASSES)
    y_test_raw_cat = to_categorical(y_test_raw, num_classes=NUM_CLASSES)
    y_train_std_cat = to_categorical(y_train_std, num_classes=NUM_CLASSES)
    y_test_std_cat = to_categorical(y_test_std, num_classes=NUM_CLASSES)
    y_train_minmax_cat = to_categorical(y_train_minmax, num_classes=NUM_CLASSES)
    y_test_minmax_cat = to_categorical(y_test_minmax, num_classes=NUM_CLASSES)

    latent_train_raw = models["raw"][1].predict(X_train_raw, verbose=0)
    latent_test_raw = models["raw"][1].predict(X_test_raw, verbose=0)
    latent_train_std = models["std"][1].predict(X_train_std, verbose=0)
    latent_test_std = models["std"][1].predict(X_test_std, verbose=0)
    latent_train_minmax = models["minmax"][1].predict(X_train_minmax, verbose=0)
    latent_test_minmax = models["minmax"][1].predict(X_test_minmax, verbose=0)


    def build_classifier(input_dim, num_classes=NUM_CLASSES):
        model = Sequential([
            Input(shape=(input_dim,), name="latent_input"),
            Dense(20, activation="relu", name="classifier_dense_20"),
            Dense(num_classes, activation="softmax", name="activity_softmax"),
        ], name="latent_classifier")
        model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
        return model

    clf_raw = build_classifier(latent_train_raw.shape[1])
    clf_std = build_classifier(latent_train_std.shape[1])
    clf_minmax = build_classifier(latent_train_minmax.shape[1])

    print("Training classifier for raw branch")
    clf_raw.fit(latent_train_raw, y_train_raw_cat, epochs=CLASSIFIER_EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    print("Training classifier for standard-scaled branch")
    clf_std.fit(latent_train_std, y_train_std_cat, epochs=CLASSIFIER_EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    print("Training classifier for min-max-scaled branch")
    clf_minmax.fit(latent_train_minmax, y_train_minmax_cat, epochs=CLASSIFIER_EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    return (
        CLASSIFIER_EPOCHS,
        clf_minmax,
        clf_raw,
        clf_std,
        latent_test_minmax,
        latent_test_raw,
        latent_test_std,
        latent_train_minmax,
        latent_train_raw,
        latent_train_std,
        to_categorical,
        y_test_minmax_cat,
        y_test_raw_cat,
        y_test_std_cat,
        y_train_minmax_cat,
        y_train_raw_cat,
        y_train_std_cat,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Evaluate the three branch classifiers

    The branch classifiers are evaluated separately before constructing the stacked ensemble.
    """)
    return


@app.cell
def _(
    NUM_CLASSES,
    clf_minmax,
    clf_raw,
    clf_std,
    label_map,
    latent_test_minmax,
    latent_test_raw,
    latent_test_std,
    np,
    plt,
    y_test_minmax,
    y_test_raw,
    y_test_std,
):
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

    def evaluate_classifier(model, X_test, y_test_true, title):
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        target_names = [label_map[i] for i in range(NUM_CLASSES)]
        print(f'\nClassification report: {title}')
        print(classification_report(y_test_true, y_pred, target_names=target_names, zero_division=0))
        cm = confusion_matrix(y_test_true, y_pred, labels=list(range(NUM_CLASSES)))
        (_fig, ax) = plt.subplots(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
        disp.plot(ax=ax, xticks_rotation=45, values_format='d', colorbar=False)
        ax.set_title(f'Confusion matrix: {title}')
        plt.tight_layout()
        plt.show()
    evaluate_classifier(clf_raw, latent_test_raw, y_test_raw, 'raw branch')
    evaluate_classifier(clf_std, latent_test_std, y_test_std, 'standard-scaled branch')
    evaluate_classifier(clf_minmax, latent_test_minmax, y_test_minmax, 'min-max-scaled branch')
    return classification_report, confusion_matrix, evaluate_classifier


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Build the stacked ensemble

    The three branch classifiers each output a 12-dimensional softmax vector. These outputs are concatenated into a 36-dimensional vector:

    `12 raw probabilities + 12 standard-scaled probabilities + 12 min-max-scaled probabilities = 36 ensemble inputs`

    The stacked meta-classifier then maps the 36-dimensional ensemble input to the final 12-class prediction.

    Meta-classifier architecture:

    `36 input neurons -> Dense(24, ReLU) -> Dense(12, Softmax)`
    """)
    return


@app.cell
def _(
    Adam,
    BATCH_SIZE,
    CLASSIFIER_EPOCHS,
    Dense,
    Input,
    NUM_CLASSES,
    Sequential,
    clf_minmax,
    clf_raw,
    clf_std,
    evaluate_classifier,
    latent_test_minmax,
    latent_test_raw,
    latent_test_std,
    latent_train_minmax,
    latent_train_raw,
    latent_train_std,
    np,
    to_categorical,
    y_test_raw,
    y_train_raw,
):
    train_logits_raw = clf_raw.predict(latent_train_raw, verbose=0)
    test_logits_raw = clf_raw.predict(latent_test_raw, verbose=0)

    train_logits_std = clf_std.predict(latent_train_std, verbose=0)
    test_logits_std = clf_std.predict(latent_test_std, verbose=0)

    train_logits_minmax = clf_minmax.predict(latent_train_minmax, verbose=0)
    test_logits_minmax = clf_minmax.predict(latent_test_minmax, verbose=0)

    X_train_ensemble = np.hstack([train_logits_raw, train_logits_std, train_logits_minmax]).astype(np.float32)
    X_test_ensemble = np.hstack([test_logits_raw, test_logits_std, test_logits_minmax]).astype(np.float32)

    y_train_ensemble_cat = to_categorical(y_train_raw, num_classes=NUM_CLASSES)
    y_test_ensemble_cat = to_categorical(y_test_raw, num_classes=NUM_CLASSES)

    meta_clf = Sequential([
        Input(shape=(36,), name="ensemble_input"),
        Dense(24, activation="relu", name="meta_dense_24"),
        Dense(NUM_CLASSES, activation="softmax", name="meta_softmax"),
    ], name="stacked_meta_classifier")

    meta_clf.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
    meta_clf.fit(X_train_ensemble, y_train_ensemble_cat, epochs=CLASSIFIER_EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    evaluate_classifier(meta_clf, X_test_ensemble, y_test_raw, "stacked ensemble")
    return (
        X_test_ensemble,
        X_train_ensemble,
        meta_clf,
        y_test_ensemble_cat,
        y_train_ensemble_cat,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. Convert the trained models to baseline TFLite models

    This section converts the trained encoder, branch classifier, and meta-classifier models to regular TFLite models and reports their file sizes. These are not yet the pruned QAT int8 models.
    """)
    return


@app.cell
def _(Path, clf_minmax, clf_raw, clf_std, meta_clf, models, tf):
    def convert_and_save_tflite(model, name):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        filename = Path(f"{name}.tflite")
        filename.write_bytes(tflite_model)

        size_kb = filename.stat().st_size / 1024
        print(f"{name}: {size_kb:.2f} KB")

    convert_and_save_tflite(models["raw"][1], "encoder_raw")
    convert_and_save_tflite(clf_raw, "clf_raw")
    convert_and_save_tflite(models["std"][1], "encoder_std")
    convert_and_save_tflite(clf_std, "clf_std")
    convert_and_save_tflite(models["minmax"][1], "encoder_minmax")
    convert_and_save_tflite(clf_minmax, "clf_minmax")
    convert_and_save_tflite(meta_clf, "stacked_meta_clf")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 12. Create flattened encoder-classifier pipelines

    For deployment, each branch can be represented as a single model that maps the 600-dimensional input window directly to a 12-class softmax output.

    Each flattened branch has this structure:

    `600 input neurons -> Dense(64, ReLU) -> Dense(32, linear) -> Dense(20, ReLU) -> Dense(12, Softmax)`
    """)
    return


@app.cell
def _(Adam, Input, Sequential, clf_minmax, clf_raw, clf_std, models):
    def combine_encoder_and_classifier(encoder, classifier, name):
        combined = Sequential(name=name)
        combined.add(Input(shape=(encoder.input_shape[1],), name="window_input"))

        for layer in encoder.layers[1:]:
            combined.add(layer)

        for layer in classifier.layers:
            combined.add(layer)

        combined.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
        return combined

    flattened_raw = combine_encoder_and_classifier(models["raw"][1], clf_raw, "encoder_clf_raw_flat")
    flattened_std = combine_encoder_and_classifier(models["std"][1], clf_std, "encoder_clf_std_flat")
    flattened_minmax = combine_encoder_and_classifier(models["minmax"][1], clf_minmax, "encoder_clf_minmax_flat")

    flattened_raw.summary()
    return flattened_minmax, flattened_raw, flattened_std


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 13. Pruning, QAT, and int8 TFLite export

    This section applies a deployment-oriented compression pipeline:

    1. Prune the model using a polynomial sparsity schedule.
    2. Strip the pruning wrappers to create a regular Keras model with sparse weights.
    3. Apply QAT to simulate quantization effects during fine-tuning.
    4. Reapply pruning masks during QAT so previously pruned weights remain zero.
    5. Convert the QAT model to a fully int8 TFLite model using representative data.
    6. Evaluate the exported TFLite model.
    """)
    return


@app.cell
def _(
    BATCH_SIZE,
    Dense,
    Path,
    classification_report,
    confusion_matrix,
    np,
    tf,
):
    from tensorflow_model_optimization.sparsity.keras import (
        prune_low_magnitude,
        PolynomialDecay,
        UpdatePruningStep,
        strip_pruning,
    )
    from tensorflow_model_optimization.quantization.keras import quantize_model, quantize_scope
    from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper

    PRUNE_EPOCHS = 5
    QAT_EPOCHS = 5
    PRUNE_END_STEP = 500
    REPRESENTATIVE_SAMPLES = 200


    def train_and_prune(model, X_train, y_train):
        pruning_schedule = PolynomialDecay(
            initial_sparsity=0.10,
            final_sparsity=0.80,
            begin_step=0,
            end_step=PRUNE_END_STEP,
        )

        pruned_model = prune_low_magnitude(model, pruning_schedule=pruning_schedule)
        pruned_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        pruned_model.fit(
            X_train,
            y_train,
            epochs=PRUNE_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[UpdatePruningStep()],
            verbose=1,
        )
        return pruned_model


    def extract_dense_weight_masks(model):
        masks = []
        for layer in model.layers:
            if isinstance(layer, Dense):
                weights = layer.get_weights()
                if weights:
                    layer_masks = [(w != 0).astype(np.float32) for w in weights]
                    masks.append((layer.name, [w.copy() for w in weights], layer_masks))
        return masks


    class MaskEnforcerCallback(tf.keras.callbacks.Callback):
        def __init__(self, masks):
            super().__init__()
            self.masks = masks

        def on_train_batch_end(self, batch, logs=None):
            self.apply_masks()

        def on_epoch_end(self, epoch, logs=None):
            self.apply_masks()

        def apply_masks(self):
            for layer in self.model.layers:
                if not isinstance(layer, QuantizeWrapper):
                    continue

                weights = layer.get_weights()
                if len(weights) < 2:
                    continue

                for original_name, original_weights, masks in self.masks:
                    if len(original_weights) < 2:
                        continue
                    if weights[0].shape == original_weights[0].shape and weights[1].shape == original_weights[1].shape:
                        updated_weights = [weights[0] * masks[0], weights[1] * masks[1]] + weights[2:]
                        layer.set_weights(updated_weights)
                        break


    def make_representative_data_gen(X_subset, max_samples=REPRESENTATIVE_SAMPLES):
        count = min(len(X_subset), max_samples)
        indices = np.linspace(0, len(X_subset) - 1, count, dtype=int)

        def generator():
            for i in indices:
                yield [X_subset[i:i + 1].astype(np.float32)]

        return generator


    def compute_model_sparsity(model):
        total_weights = 0
        total_zeros = 0
        print("\nLayer-wise sparsity:")

        for layer in model.layers:
            weights = layer.get_weights()
            if not weights:
                continue

            kernel = weights[0]
            if kernel.ndim < 2:
                continue

            zeros = np.sum(kernel == 0)
            size = np.prod(kernel.shape)
            total_weights += size
            total_zeros += zeros
            print(f"{layer.name:30s}: {100.0 * zeros / size:6.2f}% sparse ({zeros}/{size})")

        total_sparsity = 100.0 * total_zeros / total_weights if total_weights else 0.0
        print(f"Overall sparsity: {total_sparsity:.2f}% ({total_zeros}/{total_weights})")
        return total_sparsity


    def evaluate_tflite_model(tflite_model_path, X_test, y_test):
        interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_index = input_details[0]["index"]
        output_index = output_details[0]["index"]
        input_scale, input_zero_point = input_details[0]["quantization"]

        y_pred = []
        for i in range(len(X_test)):
            input_data = X_test[i:i + 1].astype(np.float32)
            if input_details[0]["dtype"] == np.int8:
                input_data = np.round(input_data / input_scale + input_zero_point).astype(np.int8)

            interpreter.set_tensor(input_index, input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_index)
            y_pred.append(int(np.argmax(output_data)))

        accuracy = float(np.mean(np.asarray(y_pred) == y_test))
        print("\nTFLite classification report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("TFLite confusion matrix:")
        print(confusion_matrix(y_test, y_pred))
        print(f"TFLite accuracy: {accuracy:.4f}")
        return accuracy


    def prune_qat_export(model, X_train, y_train, X_test, y_test, name):
        print(f"\nCompression pipeline for {name}")

        pruned_model = train_and_prune(model, X_train, y_train)
        stripped_model = strip_pruning(pruned_model)
        dense_masks = extract_dense_weight_masks(stripped_model)

        with quantize_scope():
            qat_model = quantize_model(stripped_model)

        qat_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        mask_callback = MaskEnforcerCallback(dense_masks)

        print("Training QAT model with pruning-mask enforcement")
        qat_model.fit(
            X_train,
            y_train,
            epochs=QAT_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[mask_callback],
            verbose=1,
        )
        mask_callback.apply_masks()

        compute_model_sparsity(qat_model)

        converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = make_representative_data_gen(X_train)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        tflite_model = converter.convert()
        output_path = Path(f"{name}_pruned_qat_int8.tflite")
        output_path.write_bytes(tflite_model)

        print(f"Saved {output_path.name}: {output_path.stat().st_size / 1024:.2f} KB")
        evaluate_tflite_model(output_path, X_test, np.argmax(y_test, axis=1))
        return output_path

    return (prune_qat_export,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 14. Export compressed branch models and the meta-classifier

    The following cells produce four int8 TFLite models:

    - Raw encoder-classifier branch
    - Standard-scaled encoder-classifier branch
    - Min-max-scaled encoder-classifier branch
    - Stacked meta-classifier
    """)
    return


@app.cell
def _(
    X_test_ensemble,
    X_test_minmax,
    X_test_raw,
    X_test_std,
    X_train_ensemble,
    X_train_minmax,
    X_train_raw,
    X_train_std,
    flattened_minmax,
    flattened_raw,
    flattened_std,
    meta_clf,
    prune_qat_export,
    y_test_ensemble_cat,
    y_test_minmax_cat,
    y_test_raw_cat,
    y_test_std_cat,
    y_train_ensemble_cat,
    y_train_minmax_cat,
    y_train_raw_cat,
    y_train_std_cat,
):
    compressed_model_paths = []

    compressed_model_paths.append(
        prune_qat_export(
            flattened_raw,
            X_train_raw,
            y_train_raw_cat,
            X_test_raw,
            y_test_raw_cat,
            "encoder_clf_raw",
        )
    )

    compressed_model_paths.append(
        prune_qat_export(
            flattened_std,
            X_train_std,
            y_train_std_cat,
            X_test_std,
            y_test_std_cat,
            "encoder_clf_std",
        )
    )

    compressed_model_paths.append(
        prune_qat_export(
            flattened_minmax,
            X_train_minmax,
            y_train_minmax_cat,
            X_test_minmax,
            y_test_minmax_cat,
            "encoder_clf_minmax",
        )
    )

    compressed_model_paths.append(
        prune_qat_export(
            meta_clf,
            X_train_ensemble,
            y_train_ensemble_cat,
            X_test_ensemble,
            y_test_ensemble_cat,
            "stacked_meta_clf",
        )
    )
    return (compressed_model_paths,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 15. Report compressed model sizes
    """)
    return


@app.cell
def _(Path, compressed_model_paths):
    print('Compressed int8 TFLite model sizes')
    print(f"{'Model':45s} {'Size (KB)':>10s}")
    print('-' * 58)
    for _path in compressed_model_paths:
        _path = Path(_path)
        if _path.exists():
            print(f'{_path.name:45s} {_path.stat().st_size / 1024:10.2f}')
        else:
            print(f"{_path.name:45s} {'missing':>10s}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 16. Convert TFLite files to C arrays for Arduino deployment

    The Arduino sketch can include model files as C arrays. This cell uses `xxd`, which is available by default on macOS and Linux.
    """)
    return


@app.cell
def _(Path, compressed_model_paths):
    import subprocess
    print(compressed_model_paths)
    for _path in compressed_model_paths:
        _path = Path(_path)
        output_cc = _path.with_suffix('.cc')
        if _path.exists():
            with output_cc.open('w') as f:
                subprocess.run(['xxd', '-i', str(_path)], stdout=f, check=True)
            print(f'Created {output_cc.name}')
        else:
            print(f'Skipped {_path.name}; file not found')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 17. Scaling parameters for Arduino implementation

    The Arduino sketch needs the same scaling parameters used during training. Use these printed values when implementing standard scaling and min-max scaling on the board.
    """)
    return


@app.cell
def _(minmax_scaler, np, sensor_features, std_scaler):
    print("Standard-scaling parameters")
    for feature, mean, std in zip(sensor_features, std_scaler.mean_, np.sqrt(std_scaler.var_)):
        print(f"{feature:8s}: mean = {mean: .6f}, std = {std: .6f}")

    print("\nMin-max-scaling parameters")
    for feature, min_val, max_val in zip(sensor_features, minmax_scaler.data_min_, minmax_scaler.data_max_):
        print(f"{feature:8s}: min = {min_val: .6f}, max = {max_val: .6f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 18. Optional recorded-IMU validation

    If you recorded a new IMU file named `imu_500_rows.csv`, this section runs the trained Keras models on that file. The expected CSV columns are:

    `acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z`

    This section is useful for checking whether real board data behaves similarly to the training data. If the CSV file is not present, the cell will skip this validation.
    """)
    return


@app.cell
def _(
    Path,
    STRIDE,
    WINDOW_SIZE,
    flattened_minmax,
    flattened_raw,
    flattened_std,
    label_map,
    meta_clf,
    minmax_scaler,
    np,
    pd,
    sensor_features,
    std_scaler,
):
    RECORDED_IMU_FILE = Path("imu_500_rows.csv")

    if not RECORDED_IMU_FILE.exists():
        print("imu_500_rows.csv was not found. Skipping recorded-IMU validation.")
    else:
        df_new = pd.read_csv(RECORDED_IMU_FILE)
        df_new = df_new.rename(columns={
            "acc_x": "Accel_X",
            "acc_y": "Accel_Y",
            "acc_z": "Accel_Z",
            "gyro_x": "Gyro_X",
            "gyro_y": "Gyro_Y",
            "gyro_z": "Gyro_Z",
        })

        missing_columns = [col for col in sensor_features if col not in df_new.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in imu_500_rows.csv: {missing_columns}")

        X_new_raw_features = df_new[sensor_features].to_numpy(dtype=np.float32)
        X_new_std_features = std_scaler.transform(X_new_raw_features).astype(np.float32)
        X_new_minmax_features = minmax_scaler.transform(X_new_raw_features).astype(np.float32)

        def create_windows_from_array(X, window_size=WINDOW_SIZE, stride=STRIDE):
            return np.asarray(
                [X[start:start + window_size].reshape(-1) for start in range(0, len(X) - window_size + 1, stride)],
                dtype=np.float32,
            )

        X_raw_new = create_windows_from_array(X_new_raw_features)
        X_std_new = create_windows_from_array(X_new_std_features)
        X_minmax_new = create_windows_from_array(X_new_minmax_features)

        if len(X_raw_new) == 0:
            raise ValueError("The recorded IMU file does not contain enough rows to create one full window.")

        logits_raw = flattened_raw.predict(X_raw_new, verbose=0)
        logits_std = flattened_std.predict(X_std_new, verbose=0)
        logits_minmax = flattened_minmax.predict(X_minmax_new, verbose=0)

        pred_raw = np.argmax(logits_raw, axis=1)
        pred_std = np.argmax(logits_std, axis=1)
        pred_minmax = np.argmax(logits_minmax, axis=1)

        X_meta_new = np.hstack([logits_raw, logits_std, logits_minmax]).astype(np.float32)
        meta_logits = meta_clf.predict(X_meta_new, verbose=0)
        pred_meta = np.argmax(meta_logits, axis=1)

        print("Recorded-IMU input windows:")
        print("Raw:", X_raw_new.shape)
        print("Standard scaled:", X_std_new.shape)
        print("Min-max scaled:", X_minmax_new.shape)

        print("\nFirst 10 predicted class IDs")
        print("Raw branch:             ", pred_raw[:10])
        print("Standard-scaled branch: ", pred_std[:10])
        print("Min-max-scaled branch:  ", pred_minmax[:10])
        print("Stacked meta-classifier:", pred_meta[:10])

        print("\nFirst 10 stacked meta-classifier labels")
        print([label_map[int(label)] for label in pred_meta[:10]])
    return


if __name__ == "__main__":
    app.run()
