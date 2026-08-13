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
    # EE446 - TinyML - Assignment 2
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Importing Libraries
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import seaborn as sns
    from pylab import rcParams
    from sklearn.model_selection import train_test_split
    from tensorflow.keras import datasets, layers,models
    from tensorflow.keras.models import Model, load_model, Sequential
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers import Input, Dense, Activation
    from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
    from tensorflow.keras import regularizers
    from sklearn.preprocessing import LabelEncoder
    import warnings
    warnings.filterwarnings("ignore")
    from sklearn.utils import class_weight
    from sklearn.metrics import accuracy_score
    import c_writer
    from os.path import join

    return (
        Dense,
        LabelEncoder,
        Sequential,
        StandardScaler,
        c_writer,
        classification_report,
        confusion_matrix,
        np,
        pd,
        plt,
        tf,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reading Data
    """)
    return


@app.cell
def _(pd):
    # Reading the data and adding column header (feature) names
    data = pd.read_csv("Network_anomaly_data.txt",sep=",",names=["duration","protocoltype","service",
    "flag","srcbytes","dstbytes","land", "wrongfragment","urgent","hot","numfailedlogins","loggedin", "numcompromised",
    "rootshell","suattempted","numroot","numfilecreations", "numshells","numaccessfiles","numoutboundcmds","ishostlogin",
    "isguestlogin","count","srvcount","serrorrate", "srvserrorrate","rerrorrate","srvrerrorrate","samesrvrate",
    "diffsrvrate", "srvdiffhostrate","dsthostcount","dsthostsrvcount","dsthostsamesrvrate", "dsthostdiffsrvrate",
    "dsthostsamesrcportrate","dsthostsrvdiffhostrate","dsthostserrorrate","dsthostsrvserrorrate","dsthostrerrorrate",
    "dsthostsrvrerrorrate","attack", "lastflag"])
    return (data,)


@app.cell
def _(data):
    data # printing the dataframe
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Question 1: Data Preprocessing
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### (a) Drop the 'land', 'urgent', 'numfailedlogins', 'numoutboundcmds' columns from the dataframe "data".
    """)
    return


@app.cell
def _(data):
    # (a) drop the four requested columns in place
    data.drop(['land', 'urgent', 'numfailedlogins', 'numoutboundcmds'], axis=1, inplace=True)
    print(data.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### (b) Change any label that is not named normal to attack in the {'attack'} column of the dataframe data.
    """)
    return


@app.cell
def _(data):
    # (b) binarize the label: everything that is not "normal" becomes "attack"
    data.loc[data['attack'] != 'normal', 'attack'] = 'attack'
    cat_cols = ['protocoltype', 'service', 'flag', 'attack']
    print(data['attack'].value_counts())
    return (cat_cols,)


@app.cell
def _(data):
    data
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### (c) Use LabelEncoder() function from the sklearn.preprocessing library to convert non-numerical attributes in the {'protocoltype', 'service', 'flag', 'attack'} columns of the dataframe data to numerical values.
    """)
    return


@app.cell
def _(LabelEncoder, cat_cols, data):
    # (c) label-encode the non-numerical columns (attack -> 0, normal -> 1)
    for _col in cat_cols:
        data[_col] = LabelEncoder().fit_transform(data[_col])
    print(data[cat_cols].dtypes)
    return


@app.cell
def _(data, pd):
    pd.DataFrame(data) #<--------- Print your modified dataframe "data"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feature Scaling and Train/Test Split
    """)
    return


@app.cell
def _(StandardScaler, data, train_test_split):
    # All the features apart from Attack are what we are going to use to predict the attack status of the data
    # attack = 1 (normal/not an attack) and attack = 0 (attack)
    X = data.drop(['attack'],axis=1).to_numpy()

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit and transform the data
    X_normalized = scaler.fit_transform(X)

    Y = data['attack'].to_numpy()
    # Splitting X and y testing and training data
    # we are taking 20% of the data for testing and 80% of the data for training
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, Y, test_size = 0.20)
    # reshaping y test and train array
    y_train = y_train.reshape(len(y_train),1)
    y_test = y_test.reshape(len(y_test),1)
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Question 2: Dimensionality Reduction for Visualization
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### (a) Use TSNE from the sklearn.manifold library to visualize the data in the test set (X_test) in 2D. In your figure, use color "red" to mark {attack} data points and color "blue" to mark {normal} data points.
    """)
    return


@app.cell
def _(X_test, np, plt, y_test):
    from sklearn.manifold import TSNE

    # t-SNE and kernel PCA are O(n^2), so use a random 2000-point subsample of the test set
    _rng = np.random.default_rng(0)
    sub = _rng.choice(len(X_test), 2000, replace=False)
    X_sub, y_sub = X_test[sub], y_test[sub].ravel()

    def plot_2d(Z, title):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(Z[y_sub == 0, 0], Z[y_sub == 0, 1], c='red', s=5, label='attack')
        ax.scatter(Z[y_sub == 1, 0], Z[y_sub == 1, 1], c='blue', s=5, label='normal')
        ax.set_title(title)
        ax.legend()
        fig.savefig('../assets/' + title.lower().replace('-', '') + '.png', dpi=120, bbox_inches='tight')
        return fig

    plot_2d(TSNE(n_components=2, random_state=0).fit_transform(X_sub), 'TSNE')
    return X_sub, plot_2d


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### (b) Use PCA from the sklearn.decomposition library to visualize the data in the test set (X_test) in 2D. In your figure, use color "red" to mark {attack} data points and color "blue" to mark {normal} data points.
    """)
    return


@app.cell
def _(X_sub, plot_2d):
    from sklearn.decomposition import PCA

    plot_2d(PCA(n_components=2).fit_transform(X_sub), 'PCA')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### (c) Use KernelPCA from the sklearn.decomposition library to visualize the data in the test set (X_test) in 2D. Use radial basis function (rbf) as the kernel. In your figure, use color "red" to mark {attack} data points and color "blue" to mark {normal} data points.
    """)
    return


@app.cell
def _(X_sub, plot_2d):
    from sklearn.decomposition import KernelPCA

    plot_2d(KernelPCA(n_components=2, kernel='rbf').fit_transform(X_sub), 'KernelPCA')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Question 3: Implementing a DNN on the dataset
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### (a) Implement a deep neural network (DNN) on the Network Anomaly Dataset. Ensure that the output layer contains one neuron with sigmoid activation.
    """)
    return


@app.cell
def _(Dense, Sequential, X_train):
    # Define the DNN model
    base_model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    base_model.summary()
    return (base_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### (b) Compile and train your DNN model on the training set (X_train). Denote the trained model as base_model.
    """)
    return


@app.cell
def _(X_train, base_model, y_train):
    base_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    base_model.fit(X_train, y_train, epochs=10, batch_size=256, validation_split=0.1, verbose=2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### (c) Evaluate the base_model on the test set (X_test) using classification_report and confusion_matrix from the sklearn.metrics library. Report these numbers in your .pdf writeup file using screenshots.
    """)
    return


@app.cell
def _(X_test, base_model, classification_report, confusion_matrix, y_test):
    y_pred = (base_model.predict(X_test, verbose=0) > 0.5).astype(int)
    print(classification_report(y_test, y_pred, target_names=['attack', 'normal']))
    print(confusion_matrix(y_test, y_pred))
    return


@app.cell
def _(base_model):
    # Save the original Keras model to HDF5 file
    base_model.save('original_model.h5')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Question 4: Implementing Quantized Model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### (a) Implement full-integer INT8 post-training quantization on the base_model. Use a representative dataset from X_train to calibrate the activation ranges. Configure the converter to use INT8 operators with INT8 input and output tensors. Designate the resulting quantized model as tflite_quant_model.
    """)
    return


@app.cell
def _(X_train, np, tf):
    # Load the trained model
    base_model_1 = tf.keras.models.load_model('original_model.h5')

    def representative_dataset():
        # calibrate activation ranges on a slice of the training set
        for _x in X_train[:500].astype(np.float32):
            yield [_x.reshape(1, -1)]

    converter = tf.lite.TFLiteConverter.from_keras_model(base_model_1)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_quant_model = converter.convert()
    return (tflite_quant_model,)


@app.cell
def _(tflite_quant_model):
    import os

    # Save the quantized model
    with open('quantized_model.tflite', 'wb') as f:
        f.write(tflite_quant_model)

    # Get the file sizes
    original_model_size = os.path.getsize('original_model.h5')
    quantized_model_size = os.path.getsize('quantized_model.tflite')

    # Print the model sizes
    print(f"Original model size: {original_model_size / 1024:.2f} KB")
    print(f"Quantized model size: {quantized_model_size / 1024:.2f} KB")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### (b) Evaluate the tflite_quant_model on the test set (X_test) using classification_report and confusion_matrix from the sklearn.metrics library. Report these numbers in your .pdf writeup file using screenshots.
    """)
    return


@app.cell
def _(
    X_test,
    classification_report,
    confusion_matrix,
    np,
    tf,
    tflite_quant_model,
    y_test,
):
    interpreter = tf.lite.Interpreter(model_content=tflite_quant_model)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    in_scale, in_zp = in_det['quantization']
    out_scale, out_zp = out_det['quantization']
    print(f"input scale={in_scale}, zero_point={in_zp}")

    y_pred_quant = np.empty((len(X_test), 1), dtype=int)
    for _i, _x in enumerate(X_test):
        # quantize the sample with the input tensor's scale and zero point
        _xq = np.clip(np.round(_x / in_scale) + in_zp, -128, 127).astype(np.int8)
        interpreter.set_tensor(in_det['index'], _xq.reshape(1, -1))
        interpreter.invoke()
        _out = interpreter.get_tensor(out_det['index'])[0][0]
        y_pred_quant[_i] = ((_out - out_zp) * out_scale) > 0.5

    print(classification_report(y_test, y_pred_quant, target_names=['attack', 'normal']))
    print(confusion_matrix(y_test, y_pred_quant))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Converting tflite_quant_model to C and creating the header file
    """)
    return


@app.cell
def _(c_writer, np, tflite_quant_model):
    # c_writer is a py file in the same folder and has been imported at the beginning of the notebook
    # Reference : https://github.com/ShawnHymel/tinyml-example-anomaly-detection/blob/master/utils/c_writer.py
    # We use #04x to pad the output to 2 digits with a 0x prefix
    hex_array = [format(val, '#04x') for val in tflite_quant_model]
    # Calling function to convert an array into a C string (requires Numpy)
    # create_array(np_array, var_type, var_name, line_limit=80, indent=4)
    c_model = c_writer.create_array(np.array(hex_array), 'unsigned char', "network_model")
    # Calling Function to create a header file with given C code as a string
    header_str = c_writer.create_header(c_model, "network_model")
    return (header_str,)


@app.cell
def _(header_str):
    #Writing to the header file
    with open('network_model.h', 'w') as file:
        file.write(header_str)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generating Samples for Inference on Arduino
    """)
    return


@app.cell
def _(X_test, c_writer):
    # Converting a sample piece of the X test and y test data to C (for the purpose of ino code (arduino) to load and test
    # the sample and compare

    Xtest = X_test[0:5,:]
    print(c_writer.create_array(Xtest,"float","X_test"))
    return


@app.cell
def _(c_writer, y_test):
    ytest=y_test[0:5]
    print(c_writer.create_array(ytest,"uint8_t","y_test"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### Question 5 (d): ten more samples, excluding the first five
    """)
    return


@app.cell
def _(X_test, c_writer):
    Xtest10 = X_test[5:15,:]
    print(c_writer.create_array(Xtest10,"float","X_test"))
    return


@app.cell
def _(c_writer, y_test):
    ytest10 = y_test[5:15].ravel()
    print(c_writer.create_array(ytest10,"uint8_t","y_test"))
    return


if __name__ == "__main__":
    app.run()
