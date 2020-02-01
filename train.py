"""
The training Module to train the SpatioTemporal AutoEncoder

Usage:
>>python train.py n_epochs model_path_to_save dataset

eg;python train.py 50 dev-25fps-50epochs.h5 dev-train-25fps.npy
"""

from keras.callbacks import ModelCheckpoint, EarlyStopping
from model import load_model
import numpy as np
import argparse
import tensorflow as tf

# Config backend
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
# config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))

# Parsing parameters
parser = argparse.ArgumentParser()
parser.add_argument('n_epochs', type=int)
parser.add_argument('model_path_to_save', type=str)
parser.add_argument('dataset', type=str)
args = parser.parse_args()

# Load and Prepare data
X_train = np.load(args.dataset)
frames = X_train.shape[2]  # X x Y x batch

# Need to make number of frames divisible by 10
frames = frames - frames % 10
X_train = X_train[:, :, :frames]

# Reshape data
X_train = X_train.reshape(-1, 227, 227, 10)  # 10씩 끊어서
X_train = np.expand_dims(X_train, axis=4)
Y_train = X_train.copy()

# Set hyper-parameters
epochs = args.n_epochs
batch_size = 1

if __name__ == "__main__":
    model = load_model()

    callback_save = ModelCheckpoint(args.model_path_to_save, monitor="mean_squared_error", save_best_only=False)
    callback_early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    print('Model has been loaded')

    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=epochs,
              callbacks=[callback_save, callback_early_stopping]
              )
