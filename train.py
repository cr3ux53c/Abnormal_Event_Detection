"""
The training Module to train the SpatioTemporal AutoEncoder

Run:

>>python3 train.py n_epochs(enter integer) to begin training.

Author: Harsh Tiku
"""

from keras.callbacks import ModelCheckpoint, EarlyStopping
from model import load_model
import numpy as np
import argparse
import tensorflow as tf


config = tf.ConfigProto()

config.gpu_options.per_process_gpu_memory_fraction = 0.8

tf.keras.backend.set_session(tf.Session(config=config))
# Running arguments
parser = argparse.ArgumentParser()
parser.add_argument('n_epochs', type=int)
parser.add_argument('model_path_to_save', type=str)
parser.add_argument('dataset', type=str)

args = parser.parse_args()

X_train = np.load(args.dataset)
frames = X_train.shape[2]  # X x Y x batch

# Need to make number of frames divisible by 10
frames = frames - frames % 10
X_train = X_train[:, :, :frames]


# X_train_overlapped = np.empty((227, 227, (int(frames*2))), float)
#
# for width in range(227):
#     for height in range(227):
#         for start_idx in range(0, frames - 9, 5):
#             X_train_overlapped[width, height, start_idx*2:start_idx*2+10] = X_train[width, height, start_idx:start_idx + 10]
#
# X_train_overlapped = X_train_overlapped[:, :, :frames*2]
# X_train_overlapped = X_train_overlapped.reshape(-1, 227, 227, 10)  # 10씩 끊어서
# X_train_overlapped = np.expand_dims(X_train_overlapped, axis=4)
# Y_train = X_train_overlapped.copy()

X_train = X_train.reshape(-1, 227, 227, 10)  # 10씩 끊어서
X_train = np.expand_dims(X_train, axis=4)
Y_train = X_train.copy()


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
