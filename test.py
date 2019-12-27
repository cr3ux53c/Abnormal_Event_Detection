"""
Testing module to test the presence of Anomalous Events in a Video

The module computes reconstruction loss between input bunch and
the reconstructed batch from the model, and flagges the batch as anomalous
if loss value is greater than a given threshold.

Author: Harsh Tiku
"""

from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, minmax_scale
import numpy as np
from matplotlib import pyplot as plt
import argparse
import tensorflow as tf

config = tf.ConfigProto()

config.gpu_options.per_process_gpu_memory_fraction = 0.8

tf.keras.backend.set_session(tf.Session(config=config))

# Parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument('model', type=str)
parser.add_argument('dataset', type=str)
args = parser.parse_args()


def mean_squared_loss(x1, x2):
    ''' Compute Euclidean Distance Loss  between
    input frame and the reconstructed frame'''

    diff = x1 - x2
    a, b, c, d, e = diff.shape
    n_samples = a * b * c * d * e
    sq_diff = diff ** 2
    Sum = sq_diff.sum()
    dist = np.sqrt(Sum)
    mean_dist = dist / n_samples

    return mean_dist


'''Define threshold for Sensitivity
Lower the Threshhold,higher the chances that a bunch of frames will be flagged as Anomalous.

'''

# threshold = 0.0004 # 사람이 큰 것들이 나타났을 때 검출
# threshold = 0.0003795
# threshold = 0.00017
threshold = 0.000179

model = load_model(args.model)

X_test = np.load(args.dataset)
frames = X_test.shape[2]
# Need to make number of frames divisible by 10


flag = 0  # Overall video flag

frames = frames - frames % 10

X_test = X_test[:, :, :frames]

# X_test_overlapped = np.empty((227, 227, (int(frames * 2))), float)
#
# for width in range(227):
#     for height in range(227):
#         for start_idx in range(0, frames - 9, 5):
#             X_test_overlapped[width, height, start_idx * 2:start_idx * 2 + 10] = X_test[width, height, start_idx:start_idx + 10]
#
# X_test_overlapped = X_test_overlapped[:, :, :frames * 2]
# X_test_overlapped = X_test_overlapped.reshape(-1, 227, 227, 10)  # 10씩 끊어서
# X_test_overlapped = np.expand_dims(X_test_overlapped, axis=4)

X_test = X_test.reshape(-1, 227, 227, 10)
X_test = np.expand_dims(X_test, axis=4)

reconstruction_errors = []
predictions = np.empty(0)

for number, bunch in enumerate(X_test):
    """
    bunch <class 'tuple'>: (227, 227, 10, 1)
    """
    n_bunch = np.expand_dims(bunch, axis=0)
    reconstructed_bunch = model.predict(n_bunch)

    # Reconstruction Error
    loss = mean_squared_loss(n_bunch, reconstructed_bunch)
    reconstruction_errors.append(loss)

    if loss > threshold:
        print('Anomalous frames at bunch number %3d with loss: %.10f' % (number + 1, loss))
        flag = 1
        predictions = np.append(predictions, 1)
    else:
        print('Normal frames at bunch number %3d with loss: %.10f' % (number + 1, loss))
        predictions = np.append(predictions, 0)

if flag == 1:
    print("Anomalous Events detected")

# Evaluate Regularity score
reconstruction_errors = np.asarray(reconstruction_errors)
abnormal_score = minmax_scale(reconstruction_errors)
regularity_score = 1 - abnormal_score
# regularity_score = np.clip(regularity_score, 0, 1)
plt.figure(figsize=(20, 4))
plt.xticks(np.arange(0, 1600, 100.0))
plt.plot(regularity_score)
plt.title('Regularity Error (Threshold: %f)' % threshold)
plt.show()

plt.clf()
plt.figure(figsize=(20, 4))
plt.xticks(np.arange(0, 1600, 100.0))
plt.plot(predictions)
plt.title('Predictions (Threshold: %f)' % threshold)
plt.show()
