"""
Testing module to test the presence of Anomalous Events in a Video

The module computes reconstruction loss between input bunch and
the reconstructed batch from the model, and flagges the batch as anomalous
if loss value is greater than a given threshold.
"""

import os
from keras.models import load_model

import numpy as np
import argparse
import scipy.io
import matlab.engine
from sklearn.preprocessing import minmax_scale

# Config backend
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
# config.gpu_options.allow_growth = True
# tf.keras.backend.set_session(tf.Session(config=config))

# Parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument('model', type=str)
parser.add_argument('dataset', type=str)
parser.add_argument('ground_truth_path', type=str)
args = parser.parse_args()

# Set environment variables
reconstruction_errors = []

# Prepare Ground Truth
gt_files = os.listdir(args.ground_truth_path)
gt_files = [video for video in gt_files if video.endswith('.mat')]
gt_by_all_video = []
for gt_file in gt_files:
    ground_truth = scipy.io.loadmat(os.path.join(args.ground_truth_path, gt_file))
    frame_size = int(ground_truth['volLabel'][0].size - ground_truth['volLabel'][0].size % 10)
    gt_by_video = []

    for frame in range(frame_size):
        if np.max(ground_truth['volLabel'][0][frame]) == 0:
            gt_by_video.append(0)
        else:
            gt_by_video.append(1)
    gt_by_all_video.append(gt_by_video)

gt_by_all_video_flatten = [y for x in gt_by_all_video for y in x]
np.save('gt_by_all_video_flatten.npy', gt_by_all_video_flatten)

def mean_squared_loss(x1, x2):
    """ Compute Euclidean Distance Loss  between
    input frame and the reconstructed frame"""
    diff = x1 - x2
    a, b, c, d, e = diff.shape
    n_samples = a * b * c * d * e
    sq_diff = diff ** 2
    Sum = sq_diff.sum()
    dist = np.sqrt(Sum)
    return dist / n_samples


# Load trained model
model = load_model(args.model)

# Load data
X_test = np.load(args.dataset)
frames = X_test.shape[2]
# Need to make number of frames divisible by 10
frames = frames - frames % 10
# Reshape data
X_test = X_test[:, :, :frames]
X_test = X_test.reshape(-1, 227, 227, 10)
X_test = np.expand_dims(X_test, axis=4)

# Calc Reconstruction errors
for number, bunch in enumerate(X_test):
    # Create a bunch <class 'tuple'>: (227, 227, 10, 1)
    n_bunch = np.expand_dims(bunch, axis=0)
    reconstructed_bunch = model.predict(n_bunch)

    # Reconstruction Error
    loss = mean_squared_loss(n_bunch, reconstructed_bunch)
    reconstruction_errors.append(loss)

reconstruction_errors = np.asarray(reconstruction_errors)
np.save('reconstruction_errors.npy', reconstruction_errors)


# Processing Reconstruct1D
# Evaluate Regularity score
abnormal_score = minmax_scale(reconstruction_errors)
regularity_score = 1 - abnormal_score
regularity_score = np.clip(regularity_score, 0, 1)
# Running matlab
scipy.io.savemat('regularity_score.mat', mdict={'regularity_score': regularity_score})  # For running in matlab
matlab_engine = matlab.engine.start_matlab()
matlab_engine.addpath('persistence1d/reconstruct1d/examples')
matlab_engine.addpath('persistence1d/reconstruct1d')
matlab_engine.reconstruct_numpy(nargout=0)  # Smooth regularity-score
matlab_engine.exit()
file = open('regularity_score_with_reconstruct1d-x_bi_smooth.txt', 'r')
reconstruct1d_errors = file.readlines()
file.close()
reconstruct1d_errors = np.array(reconstruct1d_errors).astype(np.float)
np.save('reconstruct1d_errors.npy', reconstruct1d_errors)
