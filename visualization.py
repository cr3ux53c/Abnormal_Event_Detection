import numpy as np
import argparse
from matplotlib import pyplot as plt
from sklearn.preprocessing import minmax_scale
import math

# Parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument('gt_by_all_video_flatten', type=str)
parser.add_argument('reconstruction_errors', type=str)
args = parser.parse_args()

# Set environment variables
flag = 0  # Overall video flag
# Define threshold for Sensitivity: Lower the Threshold, higher the chances that a bunch of frames will be flagged as Anomalous.
# threshold = 0.0004 # 사람이 큰 것들이 나타났을 때 검출
# threshold = 0.0003795
# threshold = 0.00017
threshold = 0.000170
threshold_reconstruct1d = 0.75
predictions = np.empty(0)
predictions_reconstruct1d = np.empty(0)


def IncreaseByTenfold(data):
    expanded_data = np.empty(0)
    for index, val in enumerate(data):
        expanded_data = np.append(expanded_data, val)
        for i in range(9):
            expanded_data = np.append(expanded_data, val)
    return expanded_data


# Paint Ground Truth
gt_by_all_video_flatten = np.load(args.gt_by_all_video_flatten)

# plt.figure(figsize=(80, 5))
# plt.title('Abnormal detection from Regularity score by threshold (Threshold: %f)' % 0.000179)
# plt.xticks(np.arange(0, gt_by_all_video_flatten.shape[0], step=50), rotation=45)
# for i, value in enumerate(gt_by_all_video_flatten):
#     if value == 1:
#         plt.axvspan(i, i, alpha=0.7, color='red')
# plt.axhline(y=threshold, color='black', linestyle='dotted')
#
# # Evaluate reconstruction errors with threshold
reconstruction_errors = np.load(args.reconstruction_errors)
# for number, reconstruction_error in enumerate(reconstruction_errors):
#     if reconstruction_error > threshold:
#         print('Anomalous frames at bunch number %3d with loss: %.10f' % (number + 1, reconstruction_error))
#         flag = 1
#         predictions = np.append(predictions, 1)
#     else:
#         print('Normal frames at bunch number %3d with loss: %.10f' % (number + 1, reconstruction_error))
#         predictions = np.append(predictions, 0)
# if flag == 1:
#     print("Anomalous Events detected")
#
# # Evaluate regularity-score (paper p.09)
abnormal_score = minmax_scale(reconstruction_errors)
regularity_score = 1 - abnormal_score
regularity_score = np.clip(regularity_score, 0, 1)
#
# # Expand data
# expanded_regularity_score = IncreaseByTenfold(regularity_score)
# expanded_predictions = IncreaseByTenfold(predictions)
#
# # Draw regularity-score with predictions
# plt.plot(expanded_regularity_score, color='blue')
# for i, value in enumerate(expanded_predictions):
#     if value == 1:
#         plt.axvspan(i, i, alpha=0.3, color='orange')
#
# plt.show()
# plt.clf()

# Reconstruct1D data
# plt.figure(figsize=(80, 5))
# plt.title('Abnormal detection from Reconstruct1D score by threshold (Threshold: %f)' % 0.000179)
# plt.xticks(np.arange(0, gt_by_all_video_flatten.shape[0], step=50), rotation=45)
# for i, value in enumerate(gt_by_all_video_flatten):
#     if value == 1:
#         plt.axvspan(i, i, alpha=0.7, color='red')
# plt.axhline(y=threshold_reconstruct1d, color='black', linestyle='dotted')
#
# # Evaluate Reconstruct1D errors with threshold
# reconstruct1d_errors = np.load('reconstruct1d_errors.npy')
# for number, reconstruct1d_error in enumerate(reconstruct1d_errors):
#     if reconstruct1d_error < threshold_reconstruct1d:
#         print('Anomalous frames at bunch number %3d with loss: %.10f' % (number + 1, reconstruct1d_error))
#         flag = 1
#         predictions_reconstruct1d = np.append(predictions_reconstruct1d, 1)
#     else:
#         print('Normal frames at bunch number %3d with loss: %.10f' % (number + 1, reconstruct1d_error))
#         predictions_reconstruct1d = np.append(predictions_reconstruct1d, 0)
# if flag == 1:
#     print("Anomalous Events detected")
#
# # Expand data
# expanded_reconstrct1d_score = IncreaseByTenfold(reconstruct1d_errors)
# expanded_predictions_reconstruct1d = IncreaseByTenfold(predictions_reconstruct1d)
# plt.plot(expanded_regularity_score, color='blue')
# plt.plot(expanded_reconstrct1d_score, color='magenta')
# for i, value in enumerate(expanded_predictions_reconstruct1d):
#     if value == 1:
#         plt.axvspan(i, i, alpha=0.3, color='orange')
#
# plt.show()

# New Algorithm

least_threshold = 0.6
new_predictions = np.empty(0)

plt.figure(figsize=(80, 5))
plt.title('Abnormal detection from Reconstruct1D score by threshold (Threshold: %f)' % 0.000179)
plt.xticks(np.arange(0, gt_by_all_video_flatten.shape[0], step=50), rotation=45)
for i, value in enumerate(gt_by_all_video_flatten):
    if value == 1:
        plt.axvspan(i, i, alpha=0.7, color='red')
# Draw threshold
plt.axhline(y=least_threshold, color='black', linestyle='dotted')

reconstruct1d_errors = np.load('reconstruct1d_errors.npy')
skip_count = 0
for number, reconstruct1d_error in enumerate(reconstruct1d_errors):
    if skip_count > 0:
        skip_count -= 1
        continue

    is_abnormal = False

    # 절댓값 기준치
    # if reconstruct1d_error < least_threshold:
    #     new_predictions = np.append(new_predictions, 1)
    #     is_abnormal = True
    #
    # if not is_abnormal:
    #     new_predictions = np.append(new_predictions, 0)

    # 50Frames 이상 감소
    count = 0
    for i in range(number, len(reconstruct1d_errors)-1):
        if round(reconstruct1d_errors[i], 3) > round(reconstruct1d_errors[i + 1], 3):
            count += 1
        else:
            break

    if count > 5:
        skip_count = count
        for i in range(count):
            new_predictions = np.append(new_predictions, 1)
    else:
        new_predictions = np.append(new_predictions, 0)

expanded_reconstruct1d_score = IncreaseByTenfold(reconstruct1d_errors)
plt.plot(IncreaseByTenfold(regularity_score), color='blue')
plt.plot(expanded_reconstruct1d_score, color='magenta')
for i, value in enumerate(IncreaseByTenfold(new_predictions)):
    if value == 1:
        plt.axvspan(i, i, alpha=0.3, color='orange')

plt.show()
