import os
import scipy.io
import numpy as np
from matplotlib import pyplot as plt
import itertools

file_list = os.listdir('./testing_label_mask')

ground_truth_files = [video for video in file_list if video.endswith('.mat')]

gt_by_all_video = []

for ground_truth_file in ground_truth_files:
    ground_truth = scipy.io.loadmat(os.path.join('./testing_label_mask', ground_truth_file))
    frame_size = int(ground_truth['volLabel'][0].size - ground_truth['volLabel'][0].size % 10)

    # np.array()
    gt_by_video = []

    for frame in range(frame_size):
        if np.max(ground_truth['volLabel'][0][frame]) == 0:
            gt_by_video.append(0)
        else:
            gt_by_video.append(1)
    gt_by_all_video.append(gt_by_video)

# gt_by_all_video = np.asarray(gt_by_all_video, dtype=int)
# gt_by_all_video.reshape(-1)

gt_by_all_video_flatten = [y for x in gt_by_all_video for y in x]
# gt_by_all_video_flatten = np.asarray(gt_by_all_video_flatten)
plt.figure(figsize=(25, 5))
plt.xticks(np.arange(0, 16000, step=500))
# plt.scatter([2, 3, 4, 5], [2, 3, 4])
# plt.hist(gt_by_all_video_flatten)
plt.plot(gt_by_all_video_flatten, color='red')
# plt.axis([0, 13000, 0, 1.1])
plt.show()
