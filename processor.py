''' This module extracts frames a Video
performs preprocessing on the frames and stores them in a Numpy array for
furthur use by the spatiotemporal autoencoder

___________________________________________________________________

Dependencies: ffmpeg

If you dont have ffmpeg installed:

Install it with :


1. sudo apt-get install ffmpeg for Linux Users
2. brew install ffmpeg for macOS

__________________________________________________________________

Usage:

python3 processor.py video_dir_path time_in_seconds_to_extract_one_frame

eg;python3 processor.py ./train 5   will search for train directory and for each video in train directory

It will extract 1 frame every 5 seconds and store it.



__________________________________________________________


Author: Harsh Tiku
'''

from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import glob
import os
from scipy.misc import imresize
import argparse
from PIL import Image

# parsing parameters
parser = argparse.ArgumentParser(description='Source Video path')
parser.add_argument('source_vid_path', type=str)
parser.add_argument('fps', type=int)
parser.add_argument('save_file_name', type=str)
args = parser.parse_args()

video_source_path = args.source_vid_path
fps = args.fps
image_store = []


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def remove_old_images(path):
    file_list = glob.glob(os.path.join(path, "*.*"))
    for f in file_list:
        os.remove(f)


def convert_to_vector(path):
    img = load_img(path)
    img = img_to_array(img)

    # Resize the Image to (227,227,3) for the network to be able to process it.
    img = imresize(img, (227, 227, 3))  # `imresize` is deprecated in SciPy 1.0.0, and will be removed in 1.3.0.

    # Convert the Image to Grayscale
    return 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]  # RGB to YIQ Convert (RGB 색상에 대한 인간의 인지/감각의 차이에 의한 상수값


# List of all Videos in the Source Directory.
videos = os.listdir(video_source_path)
print("Found ", len(videos), " training video")

# Make a temp dir to store all the frames
create_dir(video_source_path + '/frames')
# Remove old images
# remove_old_images(video_source_path + '/frames')

# Capture video frames
for video in videos:
    # pass
    os.system('ffmpeg -i {}/{} -r 1/{}  {}/frames/{}-%03d.jpg'.format(video_source_path, video, fps, video_source_path, video))
images = os.listdir(video_source_path + '/frames')
for image in images:
    image_path = video_source_path + '/frames' + '/' + image
    vector = convert_to_vector(image_path)
    image_store.append(vector)

image_store = np.array(image_store)

a, b, c = image_store.shape
# Reshape to (227,227,batch_size)
image_store.resize(b, c, a)
# Normalize
image_store = (image_store - image_store.mean()) / (image_store.std())
# Clip negative Values
image_store = np.clip(image_store, 0, 1)  # 0, 1 사이를 벗어나는 값은 0, 1로 치환
np.save(args.save_file_name, image_store)
# Remove Buffer Directory
# os.system('rm -r {}'.format(video_source_path + '/frames')) # Linux only
