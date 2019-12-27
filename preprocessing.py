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

python3 preprocessing.py video_dir_path time_in_seconds_to_extract_one_frame

eg;python3 preprocessing.py ./train 5   will search for train directory and for each video in train directory

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


def move_to_evaluation(path, eval_path):
    os.system('move %s %s' % (path, eval_path))

def move_to_evaluation_with_rename(path, eval_path, index):
    images = os.listdir(path)
    images = [image for image in images if image.endswith('.jpg')]
    for image in images:
        os.rename(path + image, eval_path + index + '.jpg')

# def create_bunch_video(video_filename, bunch_numer, path, path_to_save):
#     os.system('ffmpeg -r 25 -i {0}/{1}-%03d.jpg -vcodec mpeg4 -b 100k {2}/{3}-{4}.avi'.format(path, video_filename, path_to_save, video_filename, bunch_numer))


# List of all Videos in the Source Directory.
videos = os.listdir(video_source_path)
videos = [video for video in videos if video.endswith('.avi')]

print("Found ", len(videos), " training video")

# Make a temp dir to store all the frames
create_dir(video_source_path + '/frames')
# Remove old images
remove_old_images(video_source_path + '/frames')

index = 1

# Capture video frames
for video in videos:
    # pass
    print('ffmpeg capturing %s file...' % video)
    os.system(
        'ffmpeg -i {}/{} -r {}  {}/frames/{}-%04d.jpg'.format(video_source_path, video, args.fps, video_source_path,
                                                              video))
    images = os.listdir(video_source_path + '/frames')
    for i in range(images.__len__() - int(images.__len__() % 10)):
        print('\tAdding %04dth image to vector' % i)
        image_path = video_source_path + '/frames' + '/' + images[i]
        vector = convert_to_vector(image_path)
        image_store.append(vector)

    # create_bunch_video(video.__str__(), 1, video_source_path + '/frames', 'evaluation/images')
    for f in images[images.__len__() - int(images.__len__() % 10):]:
        os.remove(video_source_path + '/frames/' + f)
    # move_to_evaluation('.\\test\\frames\\*.jpg', 'evaluation\\images')
    images = os.listdir(video_source_path + '/frames')
    images = [image for image in images if image.endswith('.jpg')]
    # for image in images:
    #     os.rename(video_source_path + '/frames/' + image, 'evaluation/images/' + str(index) + '.jpg')
    #     index += 1

    remove_old_images(video_source_path + '/frames')

image_store = np.array(image_store)

a, b, c = image_store.shape
# Reshape to (227,227,batch_size)
image_store.resize(b, c, a)
# Normalize
image_store = (image_store - image_store.mean()) / (image_store.std())
# Scaling - Clip negative Values
image_store = np.clip(image_store, 0, 1)  # 0, 1 사이를 벗어나는 값은 0, 1로 치환
np.save(args.save_file_name, image_store)
# Remove Buffer Directory
# os.system('rm -r {}'.format(video_source_path + '/frames')) # Linux only
