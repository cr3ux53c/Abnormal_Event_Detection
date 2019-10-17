import tensorflow as tf
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

# print('============')
# print(tf.test.gpu_device_name())
# print('============')
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     print("Name:", gpu.name, "  Type:", gpu.device_type)
# print(tf.__version__())
