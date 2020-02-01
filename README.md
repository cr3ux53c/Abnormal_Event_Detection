# Abnormal Event Detection in Videos using SpatioTemporal AutoEncoder

Code for the [paper](https://arxiv.org/abs/1701.01546).

# Test Environment:

Windows NT 10.0

ffmpeg for Windows

Python for Windows (Refer to requirements.txt)

# Instructions

Run preprocessing.py with args the path of Video directory, frame rate and filename for save.

Run train.py to train the model

Run test.py to test on testing data.

Run start_live_feed.py to test the model on live webcam feed.

You can adjust the threshold parameter in test.py to different values to adjust sensitivity

Datasets Recommended: Avenue Dataset for Anomaly Detection