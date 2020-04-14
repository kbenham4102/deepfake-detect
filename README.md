# Repo for kaggle project concerning deepfake detection

### Requirements
* Tensorflow 2.1
* CUDA 10.1 and CuDNN 7.6.5
* OpenCV == 4.2.0
* Latest nvidia driver for your GPU
* numpy, pandas, other basic python libraries.

### Setup
* Basic setup involves downloading each of the ~ 10 GB video files from the kaggle page [here](https://www.kaggle.com/c/deepfake-detection-challenge/data)
* Once downloaded (unfortunately you have to do it manually) `setup_data.py` will take care of the rest and create a directory of source data:

###
    .
    |--...
    |--data
    |   |--source
    |       |--train
    |           |--REAL
    |           |--FAKE
    |       |--val
    |           |--REAL
    |           |--FAKE

* However, this sort does not keep matching pairs together. Therefore adhoc methods were used to map the exact file locations of each fake video to it's corresponding real video.
* This can be done using the master json created via running `setup_data.py`.
* Video pairs are then loaded to be processed as a batch together
* Real videos only make up about ~10% of the dataset.

### Data loading on the fly
* Data is loaded on the fly via several dataloader definitions in `video_loader.py`
* The latest used was `DeepFakeLoadExtractFeatures` which uses a `tflite` implementation of [blazeface](https://github.com/google/mediapipe/tree/master/mediapipe/models) to extract object feature maps from a real, fake video pair and pass to a custom CNN-LSTM for real, fake classification.


