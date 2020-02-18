import tensorflow as tf
from tensorflow.keras import models, layers
import pandas as pd
from datetime import datetime
from utils import *
import gc


def model(input_dim):
    model = models.Sequential()
    model.add(layers.ConvLSTM2D(64, (3,3), strides=(1,1), 
                                padding='valid', data_format='channels_last',
                                return_sequences=True, 
                                input_shape=input_dim))
    model.add(layers.Conv3D(64, (3,3,3), strides=(2,1,1), activation='relu',
                            data_format='channels_last'))
    model.add(layers.ConvLSTM2D(32, (3,3), strides=(1,1), 
                                padding='valid', data_format='channels_last',
                                return_sequences=True))
    model.add(layers.Conv3D(32, (3,3,3), strides=(2,1,1), activation='relu',
                            data_format='channels_last'))
    model.add(layers.ConvLSTM2D(16, (3,3), strides=(1,1), 
                                padding='valid', data_format='channels_last',
                                return_sequences=True))
    model.add(layers.Conv3D(16, (3,3,3), strides=(2,1,1), activation='relu',
                            data_format='channels_last'))
    model.add(layers.ConvLSTM2D(128, (3,3), strides=(1,1), 
                                padding='valid', data_format='channels_last'))
    model.add(layers.Conv2D(64, (3,3), strides=(2,2), data_format='channels_last'))
    model.add(layers.Conv2D(32, (3,3), strides=(2,2), data_format='channels_last'))
    model.add(layers.Conv2D(16, (3,3), strides=(2,2), data_format='channels_last'))
    model.add(layers.Conv2D(8, (3,3), strides=(2,2), data_format='channels_last'))
    model.add(layers.Conv2D(4, (3,3), strides=(2,2), data_format='channels_last'))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    print(model.summary())
    return model


def train(data, batch_size = 1, epochs = 1, input_shape = (1,300,300,300,3)):
    

    clf_model = model(input_shape)


    clf_model.compile(
        optimizer='sgd',
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.metrics.AUC(), 'binary_crossentropy', 'binary_accuracy'])


    clf_model.fit(
        x = data,
        epochs=epochs,
        verbose=2,
        # validation_split=0.1,
        # validation_freq=[1,2,10],
        # max_queue_size=10,
        # workers=4,
        use_multiprocessing=False
    )
    return clf_model

if __name__=="__main__":

    # Define some params
    epochs=1
    batch_size=1
    train_label_path = '../data/source/labels/train_meta.json'
    train_path = '../data/source/train/'

    # Get input dims from a single image
    d = load_transform_batch([train_path + 'acxwigylke.mp4']).shape
    print('Inputs have dims of ', d[1:])
    gc.collect()


    df = load_process_train_targets(train_label_path, train_path)
    data = DeepFakeDataSeq(df.filepath.to_list(), df.target_class.to_list(), batch_size)

    clf_trained = train(data, batch_size=batch_size, epochs=epochs, input_shape=d[1:])
    # TODO make the custom training protocol in tensorflow, 
    # enforce data loading on cpu and training run on gpu, 
    # make sure gpu stays clear and healthy
    # 1080Ti should be able to handle ~550 frames at a time.

    