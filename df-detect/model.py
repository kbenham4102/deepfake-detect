import tensorflow as tf
from tensorflow.keras import models, layers
import pandas as pd
from datetime import datetime
from utils import *



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


def train_validate(train_label_path, train_path, test_path, batch_size, input_shape):

    df = load_process_train_targets(train_label_path, train_path)


    data = DeepFakeDataSeq(df.filepath.to_list(), df.target_class.to_list(), batch_size)

    model = model()

    model.fit(
        x = data,
        epochs = 1,
        verbose=2,
        validation_split=0.1
        validation_freq=[1,2,10]
        max_queue_size=16,
        workers=4
        use_multiprocessing=True
    )

if __name__=="__main__":
    