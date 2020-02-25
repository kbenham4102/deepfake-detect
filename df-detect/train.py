import tensorflow as tf
import numpy as np
import gc
import pandas as pd 
from tensorflow.keras.layers import ConvLSTM2D, Conv3D, Conv2D, Flatten, Dense
from utils import *
from sklearn.model_selection import train_test_split
import sys

# Set some logging information
# tf.debugging.set_log_device_placement(True)


class DeepFakeDetector(tf.keras.Model):
    def __init__(self):
        super(DeepFakeDetector, self).__init__()
        self.seq_det1 = ConvLSTM2D(32, (3,3), strides=(1,1), 
                                   padding='same', data_format='channels_last', 
                                   return_sequences=True)
        self.seq_reduce1 = Conv3D(32 , (3,3,3), strides=(2,1,1), 
                                  padding='same', data_format='channels_last', 
                                  activation='relu')
        self.seq_det2 = ConvLSTM2D(64, (3,3), strides=(1,1), 
                                   padding='same', data_format='channels_last', 
                                   return_sequences=True)
        self.seq_reduce2 = Conv3D(64 , (3,3,3), strides=(2,1,1), 
                                  padding='same', data_format='channels_last', 
                                  activation='relu')
        self.seq_agg1 = ConvLSTM2D(128, (3,3), strides=(1,1), 
                                   padding='same', data_format='channels_last', 
                                   return_sequences=False)
        self.region_suggest1 = Conv2D(128 , (3,3), strides=(2,2), 
                                  padding='valid', data_format='channels_last', 
                                  activation='relu')
        self.region_suggest2 = Conv2D(128 , (3,3), strides=(2,2), 
                                  padding='valid', data_format='channels_last', 
                                  activation='relu')
        self.region_suggest3 = Conv2D(256 , (3,3), strides=(2,2), 
                                  padding='valid', data_format='channels_last', 
                                  activation='relu')
        self.region_suggest4 = Conv2D(256 , (3,3), strides=(2,2), 
                                  padding='valid', data_format='channels_last', 
                                  activation='relu')
        self.region_suggest5 = Conv2D(256 , (3,3), strides=(2,2), 
                                  padding='valid', data_format='channels_last', 
                                  activation='relu')
        self.flatten = Flatten()
        self.dense1 = Dense(1, activation='sigmoid')
    
    def call(self, input_data):
        x = self.seq_det1(input_data)
        x = self.seq_reduce1(x)
        x = self.seq_det2(x)
        x = self.seq_reduce2(x)
        x = self.seq_agg1(x)
        x = self.region_suggest1(x)
        x = self.region_suggest2(x)
        x = self.region_suggest3(x)
        x = self.region_suggest4(x)
        x = self.region_suggest5(x)
        x = self.flatten(x)
        out = self.dense1(x)
        return out
    
    def build_graph(self, input_shape):
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)

        if not hasattr(self, 'call'):
            raise AttributeError('User should define `call` method in subclass model!')

        _ = self.call(inputs)


class CollectBatchStats(tf.keras.callbacks.Callback):
    def __init__(self):
        self.batch_losses = []
        self.batch_acc = []

    def on_train_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs['loss'])
        self.batch_acc.append(logs['accuracy'])
        self.model.reset_metrics()


# Callback for printing the LR at the end of each epoch.
class PrintLR(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    print('\nLearning rate for epoch {} is {}'.format(epoch + 1,
                                                      model.optimizer.lr.numpy()))

def decay(epoch):
  if epoch < 3:
    return 1e-3
  elif epoch >= 3 and epoch < 7:
    return 1e-4
  else:
    return 1e-5


if __name__ == "__main__":

    # Define some params
    EPOCHS=1
    validate_epochs = [1,2,10]
    batch_size=1
    test_fraction = 0.2
    train_label_path = '../data/source/labels/train_meta.json'
    train_path = '../data/source/train/'
    checkpoint_prefix = 'models/ckpt_{epoch}'
    resize_shape = (224,224)
    sequence_len = 16
    n_workers = 1
    use_mult_prc = False


    # Get device names
    # Note this is for tf 2.1 when upgrading
    print(tf.config.experimental.list_physical_devices('CPU')[0].name)
    print(x[0].name for x in tf.config.experimental.list_physical_devices('GPU'))
    # devs = tf.config.experimental_list_devices()
    # cpu = devs[0]
    # gpu = devs[-1]
    # print("Found CPU at ", cpu)
    # print("Found GPU at ", gpu)

    # Run an input test to get a model summary
    # Get input dims from a single image
    test_dims = load_transform_batch([train_path + 'acxwigylke.mp4'], 
                                        resize_shape=resize_shape, 
                                        seq_length=sequence_len).shape
    print('Inputs have dims of ', test_dims[1:])


    strategy = tf.distribute.MirroredStrategy()



    # test_loss = tf.keras.metrics.Mean(name='test_loss')
    # test_auc = tf.keras.metrics.AUC()
    # test_acc = tf.keras.metrics.BinaryAccuracy()

    # Load train filepaths into a dataframe
    df = load_process_train_targets(train_label_path, train_path)

    # Split those according to train test split
    train_df, test_df = train_test_split(df, test_size=test_fraction)

    train_data = DeepFakeDataSeq(train_df.filepath.to_list(), 
                                 train_df.target_class.to_list(), 
                                 batch_size, 
                                 resize_shape=resize_shape, 
                                 sequence_len=sequence_len)
    test_data = DeepFakeDataSeq(test_df.filepath.to_list(), 
                                test_df.target_class.to_list(), 
                                batch_size, 
                                resize_shape=resize_shape, 
                                sequence_len=sequence_len)
    X_len = len(train_df.filepath)
    X_test_len = len(test_df.filepath)
    num_train_batches = int(np.ceil(X_len/batch_size))
    num_test_batches = np.ceil(X_test_len/batch_size)

    # Define model and get structure within distribution strategy
    with strategy.scope():
        model = DeepFakeDetector()
        model.build_graph(test_dims)
        # Loss object
        loss_object = tf.keras.losses.BinaryCrossentropy()

        # Optimizer
        optimizer = tf.keras.optimizers.SGD()
        # Compile the model under the strategy scope
        model.compile(optimizer=optimizer, loss=loss_object, metrics=['accuracy'])

    
    print(model.summary())

    # Define something to keep the callbacks for plot gen
    batch_stats_callback = CollectBatchStats()

    callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                       save_weights_only=True),
    tf.keras.callbacks.LearningRateScheduler(decay),
    PrintLR(), 
    batch_stats_callback
    ]


    history = model.fit(train_data, 
                        epochs=EPOCHS, 
                        steps_per_epoch=10, 
                        callbacks=callbacks,
                        validation_data=test_data, 
                        validation_steps=10, 
                        )
    

