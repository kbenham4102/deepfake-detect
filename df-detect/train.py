import tensorflow as tf
import numpy as np
import gc
import pandas as pd 
from tensorflow.keras.layers import ConvLSTM2D, Conv3D, Conv2D, Flatten, Dense, BatchNormalization
from utils import *
from sklearn.model_selection import train_test_split
from video_loader import DeepFakeTransformer
import sys
import pathlib
import datetime
import matplotlib.pyplot as plt



class CollectBatchStats(tf.keras.callbacks.Callback):
    def __init__(self):
        self.batch_losses = []
        self.batch_acc = []

    def on_train_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs['loss'])
        self.batch_acc.append(logs['binary_accuracy'])
        self.model.reset_metrics()


# Callback for printing the LR at the end of each epoch.
class PrintLR(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    print('\nLearning rate for epoch {} is {}'.format(epoch + 1,
                                                      model.optimizer.lr.numpy()))

def decay(epoch):
  if epoch < 3:
    return 1e-3
  elif epoch >= 5 and epoch < 7:
    return 1e-4
  else:
    return 1e-5



if __name__ == "__main__":
  # Define some params
  EPOCHS=10
  batch_size=3
  test_fraction = 0.2
  train_root = '../data/source/train_val_sort/train'
  val_root = '../data/source/train_val_sort/val'
  checkpoint_prefix = 'models/ckpt_{epoch}'
  resize_shape = (224,224)
  sequence_len = 30
  n_workers = 1
  use_mult_prc = False
  trunc_epochs = True
  epoch_steps = 1000



  # Define dummy test dims based on parameters
  test_dims = (batch_size, None, *resize_shape, 3)

  # Get device names

  cpu = tf.config.experimental.list_physical_devices('CPU')[0].name
  print("FOUND CPU AT: ", cpu)
  print([x[0] for x in tf.config.experimental.list_physical_devices('GPU')])



  
  
  train_ds = tf.data.Dataset.list_files(str(pathlib.Path(train_root)/'*/*'))
  val_ds = tf.data.Dataset.list_files(str(pathlib.Path(val_root)/'*/*'))
  train_ds = train_ds.shuffle(10000)
  val_ds = val_ds.shuffle(10000)

  X_train_len = len(list(train_ds))
  X_val_len = len(list(val_ds))
  num_train_batches = int(np.ceil(X_train_len/batch_size))
  num_val_batches = int(np.ceil(X_val_len/batch_size))

  train_transformer = DeepFakeTransformer(resize_shape=(224,224), seq_length=sequence_len)

  # TODO add in random crops, rotations, etc to make this non-redundant
  val_transformer = DeepFakeTransformer(resize_shape=(224,224), seq_length=sequence_len)

  # Map transformations to videos loaded from filenames and batch the datasets
  train_ds = train_ds.map(lambda x: train_transformer.transform_map(x)).batch(batch_size).prefetch(2)
  val_ds = val_ds.map(lambda x: val_transformer.transform_map(x)).batch(batch_size).prefetch(2)



  # Define model and get structure within distribution strategy
  # strategy = tf.distribute.MirroredStrategy()
  # with strategy.scope():

  reg = tf.keras.regularizers.l2(l=0.001)
  model = tf.keras.models.Sequential()
  model.add(ConvLSTM2D(64, (3,3), strides=(2,2), 
                              padding='valid', 
                              data_format='channels_last',
                              recurrent_regularizer=reg,
                              kernel_regularizer=reg,
                              bias_regularizer=reg, 
                              return_sequences=True,
                              input_shape=test_dims[1:]))
  model.add(ConvLSTM2D(128, (3,3), strides=(2,2), 
                              padding='valid', 
                              data_format='channels_last',
                              recurrent_regularizer=reg,
                              kernel_regularizer=reg,
                              bias_regularizer=reg,  
                              return_sequences=True,
                              ))
  model.add(ConvLSTM2D(128, (3,3), strides=(2,2), 
                              padding='valid', 
                              data_format='channels_last',
                              recurrent_regularizer=reg,
                              kernel_regularizer=reg,
                              bias_regularizer=reg, 
                              return_sequences=True,
                              ))
  model.add(ConvLSTM2D(256, (3,3), strides=(2,2), 
                              padding='valid', 
                              data_format='channels_last',
                              recurrent_regularizer=reg,
                              kernel_regularizer=reg,
                              bias_regularizer=reg,  
                              return_sequences=False,
                              ))
  model.add(BatchNormalization())
  model.add(Conv2D(128 , (3,3), strides=(2,2), 
                          padding='valid', data_format='channels_last',
                          kernel_regularizer=reg,
                          bias_regularizer=reg,
                          activation='relu'))
  model.add(BatchNormalization())
  model.add(Conv2D(64 , (3,3), strides=(2,2), 
                          padding='valid', data_format='channels_last',
                          kernel_regularizer=reg,
                          bias_regularizer=reg,
                          activation='relu'))
  model.add(Flatten())
  model.add(Dense(64, 
                  kernel_regularizer=reg,
                  bias_regularizer=reg,
                  activation='relu'))
  model.add(Dense(1, 
                  kernel_regularizer=reg,
                  bias_regularizer=reg,
                  activation='sigmoid'))

  print(model.summary())
  # sys.exit(0)
  # Loss object
  loss_object = tf.keras.losses.BinaryCrossentropy()

  # Optimizer
  optimizer = tf.keras.optimizers.Adam()
  # Compile the model under the strategy scope
  model.compile(optimizer=optimizer, loss=loss_object, metrics=[tf.keras.metrics.BinaryAccuracy()])


  # Define something to keep the callbacks for plot gen
  batch_stats_callback = CollectBatchStats()

  callbacks = [
  tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                    save_weights_only=True),
  tf.keras.callbacks.LearningRateScheduler(decay),
  PrintLR(), 
  batch_stats_callback
  ]


  if trunc_epochs == True:

    history = model.fit(train_ds, 
                        epochs=EPOCHS, 
                        steps_per_epoch=epoch_steps, 
                        callbacks=callbacks,
                        validation_data=val_ds, 
                        validation_steps=100,
                        )
  else:
    history = model.fit(train_ds, 
                        epochs=EPOCHS, 
                        callbacks=callbacks,
                        validation_data=val_ds, 
                        )

  # Save some figures
  outdir = '../data/output/figures/'
  dt = datetime.datetime.now()
  tstamp = f"{dt.year}{dt.month}{dt.day}_{dt.hour}:{dt.minute}:{dt.second}"
  
  plt.figure()
  plt.ylabel("Acc")
  plt.xlabel("Training Steps")
  plt.ylim([0,2])
  plt.plot(batch_stats_callback.batch_acc)
  plt.savefig(outdir + tstamp+"_accuracy.png")

  plt.figure()
  plt.ylabel("Loss")
  plt.xlabel("Training Steps")
  plt.ylim([0,2])
  plt.plot(batch_stats_callback.batch_losses)
  plt.savefig(outdir + tstamp+"_losses.png")

  plt.figure()
  plt.plot(history.history['binary_accuracy'], label='accuracy')
  plt.plot(history.history['val_binary_accuracy'], label = 'val_accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.ylim([0.1, 1])
  plt.legend(loc='best')
  plt.savefig(outdir + tstamp+"_train_val_acc.png")
  

  plt.figure()
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label = 'val_loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.ylim([0, 2])
  plt.legend(loc='best')
  plt.savefig(outdir + tstamp+"_train_val_loss.png")


