import tensorflow as tf
import numpy as np
import gc
import pandas as pd 
from tensorflow.keras.layers import ConvLSTM2D, Conv3D, Conv2D, Flatten, Dense, BatchNormalization
import tensorflow.math as M
from utils import *
from sklearn.model_selection import train_test_split
from video_loader import DeepFakeDualTransformer, DeepFakeTransformer
import sys
import pathlib
import datetime
import matplotlib.pyplot as plt


class SimilarityLoss(tf.keras.losses.Loss):
  # Loss function subclassed to penalize for returning a pair 
  # with a high cosine sim
  def call(self, y_real, y_fake):

    den = M.multiply(M.sqrt(M.reduce_sum(M.pow(y_real,2))), M.sqrt(M.reduce_sum(M.pow(y_fake,2))))
    cosine_sim = M.divide(M.reduce_sum(M.multiply(y_real, y_fake)), den)
    loss = M.scalar_mul(-1, M.log(M.subtract(1, cosine_sim)))

    return loss

@tf.function
def similarity_loss(y_real, y_fake):
    
    cs = tf.keras.losses.cosine_similarity(y_real, y_fake)
    loss = M.log(1.00001 + cs)

    return -1*loss

@tf.function
def grad(model, x, loss):
    with tf.GradientTape() as tape:
        y_pair = model(x)
        loss_value = loss(y_pair[0], y_pair[1])
    grads = tape.gradient(loss_value, model.trainable_variables)
    return loss_value, grads

@tf.function
def apply_grads(optimizer, grads, model):
  optimizer.apply_gradients(zip(grads, model.trainable_variables))


def validation_step(val_pair, model, loss):
  y_pair = model(val_pair)
  loss_value = loss(y_pair[0], y_pair[1])
  return loss_value

if __name__ == "__main__":
    # Define some params
    # model params
    load_ckpt = False # set false if not loading


    # Train params
    EPOCHS=10
    batch_size=1
    reg_penalty = 0.001
    cls_wt = {0:3, 1:2.25}
    validate_epochs = list(np.arange(EPOCHS))
    

    # Dataset params
    data_pairs_path = '../data/source/labels/fake_to_real_mapping.csv'
    resize_shape = (224,224)
    sequence_len = 30
    prefetch_num = 4
    train_val_split = 0.015


    # Final model params
    dt = datetime.datetime.now()
    tstamp = f'{dt.year}{dt.month}{dt.day}{dt.hour}{dt.minute}{dt.second}'
    regstr = str(reg_penalty).split('.')[1]

    model_stamp = tstamp + f'_model_{regstr}_reg'
    final_model_path = f'models/{model_stamp}/model.h5'
    checkpoint_prefix = f'models/{model_stamp}/'

    # Losses, metrics, optimizer

    loss = SimilarityLoss()

    # Define dummy test dims based on parameters
    test_dims = (None, None, *resize_shape, 3)

    # Get device names

    cpu = tf.config.experimental.list_physical_devices('CPU')[0].name
    print("FOUND CPU AT: ", cpu)
    print([x[0] for x in tf.config.experimental.list_physical_devices('GPU')])


    # Define and load the datasets
    df_pairs = pd.read_csv(data_pairs_path)[['real', 'fake']]
    train_df, val_df = train_test_split(df_pairs, test_size = train_val_split)

    
    # Create dataset from pairs of path strings
    train_ds = tf.data.Dataset.from_tensor_slices(train_df.to_numpy())
    val_ds = tf.data.Dataset.from_tensor_slices(val_df.to_numpy())

    # TODO add in random crops, rotations, etc to make this non-redundant
    # define the transformer to load data on the fly
    train_transformer = DeepFakeDualTransformer(resize_shape=resize_shape, seq_length=sequence_len)
    val_transformer = DeepFakeDualTransformer(resize_shape=resize_shape, seq_length=sequence_len)

    # map the transformer on the dataset entries
    train_ds = train_ds.map(lambda x: train_transformer.transform_map(x)).prefetch(prefetch_num)
    val_ds = val_ds.map(lambda x: val_transformer.transform_map(x)).prefetch(prefetch_num)


    X_train_len = len(train_df)
    X_val_len = len(val_df)
    num_train_batches = int(np.ceil(X_train_len/batch_size))
    num_val_batches = int(np.ceil(X_val_len/batch_size))

    lr_fn = tf.keras.optimizers.schedules.PolynomialDecay(0.01, num_train_batches*EPOCHS, 
                                                          end_learning_rate=1e-5)
    optimizer = tf.keras.optimizers.SGD(lr_fn)

    # Define the model, currently copying model archs to model.py manually
    model = tf.keras.models.Sequential()
    model.add(ConvLSTM2D(64, (3,3), strides=(2,2), 
                                padding='valid', 
                                data_format='channels_last',
                                return_sequences=True,
                                input_shape=test_dims[1:]))
    model.add(ConvLSTM2D(128, (3,3), strides=(2,2), 
                                padding='valid', 
                                data_format='channels_last',
                                return_sequences=True,
                                ))
    model.add(ConvLSTM2D(128, (3,3), strides=(2,2), 
                                padding='valid', 
                                data_format='channels_last',
                                return_sequences=True,
                                ))
    model.add(ConvLSTM2D(256, (3,3), strides=(2,2), 
                                padding='valid', 
                                data_format='channels_last', 
                                return_sequences=False,
                                ))
    model.add(Conv2D(128 , (3,3), strides=(2,2), 
                            padding='valid', data_format='channels_last',
                            activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64 , (3,3), strides=(2,2), 
                            padding='valid', data_format='channels_last',
                            activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(128, activation='softmax'))
    print(model.summary())

    # sys.exit(0)

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_prefix, max_to_keep=5)

    # Keep results for plotting
    train_loss_results = []
    validation_loss_results = []

    if manager.latest_checkpoint:
      print("Restored from {}".format(manager.latest_checkpoint))
    else:
      print("Initializing from scratch.")

    
    for epoch in range(EPOCHS):
        epoch_loss_avg = tf.keras.metrics.Mean()
        # prgbar = tf.keras.utils.Progbar(X_train_len)
        

        i = 0
        # Training loop
        for train_pair in train_ds:

            # Optimize the model
            loss_value, grads = grad(model, train_pair, similarity_loss)
            apply_grads(optimizer, grads, model)
            
            if i%1000 == 0:
              print("Training batch {} accum mean loss: {:.10f}".format(i, epoch_loss_avg.result()))
            # prgbar.update(i+1, values=[('loss', loss_value.numpy())])
            # Track progress
            epoch_loss_avg(loss_value)  # Add current batch loss
            i += 1

        # End epoch ops
        train_loss_results.append(epoch_loss_avg.result())
        manager.save(checkpoint_number=epoch)

        
        print("Epoch {:03d}: Loss: {:.10f}".format(epoch, epoch_loss_avg.result()))

        if epoch in validate_epochs:
          valid_epoch_loss_avg = tf.keras.metrics.Mean()

          print("Validating {} batches".format(num_val_batches))

          for val_pair in val_ds.as_numpy_iterator():

            loss_value = validation_step(val_pair, model, similarity_loss)
            valid_epoch_loss_avg(loss_value)
            
          
          validation_loss_results.append(valid_epoch_loss_avg.result())
          print("Validation {:03d}: Loss: {:.3f}".format(epoch, valid_epoch_loss_avg.result()))

    model.save(final_model_path)


    # Save some figures
    outdir = '../data/output/figures/'


    plt.figure()
    plt.plot(np.arange(EPOCHS), train_loss_results, label='loss')
    plt.plot(np.arange(EPOCHS), validation_loss_results, label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.savefig(outdir + model_stamp + "_train_val_loss.png")

