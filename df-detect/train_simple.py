import tensorflow as tf
import numpy as np
import gc
import pandas as pd 
from tensorflow.keras.layers import ConvLSTM2D, Conv3D, Conv2D, Flatten, Dense, BatchNormalization, MaxPool2D
import tensorflow.math as M
from utils import *
from sklearn.model_selection import train_test_split
from video_loader import DeepFakeDualTransformer, DeepFakeTransformer
import sys
import pathlib
import datetime
import matplotlib.pyplot as plt

@tf.function
def loss(labels, logits):

    cross_entropies = tf.keras.losses.categorical_crossentropy(labels, logits)
    batch_loss = tf.nn.compute_average_loss(cross_entropies, global_batch_size=cross_entropies.shape[0])

    return batch_loss


@tf.function
def grad(model, x, labels, loss):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss(labels, logits)
    grads = tape.gradient(loss_value, model.trainable_variables)
    return loss_value, grads

@tf.function
def apply_grads(optimizer, grads, model):
  optimizer.apply_gradients(zip(grads, model.trainable_variables))


def validation_step(model, x, labels, loss):
  logits = model(x)
  loss_value = loss(labels, logits)
  return loss_value

if __name__ == "__main__":
    # Define some params
    # model params
    
    # Train params
    EPOCHS=10
    batch_size=1
    reg_penalty = 0.001
    validate_epochs = list(np.arange(EPOCHS))
    dropout_frac = 0.2
    

    # Dataset params
    data_pairs_path = '../data/source/labels/fake_to_real_mapping.csv'
    resize_shape = (224,224)
    sequence_len = 32
    prefetch_num = 4
    train_val_split = 0.015


    # Final model params
    dt = datetime.datetime.now()
    tstamp = f'{dt.year}{dt.month}{dt.day}{dt.hour}{dt.minute}{dt.second}'
    regstr = str(reg_penalty).split('.')[1]

    model_stamp = tstamp + f'_classifier_model_{regstr}_reg'
    final_model_path = f'models/{model_stamp}/model.h5'

    # Use `model_stamp` variable if new model is desired, otherwise enter manually
    checkpoint_prefix = f'models/{model_stamp}/'

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
    train_ds = train_ds.map(lambda x: train_transformer.transform_map(x)).batch(batch_size).prefetch(prefetch_num)
    val_ds = val_ds.map(lambda x: val_transformer.transform_map(x)).batch(batch_size).prefetch(prefetch_num)


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
                                padding='same', 
                                data_format='channels_last',
                                return_sequences=True,
                                dropout=dropout_frac,
                                input_shape=test_dims[1:]))
    model.add(ConvLSTM2D(128, (3,3), strides=(2,2), 
                                padding='same', 
                                data_format='channels_last',
                                dropout=dropout_frac,
                                return_sequences=True,
                                ))
    model.add(ConvLSTM2D(128, (3,3), strides=(2,2), 
                                padding='same', 
                                data_format='channels_last',
                                dropout=dropout_frac,
                                return_sequences=True,
                                ))
    model.add(ConvLSTM2D(256, (3,3), strides=(2,2), 
                                padding='same', 
                                data_format='channels_last',
                                dropout=dropout_frac, 
                                return_sequences=False,
                                ))
    model.add(MaxPool2D())
    model.add(BatchNormalization())
    model.add(Conv2D(128 , (3,3), strides=(1,1), 
                            padding='same', data_format='channels_last',
                            activation='relu'))
    model.add(MaxPool2D())                       
    model.add(BatchNormalization())
    model.add(Conv2D(64 , (3,3), strides=(1,1), 
                            padding='same', data_format='channels_last',
                            activation='relu'))
    model.add(MaxPool2D())
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    print(model.summary())
   
    #sys.exit(0)
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_prefix, max_to_keep=10)

    # Keep results for plotting
    train_loss_results = []
    validation_loss_results = []

    if manager.latest_checkpoint:
      status = ckpt.restore(manager.latest_checkpoint)
      print("Restored from {} with status {}".format(manager.latest_checkpoint, status))
    else:
      print("Initializing from scratch.")
       
    for epoch in range(EPOCHS):
        epoch_loss_avg = tf.keras.metrics.Mean()
        prgbar = tf.keras.utils.Progbar(num_train_batches, stateful_metrics='loss')
        

        i = 0
        # Training loop
        for train_pair, labels in train_ds:

            # Take care of batching shapes
            train_pair = tf.reshape(train_pair, shape=(batch_size*2, *train_pair.shape[2:]))
            labels = tf.reshape(labels, shape=(batch_size*2, *labels.shape[2:]))

            # Optimize the model
            loss_value, grads = grad(model, train_pair, labels, loss)
            apply_grads(optimizer, grads, model)
            
            if i%1000 == 0:
              print(" accum mean loss: {:.6f}".format(epoch_loss_avg.result()))

            if (i%5000 == 0) or (i == (X_train_len - 1)):
                if i != 0:
                    manager.save(checkpoint_number=(epoch*X_train_len) + i)
            
            prgbar.update(i+1, values=[('loss', loss_value.numpy())])
            # print(" loss_long: {:.8f}".format(loss_value.numpy()))
            # Track progress
            epoch_loss_avg(loss_value)  # Add current batch loss
            i += 1

        # End epoch ops
        train_loss_results.append(epoch_loss_avg.result())
        print("\nEpoch {:03d}: Loss: {:.10f}".format(epoch+1, epoch_loss_avg.result()))

        if epoch in validate_epochs:
          val_prgbar = tf.keras.utils.Progbar(num_val_batches)
          valid_epoch_loss_avg = tf.keras.metrics.Mean()

          print("Validating {} batches".format(num_val_batches))
          i=0
          for val_pair, labels in val_ds:

            # Take care of batching shapes
            val_pair = tf.reshape(val_pair, shape=(batch_size*2, *val_pair.shape[2:]))
            labels = tf.reshape(labels, shape=(batch_size*2, *labels.shape[2:]))

            loss_value = validation_step(model, val_pair, labels, loss)
            valid_epoch_loss_avg(loss_value)
            val_prgbar.update(i+1, values=[('loss', loss_value.numpy())])
            i+=1
          
          validation_loss_results.append(valid_epoch_loss_avg.result())
          print("Validation {:03d}: Loss: {:.3f}".format(epoch+1, valid_epoch_loss_avg.result()))

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

