import tensorflow as tf
import numpy as np
import gc
import pandas as pd 
from utils import *
from sklearn.model_selection import train_test_split
import sys
from model import DeepFakeDetector
from tensorflow.keras.layers import ConvLSTM2D, Conv3D, Conv2D, Flatten, Dense


if __name__ == "__main__":

    # Define some params
    EPOCHS=1
    validate_epochs = [0,1,2,10]
    batch_size=3
    test_fraction = 0.2
    train_root = '../data/source/train/'
    meta_path = '../data/source/labels/train_meta.json'
    checkpoint_prefix = 'models/ckpt_{epoch}'
    resize_shape = (224,224)
    sequence_len = 32
    n_workers = 1
    use_mult_prc = False
    debug = True

    test_dims = (batch_size, sequence_len, *resize_shape, 3)

    df = load_process_train_targets(meta_path, train_root)
    train_df, val_df = train_test_split(df, test_size=test_fraction)

    train_size = len(train_df)
    val_size = len(val_df)
    train_batches = int(np.ceil(train_size/batch_size))
    val_batches = int(np.ceil(val_size/batch_size))
    
    if debug:
        train_batches=3
        val_batches=3
        # i = 1
        # batch_df = train_df[i*batch_size:(i+1)*batch_size]
        # batch_fnames = batch_df.filepath.to_list()
        # print(batch_fnames)
        # y = np.array(batch_df.target_class.to_list())
        # x = load_transform_batch(batch_fnames, 
        #                         resize_shape=resize_shape, 
        #                         seq_length=sequence_len)
        # print(x)



    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)

    def loss(model, x, y):
        y_ = model(x)
        return loss_object(y_true=y, y_pred=y_)
    
    def grad(model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = loss(model, inputs, targets)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    
    model = tf.keras.models.Sequential()
    model.add(ConvLSTM2D(32, (3,3), strides=(2,2), 
                                    padding='valid', 
                                    data_format='channels_last', 
                                    return_sequences=True,
                                    input_shape=test_dims[1:]))
    model.add(ConvLSTM2D(32, (3,3), strides=(2,2), 
                                    padding='valid', 
                                    data_format='channels_last', 
                                    return_sequences=True,
                                    ))
    model.add(ConvLSTM2D(32, (3,3), strides=(2,2), 
                                    padding='valid', 
                                    data_format='channels_last', 
                                    return_sequences=True,
                                    ))
    model.add(ConvLSTM2D(32, (3,3), strides=(2,2), 
                                    padding='valid', 
                                    data_format='channels_last', 
                                    return_sequences=False,
                                    ))
    model.add(Conv2D(16 , (3,3), strides=(2,2), 
                                  padding='valid', data_format='channels_last', 
                                  activation='relu'))
    model.add(Conv2D(16 , (3,3), strides=(2,2), 
                                  padding='valid', data_format='channels_last', 
                                  activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())

    # sys.exit(0)


    # Keep results for plotting
    train_loss_results = []
    train_accuracy_results = []
    validation_loss_results = []
    validation_accuracy_results = []

    for epoch in range(EPOCHS):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.BinaryAccuracy()

        # Training loop
        for i in range(train_batches):

            with tf.device('CPU:0'):
                
                # Get the batch
                batch_df = train_df[i*batch_size:(i+1)*batch_size]
                batch_fnames = batch_df.filepath.to_list()
                y = np.array(batch_df.target_class.to_list()).reshape((batch_size,1))
                x = load_transform_batch(batch_fnames, 
                                        resize_shape=resize_shape, 
                                        seq_length=sequence_len)

            with tf.device('GPU:0'):
                # Optimize the model
                loss_value, grads = grad(model, x, y)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
            print("Training batch {} loss: {:.3f}".format(i, loss_value))
            # Track progress
            epoch_loss_avg(loss_value)  # Add current batch loss
            # Compare predicted label to actual label
            # training=True is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            epoch_accuracy(y, model(x))

        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        if epoch % 1 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                        epoch_loss_avg.result(),
                                                                        epoch_accuracy.result()))
        if epoch in validate_epochs:
            valid_epoch_loss_avg = tf.keras.metrics.Mean()
            valid_epoch_accuracy = tf.keras.metrics.BinaryAccuracy()

            print("Validating {} batches".format(val_batches))

            for i in range(val_batches):
                
                # Get the batch
                batch_df = val_df[i*batch_size:(i+1)*batch_size]
                batch_fnames = batch_df.filepath.to_list()
                y = np.array(batch_df.target_class.to_list()).reshape((batch_size,1))
                x = load_transform_batch(batch_fnames, 
                                        resize_shape=resize_shape, 
                                        seq_length=sequence_len)
                loss_value, grads = grad(model, x, y)

                valid_epoch_loss_avg(loss_value)
                valid_epoch_accuracy(y, model(x))
            
            validation_loss_results.append(valid_epoch_loss_avg)
            validation_accuracy_results.append(valid_epoch_accuracy)

            print("Validation {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                        valid_epoch_loss_avg.result(),
                                                                        valid_epoch_accuracy.result()))

