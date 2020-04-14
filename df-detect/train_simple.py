import tensorflow as tf
import numpy as np
import gc
import pandas as pd 
from tensorflow.keras.layers import ConvLSTM2D, Conv3D, Conv2D, Flatten, Dense, BatchNormalization, MaxPool2D
import tensorflow.math as M
from sklearn.model_selection import train_test_split
from video_loader import DeepFakeLoadExtractFeatures
from model import FeatureDifferentiatorModel
import sys
import datetime
import matplotlib.pyplot as plt
import os

@tf.function
def loss(labels, logits):

    cross_entropies = tf.keras.losses.binary_crossentropy(labels, logits)
    batch_loss = tf.nn.compute_average_loss(cross_entropies, global_batch_size=cross_entropies.shape[0])

    return batch_loss


@tf.function
def grad(model, x16, x8, labels, loss):
    with tf.GradientTape() as tape:
        logits = model([x16, x8], training=True)
        loss_value = loss(labels, logits)
    grads = tape.gradient(loss_value, model.trainable_variables)
    return loss_value, grads

@tf.function
def apply_grads(optimizer, grads, model):
  optimizer.apply_gradients(zip(grads, model.trainable_variables))


def validation_step(model, x16, x8, labels, loss):
  logits = model([x16, x8])
  loss_value = loss(labels, logits)
  return loss_value

def force_reload_weights(checkpoint_prefix, model):
  """Solves the issue where the checkpoint managet refuses to refresh
  Only good for the specific instance of the Feature Differentiator model
  
  Arguments:
      checkpoint_prefix {str} -- prefix directory to load latest checkpoint 
      model -- subclassed model instance to load weights to
  
  Returns:
      model
  """
  ckpt_reader = tf.train.load_checkpoint(checkpoint_prefix)

  layer_list = ['x1_16', 'x1_8', 'x2_16', 'x2_8', 'x_3', 'x_4','x_5', 'x_7', 'x_bn1', 'x_bn2', 'x_bn3']

  weights_dict = {}

  for layer_pre in layer_list:
      if layer_pre in ['x1_16', 'x1_8', 'x2_16', 'x2_8']:
          recurr_name = f'net/{layer_pre}/cell/recurrent_kernel/.ATTRIBUTES/VARIABLE_VALUE'
          kern_name = f'net/{layer_pre}/cell/kernel/.ATTRIBUTES/VARIABLE_VALUE'
          bias_name = f'net/{layer_pre}/cell/bias/.ATTRIBUTES/VARIABLE_VALUE'
          r = ckpt_reader.get_tensor(recurr_name)
          k = ckpt_reader.get_tensor(kern_name)
          b = ckpt_reader.get_tensor(bias_name)
          weights_dict[layer_pre] = np.array([k, r, b])
      elif layer_pre in ['x_3', 'x_4']:
          kern_name = f'net/{layer_pre}/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE'
          bias_name = f'net/{layer_pre}/bias/.ATTRIBUTES/VARIABLE_VALUE'
          k = ckpt_reader.get_tensor(kern_name)
          b = ckpt_reader.get_tensor(bias_name)
          weights_dict[layer_pre] = np.array([k, b])
      elif layer_pre in ['x_5', 'x_7']:
          kern_name = f'net/{layer_pre}/kernel/.ATTRIBUTES/VARIABLE_VALUE'
          bias_name = f'net/{layer_pre}/bias/.ATTRIBUTES/VARIABLE_VALUE'
          k = ckpt_reader.get_tensor(kern_name)
          b = ckpt_reader.get_tensor(bias_name)
          weights_dict[layer_pre] = np.array([k, b])
      elif layer_pre in ['x_bn1', 'x_bn2', 'x_bn3']:
          beta = f'net/{layer_pre}/beta/.ATTRIBUTES/VARIABLE_VALUE'
          gamma = f'net/{layer_pre}/gamma/.ATTRIBUTES/VARIABLE_VALUE'
          moving_mean = f'net/{layer_pre}/moving_mean/.ATTRIBUTES/VARIABLE_VALUE'
          moving_var = f'net/{layer_pre}/moving_variance/.ATTRIBUTES/VARIABLE_VALUE'
          b = ckpt_reader.get_tensor(beta)
          g = ckpt_reader.get_tensor(gamma)
          mm = ckpt_reader.get_tensor(moving_mean)
          mv = ckpt_reader.get_tensor(moving_var)
          weights_dict[layer_pre] = np.array([b, g, mm, mv])
    
  model.x1_16.set_weights(weights_dict['x1_16'])
  model.x2_16.set_weights(weights_dict['x2_16'])
  model.x1_8.set_weights(weights_dict['x1_8'])
  model.x2_8.set_weights(weights_dict['x2_8'])
  model.x_3.set_weights(weights_dict['x_3'])
  model.x_4.set_weights(weights_dict['x_4'])
  model.x_5.set_weights(weights_dict['x_5'])
  model.x_7.set_weights(weights_dict['x_7'])
  model.x_bn1.set_weights(weights_dict['x_bn1'])
  model.x_bn2.set_weights(weights_dict['x_bn2'])
  model.x_bn3.set_weights(weights_dict['x_bn3'])

  return model


class FeatureDifferentiatorModel(tf.keras.Model):
    def __init__(self):
        super(FeatureDifferentiatorModel, self).__init__()
        self.x1_8 = ConvLSTM2D(64, (3,3), strides=(1,1), 
                            padding='same', 
                            data_format='channels_last',
                            return_sequences=True,
                            dropout=0.2)
        self.x2_8 = ConvLSTM2D(128, (3,3), strides=(1,1), 
                                padding='same', 
                                data_format='channels_last',
                                return_sequences=False,
                                dropout=0.2)

        self.x1_16 = ConvLSTM2D(64, (3,3), strides=(1,1), 
                                    padding='same', 
                                    data_format='channels_last',
                                    return_sequences=True,
                                    dropout=0.2)
        self.x2_16 = ConvLSTM2D(128, (3,3), strides=(2,2), 
                            padding='same', 
                            data_format='channels_last',
                            return_sequences=False,
                            dropout=0.2)
        self.x_bn1 = tf.keras.layers.BatchNormalization()

        self.x_3 = tf.keras.layers.DepthwiseConv2D((3,3), 
                                              strides=(2,2), 
                                              depth_multiplier=2, 
                                              padding='same')
        self.x_bn2 = tf.keras.layers.BatchNormalization()
        self.x_4 = tf.keras.layers.DepthwiseConv2D((3,3), 
                                              strides=(2,2), 
                                              depth_multiplier=2, 
                                              padding='same')
        self.x_5 = tf.keras.layers.Conv2D(512, (1,1), strides=(2,2))
        self.x_bn3 = tf.keras.layers.BatchNormalization()
        self.x_7 = Dense(1, activation='sigmoid')

    def call(self, inputs):

        inputs16 = inputs[0]
        inputs8 = inputs[1]
        x8 = self.x1_8(inputs8)
        x8 = self.x2_8(x8)
        x16 = self.x1_16(inputs16)
        x16 = self.x2_16(x16)
        x_comb = tf.keras.layers.Add()([x16, x8])
        x = self.x_bn1(x_comb)
        x = self.x_3(x)
        x = self.x_bn2(x)
        x = self.x_4(x)
        x = self.x_5(x)
        x = self.x_bn3(x)
        x = tf.keras.layers.Flatten()(x)
        out = self.x_7(x)
        return out


      

def train(model, optimizer, train_ds, val_ds, X_train_len, EPOCHS, batch_size, num_train_batches, manager, num_val_batches, validate_epochs):

    train_loss_results = []
    validation_loss_results = []
    for epoch in range(EPOCHS):
      epoch_loss_avg = tf.keras.metrics.Mean()
      prgbar = tf.keras.utils.Progbar(num_train_batches, stateful_metrics='loss')
      

      i = 0
      # Training loop
      for train_pair16, train_pair8, labels in train_ds:

          # Take care of batching shapes
          n_pairs = train_pair16.shape[0]
          train_pair16 = tf.reshape(train_pair16, shape=(n_pairs*2, *train_pair16.shape[2:]))
          train_pair8 = tf.reshape(train_pair8, shape=(n_pairs*2, *train_pair8.shape[2:]))
          labels = tf.reshape(labels, shape=(n_pairs*2, *labels.shape[2:]))

          # Optimize the model
          loss_value, grads = grad(model, train_pair16, train_pair8, labels, loss)
          apply_grads(optimizer, grads, model)
          
          if i%1000 == 0:
            print(" accum mean loss: {:.6f}".format(epoch_loss_avg.result()))

          if (i%1000 == 0) or (i == (X_train_len - 1)):
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
        for val_pair16, val_pair8, labels in val_ds:

          # Take care of batching shapes
          n_pairs = val_pair16.shape[0]
          val_pair16 = tf.reshape(val_pair16, shape=(n_pairs*2, *val_pair16.shape[2:]))
          val_pair8 = tf.reshape(val_pair8, shape=(n_pairs*2, *val_pair8.shape[2:]))
          labels = tf.reshape(labels, shape=(n_pairs*2, *labels.shape[2:]))

          loss_value = validation_step(model, val_pair16, val_pair8, labels, loss)
          valid_epoch_loss_avg(loss_value)
          val_prgbar.update(i+1, values=[('loss', loss_value.numpy())])
          i+=1
          
        
        validation_loss_results.append(valid_epoch_loss_avg.result())
        print("Validation {:03d}: Loss: {:.3f}".format(epoch+1, valid_epoch_loss_avg.result()))
    
    model.save(final_model_path)

    return train_loss_results, validation_loss_results


if __name__ == "__main__":
    # Define some params
    # model params
    
    # Train params
    EPOCHS=10
    batch_size=8 # This actually is 16, 
                 # corresponding real videos are loaded 
                 # with every fake to oversample
    reg_penalty = 0.001
    validate_epochs = list(np.arange(EPOCHS))
    dropout_frac = 0.2
    

    # Dataset params
    data_pairs_path = '../data/source/fake_to_real_maps_82k.csv'
    resize_shape = (128,128)
    sequence_len = 64
    prefetch_num = 32
    train_val_split = 0.015
    n_calls= None


    # Final model params
    dt = datetime.datetime.now()
    tstamp = f'{dt.year}{dt.month}{dt.day}{dt.hour}{dt.minute}{dt.second}'
    regstr = str(reg_penalty).split('.')[1]

    model_stamp = tstamp + f'_bzface_model_{dropout_frac}_drop'
    final_model_path = f'models/{model_stamp}/model.h5'

    # Use `model_stamp` variable if new model is desired, otherwise enter manually
    # checkpoint_prefix = f'models/{model_stamp}/'
    checkpoint_prefix = 'models/202032811529_bzface_model_0.2_drop/'
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
    train_transformer =DeepFakeLoadExtractFeatures(
                      feat_extractor_path='models/blazeface/face_detection_front.tflite',
                      seq_length=sequence_len,
                      resize_shape=resize_shape)
    val_transformer = DeepFakeLoadExtractFeatures(
                      feat_extractor_path='models/blazeface/face_detection_front.tflite',
                      seq_length=sequence_len,
                      resize_shape=resize_shape)

    # map the transformer on the dataset entries
    train_ds = train_ds.map(lambda x: train_transformer.transform_map(x), 
                            num_parallel_calls=n_calls).batch(batch_size).prefetch(prefetch_num)
    val_ds = val_ds.map(lambda x: val_transformer.transform_map(x), 
                        num_parallel_calls=n_calls).batch(batch_size).prefetch(prefetch_num)


    X_train_len = len(train_df)
    X_val_len = len(val_df)
    num_train_batches = int(np.ceil(X_train_len/batch_size))
    num_val_batches = int(np.ceil(X_val_len/batch_size))

    lr_fn = tf.keras.optimizers.schedules.PolynomialDecay(0.01, num_train_batches*EPOCHS, 
                                                          end_learning_rate=1e-5)
    optimizer = tf.keras.optimizers.Adam(lr_fn)

    # Define the model, currently copying model archs to model.py manually

    test_input8 = tf.keras.Input((None,8, 8, 96))
    test_input16 = tf.keras.Input(( None, 16, 16, 88))
    inputs = (test_input16, test_input8)

    model = FeatureDifferentiatorModel()

    _ = model(inputs)

    print(model.summary())
   
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_prefix, max_to_keep=10)

    if os.path.exists(checkpoint_prefix):
      model = force_reload_weights(checkpoint_prefix, model)

    train_loss_results, validation_loss_results = train(model, optimizer, train_ds, val_ds, X_train_len, EPOCHS, batch_size, num_train_batches, manager, num_val_batches, validate_epochs)
    # Save some figures
    outdir = '../data/output/figures/'
    plt.figure()
    plt.plot(np.arange(EPOCHS), train_loss_results, label='loss')
    plt.plot(np.arange(EPOCHS), validation_loss_results, label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.savefig(outdir + model_stamp + "_train_val_loss.png")

