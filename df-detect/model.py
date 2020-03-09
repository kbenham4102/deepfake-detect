import tensorflow as tf
from tensorflow.keras.layers import ConvLSTM2D, Conv3D, Conv2D, Flatten, Dense, BatchNormalization

class DeepFakeDetector7IK(tf.keras.Model):
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




def model_0302():
    test_dims = (None, None, 224, 224, 3)
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
    return model

def model_0303():
    test_dims = (None, None, 224, 224, 3)
    reg = tf.keras.regularizers.l2(l=0.1)
    model = tf.keras.models.Sequential()
    model.add(ConvLSTM2D(32, (3,3), strides=(2,2), 
                                padding='valid', 
                                data_format='channels_last',
                                recurrent_regularizer=reg,
                                kernel_regularizer=reg,
                                bias_regularizer=reg, 
                                return_sequences=True,
                                input_shape=test_dims[1:]))
    model.add(ConvLSTM2D(32, (3,3), strides=(2,2), 
                                padding='valid', 
                                data_format='channels_last',
                                recurrent_regularizer=reg,
                                kernel_regularizer=reg,
                                bias_regularizer=reg,  
                                return_sequences=True,
                                ))
    model.add(ConvLSTM2D(32, (3,3), strides=(2,2), 
                                padding='valid', 
                                data_format='channels_last',
                                recurrent_regularizer=reg,
                                kernel_regularizer=reg,
                                bias_regularizer=reg, 
                                return_sequences=True,
                                ))
    model.add(ConvLSTM2D(32, (3,3), strides=(2,2), 
                                padding='valid', 
                                data_format='channels_last',
                                recurrent_regularizer=reg,
                                kernel_regularizer=reg,
                                bias_regularizer=reg,  
                                return_sequences=False,
                                ))
    model.add(Conv2D(16 , (3,3), strides=(2,2), 
                            padding='valid', data_format='channels_last',
                            kernel_regularizer=reg,
                            bias_regularizer=reg,
                            activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(16 , (3,3), strides=(2,2), 
                            padding='valid', data_format='channels_last',
                            kernel_regularizer=reg,
                            bias_regularizer=reg,
                            activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, kernel_regularizer=reg,
                    bias_regularizer=reg,
                    activation='sigmoid'))

    return model

def model_0304():
    test_dims = (None, None, 224, 224, 3)
    reg = tf.keras.regularizers.l2(l=0.01)
    model = tf.keras.models.Sequential()
    model.add(ConvLSTM2D(32, (3,3), strides=(2,2), 
                                padding='valid', 
                                data_format='channels_last',
                                recurrent_regularizer=reg,
                                kernel_regularizer=reg,
                                bias_regularizer=reg, 
                                return_sequences=True,
                                input_shape=test_dims[1:]))
    model.add(ConvLSTM2D(32, (3,3), strides=(2,2), 
                                padding='valid', 
                                data_format='channels_last',
                                recurrent_regularizer=reg,
                                kernel_regularizer=reg,
                                bias_regularizer=reg,  
                                return_sequences=True,
                                ))
    model.add(ConvLSTM2D(32, (3,3), strides=(2,2), 
                                padding='valid', 
                                data_format='channels_last',
                                recurrent_regularizer=reg,
                                kernel_regularizer=reg,
                                bias_regularizer=reg, 
                                return_sequences=True,
                                ))
    model.add(ConvLSTM2D(32, (3,3), strides=(2,2), 
                                padding='valid', 
                                data_format='channels_last',
                                recurrent_regularizer=reg,
                                kernel_regularizer=reg,
                                bias_regularizer=reg,  
                                return_sequences=False,
                                ))
    model.add(Conv2D(16 , (3,3), strides=(2,2), 
                            padding='valid', data_format='channels_last',
                            kernel_regularizer=reg,
                            bias_regularizer=reg,
                            activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(16 , (3,3), strides=(2,2), 
                            padding='valid', data_format='channels_last',
                            kernel_regularizer=reg,
                            bias_regularizer=reg,
                            activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, kernel_regularizer=reg,
                    bias_regularizer=reg,
                    activation='sigmoid'))

    return model

def model_0304_v2():
    test_dims = (None, None, 224, 224, 3)
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

    return model