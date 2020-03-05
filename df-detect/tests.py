import tensorflow as tf
from utils import *
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def generator_test():

    batch_size=1
    train_label_path = '../data/source/labels/train_meta.json'
    train_path = '../data/source/train/'
    resize_shape = (224,224)
    sequence_len = 16
    n_workers = 1
    use_mult_prc = False


    df = load_process_train_targets(train_label_path, train_path)
    seq = DeepFakeDataSeq(df.filepath.to_list(), 
                                 df.target_class.to_list(), 
                                 batch_size, 
                                 resize_shape=resize_shape, 
                                 sequence_len=sequence_len)

    with tf.device(get_available_gpus()[0]):
        vids = []
        for i in range(len(df)):
            print(i)
            vid = seq.__getitem__(i)
            vids.append(vid)


if __name__ == "__main__":
