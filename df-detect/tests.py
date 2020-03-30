import tensorflow as tf
from utils import *
from tensorflow.python.client import device_lib

# WIP

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']




if __name__ == "__main__":
