import numpy as np
import cv2
import pandas as pd
# from tensorflow.image import resize, transpose, random_flip_up_down, random_flip_left_right, random_brightness, random_contrast, random_crop
import tensorflow as tf
import math
import os

def get_frames(filepath):
    '''
    method for getting the frames from a video file
    args: 
        filepath: exact path of the video file
        first_only: whether to detect the first frame only or all of the frames
    out:
        frames:  
    '''

    cap = cv2.VideoCapture(filepath) 
    # captures the video. Think of it as if life is a movie so we ask the method to focus on patricular event
    # that is our video in this case. It will concentrate on the video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idxs = np.linspace(0, frame_count, num=frame_count, dtype=np.int)
    
    
    all_frames, _ = _read_frames_at_indices(filepath, cap, frame_idxs)
                
    cap.release()
    # release whatever was held by the method for say, resources and the video itself
    return all_frames

def _read_frames_at_indices(path, capture, frame_idxs):
    try:
        frames = []
        idxs_read = []
        for frame_idx in range(frame_idxs[0], frame_idxs[-1] + 1):
            # Get the next frame, but don't decode if we're not using it.
            ret = capture.grab()

            # Need to look at this frame?
            current = len(idxs_read)
            if frame_idx == frame_idxs[current]:
                ret, frame = capture.retrieve()

                frames.append(frame)
                idxs_read.append(frame_idx)

        if len(frames) > 0:
            return np.stack(frames), idxs_read

    except:
        print("Exception while reading movie %s" % path)
        return None    


def load_process_train_targets(meta_path, train_path, return_df = True):
    """[summary]
    
    Arguments:
        meta_path {[type]} -- [description]
        train_path {[type]} -- [description]
    
    Keyword Arguments:
        return_df {bool} -- [description] (default: {True})
    
    Returns:
        [type] -- [description]
    """

    df = pd.read_json(meta_path)
    df['target_class'] = (df['label'] == 'FAKE').astype('float')
    df['filepath'] = df['index'].apply(lambda x: os.path.join(train_path, x))
    df.drop(['index', 'original', 'label'], axis=1, inplace=True)
    df.to_csv(meta_path + 'paths_labels.csv', header=True)

    if return_df == True:
        return df

def normalize(video, chan_means, chan_std_dev):
    """[summary]
    
    Arguments:
        video {tf.Tensor} -- tensorflow reshaped video data
        chan_means {array} -- [description]
        chan_std_dev {array} -- [description]
    
    Returns:
        [tf.Tensor] -- normalized video data
    """

    video /= 255
    video -= chan_means
    video /= chan_std_dev

    return video

def load_transform_batch(paths, 
                         chan_means=[0.485, 0.456, 0.406], 
                         chan_std_dev = [0.229, 0.224, 0.225], 
                         resize_shape=(300,300),
                         seq_length=298):
    """[summary]
    
    Arguments:
        paths {[type]} -- [description]
    
    Keyword Arguments:
        chan_means {list} -- [description] (default: {[0.485, 0.456, 0.406]})
        chan_std_dev {list} -- [description] (default: {[0.229, 0.224, 0.225]})
        resize_shape {tuple} -- [description] (default: {(300,300)})
    
    Returns:
        [type] -- [description]
    """

    batch = []
    for p in paths:
        # On Acer predator, max size for prediction is 100 frames, 
        # Don't want to exceed frames, available

        vid = get_frames(p)
        n_frames = vid.shape[0]
        
        start = np.random.randint(n_frames - seq_length)
        vid = vid[start:(start+seq_length),:,:,:]
        vid = tf.image.resize(vid, size=resize_shape).numpy()
        vid = normalize(vid, chan_means, chan_std_dev)
        # Add more augmentations on load in below here
        #
        #
        batch.append(vid)

    return tf.stack(batch)
    

class DeepFakeDataSeq(tf.keras.utils.Sequence):


    # Pass in two lists, one of filename paths for X, labels
    # for y and then batch_size
    def __init__(self, x_set, y_set, batch_size, resize_shape=(300,300), sequence_len=298):
            self.x, self.y = x_set, y_set
            self.batch_size = batch_size
            self.resize_shape = resize_shape
            self.sequence_len = sequence_len
    
    def __len__(self):
            return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        print("Loading batch ", idx)

        batch_x_loaded = load_transform_batch([fn for fn in batch_x], 
                                                resize_shape=self.resize_shape,
                                                seq_length=self.sequence_len)
        return (batch_x_loaded, tf.constant([batch_y]), [None])





    
