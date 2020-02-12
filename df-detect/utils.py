import numpy as np
import cv2
import pandas as pd
# from tensorflow.image import resize, transpose, random_flip_up_down, random_flip_left_right, random_brightness, random_contrast, random_crop
import tensorflow as tf
import math

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
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    all_frames = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
    
    

    fc = 0
    while(cap.isOpened() and fc < frameCount): # as long as all the frames have been traversed
        ret, frame = cap.read()
        # capture the frame. Again, if life is a movie, this function acts as camera
        
        if ret==True:
            all_frames[fc] = frame
            fc += 1
            if cv2.waitKey(1) & 0xFF == ord('q'): # break in between by pressing the key given
                break
        else:
            break
                
    cap.release()
    # release whatever was held by the method for say, resources and the video itself
    return all_frames
    


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

    df = pd.read_json(meta_path, orient='index').reset_index()
    df['target_class'] = (df['label'] == 'FAKE').astype('float')
    df['filepath'] = train_path + df['index']
    df.drop(['index', 'split', 'original', 'label'], axis=1, inplace=True)
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
                         resize_shape=(300,300)):
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
        # Some videos are cut at 298 frames so going with that
        # as the standard
        vid = get_frames(p)[:298,:,:,:]
        vid = tf.image.resize(vid, size=resize_shape)
        vid = normalize(vid, chan_means, chan_std_dev)
        # Add more augmentations on load in below here
        #
        #
        batch.append(vid)
        print(p)
        print(vid.shape)

    print("Batch length is ", len(paths))

    return tf.stack(batch)
    

class DeepFakeDataSeq(tf.keras.utils.Sequence):


    # Pass in two lists, one of filename paths for X, labels
    # for y and then batch_size
    def __init__(self, x_set, y_set, batch_size):
            self.x, self.y = x_set, y_set
            self.batch_size = batch_size
    
    def __len__(self):
            return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
            batch_x = self.x[idx * self.batch_size:(idx + 1) *
            self.batch_size]
            batch_y = self.y[idx * self.batch_size:(idx + 1) *
            self.batch_size]


            batch_x_loaded = load_transform_batch([fn for fn in batch_x])
            return (batch_x_loaded, tf.Tensor(batch_y))



def main():
    meta_path = '../data/source/labels/train_meta.json'
    # TODO make the apply method for os.path.join
    train_path = '../data/source/train/'
    batch_sz = 10

    df = load_process_train_targets(meta_path, train_path)

    #batch = load_transform_batch(df.filepath[:batch_sz])

    #print(batch)

    data = DeepFakeDataSeq(df.filepath.to_list(), df.target_class.to_list(), 10)

    print(data.__getitem__(0))

    

if __name__ == "__main__":
    main()



    
