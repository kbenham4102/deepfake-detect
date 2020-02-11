import numpy as np
import cv2
import pandas as pd
# from tensorflow.image import resize, transpose, random_flip_up_down, random_flip_left_right, random_brightness, random_contrast, random_crop
import tensorflow as tf
import sys

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
        vid = tf.image.resize_with_pad(vid, resize_shape[0], resize_shape[1])
        vid = normalize(vid, chan_means, chan_std_dev)
        # Add more augmentations on load in below here
        #
        #
        batch.append(vid)
        print(p)
        print(vid.shape)

    print("Batch length is ", len(paths))

    return tf.stack(batch)
    





def main():
    meta_path = '../data/source/labels/train_meta.json'
    # TODO make the apply method for os.path.join
    train_path = '../data/source/train/'
    batch_sz = 10

    df = load_process_train_targets(meta_path, train_path)

    batch = load_transform_batch(df.filepath[:batch_sz])

    

if __name__ == "__main__":
    main()



    
