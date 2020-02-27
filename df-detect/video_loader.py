import numpy as np
import tensorflow as tf
import cv2

def get_frames(filename):
    '''
    method for getting the frames from a video file
    args: 
        filename: exact path of the video file
        first_only: whether to detect the first frame only or all of the frames
    out:
        video_frames, label:  
    '''

    filepath = filename.numpy().decode('utf-8')


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

class DeepFakeTransformer(object):
    def __init__(self, chan_means=[0.485, 0.456, 0.406],
                       chan_std_dev=[0.229, 0.224, 0.225],
                       resize_shape=(300,300),
                       seq_length=298):
        """[summary]
        
        Keyword Arguments:
            chan_means {list} -- [description] (default: {[0.485, 0.456, 0.406]})
            chan_std_dev {list} -- [description] (default: {[0.229, 0.224, 0.225]})
            resize_shape {tuple} -- [description] (default: {(300,300)})
            seq_length {int} -- [description] (default: {298})
        """
        self.chan_means = chan_means
        self.chan_std_dev = chan_std_dev
        self.resize_shape = resize_shape
        self.seq_length = seq_length
        

    def transform_vid(self, filename):
        """[summary]
        
        Arguments:
            filename {[type]} -- [description]
        
        Returns:
            vid -- [description]
            label -- tensfor for the label of the video
        """
        
        chan_means = self.chan_means
        chan_std_dev = self.chan_std_dev
        resize_shape = self.resize_shape
        seq_length = self.seq_length
 
        parts = tf.strings.split(filename, '/')
        label = parts[-2]
        # Don't want to exceed frames, available, using 198 as limit
        if seq_length == 298:
            start = 0
        else:
            start = np.random.randint(298 - seq_length)

        vid = get_frames(filename)[start:(start+seq_length),:,:,:]
        vid = tf.image.resize(vid, size=resize_shape).numpy()
        vid = normalize(vid, chan_means, chan_std_dev)

        return vid, label



if __name__ == "__main__":
    
    import pathlib
    vid_root = '/home/kevin/deepfake-proj/data/source/train_sorted'
    vid_root = pathlib.Path(vid_root)
    vid_ds = tf.data.Dataset.list_files(str(vid_root/'*/*'))
    transformer = DeepFakeTransformer(resize_shape=(224,224), seq_length=16)
    trf_func = transformer.transform_vid

    vid_ds = vid_ds.map(lambda x: tf.py_function(trf_func, [x], [tf.float32, tf.string]))

    print("TESTING VIDEO LOADER FUNCTIONALITY")

    for vid, label in vid_ds.take(3):
        print(label)
        print(vid.shape)