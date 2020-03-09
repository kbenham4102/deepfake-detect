import numpy as np
import tensorflow as tf
import cv2


class VideoReader:
    """Helper class for reading one or more frames from a video file."""

    def __init__(self, verbose=True, insets=(0, 0)):
        """Creates a new VideoReader.

        Arguments:
            verbose: whether to print warnings and error messages
            insets: amount to inset the image by, as a percentage of 
                (width, height). This lets you "zoom in" to an image 
                to remove unimportant content around the borders. 
                Useful for face detection, which may not work if the 
                faces are too small.
        """
        self.verbose = verbose
        self.insets = insets

    def read_frames(self, path, num_frames, jitter=0, seed=None):
        """Reads frames that are always evenly spaced throughout the video.

        Arguments:
            path: the video file
            num_frames: how many frames to read, -1 means the entire video
                (warning: this will take up a lot of memory!)
            jitter: if not 0, adds small random offsets to the frame indices;
                this is useful so we don't always land on even or odd frames
            seed: random seed for jittering; if you set this to a fixed value,
                you probably want to set it only on the first video 
        """
        assert num_frames > 0

        capture = cv2.VideoCapture(path)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0: return None

        start = np.random.randint(frame_count-num_frames)
        frame_idxs = np.linspace(start, start+num_frames, num=num_frames, dtype=np.int)
        if jitter > 0:
            np.random.seed(seed)
            jitter_offsets = np.random.randint(-jitter, jitter, len(frame_idxs))
            frame_idxs = np.clip(frame_idxs + jitter_offsets, 0, frame_count - 1)

        result, _ = self._read_frames_at_indices(path, capture, frame_idxs)
        capture.release()
        return result

    def read_random_frames(self, path, num_frames, seed=None):
        """Picks the frame indices at random.
        
        Arguments:
            path: the video file
            num_frames: how many frames to read, -1 means the entire video
                (warning: this will take up a lot of memory!)
        """
        assert num_frames > 0
        np.random.seed(seed)

        capture = cv2.VideoCapture(path)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0: return None

        frame_idxs = sorted(np.random.choice(np.arange(0, frame_count), num_frames))
        result = self._read_frames_at_indices(path, capture, frame_idxs)

        capture.release()
        return result

    def read_frames_at_indices(self, path, frame_idxs):
        """Reads frames from a video and puts them into a NumPy array.

        Arguments:
            path: the video file
            frame_idxs: a list of frame indices. Important: should be
                sorted from low-to-high! If an index appears multiple
                times, the frame is still read only once.

        Returns:
            - a NumPy array of shape (num_frames, height, width, 3)
            - a list of the frame indices that were read

        Reading stops if loading a frame fails, in which case the first
        dimension returned may actually be less than num_frames.

        Returns None if an exception is thrown for any reason, or if no
        frames were read.
        """
        assert len(frame_idxs) > 0
        capture = cv2.VideoCapture(path)
        result = self._read_frames_at_indices(path, capture, frame_idxs)
        capture.release()
        return result

    def _read_frames_at_indices(self, path, capture, frame_idxs):
        try:
            frames = []
            idxs_read = []
            for frame_idx in range(frame_idxs[0], frame_idxs[-1] + 1):
                # Get the next frame, but don't decode if we're not using it.
                ret = capture.grab()
                if not ret:
                    if self.verbose:
                        print("Error grabbing frame %d from movie %s" % (frame_idx, path))
                    break

                # Need to look at this frame?
                current = len(idxs_read)
                if frame_idx == frame_idxs[current]:
                    ret, frame = capture.retrieve()
                    if not ret or frame is None:
                        if self.verbose:
                            print("Error retrieving frame %d from movie %s" % (frame_idx, path))
                        break

                    frame = self._postprocess_frame(frame)
                    frames.append(frame)
                    idxs_read.append(frame_idx)

            if len(frames) > 0:
                return np.stack(frames), idxs_read
            if self.verbose:
                print("No frames read from movie %s" % path)
            return None
        except:
            if self.verbose:
                print("Exception while reading movie %s" % path)
            return None    

    def read_middle_frame(self, path):
        """Reads the frame from the middle of the video."""
        capture = cv2.VideoCapture(path)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        result = self._read_frame_at_index(path, capture, frame_count // 2)
        capture.release()
        return result

    def read_frame_at_index(self, path, frame_idx):
        """Reads a single frame from a video.
        
        If you just want to read a single frame from the video, this is more
        efficient than scanning through the video to find the frame. However,
        for reading multiple frames it's not efficient.
        
        My guess is that a "streaming" approach is more efficient than a 
        "random access" approach because, unless you happen to grab a keyframe, 
        the decoder still needs to read all the previous frames in order to 
        reconstruct the one you're asking for.

        Returns a NumPy array of shape (1, H, W, 3) and the index of the frame,
        or None if reading failed.
        """
        capture = cv2.VideoCapture(path)
        result = self._read_frame_at_index(path, capture, frame_idx)
        capture.release()
        return result

    def _read_frame_at_index(self, path, capture, frame_idx):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = capture.read()    
        if not ret or frame is None:
            if self.verbose:
                print("Error retrieving frame %d from movie %s" % (frame_idx, path))
            return None
        else:
            frame = self._postprocess_frame(frame)
            return np.expand_dims(frame, axis=0), [frame_idx]
    
    def _postprocess_frame(self, frame):
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # if self.insets[0] > 0:
        #     W = frame.shape[1]
        #     p = int(W * self.insets[0])
        #     frame = frame[:, p:-p, :]

        # if self.insets[1] > 0:
        #     H = frame.shape[1]
        #     q = int(H * self.insets[1])
        #     frame = frame[q:-q, :, :]

        return frame

class DeepFakeTransformer(object):
    def __init__(self, chan_means=[0.485, 0.456, 0.406],
                       chan_std_dev=[0.229, 0.224, 0.225],
                       resize_shape=(300,300),
                       seq_length=298,
                       mode="train"):
        """[summary]
        
        Keyword Arguments:
            chan_means {list} -- [description] (default: {[0.485, 0.456, 0.406]})
            chan_std_dev {list} -- [description] (default: {[0.229, 0.224, 0.225]})
            resize_shape {tuple} -- [description] (default: {(300,300)})
            seq_length {int} -- [description] (default: {298})
            mode {str} -- [description] (default: {"train"})
        """

        self.chan_means = chan_means
        self.chan_std_dev = chan_std_dev
        self.resize_shape = resize_shape
        self.seq_length = seq_length
        self.mode = mode
        self.reader = VideoReader()
        
    def get_frames(self, filename):
        '''
        method for getting the frames from a video file
        args: 
            filename: exact path of the video file
        out:
            video_frames, label:  
        '''

        filepath = filename.numpy().decode('utf-8')

        all_frames = self.reader.read_frames(filepath, num_frames=self.seq_length)
        
        return all_frames
    
    def normalize(self, video, chan_means, chan_std_dev):
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
        # print('*'*1000, filename.numpy())
 
        parts = tf.strings.split(filename, '/')
        label = parts[-2].numpy().decode('utf-8')
        
        # For kaggle only
        # fname = parts[-1].numpy().decode('utf-8')
        # global filelog
        # filelog.append(fname)
        
        if label == 'FAKE':
            out_label = tf.constant([1.0])
        else:
            out_label = tf.constant([0.0])
        
        vid = self.get_frames(filename)
        assert len(vid.shape) == 4, f'{filename} has incorrect video dimensions'
        # assert np.sum(np.isnan(vid)) == 0, f'{filename} has null pixels'
        # vid = vid.astype(np.float)
        vid = tf.image.resize(vid, size=resize_shape)
        vid = self.normalize(vid, chan_means, chan_std_dev)


        return vid, out_label
    
    def transform_map(self, x):
        result_tensors = tf.py_function(func=self.transform_vid,
                                        inp=[x],
                                        Tout=[tf.float32, tf.float32])
        result_tensors[0].set_shape((None,None,None,None))
        result_tensors[1].set_shape((1,))
        return result_tensors



if __name__ == "__main__":
    
    import pathlib
    import glob 


    vid_root = '../data/source/train_val_sort/*/*/*.mp4'
    vid_ds = tf.data.Dataset.list_files(vid_root, shuffle=False)
    transformer = DeepFakeTransformer(resize_shape=(224,224), seq_length=30)
    
    # vid_ds = vid_ds.map(lambda x: transformer.transform_map(x))
    # vid_ds = vid_ds.batch(2).prefetch(4)

    # print("TESTING VIDEO LOADER FUNCTIONALITY")

    # for item in vid_ds.as_numpy_iterator():
    #     tf.convert_to_tensor(item)

    # vidreader = VideoReader()

    vid_files = glob.glob(vid_root)

    for f in vid_files:

        vid, label = transformer.transform_map(tf.convert_to_tensor(f.encode('utf-8')))

