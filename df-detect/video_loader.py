import numpy as np
import tensorflow as tf
import cv2
import pandas as pd
import gc
import os


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
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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
        if isinstance(filename, str):
            filepath= filename
        else:
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
            out_label = tf.constant([0.0, 1.0])
        else:
            out_label = tf.constant([1.0 , 0.0])
        
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
        result_tensors[1].set_shape((2,))
        return result_tensors

class DeepFakeDualTransformer(object):
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
        
    def get_frames(self, fnames):

        num_frames = self.seq_length
        
        real = fnames.numpy()[0].decode('utf-8')
        fake = fnames.numpy()[1].decode('utf-8')
        
        real_capture = cv2.VideoCapture(real)
        fake_capture = cv2.VideoCapture(fake)
        
        
        # Counts should be equal between real and fakes
        frame_count = int(fake_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        # Base inds on same frame grab to use matching video frames
        start = np.random.randint(frame_count-num_frames)
        frame_idxs = np.linspace(start, start+num_frames, num=num_frames, dtype=np.int)
        
        real_vid, _ = self.reader._read_frames_at_indices(real, real_capture, frame_idxs)
        fake_vid, _ = self.reader._read_frames_at_indices(fake, fake_capture, frame_idxs)
        
        real_capture.release()
        fake_capture.release()
        
        return real_vid, fake_vid
    
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
    
    def transform_vid(self, filenames):

        
        chan_means = self.chan_means
        chan_std_dev = self.chan_std_dev
        resize_shape = self.resize_shape
        
        # For kaggle only
        # fname = parts[-1].numpy().decode('utf-8')
        # global filelog
        # filelog.append(fname)
        
        real_vid, fake_vid = self.get_frames(filenames)
        

        real_vid = tf.image.resize(real_vid, size=resize_shape)
        fake_vid = tf.image.resize(fake_vid, size=resize_shape)
        real_vid = self.normalize(real_vid, chan_means, chan_std_dev)
        fake_vid = self.normalize(fake_vid, chan_means, chan_std_dev)

        return tf.stack((real_vid, fake_vid))
    
    def transform_map(self, x):
        result_tensor = tf.py_function(func=self.transform_vid,
                                        inp=[x],
                                        Tout=[tf.float32])
        # Convention is that result_tensor[0] = real_vid, result_tensor[1] = fake_vid
        result_tensor[0].set_shape((2,None,None,None,None))
        labels = tf.constant([[1.0, 0.0], [0.0, 1.0]])
        return result_tensor[0], labels


class EfficientNetLite(object):
    
    def __init__(self, path,  output_layer_ind=158):
        self.interpreter = tf.lite.Interpreter(model_path=path)
        self.interpreter.allocate_tensors()
        self.out_ind = output_layer_ind
    
    def extract_from_image(self, img):
        # Note strange behavior with RuntimeError has been observed
        
        self.interpreter.set_tensor(0, img)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.out_ind)
        
        return output_data
    
    def get_output_shapes(self):
        
        out_shapes = self.interpreter.get_tensor(self.out_ind).shape
        
        return out_shapes
        

    

class BlazefaceNetLite(object):
    
    def __init__(self, path,  output_layer_inds=[114, 157]):
        """Builds an efficient tflite model from pretrained blazeface
        to extract features
        
        Arguments:
            path {[type]} -- [description]
        
        Keyword Arguments:
            output_layer_inds {list} -- These are the indices 
            of the feature map outputs in the model, 
            defaults are recommended (default: {[114, 157]})
        """
        self.interpreter = tf.lite.Interpreter(model_path=path)
        self.interpreter.allocate_tensors()
        self.out_ind = output_layer_inds
    
    def extract_from_image(self, img):
        # Note strange behavior with RuntimeError has been observed
        
        self.interpreter.set_tensor(0, img)
        self.interpreter.invoke()
        out16 = self.interpreter.get_tensor(self.out_ind[0])
        out8 = self.interpreter.get_tensor(self.out_ind[1])
        
        return out16, out8
    
    def get_output_shapes(self):
        
        out_shape1 = self.interpreter.get_tensor(self.out_ind[0]).shape[1:]
        out_shape2 = self.interpreter.get_tensor(self.out_ind[1]).shape[1:]
        
        return out_shape1, out_shape2
    

class DeepFakeLoadExtractFeatures(object):
    def __init__(self, chan_means=[0.485*255, 0.456*255, 0.406*255],
                       chan_std_dev=[0.229*255, 0.224*255, 0.225*255],
                       resize_shape=(128,128),
                       seq_length=64,
                       feat_extractor_path='',
                       feat_extractor_output_layers=[114, 157],
                       mode="train"):
        """[summary]
        
        Keyword Arguments:
            chan_means {list} -- unused for blazeface (default: {[0.485*255, 0.456*255, 0.406*255]})
            chan_std_dev {list} -- unused for blazeface (default: {[0.229*255, 0.224*255, 0.225*255]})
            resize_shape {tuple} -- shape to resize all frames by, blazeface uses (default: {(128,128)})
            seq_length {int} -- number of frames to use (default: {64})
            feat_extractor_path {str} -- path where .tflite32 model is stored (default: {''})
            feat_extractor_output_layers {list} -- These are the indices 
                of the feature map outputs in the model, 
                defaults are recommended  (default: {[114, 157]})
            mode {str} -- WIP To use if validation requires different transformations (default: {"train"})
        
        Returns:
            [object] -- Feature extractor object that whose functions can be used to map over a 
            tensorflow dataset of filepaths
        """
        self.chan_means = chan_means
        self.chan_std_dev = chan_std_dev
        self.resize_shape = resize_shape
        self.seq_length = seq_length
        self.mode = mode
        self.reader = VideoReader()
        self.efficientnet_extractor = BlazefaceNetLite(path=feat_extractor_path, 
                                      output_layer_inds=feat_extractor_output_layers)
        
        self.frame_feature_shapes = self.efficientnet_extractor.get_output_shapes()
        
    def get_frames(self, fnames):

        num_frames = self.seq_length
        
        real = fnames.numpy()[0].decode('utf-8')
        fake = fnames.numpy()[1].decode('utf-8')
        
        real_capture = cv2.VideoCapture(real)
        fake_capture = cv2.VideoCapture(fake)
        
        
        # Counts should be equal between real and fakes
        frame_count = int(fake_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Useful if loading total video size
        if frame_count < num_frames:
            num_frames = frame_count
        
        # Base inds on same frame grab to use matching video frames
        start = np.random.randint(frame_count-num_frames)
        frame_idxs = np.linspace(start, start+num_frames, num=num_frames, dtype=np.int)
        
        real_vid, _ = self.reader._read_frames_at_indices(real, real_capture, frame_idxs)

        if np.random.choice([0,1], p=[0.9, 0.1]) == 1:
            # Throw an easy differable example randomly
            start = np.random.randint(frame_count-num_frames)
            frame_idxs = np.linspace(start, start+num_frames, num=num_frames, dtype=np.int)

        fake_vid, _ = self.reader._read_frames_at_indices(fake, fake_capture, frame_idxs)
        
        real_capture.release()
        fake_capture.release()
        
        return real_vid, fake_vid
    
    def normalize(self, video, chan_means, chan_std_dev):
        """[summary]

        Arguments:
            video {tf.Tensor} -- tensorflow reshaped video data
            chan_means {array} -- [description]
            chan_std_dev {array} -- [description]

        Returns:
            [tf.Tensor] -- normalized video data
        """
        
        # video -= chan_means
        # video /= chan_std_dev
        video /= 127.5
        video -= 1.0

        return video
    
    def transform_vid(self, filenames):

        
        chan_means = self.chan_means
        chan_std_dev = self.chan_std_dev
        resize_shape = self.resize_shape
        
        # For kaggle only
        # fname = parts[-1].numpy().decode('utf-8')
        # global filelog
        # filelog.append(fname)
        
        real_vid, fake_vid = self.get_frames(filenames)
        

        real_vid = tf.image.resize(real_vid, size=resize_shape)
        fake_vid = tf.image.resize(fake_vid, size=resize_shape)
        real_vid = self.normalize(real_vid, chan_means, chan_std_dev)
        fake_vid = self.normalize(fake_vid, chan_means, chan_std_dev)

        return real_vid, fake_vid
    
    def extract_features(self, filenames):
        
        rvid, fvid = self.transform_vid(filenames)
        
        real_output16 = np.empty((self.seq_length, *self.frame_feature_shapes[0]), dtype=np.float32)
        fake_output16 = np.empty((self.seq_length, *self.frame_feature_shapes[0]), dtype=np.float32)
        
        real_output8 = np.empty((self.seq_length, *self.frame_feature_shapes[1]), dtype=np.float32)
        fake_output8 = np.empty((self.seq_length, *self.frame_feature_shapes[1]), dtype=np.float32)
        
        for i in range(self.seq_length):
            
            real_output16[i], real_output8[i] = self.efficientnet_extractor.\
                                                extract_from_image(tf.reshape(rvid[i], 
                                                (1, *self.resize_shape, 3)))
            
            fake_output16[i], fake_output8[i] = self.efficientnet_extractor.\
                                                extract_from_image(tf.reshape(fvid[i], 
                                                (1, *self.resize_shape, 3)))
        del rvid, fvid
        gc.collect()
        
        out = (tf.stack((real_output16, fake_output16)), tf.stack((real_output8, fake_output8)))
        return out
    
    def transform_map(self, x):
        results16, results8 = tf.py_function(func=self.extract_features,
                                        inp=[x],
                                        Tout=[tf.float32, tf.float32])
        # Convention is 
        # result_tensor.set_shape((2,None,None,None,None))
        labels = tf.constant([[0.0], [1.0]])
        return results16, results8, labels

class DeepFakeLoadExtractFeaturesV2(object):
    def __init__(self, chan_means=[0.485*255, 0.456*255, 0.406*255],
                       chan_std_dev=[0.229*255, 0.224*255, 0.225*255],
                       resize_shape=(300,300),
                       seq_length=298,
                       feat_extractor_path='',
                       feat_extractor_output_layers=[114, 157],
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
        self.efficientnet_extractor = BlazefaceNetLite(path=feat_extractor_path, 
                                      output_layer_inds=feat_extractor_output_layers)
        
        self.frame_feature_shapes = self.efficientnet_extractor.get_output_shapes()
        
    def get_frames(self, fnames):

        num_frames = self.seq_length
        
        real = fnames.numpy()[0].decode('utf-8')
        fake = fnames.numpy()[1].decode('utf-8')
        
        real_capture = cv2.VideoCapture(real)
        fake_capture = cv2.VideoCapture(fake)
        
        
        # Counts should be equal between real and fakes
        frame_count = int(fake_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Useful if loading total video size
        if frame_count < num_frames:
            num_frames = frame_count
        
        # Base inds on same frame grab to use matching video frames
        start = np.random.randint(frame_count-num_frames)
        frame_idxs = np.linspace(start, start+num_frames, num=num_frames, dtype=np.int)
        
        real_vid, _ = self.reader._read_frames_at_indices(real, real_capture, frame_idxs)

        if np.random.choice([0,1], p=[0.9, 0.1]) == 1:
            # Throw an easy differable example randomly
            start = np.random.randint(frame_count-num_frames)
            frame_idxs = np.linspace(start, start+num_frames, num=num_frames, dtype=np.int)

        fake_vid, _ = self.reader._read_frames_at_indices(fake, fake_capture, frame_idxs)
        
        real_capture.release()
        fake_capture.release()
        
        return real_vid, fake_vid
    
    def normalize(self, video, chan_means, chan_std_dev):
        """[summary]

        Arguments:
            video {tf.Tensor} -- tensorflow reshaped video data
            chan_means {array} -- [description]
            chan_std_dev {array} -- [description]

        Returns:
            [tf.Tensor] -- normalized video data
        """
        
        # video -= chan_means
        # video /= chan_std_dev
        video /= 127.5
        video -= 1.0

        return video
    
    def transform_vid(self, filenames):

        
        chan_means = self.chan_means
        chan_std_dev = self.chan_std_dev
        resize_shape = self.resize_shape
        
        # For kaggle only
        # fname = parts[-1].numpy().decode('utf-8')
        # global filelog
        # filelog.append(fname)
        
        real_vid, fake_vid = self.get_frames(filenames)
        

        real_vid = tf.image.resize(real_vid, size=resize_shape)
        fake_vid = tf.image.resize(fake_vid, size=resize_shape)
        real_vid = self.normalize(real_vid, chan_means, chan_std_dev)
        fake_vid = self.normalize(fake_vid, chan_means, chan_std_dev)

        return real_vid, fake_vid
    
    def extract_features(self, videos):
        
        rvid, fvid = videos[0], videos[1]
        
        real_output16 = np.empty((self.seq_length, *self.frame_feature_shapes[0]), dtype=np.float32)
        fake_output16 = np.empty((self.seq_length, *self.frame_feature_shapes[0]), dtype=np.float32)
        
        real_output8 = np.empty((self.seq_length, *self.frame_feature_shapes[1]), dtype=np.float32)
        fake_output8 = np.empty((self.seq_length, *self.frame_feature_shapes[1]), dtype=np.float32)
        
        for i in range(self.seq_length):
            
            real_output16[i], real_output8[i] = self.efficientnet_extractor.\
                                                extract_from_image(tf.reshape(rvid[i], 
                                                (1, *self.resize_shape, 3)))
            
            fake_output16[i], fake_output8[i] = self.efficientnet_extractor.\
                                                extract_from_image(tf.reshape(fvid[i], 
                                                (1, *self.resize_shape, 3)))
        del rvid, fvid
        gc.collect()
        
        out = (tf.stack((real_output16, fake_output16)), tf.stack((real_output8, fake_output8)))
        return out
    
    def load_videos_map(self, x):
        # Loading map function for parallel loading call

        real_vid, fake_vid = tf.py_function(func=self.transform_vid,
                                inp=[x],
                                Tout=[tf.float32, tf.float32])
        return tf.stack((real_vid, fake_vid))


    def extract_feats_map(self, x):
        results16, results8 = tf.py_function(func=self.extract_features,
                                        inp=[x],
                                        Tout=[tf.float32, tf.float32])
        # Convention is 
        # result_tensor.set_shape((2,None,None,None,None))
        labels = tf.constant([[0.0], [1.0]])
        return results16, results8, labels


class ExtractedFeatureLoader():
    def __init__(self):
        self.initial_var = 0

    def read_numpy(self, filepath):

        file_str = filepath.numpy().decode('utf-8')
        real_fake_feats = np.load(file_str, allow_pickle=True)

        return real_fake_feats
    
    def tflow_map(self, x):

        result_tensor = tf.py_function(func=self.read_numpy,
                                        inp=[x],
                                        Tout=[tf.float32])
        # Convention is that result_tensor[0] = real_vid, result_tensor[1] = fake_vid
        result_tensor[0].set_shape((2,None,None,None,None))
        labels = tf.constant([[0.0], [1.0]])
        return result_tensor[0], labels




def etl_video_pairs_numpy(data_pairs_path,
                          out_dir,
                          resize_shape, 
                          sequence_len, 
                          feature_extractor_path
                          ):


    df_pairs = pd.read_csv(data_pairs_path)[['real', 'fake']]


    extractor = DeepFakeLoadExtractFeatures(resize_shape=resize_shape, seq_length=sequence_len,
                                            feat_extractor_path=feature_extractor_path)
    
    out_paths = []
    i = 0
    for r, f in df_pairs.to_numpy():

        real_file = r.split('/')[-1].split('.')[0]
        fake_file = f.split('/')[-1].split('.')[0]

        npy_file = real_file + '_' + fake_file + '.npy'

        extract = extractor.extract_features(tf.constant([r,f]))

        out_path = os.path.join(out_dir, npy_file)

        np.save(out_path, extract)

        out_paths.append(out_path)

        
        if i%100 == 0:
            print(f"Saved {i+1} npy files latest path: {out_path}")
        
        i+=1

    
    # df = pd.DataFrame()
    # df['files'] = np.array(out_paths)
    # df.to_csv(os.path.join(out_dir, 'real_fake_npys.csv'))


if __name__ == "__main__":


    etl_video_pairs_numpy('../data/intermediate/fake_to_real_mapping_part.csv',
                          '../data/intermediate/whole_videos_7x7x320',
                          (224,224), 
                          32, 
                          'models/efficientnet-lite0/efficientnet-lite0-fp32.tflite'
                          )
    


