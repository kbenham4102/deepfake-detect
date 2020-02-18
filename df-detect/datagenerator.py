class DataGenerator(tf.keras.utils.Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, filepaths, labels, image_path,
                 to_fit=True, batch_size=4, dim=(298,300,300,3),
                 resize_dim=(300,300),
                 chan_means=[0.485, 0.456, 0.406], 
                 chan_std_dev = [0.229, 0.224, 0.225],
                 n_channels=3, shuffle=True):
        """Initialization

        :param filepaths: list of all image filepaths to use in the generator
        :param labels: list of image class labels 
        :param image_path: path to images location
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.filepaths = filepaths
        self.mean = chan_means
        self.std = chan_std_dev
        self.labels = labels
        self.image_path = image_path
        self.mask_path = mask_path
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.resize_dim = resize_dim
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch

        :return: number of batches per epoch
        """
        return int(np.floor(len(self.filepaths) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data

        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        filepaths_temp = [self.filepaths[k] for k in indexes]

        # Generate data
        X = self._generate_X(filepaths_temp)

        if self.to_fit:
            y = self._generate_y(filepaths_temp)
            return X, y
        else:
            return X

    def on_epoch_end(self):
        """Updates indexes after each epoch

        """
        self.indexes = np.arange(len(self.filepaths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _generate_X(self, filepaths_temp):
        """Generates data containing batch_size images

        :param filepaths_temp: list of images to load
        :return: batch of images
        """
        # Initialization
        X = np.empty((self.batch_size, *self.dim)

        # Generate data
        for i, fp in enumerate(filepaths_temp):
            # Store sample
            X[i,] = self._load_video(fp)

        return X

    def _generate_y(self, filepaths_temp):
        """Generates data containing batch_size masks

        :param filepaths_temp: list of label ids to load
        :return: batch if masks
        """
        y = np.empty((self.batch_size, *self.dim), dtype=int)

        # Generate data
        for i, ID in enumerate(filepaths_temp):
            # Store sample
            y[i,] = self._load_video(self.mask_path + self.labels[ID])

        return y

    def _load_video(self, vid_path):
        """Load video
        :param vid_path: path to vid to load
        :return: loaded vid
        """

        vid = get_frames(p)[:298,:,:,:]
        vid = tf.image.resize(vid, size=self.dim)
        vid = normalize(vid, self.mean, self.std)

        return vid
    
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