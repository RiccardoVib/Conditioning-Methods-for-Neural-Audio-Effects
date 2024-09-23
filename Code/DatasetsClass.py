import pickle
import os
import numpy as np
from tensorflow.keras.utils import Sequence
import tensorflow as tf
from Utils import filterAudio
from scipy.signal.windows import tukey


class DataGeneratorPickles(Sequence):

    def __init__(self, data_dir, filename, input_enc_size, input_dec_size, cond_size, model, conditioning, batch_size=10):
        """
        Initializes a data generator object for the CL1B dataset
          :param data_dir: the directory in which data are stored
          :param filename: the name of the dataset
          :param input_enc_size: the encoder input size
          :param input_dec_size: the decoder input size
          :param cond_size: the number of device's parameters
          :param model: the model object
          :param conditioning: if false the input and conditioning parameters are concatenated in the same vector
          :param batch_size: The size of each batch returned by __getitem__
        """
        # prepare the input, target, and conditioning matrix
        file_data = open(os.path.normpath('/'.join([data_dir, filename])), 'rb')
        Z = pickle.load(file_data)
        
        self.x = np.array(Z['x'][:, :], dtype=np.float32)
        self.y = np.array(Z['y'][:, :], dtype=np.float32)

        # windowning to avoid misalignments
        self.x = self.x * np.array(tukey(self.x.shape[1], alpha=0.005), dtype=np.float32).reshape(1,-1)
        self.y = self.y * np.array(tukey(self.x.shape[1], alpha=0.005), dtype=np.float32).reshape(1,-1)

        self.batch_size = batch_size

        rep = self.x.shape[1]
        self.x = self.x.reshape(-1)
        self.y = self.y.reshape(-1)
        
        # remove the last samples if not enough for a batch
        lim = int((self.x.shape[0] / self.batch_size) * self.batch_size)
        self.x = self.x[:lim]
        self.y = self.y[:lim]
        self.conditioning = conditioning
            
        self.z = np.array(Z['z'], dtype=np.float32)
        self.z = np.repeat(self.z, rep, axis=1)
    
        del Z

        self.input_enc_size = input_enc_size
        self.input_dec_size = input_dec_size
        self.window = int(self.input_enc_size + self.input_dec_size)
        
        # the number of iterations per epoch
        self.training_steps = (lim // self.batch_size)

        self.cond_size = cond_size
        self.model = model
        self.on_epoch_end()

    def on_epoch_end(self):
        # create/reset the vector containing the indices of the batches
        self.indices = np.arange(self.x.shape[0] + self.window)

    def __len__(self):
        # compute the needed number of iterations before conclude one epoch
        return int((self.x.shape[0]) / self.batch_size)-1

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            if i == self.__len__() - 1:
                self.on_epoch_end()

    def __getitem__(self, idx):
        # Initializing input, target, and conditioning batches

        X = np.empty((self.batch_size, self.window))
        Y = np.empty((self.batch_size, 1))
        Z = np.empty((self.batch_size, self.cond_size))
        
        # get the indices of the requested batch
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size] + self.window

        # fill the batches
        if self.cond_size != 0:
            c = 0
            for t in range(indices[0], indices[-1] + 1, 1):
                X[c, :] = (np.array(self.x[t - self.window: t]).T)
                Y[c, :] = (np.array(self.y[t-1]).T)
                Z[c, :] = (np.array(self.z[:, t-1]).T)
                c += 1
                
        else:
             c = 0
            for t in range(indices[0], indices[-1] + 1, 1):
                X[c, :] = (np.array(self.x[t - self.window: t]).T)
                Y[c, :] = (np.array(self.y[t-1]).T)
                c += 1
        
        if self.conditioning:

            return [Z, X], Y
        else:
            X = np.concatenate((X, Z), axis=-1)
            return X, Y
       
       
