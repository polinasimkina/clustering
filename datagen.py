import random
import numpy as np

from tensorflow import keras
from sample import create_sample
from loader import load_data

class DataGenerator(keras.utils.Sequence):
    'Generate data sample of the fly'
    def __init__(self, sample_size, reco, sim, center, indeces, batch_size=32, dim=(171,360)):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.start = 0
        self.n = 42
        self.reco = reco
        self.sim = sim
        self.centers = center
        self.ind = indeces
        self.rand = random
        self.rand.seed(self.n)
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.sample_size / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        X, y, ye = self.__data_generation()
        return X, (y, ye)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.sample_size)
        #self.reco, self.sim, self.centers, self.ind = load_data(self.start,self.start+100)
        self.n = 42
        self.rand.seed(self.n)
    
    def transform(self, x, y, ye):
        'Normalize center coordinated to be (0,1)'
        y = y.astype(np.float32)
        phi, eta = y[:,:,0], y[:,:,1]
        phi[phi!=-1]/=360
        eta[eta!=-1]+=85
        eta[eta!=-1]/=170
        y[:,:,0], y[:,:,1] = phi, eta
        x = x        
        ye = ye
        return x, y, ye
        
    def __data_generation(self):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.full((self.batch_size, 171, 360), 0.)
        y = np.full((self.batch_size, 3, 2), 0)
        ye = np.full((self.batch_size, 3, 11, 11), 0)
        # Fill arrays
        for i in range(self.batch_size):
            X[i], y[i], ye[i] = create_sample(self.reco, self.sim, self.centers, self.ind, self.rand, 1, 3)
        X, y, ye = self.transform(X,y,ye)
        return X.reshape(self.batch_size, 171, 360,1), y, ye

def net_data(k=0, n_train=37500, n_valid=12500, batch=64):
    """
    Returns data to be fed to the network: either datagenerator or loaded numpy arrays. 
    
    Args: 
    - k (default = 0): 0 for DataGenerator, 1 for numpy arrays
    - n_train (default = 37500): number of sample for training data generator
    - n_valid (default = 12500): number of sample for validation data generator
    - batch (default = 64): batch size for data generator
    """
    if k == 0: 
        reco, sim, center, indeces = load_data()
        train = DataGenerator(n_train, reco, sim, center, indeces, batch)
        valid = DataGenerator(n_valid, reco, sim, center, indeces, batch)
    if k == 1: 
        train = (np.load('Xtrain.npy'), np.load('ytrain.npy'), np.load('yetrain.npy'))
        valid = (np.load('Xtest.npy'), np.load('ytest.npy'), np.load('yetest.npy'))
    return train, valid