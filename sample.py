import numpy as np
import random
import tensorflow as tf
from loader import load_data

def create_sample(reconstructed, simulated, center, indeces, rand=random, cl_min=1, cl_max=3):
    """
    Randomly combines clusters from different events into one sample and returns data to be fed 
    into the network:
    - X: numpy array (dim: 171x360) containing reco energy information of the clusters
    - ysort: tensor with center sim coordinates of each cluster from X (padded with -1 for constant dim: (cl_max,2))
    - ye: sim energy info of each cluster from X (padded with 0 for constant dim:(cl_max,11,11))
    
    Args: 
    - reconstructed: numpy array with reco energy info of each cluster
    - simulated: numpy array with sim energy info of each cluster
    - center: numpy array with sim coordinated of the cluster center
    - indeces: all the indeces of the existing events 
    - rand (default = random): random generator
    - cl_min: minimum number of clusters per sample 
    - cl_max: maximum number of clusters per sample
    """
    # Randomly choose events and the number of clusters and fill the arrays. 
    index = random.sample(indeces, random.randint(cl_min,cl_max))
    # print(index)
    X = sum((reconstructed[ind[0]][ind[1]] for ind in index))
    y = np.asarray([(center[ind[0]][ind[1]]) for ind in index])
    ye = np.asarray([(simulated[ind[0]][ind[1]]) for ind in index])
    
    # Sort the clusters based on the distance from (0,0) coordinate of the image. 
    dr = np.sqrt(1**2 + 1**2)
    ysort = y[np.argsort((np.sqrt(np.power(y[:,0],2) + np.power(y[:,1]+85,2))//dr + y[:,0]/360))] 
    ye = ye[np.argsort((np.sqrt(np.power(y[:,0],2) + np.power(y[:,1]+85,2))//dr + y[:,0]/360))] 

    return X, tf.pad(ysort, ((0, cl_max-len(index)), (0,0)), constant_values=-1), tf.pad(ye,((0,cl_max-len(index)),(0,0), (0,0)))

def save_sample(name, nevt=10000):
    """
    Create and save data sample in .npy format.
    
    Args: 
    - name: name of the file to save into 
    - nevent (default = 10000): number of samples to save in one file
    """
    reco, sim, center, indeces = load_data(0,2500)
    
    # Create numpy arrays to be filled. 
    X = np.full((nevt, 171, 360), 0.)
    y = np.full((nevt, 3, 2), 0.)
    ye = np.full((nevt, 3, 11, 11), 0.)
    
    # Fill the arrays.
    for i in range(nevt):
        X[i], y[i], ye[i] = create_sample(reco, sim, center, indeces, cl_min=1, cl_max=3)
    # Normalize the center coordinates to (0,1). 
    phi, eta = y[:,:,0], y[:,:,1]
    phi[phi!=-1]/=360
    eta[eta!=-1]+=85
    eta[eta!=-1]/=170
    y[:,:,0], y[:,:,1] = phi, eta
    X = X.reshape(nevt,171,360,1)
        
    # Save files. 
    np.save('X' + str(name), X)
    np.save('y' + str(name), y)
    np.save('ye' + str(name), ye)