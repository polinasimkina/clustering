import uproot
import numpy as np
import matplotlib.pyplot as plt
upFile = uproot.open('../data/photon_E1.0to100GeV_closeEcal_EB_noPU_pfrhRef_seedRef_thrXtalEBXtalEE_y2023_T2_v1_t0_n30000.root')
tree = upFile['recosimdumper/caloTree']
print(f"Events = {tree.array('eventId').size }")

import pandas as pd
import awkward
import concurrent.futures

executor = concurrent.futures.ThreadPoolExecutor()

def loadSimHits(tree, entrystop = 10, entrystart = 0, branches = ['simHit_energy', 'simHit_ieta', 'simHit_iphi', 'simHit_iz'],
               renameFcn = lambda x: x.replace('simHit_', '')):
    """
    Return a pandas DataFrame with simHit information like energy, ieta, iphi, iz.
    The information is read from an uproot tree with vector<vector<T>> branches which are flattened,
    adding as index (entry, particle, hit)

    Args:
    - tree: uproot tree (caloTree from ECAL clustering ntuples)
    - entrystart (default = 0): index of the first event to read
    - entrystop (default = 10): index of the last event to read
    - branches (default = ['simHit_energy', 'simHit_ieta', 'simHit_iphi', 'simHit_iz']): branches to read
    - renameFcn: function to rename the branches (default: remove the prefix simHit_)
    - entrystart (default = 0): index of the first event to read. 
    """
    return tree.pandas.df(branches,executor=executor, blocking=True, entrystop=entrystop, entrystart=entrystart)\
                      .apply(lambda x: awkward.topandas(awkward.fromiter(x), flatten=True))\
                      .rename(columns=renameFcn)\
                      .rename_axis(['entry', 'particle', 'hit'])

def loadRecHits(tree, entrystop = 10, entrystart = 0, branches = ['energy', 'ieta', 'iphi', 'iz'],
                prefixes = ['recHit_noPF_', 'pfRecHit_unClustered_'],
                renameFcn = lambda x: x.split('_')[-1]):
    """
    Return a pandas DataFrame with recHit information like energy, ieta, iphi, iz.
    The information is read from an uproot tree with vector<T> branches like recHit_noPF_energy, ...
    (combining prefixes and branches) which are flattened, adding as index (entry, subentry).

    NB: subentry repeated for branches with different prefixes

    Args:
    - tree: uproot tree (caloTree from ECAL clustering ntuples)
    - entrystart (default = 0): index of the first event to read
    - entrystop (default = 10): index of the last event to read
    - branches (default = ['energy', 'ieta', 'iphi', 'iz']): suffixes of branches to read
    - prefixes (default = ['recHit_noPF_', 'pfRecHit_unClustered_']): prefixes of branches to read
    - renameFcn: function to rename the branches (default: remove the prefixes)
    """
    d = [tree.pandas.df([p+s for s in branches], entrystop=entrystop, flatten=True, entrystart=entrystart)\
         .rename(columns=renameFcn)\
         for p in prefixes]
    return pd.concat(d)

def plot2D_eb( data, column = 'energy', axis = None, title = '', zAxis = (None, None) , cmap = 'terrain'):
    ics = np.full( (171,360) , np.nan)
    ics[  data['ieta']+85 , data['iphi']-1] = data[column]
    if axis is None:
        axis = plt.subplot()
    eb = axis.pcolormesh( np.linspace(0.5,360.5,num=361), np.linspace(-85.5,85.5,num=172),
                          ics, cmap = cmap, vmin = zAxis[0], vmax = zAxis[1]  )
    axis.set_xlabel('i$\phi$')
    axis.set_ylabel('i$\eta$')
    axis.set_title( title )
    plt.colorbar( eb , ax = axis  )
    return eb

def load_pandas(entrystart=0, entrystop=100):
    """
    Return pandas DataFrame:
    - recHits/simHits: recHit/simHit information (ieta, iphi, iz, energy) of the particles 
    - centers: sim information (ieta, iphi, iz, energy) of the center of cluster 
    The hits occured outside the barrel or with energy less than 50 MeV are dropped. 
    
    Args: 
    - entrystart (default = 0): index of the first event to read
    - entrystop (default = 100): index of the last event to read
    """
    simHits = loadSimHits(tree, entrystop=entrystop, entrystart=entrystart)
    recHits = loadRecHits(tree, entrystop=entrystop, entrystart=entrystart)
    centers = tree.pandas.df(['caloParticle_simEnergy', 'caloParticle_simIeta',
                                        'caloParticle_simIphi', 'caloParticle_simIz'], entrystop=entrystop, 
                             entrystart=entrystart).rename(columns=lambda x: x.replace('caloParticle_sim', '').lower())
    condition = (simHits.iz == 0) & (simHits.energy>0.05)
    return recHits[(recHits.iz == 0) & (recHits.energy>0.05)], simHits[(simHits.iz == 0) & (simHits.energy>0.05)], centers[(centers.iz == 0) & (centers.energy>0.05)]

def load_data(start=0, stop=100):
    """
    Function to translate information from pandas DataFrame to numpy array. 
    
    Return:
    - reconstructed: numpy array (dim: nx10x171x360) with reco energy info (whole barrel) 
    - simulated: numpy array (dim: nx10x11x11) with sim energy info (only around the center
    of the cluster, to associate with location in barrel, center info is required)
    - center: numpy array (dim: nx10x2) with sim coordinates of the cluster center
    - indeces: array containing indeces of all occured events
    
    Args: 
    - start (default = 0): index of the first event to read
    - stop (default = 100): index of the last event to read
    """
    # Load pandas DataFrame and drop hit index information. 
    recHits, simHits, centers = load_pandas(start,stop)
    recHits = recHits.reset_index(level=1, drop=True)
    simHits = simHits.reset_index(level=2, drop=True)
    
    # Create numpy arrays to be filled. 
    n = stop-start
    reconstructed = np.full((n, 10, 171, 360), 0.)
    simulated = np.full((n, 10, 11, 11), 0.)
    center = np.full((n,10,2), 0)
    
    # Fill the numpy arrays with info from DataFrames. 
    for ind in set(centers.index):
        i, j, strt = ind[0], ind[1], centers.index[0][0]
        phi0, eta0 = centers.loc[i,j].iphi, centers.loc[i,j].ieta # Define the center of the cluster (sim info).
        event = recHits.loc[i] 
        if (start>0): eventSim = simHits.loc[i-strt,j] # Required if start index is not zero. 
        else: eventSim = simHits.loc[i,j]

        # Define the reco cluster of 36x36 around the sim center position. 
        phi, eta = np.mod(np.arange(phi0-18,phi0+18), 360), np.arange(eta0-18, eta0+18)
        particle = event[event.iphi.isin(phi)&event.ieta.isin(eta)]
        reconstructed[i-strt, j, particle['ieta']+85, particle['iphi']-1] = particle['energy']
        
        # Record the sim center info.
        center[i-strt, j, 0] = centers.loc[i,j].iphi
        center[i-strt, j, 1] = centers.loc[i,j].ieta
        
        # Define the sim cluster of 11x11 around the sim center position. 
        phi, eta = np.arange(phi0-5,phi0+6), np.arange(eta0-5,eta0+6)
        eventSim = eventSim[(eventSim.iphi.isin(phi))&(eventSim.ieta.isin(eta))]
        eventSim.iphi-= int(phi0) - 5 # Required for the center of the cluster to be in the center of the image.
        eventSim.ieta-= int(eta0) - 5
        simulated[i-strt, j, eventSim['ieta'], eventSim['iphi']] = eventSim['energy']
        
    # Record the indeces of occured events. 
    indeces = [(a[0]-strt, a[1]) for a in set(centers.index)]
    return reconstructed, simulated, center, indeces