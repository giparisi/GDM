# -*- coding: utf-8 -*-
"""
gwr-tb :: utilities
@last-modified: 17 November 2018
@author: German I. Parisi (german.parisi@gmail.com)

"""

import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt

def import_network(file_name, NetworkClass):
    """ Import pickled network from file
    """
    file = open(file_name, 'br')
    data_pickle = file.read()
    file.close()
    net = NetworkClass(ds=None, random=False)
    net.__dict__ = pickle.loads(data_pickle)
    return net
    
def export_network(file_name, net) -> None:
    """ Export pickled network to file
    """
    file = open(file_name, 'wb')
    file.write(pickle.dumps(net.__dict__))
    file.close()

def load_file(file_name) -> np.ndarray:
    """ Load dataset from file
    """
    reader = csv.reader(open(file_name, "r"), delimiter=',')
    x_rdr = list(reader)
    return  np.array(x_rdr).astype('float')

def normalize_data(data) -> np.ndarray:
    """ Normalize data vectors
    """
    for i in range(0, data.shape[1]):
        max_col = max(data[:, i])
        min_col = min(data[:, i])
        for j in range(0, data.shape[0]):
            data[j, i] = (data[j, i] - min_col) / (max_col - min_col)
    return data

def plot_network(net, edges, labels) -> None:
    """ 2D plot
    """        
    # Plot network
    # This just plots the first two dimensions of the weight vectors.
    # For better visualization, PCA over weight vectors must be performed.
    ccc = ['black','blue','red','green','yellow','cyan','magenta','0.75','0.15','1']
    plt.figure()
    for ni in range(len(net.weights)):
        plindex = np.argmax(net.alabels[ni])
        if labels:
            plt.scatter(net.weights[ni,0], net.weights[ni,1], color=ccc[plindex], alpha=.5)
        else:
            plt.scatter(net.weights[ni,0], net.weights[ni,1], alpha=.5)
        if edges:
            for nj in range(len(net.weights)):
                if  net.edges[ni,nj] > 0:
                    plt.plot([net.weights[ni,0], net.weights[nj,0]], 
                             [net.weights[ni,1], net.weights[nj,1]], 'gray', alpha=.3)
    plt.show()

def plot_gamma(net, edges, labels) -> None:
    """ 2D plot of Gamma-GWR
    """  
    # This just plots the first two dimensions of the weight vectors.
    # For better visualization, PCA over weight vectors must be performed.
    #sns.set(style="darkgrid")
    ccc = ['black','blue','red','green','yellow','cyan','magenta','0.75','0.15','1']
    plt.figure()
    for ni in range(len(net.weights)):
        plindex = np.argmax(net.alabels[ni])
        if labels:
            plt.scatter(net.weights[ni,0,0], net.weights[ni,0,1], color=ccc[plindex], alpha=.5)
        else:
            plt.scatter(net.weights[ni,0,0], net.weights[ni,0,1])
        if edges:  
            for nj in range(len(net.weights)):
                if net.edges[ni,nj] > 0:
                    plt.plot([net.weights[ni,0,0], net.weights[nj,0,0]],
                             [net.weights[ni,0,1], net.weights[nj,0,1]], 'gray', alpha=.3)
    plt.show()


class IrisDataset:
    """ Create an instance of Iris dataset
    """
    def __init__(self, file, normalize):
        self.name = 'IRIS'
        self.file = file
        self.normalize = normalize
        self.num_classes = 3
        
        raw_data = load_file(self.file)
        
        self.labels = raw_data[:, raw_data.shape[1]-1]
        self.vectors = raw_data[:, 0:raw_data.shape[1]-1]
        
        label_list = list()
        for label in self.labels:
            if label not in label_list:
                label_list.append(label)
        n_classes = len(label_list)
        
        assert self.num_classes == n_classes, "Inconsistent number of classes"
        
        if self.normalize:
            self.vectors = normalize_data(self.vectors)
                