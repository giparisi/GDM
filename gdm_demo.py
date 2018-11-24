# -*- coding: utf-8 -*-
"""
Dual-memory Incremental learning with memory replay
@last-modified: 23 November 2018
@author: German I. Parisi (german.parisi@gmail.com)

"""

import gtls
import numpy as np
from episodic_gwr import EpisodicGWR

if __name__ == "__main__":

    train_flag = True
    train_type = 1 # 0:Batch, 1: Incremental
    train_replay = True
    
    ds_iris = gtls.IrisDataset(file='iris.csv', normalize=True)
    print("%s from %s loaded." % (ds_iris.name, ds_iris.file))

    assert train_type < 2, "Invalid type of training."
    
    '''
    Episodic-GWR supports multi-class neurons.
    Set the number of label classes per neuron and possible labels per class
    e.g. e_labels = [50, 10]
    is two labels per neuron, one with 50 and the other with 10 classes.
    Setting the n. of classes is done for experimental control but it is not
    necessary for associative GWR learning.
    '''
    e_labels = [3, 3]
    s_labels = [3]
    ds_vectors = ds_iris.vectors
    ds_labels = np.zeros((len(e_labels), len(ds_iris.labels)))
    ds_labels[0] = ds_iris.labels
    ds_labels[1] = ds_iris.labels
    
    num_context = 1 # number of context descriptors
    epochs = 1 # epochs per sample for incremental learning
    a_threshold = [0.85, 0.9]
    beta = 0.7
    learning_rates = [0.2, 0.001]
    context = True
    
    g_episodic = EpisodicGWR(ds_iris, e_labels, num_context)
    g_semantic = EpisodicGWR(ds_iris, s_labels, num_context)
    
    if train_type == 0:
        # Batch training
        # Train episodic memory
        g_episodic.train_egwr(ds_vectors, ds_labels, epochs, a_threshold[0],
                              beta, learning_rates, context, regulated=0)
                              
        e_weights, e_labels = g_episodic.test(ds_vectors, ds_labels,
                                              test_accuracy=True,
                                              ret_vecs=True)
        # Train semantic memory
        g_semantic.train_egwr(e_weights, e_labels, epochs, a_threshold[1], beta, 
                          learning_rates, context, regulated=1)
                          
        g_semantic.test(e_weights, e_labels, test_accuracy=True)
        
    else:
        # Incremental training
        # Train episodic memory
        batch_size = 10 # number of samples per epoch
        for s in range(0, ds_vectors.shape[0], batch_size):
            g_episodic.train_egwr(ds_vectors[s:s+batch_size], ds_labels[:, s:s+batch_size],
                              epochs, a_threshold[0], beta, learning_rates, context,
                              regulated=0)

        e_weights, e_labels = g_episodic.test(ds_vectors, ds_labels,
                                              test_accuracy=True,
                                              ret_vecs=True)
        # Train semantic memory
        for s in range(0, ds_vectors.shape[0], batch_size):
            g_semantic.train_egwr(e_weights[s:s+batch_size], e_labels[:, s:s+batch_size],
                                  epochs, a_threshold[1], beta, learning_rates, context,
                                  regulated=1)
                          
        g_semantic.test(e_weights, e_labels, test_accuracy=True)                                          

    print("Accuracy episodic: %s, semantic: %s" % 
          (g_episodic.test_accuracy[0], g_semantic.test_accuracy[0]))