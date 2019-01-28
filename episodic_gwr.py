"""
gwr-tb :: Episodic-GWR
@last-modified: 25 January 2019
@author: German I. Parisi (german.parisi@gmail.com)

"""

import numpy as np
import math
from gammagwr import GammaGWR

class EpisodicGWR(GammaGWR):

    def __init__(self):
        self.iterations = 0
    
    def init_network(self, ds, e_labels, num_context) -> None:
        
        assert self.iterations < 1, "Can't initialize a trained network"
        assert ds is not None, "Need a dataset to initialize a network"
        
        # Lock to prevent training
        self.locked = False

        # Start with 2 neurons
        self.num_nodes = 2
        self.dimension = ds.vectors.shape[1]
        self.num_context = num_context
        self.depth = self.num_context + 1
        empty_neuron = np.zeros((self.depth, self.dimension))
        self.weights = [empty_neuron, empty_neuron]
        
        # Global context
        self.g_context = np.zeros((self.depth, self.dimension))        
        
        # Create habituation counters
        self.habn = [1, 1]
        
        # Create edge and age matrices
        self.edges = np.ones((self.num_nodes, self.num_nodes))
        self.ages = np.zeros((self.num_nodes, self.num_nodes))
        
        # Temporal connections
        self.temporal = np.zeros((self.num_nodes, self.num_nodes))

        # Label histogram
        self.num_labels = e_labels
        self.alabels = []
        for l in range(0, len(self.num_labels)):
            self.alabels.append(-np.ones((self.num_nodes, self.num_labels[l])))
        init_ind = list(range(0, self.num_nodes))
        for i in range(0, len(init_ind)):
            self.weights[i][0] = ds.vectors[i]
            
        # Context coefficients
        self.alphas = self.compute_alphas(self.depth)
            
    def update_temporal(self, current_ind, previous_ind, **kwargs) -> None:
        new_node = kwargs.get('new_node', False)
        if new_node:
            self.temporal = super().expand_matrix(self.temporal)
        if previous_ind != -1 and previous_ind != current_ind:
            self.temporal[previous_ind, current_ind] += 1

    def update_labels(self, bmu, label, **kwargs) -> None:
        new_node = kwargs.get('new_node', False)        
        if not new_node:
            for l in range(0, len(self.num_labels)):
                for a in range(0, self.num_labels[l]):
                    if a == label[l]:
                        self.alabels[l][bmu, a] += self.a_inc
                    else:
                        if label[l] != -1:
                            self.alabels[l][bmu, a] -= self.a_dec
                            if (self.alabels[l][bmu, a] < 0):
                                self.alabels[l][bmu, a] = 0              
        else:
            for l in range(0, len(self.num_labels)):
                new_alabel = np.zeros((1, self.num_labels[l]))
                if label[l] != -1:
                    new_alabel[0, int(label[l])] = self.a_inc
                self.alabels[l] = np.concatenate((self.alabels[l], new_alabel), axis=0)

    def remove_isolated_nodes(self) -> None:
        if self.num_nodes > 2:
            ind_c = 0
            rem_c = 0
            while (ind_c < self.num_nodes):
                neighbours = np.nonzero(self.edges[ind_c])            
                if len(neighbours[0]) < 1:
                    if self.num_nodes > 2:
                        self.weights.pop(ind_c)
                        self.habn.pop(ind_c)
                        for d in range(0, len(self.num_labels)):
                            d_labels = self.alabels[d]
                            self.alabels[d] = np.delete(d_labels, ind_c, axis=0)
                        self.edges = np.delete(self.edges, ind_c, axis=0)
                        self.edges = np.delete(self.edges, ind_c, axis=1)
                        self.ages = np.delete(self.ages, ind_c, axis=0)
                        self.ages = np.delete(self.ages, ind_c, axis=1)
                        self.temporal = np.delete(self.temporal, ind_c, axis=0)
                        self.temporal = np.delete(self.temporal, ind_c, axis=1)
                        self.num_nodes -= 1
                        rem_c += 1
                    else: return
                else:
                    ind_c += 1
            print ("(-- Removed %s neuron(s))" % rem_c)
         
    def train_egwr(self, ds_vectors, ds_labels, epochs, a_threshold, beta, 
                   l_rates, context, regulated) -> None:
        
        assert not self.locked, "Network is locked. Unlock to train."
        
        self.samples = ds_vectors.shape[0]        
        self.max_epochs = epochs
        self.a_threshold = a_threshold   
        self.epsilon_b, self.epsilon_n = l_rates
        self.beta = beta
        self.regulated = regulated
        self.context = context
        if not self.context:
            self.g_context.fill(0)
        self.hab_threshold = 0.1
        self.tau_b = 0.3
        self.tau_n = 0.1
        self.max_nodes = self.samples # OK for batch, bad for incremental
        self.max_neighbors = 6
        self.max_age = 600
        self.new_node = 0.5
        self.a_inc = 1
        self.a_dec = 0.1
        self.mod_rate = 0.01
            
        # Start training
        error_counter = np.zeros(self.max_epochs)
        previous_bmu = np.zeros((self.depth, self.dimension))
        previous_ind = -1
        for epoch in range(0, self.max_epochs):
            for iteration in range(0, self.samples):
                
                # Generate input sample
                self.g_context[0] = ds_vectors[iteration]
                label = ds_labels[:, iteration]
                
                # Update global context
                for z in range(1, self.depth):
                    self.g_context[z] = (self.beta * previous_bmu[z]) + ((1-self.beta) * previous_bmu[z-1])
                
                # Find the best and second-best matching neurons
                b_index, b_distance, s_index = super().find_bmus(self.g_context, s_best = True)
                
                b_label = np.argmax(self.alabels[0][b_index])
                misclassified = b_label != label[0]
                
                # Quantization error
                error_counter[epoch] += b_distance
                
                # Compute network activity
                a = math.exp(-b_distance)

                # Store BMU at time t for t+1
                previous_bmu = self.weights[b_index]

                if (not self.regulated) or (self.regulated and misclassified):
                    
                    if (a < self.a_threshold
                        and self.habn[b_index] < self.hab_threshold
                        and self.num_nodes < self.max_nodes):
                        # Add new neuron
                        n_index = self.num_nodes
                        super().add_node(b_index)
                       
                        # Add label histogram           
                        self.update_labels(n_index, label, new_node = True)                   
    
                        # Update edges and ages
                        super().update_edges(b_index, s_index, new_index = n_index)
                        
                        # Update temporal connections
                        self.update_temporal(n_index, previous_ind, new_node = True)
    
                        # Habituation counter                    
                        super().habituate_node(n_index, self.tau_b, new_node = True)
                    
                    else:
                        # Habituate BMU
                        super().habituate_node(b_index, self.tau_b)
    
                        # Update BMU's weight vector
                        b_rate, n_rate = self.epsilon_b, self.epsilon_n
                        if self.regulated and misclassified:
                            b_rate *= self.mod_rate
                            n_rate *= self.mod_rate
                        else:
                            # Update BMU's label histogram
                            self.update_labels(b_index, label)
    
                        super().update_weight(b_index, b_rate)
    
                        # Update BMU's edges // Remove BMU's oldest ones
                        super().update_edges(b_index, s_index)
    
                        # Update temporal connections
                        self.update_temporal(b_index, previous_ind)
    
                        # Update BMU's neighbors
                        super().update_neighbors(b_index, n_rate)
                        
                self.iterations += 1
                    
                previous_ind = b_index

            # Remove old edges
            super().remove_old_edges()
            
            # Average quantization error (AQE)
            error_counter[epoch] /= self.samples
            
            print ("(Epoch: %s, NN: %s, ATQE: %s)" % (epoch + 1, self.num_nodes, error_counter[epoch]))
            
        # Remove isolated neurons
        self.remove_isolated_nodes()

    def test(self, ds_vectors, ds_labels, **kwargs):
        test_accuracy = kwargs.get('test_accuracy', False)
        test_vecs = kwargs.get('ret_vecs', False)
        test_samples = ds_vectors.shape[0]
        self.bmus_index = -np.ones(test_samples)
        self.bmus_weight = np.zeros((test_samples, self.dimension))
        self.bmus_label = -np.ones((len(self.num_labels), test_samples))
        self.bmus_activation = np.zeros(test_samples)
        
        input_context = np.zeros((self.depth, self.dimension))
        
        if test_accuracy:
            acc_counter = np.zeros(len(self.num_labels))
        
        for i in range(0, test_samples):
            input_context[0] = ds_vectors[i]
            # Find the BMU
            b_index, b_distance = super().find_bmus(input_context)
            self.bmus_index[i] = b_index
            self.bmus_weight[i] = self.weights[b_index][0]
            self.bmus_activation[i] = math.exp(-b_distance)
            for l in range(0, len(self.num_labels)):
                self.bmus_label[l, i] = np.argmax(self.alabels[l][b_index])
            
            for j in range(1, self.depth):
                input_context[j] = input_context[j-1]
            
            if test_accuracy:
                for l in range(0, len(self.num_labels)):
                    if self.bmus_label[l, i] == ds_labels[l, i]:
                        acc_counter[l] += 1

        if test_accuracy: self.test_accuracy =  acc_counter / ds_vectors.shape[0]
            
        if test_vecs:
            s_labels = -np.ones((1, test_samples))
            s_labels[0] = self.bmus_label[1]
            return self.bmus_weight, s_labels
