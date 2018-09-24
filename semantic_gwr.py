# -*- coding: utf-8 -*-
"""
Semantic Gamma-GWR Memory
@last-modified: 8 September 2018
@author: German I. Parisi (german.parisi@gmail.com)
Please cite this paper: Parisi, G.I., Tani, J., Weber, C., Wermter, S. (2018) Lifelong Learning of Spatiotemporal Representations with Dual-Memory Recurrent Self-Organization. arXiv:1805.10966
"""

import numpy as np
import math

class SemanticGWR:
    def initNetwork(self, dimension, numWeights, numClasses):
        self.numNodes = 2
        self.dimension = dimension
        self.numWeights = numWeights
        self.numOfClasses = numClasses
        self.recurrentWeights = np.zeros((self.numNodes,self.numWeights,self.dimension))
        self.alabels = np.zeros((self.numNodes,self.numOfClasses))
        self.globalContext = np.zeros((self.numWeights,self.dimension))
        self.edges = np.zeros((self.numNodes,self.numNodes))
        self.ages = np.zeros((self.numNodes,self.numNodes))
        self.habn = np.ones(self.numNodes)
        self.temporal = np.zeros((self.numNodes,self.numNodes))
        self.varAlpha = self.gammaWeights(self.numWeights)
        self.context = 1
        self.updateRate = 0
 
    def gammaWeights ( self, nw ):
        iWe = np.zeros(nw)
        for h in range(0,len(iWe)):
            iWe[h] = np.exp(-h)
        iWe[:] = iWe[:] / sum(iWe)
        return iWe
            
    def habituateNeuron(self, index, tau):
            self.habn[index] += (tau * 1.05 * (1. - self.habn[index]) - tau)

    def updateNeuron(self, index, epsilon):
        deltaWeights = np.zeros((self.numWeights,self.dimension))
        for i in range(0, self.numWeights):
            deltaWeights[i] = np.array([np.dot((self.globalContext[i]-self.recurrentWeights[index,i]), epsilon)]) * self.habn[index]
        self.recurrentWeights[index] += deltaWeights
            
    def updateLabelHistogram(self, bmu, label):         
        if (label!=-1):
            for a in range(0,self.numOfClasses):
                if (a==label):
                    self.alabels[bmu, a] += self.aIncreaseFactor
                else:
                    self.alabels[bmu, a] -= self.aDecreaseFactor
                    if (self.alabels[bmu, a] < 0):
                        self.alabels[bmu, a] = 0
        
    def updateEdges(self, fi, si):
        neighboursFirst = np.nonzero(self.edges[fi])
        if (len(neighboursFirst[0]) >= self.maxNeighbours):
            remIndex = -1
            maxAgeNeighbour = 0
            for u in range(0,len(neighboursFirst[0])):
                if (self.ages[fi, neighboursFirst[0][u]]>maxAgeNeighbour):
                    maxAgeNeighbour = self.ages[fi, neighboursFirst[0][u]]
                    remIndex = neighboursFirst[0][u]
            self.edges[fi, remIndex] = 0
            self.edges[remIndex, fi] = 0
        self.edges[fi, si] = 1

    def removeOldEdges(self):
        for i in range(0, self.numNodes):
            neighbours = np.nonzero(self.edges[i])
            for j in range(0, len(neighbours[0])):
                if (self.ages[i, j] >=  self.maxAge):
                    self.edges[i, j] = 0
                    self.edges[j, i] = 0

    def removeIsolatedNeurons(self):        
        indCount = 0
        while (indCount < self.numNodes):
            neighbours = np.nonzero(self.edges[indCount])
            if (len(neighbours[0])<1):
                self.recurrentWeights = np.delete(self.recurrentWeights, indCount, axis=0)
                self.alabels = np.delete(self.alabels, indCount, axis=0)
                self.edges = np.delete(self.edges, indCount, axis=0)
                self.edges = np.delete(self.edges, indCount, axis=1)
                self.ages = np.delete(self.ages, indCount, axis=0)
                self.ages = np.delete(self.ages, indCount, axis=1)
                self.habn = np.delete(self.habn, indCount)
                self.temporal = np.delete(self.temporal, indCount, axis=0)
                self.temporal = np.delete(self.temporal, indCount, axis=1)
                self.numNodes -= 1
                print ("(--", indCount, ")")
            else:
                indCount += 1

    def train(self, dataSet, labelSet, maxEpochs, insertionT, beta, epsilon_b, epsilon_n, context, regulated):
        self.dataSet = dataSet
        self.samples = self.dataSet.shape[0]
        self.labelSet = labelSet
        self.maxEpochs = maxEpochs
        self.insertionThreshold = insertionT 
        self.varBeta = beta
        self.epsilon_b = epsilon_b
        self.epsilon_n = epsilon_n
        if not self.context:
            self.globalContext.fill(0)
        self.regulated = regulated
        
        self.habThreshold = 0.1
        self.tau_b = 0.3
        self.tau_n = 0.1
        self.maxNodes = 10000
        self.maxAge = 1200
        self.maxNeighbours = 6
        self.aIncreaseFactor = 1.
        self.aDecreaseFactor = 0.01
        
        self.nNN = np.zeros(self.maxEpochs) #nNN[0] = 2
        self.qrror = np.zeros((self.maxEpochs,2))
        self.fcounter = np.zeros((self.maxEpochs,2))
        
        if (self.recurrentWeights[0:2,0].all() == 0):
            self.recurrentWeights[0,0] = self.dataSet[0]
            self.recurrentWeights[1,0] = self.dataSet[1]
            if (self.regulated):
                self.alabels[0, int(self.labelSet[0])] = 1
                self.alabels[1, int(self.labelSet[1])] = 1
        
        previousBMU = np.zeros((1,self.numWeights,self.dimension))
        previousIndex = -1
        cu_qrror = np.zeros(self.samples)
        cu_fcounter = np.zeros(self.samples)
        print ("++++", "EE:", self.maxEpochs, "WC:", self.numWeights, "IT:", self.insertionThreshold, "LR:", self.epsilon_b, self.epsilon_n, "NN:", self.numNodes)

        # Start training
        for epoch in range(0, self.maxEpochs):
            self.updateRate = 0
            for iteration in range(0, self.samples):
                self.globalContext[0] = self.dataSet[iteration]
                label = self.labelSet[iteration]

                # Update global context
                for z in range(1, self.numWeights):
                    self.globalContext[z] = (self.varBeta * previousBMU[0,z]) + ((1-self.varBeta) * previousBMU[0,z-1])
                
                # Find the best and second-best matching neurons
                distances = np.zeros(self.numNodes)
                for i in range(0, self.numNodes):
                    gammaDistance = 0.0
                    for j in range(0, self.numWeights):
                        gammaDistance += (self.varAlpha[j] * (np.sqrt(np.sum((self.globalContext[j] - self.recurrentWeights[i,j])**2))))
                    distances[i] = gammaDistance
                    
                firstIndex = np.argmin(distances)
                firstDistance = distances[firstIndex]
                distances[firstIndex] = 99999
                secondIndex = np.argmin(distances)
                
                #sort_index = np.argsort(distances)
                #firstIndex = sort_index[0]
                #firstDistance = distances[firstIndex]
                #secondIndex = sort_index[1]

                winnerLabelIndex = np.argmax(self.alabels[firstIndex])
        
                #spatialQE = np.sqrt(np.sum((globalContext[0]+recurrentWeights[firstIndex,0])**2))
                previousBMU[0] = self.recurrentWeights[firstIndex]
                self.ages += 1
                
                # Compute network activity
                cu_qrror[iteration] = firstDistance
                h = self.habn[firstIndex]
                cu_fcounter[iteration] = h
                a = math.exp(-firstDistance)
                
                if ((not self.regulated) and (a < self.insertionThreshold) and (h < self.habThreshold) and (self.numNodes < self.maxNodes)) or ((self.regulated) and (label!=winnerLabelIndex) and (h < self.habThreshold) and (self.numNodes < self.maxNodes)):
                    # Add new neuron
                    newRecurrentWeight = np.zeros((1,self.numWeights,self.dimension))
                    for i in range(0, self.numWeights):
                        newRecurrentWeight[0,i] = np.array([np.dot(self.recurrentWeights[firstIndex,i] + self.globalContext[i], 0.5)])
                    self.recurrentWeights = np.concatenate((self.recurrentWeights,newRecurrentWeight),axis=0)
                   
                    newAlabel = np.zeros((1,self.numOfClasses))
                    if (label!=-1):
                        newAlabel[0,int(label)] = self.aIncreaseFactor
                    self.alabels = np.concatenate((self.alabels,newAlabel),axis=0)
                    
                    newIndex = self.numNodes
                    self.numNodes += 1
                    self.habn.resize(self.numNodes)
                    self.habn[newIndex] = 1
                    self.temporal.resize((self.numNodes, self.numNodes))
                    
                    # Update edges
                    self.edges.resize((self.numNodes,self.numNodes))
                    self.edges[firstIndex, secondIndex] = 0
                    self.edges[secondIndex, firstIndex] = 0
                    self.edges[firstIndex, newIndex] = 1
                    self.edges[newIndex, firstIndex] = 1
                    self.edges[newIndex, secondIndex] = 1
                    self.edges[secondIndex, newIndex] = 1
                        
                    # Update ages
                    self.ages.resize((self.numNodes,self.numNodes))
                    self.ages[firstIndex, newIndex] = 0
                    self.ages[newIndex, firstIndex] = 0
                    self.ages[newIndex, secondIndex] = 0
                    self.ages[secondIndex, newIndex] = 0

                else:
                    
                    updateRate_b = self.epsilon_b
                    updateRate_n = self.epsilon_n
                        
                    if (self.regulated) and (label!=winnerLabelIndex):
                            updateRate_b *= 0.01
                            updateRate_n *= 0.01
                    else:
                        # Adapt label histogram
                        self.updateLabelHistogram(firstIndex, label)
                        
                    # Adapt weights and context descriptors
                    self.updateNeuron(firstIndex, updateRate_b)
                    self.updateRate += updateRate_b * self.habn[firstIndex]
                                    
                    # Habituate BMU            
                    self.habituateNeuron(firstIndex, self.tau_b)
                    
                    # Update ages
                    self.ages[firstIndex, secondIndex] = 0
                    self.ages[secondIndex, firstIndex] = 0
                    
                    # Update edges // Remove oldest ones
                    self.updateEdges(firstIndex, secondIndex)
                    self.updateEdges(secondIndex, firstIndex)
                    
                    # Update topological neighbours
                    neighboursFirst = np.nonzero(self.edges[firstIndex])
                    for z in range(0, len(neighboursFirst[0])):
                        neIndex = neighboursFirst[0][z]
                        self.updateNeuron(neIndex, updateRate_n)
                        self.habituateNeuron(neIndex, self.tau_n)
                
                # Update temporal connections    
                if (previousIndex != -1) and (previousIndex != firstIndex):
                    self.temporal[previousIndex, firstIndex] += 1
                previousIndex = firstIndex
                 
            # Remove old edges
            #self.removeOldEdges()
                        
            self.nNN[epoch] = self.numNodes
            self.qrror[epoch,0] = np.mean(cu_qrror)
            self.qrror[epoch,1] = np.std(cu_qrror)
            self.fcounter[epoch,0] = np.mean(cu_fcounter)
            self.fcounter[epoch,1] = np.std(cu_fcounter)
            self.updateRate = self.updateRate / self.samples
        
            print ("- SM (E:", epoch+1,", NN:",self.numNodes, "UR:",self.updateRate, ", TQE:", self.qrror[epoch,0],")")
                
        # Remove isolated neurons
        #if (context):
        #    self.removeIsolatedNeurons()

    # REPLAY ##################################################################
    def replay(self, maxEpochs, insertionT, beta, epsilon_b, epsilon_n, context):  
        for i in range(0, self.numNodes):
            pseudoData, pseudoLabels = self.activationTrajectories(self.numWeights, i)
            self.train(pseudoData, pseudoLabels, maxEpochs, insertionT, beta, epsilon_b, epsilon_n, context)
        
    def activationTrajectories(self, size, onsetNeuron):
        pseudoDataPoints = np.zeros((size,self.dimension))
        pseudoLabels = np.zeros((size,1))
        indices = np.zeros((size,1))
        
        indices[0] = onsetNeuron
        pseudoDataPoints[0] = self.recurrentWeights[onsetNeuron][0]
        pseudoLabels[0] = np.argmax(self.alabels[onsetNeuron])
        
        for r in range(1, size):
            indices[r] = np.argmax(self.temporal[int(indices[r-1]), :])
            pseudoDataPoints[r] = self.recurrentWeights[int(indices[r])][0]
            pseudoLabels[r] = np.argmax(self.alabels[int(indices[r])])
            
        return pseudoDataPoints, pseudoLabels
    
    # Test GWR ################################################################ 
    
    def predict( self, dataSet, useContext ):
        print ("Predicting ..."),
        distances = np.zeros(self.numNodes)
        wSize = dataSet.shape[0]
        bmuWeights = np.zeros((wSize, self.dimension))
        bmuActivation = np.zeros(wSize)
        bmuLabelClasses = -np.ones(wSize)
        inputContext = np.zeros((self.numWeights, self.dimension))
        
        if (useContext):
            
            for ti in range(0, wSize):
                inputContext[0] = dataSet[ti]
                
                for i in range(0, self.numNodes):
                    gammaDistance = 0.0
                    for j in range(0, self.numWeights):
                        gammaDistance += (1 * (np.sqrt(np.sum((inputContext[j] - self.recurrentWeights[i,j])**2))))
                    distances[i] = gammaDistance
                    
                firstIndex = np.argmin(distances)
                bmuWeights[ti] = self.recurrentWeights[firstIndex,0]
                bmuActivation[ti] = math.exp(-distances[firstIndex])
                bmuLabelClasses[ti] = np.argmax(self.alabels[firstIndex,:])

                for i in range(1, self.numWeights):
                    inputContext[i] = inputContext[i-1]
        else:
            
            for ti in range(0, wSize):
                inputSample = dataSet[ti]
                
                for i in range(0, self.numNodes):
                    distances[i] = np.linalg.norm(inputSample-self.recurrentWeights[i,0])
                    
                firstIndex = np.argmin(distances)
                bmuWeights[ti] = self.recurrentWeights[firstIndex,0]
                bmuActivation[ti] = math.exp(-distances[firstIndex])
                bmuLabelClasses[ti] = np.argmax(self.alabels[firstIndex,:])
            
        return bmuWeights, bmuActivation, bmuLabelClasses

    def computeAccuracy(self, bmuLabel, labelSet):
        wSize = len(bmuLabel)
        counterAcc = 0
        
        for i in range(0, wSize):
            if (bmuLabel[i]==labelSet[i]):
                counterAcc +=1
        
        return (counterAcc * 100) / wSize
