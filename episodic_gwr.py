# -*- coding: utf-8 -*-
"""
Episodic Gamma-GWR
@last-modified: 20 October 2018
@author: German I. Parisi (german.parisi@gmail.com)
Please cite this paper: Parisi, G.I., Tani, J., Weber, C., Wermter, S. (2018) Lifelong Learning of Spatiotemporal Representations with Dual-Memory Recurrent Self-Organization. arXiv:1805.10966
"""

import numpy as np
import math

class EpisodicGWR:

    def initNetwork(self, dimension, numWeights, numClasses, numInstances) -> None:
        self.numNodes = 2
        self.dimension = dimension
        self.numWeights = numWeights
        self.numOfClasses = numClasses
        self.numOfInstances = numInstances
        self.recurrentWeights = np.zeros((self.numNodes, self.numWeights, self.dimension))
        self.labelClasses = np.zeros((self.numNodes, self.numOfClasses))
        self.labelInstances = np.zeros((self.numNodes, self.numOfInstances))
        self.globalContext = np.zeros((self.numWeights, self.dimension))
        self.edges = np.zeros((self.numNodes, self.numNodes))
        self.ages = np.zeros((self.numNodes, self.numNodes))
        self.habn = np.ones(self.numNodes)
        self.temporal = np.zeros((self.numNodes, self.numNodes))
        self.varAlpha = self.gammaWeights(self.numWeights)
        self.updateRate = 0
 
    def gammaWeights (self, nw) -> np.ndarray:
        iWe = np.zeros(nw)
        for h in range(0, len(iWe)):
            iWe[h] = np.exp(-h)
        iWe[:] = iWe[:] / sum(iWe)
        return iWe
            
    def habituateNeuron(self, index, tau) -> None:
            self.habn[index] += (tau * 1.05 * (1. - self.habn[index]) - tau)

    def updateNeuron(self, index, epsilon):
        deltaWeights = np.zeros((self.numWeights, self.dimension))
        for i in range(0, self.numWeights):
            deltaWeights[i] = np.array([np.dot((self.globalContext[i]-self.recurrentWeights[index,i]), epsilon)]) * self.habn[index]
        self.recurrentWeights[index] += deltaWeights
            
    def updateLabelHistogram(self, bmu, labelClass, labelInstance) -> None:         
        for a in range(0, self.numOfClasses):
            if (a == labelClass):
                self.labelClasses[bmu, a] += self.aIncreaseFactor
            else:
                self.labelClasses[bmu, a] -= self.aDecreaseFactor
                if (self.labelClasses[bmu, a] < 0):
                    self.labelClasses[bmu, a] = 0
                    
        for a in range(0, self.numOfInstances):
            if (a == labelInstance):
                self.labelInstances[bmu, a] += self.aIncreaseFactor
            else:
                self.labelInstances[bmu, a] -= self.aDecreaseFactor
                if (self.labelInstances[bmu, a] < 0):
                    self.labelInstances[bmu, a] = 0
        
    def updateEdges(self, fi, si) -> None:
        neighboursFirst = np.nonzero(self.edges[fi])
        if (len(neighboursFirst[0]) >= self.maxNeighbours):
            remIndex = -1
            maxAgeNeighbour = 0
            for u in range(0, len(neighboursFirst[0])):
                if (self.ages[fi, neighboursFirst[0][u]] > maxAgeNeighbour):
                    maxAgeNeighbour = self.ages[fi, neighboursFirst[0][u]]
                    remIndex = neighboursFirst[0][u]
            self.edges[fi, remIndex] = 0
            self.edges[remIndex, fi] = 0
        self.edges[fi, si] = 1

    def removeOldEdges(self) -> None:
        for i in range(0, self.numNodes):
            neighbours = np.nonzero(self.edges[i])
            for j in range(0, len(neighbours[0])):
                if (self.ages[i, j] >=  self.maxAge):
                    self.edges[i, j] = 0
                    self.edges[j, i] = 0

    def removeIsolatedNeurons(self) -> None:
        indCount = 0
        while (indCount < self.numNodes):
            neighbours = np.nonzero(self.edges[indCount])
            if (len(neighbours[0]) < 1):
                self.recurrentWeights = np.delete(self.recurrentWeights, indCount, axis=0)
                self.labelClasses = np.delete(self.labelClasses, indCount, axis=0)
                self.labelInstances = np.delete(self.labelInstances, indCount, axis=0)
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

    def train(self, dataSet, labelSet, maxEpochs, insertionT, beta, epsilon_b, epsilon_n, context, regulated) -> None:
        self.dataSet = dataSet
        self.samples = self.dataSet.shape[0]
        self.labelSet = labelSet
        self.maxEpochs = maxEpochs
        self.insertionThreshold = insertionT 
        self.varBeta = beta
        self.epsilon_b = epsilon_b
        self.epsilon_n = epsilon_n

        self.context = context
        if not self.context:
            self.globalContext.fill(0)

        self.regulated = regulated
        
        self.habThreshold = 0.1
        self.tau_b = 0.3
        self.tau_n = 0.1
        self.maxNodes = 10000
        self.maxAge = 1200
        self.distanceMax = 99999
        self.maxNeighbours = 6
        self.aIncreaseFactor = 1.
        self.aDecreaseFactor = 0.01

        self.nNN = np.zeros(self.maxEpochs)
        self.qrror = np.zeros((self.maxEpochs, 2))
        self.fcounter = np.zeros((self.maxEpochs, 2))
        
        if (self.recurrentWeights[0:2,0].all() == 0):
            self.recurrentWeights[0,0] = self.dataSet[0]
            self.recurrentWeights[1,0] = self.dataSet[1]

        previousBMU = np.zeros((1,self.numWeights,self.dimension))
        previousIndex = -1
        cu_qrror = np.zeros(self.samples)
        cu_fcounter = np.zeros(self.samples)
        print ("+++", "EE:", self.maxEpochs, "WC:", self.numWeights, "IT:", self.insertionThreshold, "LR:", self.epsilon_b, self.epsilon_n, "NN:", self.numNodes)

        # Start training
        for epoch in range(0, self.maxEpochs):
            self.updateRate = 0

            for iteration in range(0, self.samples):
                self.globalContext[0] = self.dataSet[iteration]
                
                labelClass = self.labelSet[iteration,0]
                labelInstance = self.labelSet[iteration,1]
                
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
                distances[firstIndex] = self.distanceMax
                secondIndex = np.argmin(distances)
                
                #sort_index = np.argsort(distances)
                #firstIndex = sort_index[0]
                #firstDistance = distances[firstIndex]
                #secondIndex = sort_index[1]
                
                winnerLabelIndex = np.argmax(self.labelInstances[firstIndex,:])
        
                #spatialQE = np.sqrt(np.sum((globalContext[0]+recurrentWeights[firstIndex,0])**2))
                previousBMU[0] = self.recurrentWeights[firstIndex]
                
                self.ages += 1
                
                # Compute network activity
                cu_qrror[iteration] = firstDistance
                h = self.habn[firstIndex]
                cu_fcounter[iteration] = h
                a = math.exp(-firstDistance)
                
                if ((not self.regulated) and (a < self.insertionThreshold) and (h < self.habThreshold) and (self.numNodes < self.maxNodes)) or ((self.regulated) and (labelInstance!=winnerLabelIndex) and (h < self.habThreshold) and (self.numNodes < self.maxNodes)):
                    # Add new neuron
                    newRecurrentWeight = np.zeros((1,self.numWeights,self.dimension))
                    for i in range(0, self.numWeights):
                        newRecurrentWeight[0,i] = np.array([np.dot(self.recurrentWeights[firstIndex,i] + self.globalContext[i], 0.5)])
                    self.recurrentWeights = np.concatenate((self.recurrentWeights, newRecurrentWeight), axis=0)
                   
                    newLabelClass = np.zeros((1, self.numOfClasses))
                    newLabelInstance = np.zeros((1, self.numOfInstances))
                    if (labelClass!=-1):
                        newLabelClass[0, int(labelClass)] = self.aIncreaseFactor
                    if (labelInstance!=-1):
                        newLabelInstance[0, int(labelInstance)] = self.aIncreaseFactor
                    self.labelClasses = np.concatenate((self.labelClasses, newLabelClass), axis=0)
                    self.labelInstances = np.concatenate((self.labelInstances, newLabelInstance), axis=0)
                    
                    newIndex = self.numNodes
                    self.numNodes += 1
                    self.habn.resize(self.numNodes)
                    self.habn[newIndex] = 1
                    self.temporal.resize((self.numNodes, self.numNodes))
                    
                    # Update edges
                    self.edges.resize((self.numNodes, self.numNodes))
                    self.edges[firstIndex, secondIndex] = 0
                    self.edges[secondIndex, firstIndex] = 0
                    self.edges[firstIndex, newIndex] = 1
                    self.edges[newIndex, firstIndex] = 1
                    self.edges[newIndex, secondIndex] = 1
                    self.edges[secondIndex, newIndex] = 1
                        
                    # Update ages
                    self.ages.resize((self.numNodes, self.numNodes))
                    self.ages[firstIndex, newIndex] = 0
                    self.ages[newIndex, firstIndex] = 0
                    self.ages[newIndex, secondIndex] = 0
                    self.ages[secondIndex, newIndex] = 0

                else:
                    
                    updateRate_b = self.epsilon_b
                    updateRate_n = self.epsilon_n
                        
                    if (self.regulated) and (labelInstance != winnerLabelIndex):
                            updateRate_b *= 0.01
                            updateRate_n *= 0.01
                    else:
                        # Adapt label histogram
                        self.updateLabelHistogram(firstIndex, labelClass, labelInstance)

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
        
            print ("+ EM (E:", epoch+1,", NN:",self.numNodes, "UR:",self.updateRate, ", TQE:", self.qrror[epoch,0],")")
                
        # Remove isolated neurons
        #if (context):
        #    self.removeIsolatedNeurons()

    # Memory replay ################################################################
    def replayData(self, size) -> (np.ndarray, np.ndarray):
        indices = np.zeros(size)
        pWeights = np.zeros((self.numNodes, size, self.dimension))
        pLabels = np.zeros((self.numNodes, size, 2))
        
        for i in range(0, self.numNodes):
                indices[0] = i
                pWeights[i,0] = self.recurrentWeights[i,0]
                pLabels[i,0,0] = np.argmax(self.labelClasses[i])
                pLabels[i,0,1] = np.argmax(self.labelInstances[i])
                
                for r in range(1, size):
                    indices[r] = np.argmax(self.temporal[int(indices[r-1]), :])
                    pWeights[i,r] = self.recurrentWeights[int(indices[r]), 0]
                    pLabels[i,r,0] = np.argmax(self.labelClasses[int(indices[r])])
                    pLabels[i,r,1] = np.argmax(self.labelInstances[int(indices[r])])          

        return pWeights, pLabels
    
    # Test GWR ###################################################################
    def predict(self, dataSet, useContext) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        print ("Predicting ..."),
        distances = np.zeros(self.numNodes)
        wSize = dataSet.shape[0]
        bmuWeights = np.zeros((wSize, self.dimension))
        bmuActivation = np.zeros(wSize)
        bmuLabelClasses = -np.ones(wSize)
        bmuLabelInstances = -np.ones(wSize)
        inputContext = np.zeros((self.numWeights, self.dimension))
        
        if (useContext):
            for ti in range(0, wSize):
                inputContext[0] = dataSet[ti]
                
                for i in range(0, self.numNodes):
                    gammaDistance = 0.0
                    for j in range(0, self.numWeights):
                        gammaDistance += (self.varAlpha[j] * (np.sqrt(np.sum((inputContext[j] - self.recurrentWeights[i,j])**2))))
                    distances[i] = gammaDistance
                    
                firstIndex = np.argmin(distances)
                bmuWeights[ti] = self.recurrentWeights[firstIndex,0]
                bmuActivation[ti] = math.exp(-distances[firstIndex])
                bmuLabelClasses[ti] = np.argmax(self.labelClasses[firstIndex,:])
                bmuLabelInstances[ti] = np.argmax(self.labelInstances[firstIndex,:])

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
                bmuLabelClasses[ti] = np.argmax(self.labelClasses[firstIndex,:])
                bmuLabelInstances[ti] = np.argmax(self.labelInstances[firstIndex,:])
            
        return bmuWeights, bmuActivation, bmuLabelClasses, bmuLabelInstances

    def computeAccuracy(self, bmuLabel, labelSet) -> float:
        wSize = len(bmuLabel)
        counterAcc = 0
        
        for i in range(0, wSize):
            if (bmuLabel[i]==labelSet[i]):
                counterAcc +=1
        
        return counterAcc / wSize