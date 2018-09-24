# -*- coding: utf-8 -*-
"""
Dual-memory Incremental learning with memory replay (Python 3)

@last-modified: 8 September 2018

@author: German I. Parisi (german.parisi@gmail.com)

Please cite this paper: Parisi, G.I., Tani, J., Weber, C., Wermter, S. (2018) Lifelong Learning of Spatiotemporal Representations with Dual-Memory Recurrent Self-Organization. arXiv:1805.10966
"""

import csv
from episodic_gwr import EpisodicGWR
from semantic_gwr import SemanticGWR
import numpy as np
import random
import matplotlib.pyplot as plt

# Main ########################################################################

if __name__ == "__main__":
    # Set working path
    #os.getcwd()
    dataFlag = 0             # Import dataset from file
    importFlag = 0           # Import saved network
    trainFlag = 1            # Train AGWR with imported dataset
    saveFlag = 0             # Save trained network to file
    testFlag = 1             # Compute classification accuracy
    plotFlag = 0             # Plot 2D map
        
    if (dataFlag):
       
        print ("Loading training and test set...")
        
        numClasses = 10
        numInstances = 50
        
        reader = csv.reader(open("VGG16-training-classes.csv","r"),delimiter=',')
        x = list(reader)
        trainingSet = np.array(x).astype('float')

        reader = csv.reader(open("VGG16-test-classes.csv","r"),delimiter=',')
        x = list(reader)
        testSet = np.array(x).astype('float')
        
        iTr = np.zeros((numClasses,2))
        iTe = np.zeros((numClasses,2))

        c = 0
        segData = 0
        segIndex = 0
        for i in range(0, trainingSet.shape[0]):
            if (trainingSet[i, 258] != segData) or (i==trainingSet.shape[0]-1):                
                iTr[c,0] = segIndex
                iTr[c,1] = i
                c += 1
                segData = trainingSet[i, 258]
                segIndex = i

        c = 0
        segData = 0
        segIndex = 0
        for i in range(0, testSet.shape[0]):
            if (testSet[i, 258] != segData) or (i==testSet.shape[0]-1):                
                iTe[c,0] = segIndex
                iTe[c,1] = i
                c += 1
                segData = testSet[i, 258]
                segIndex = i
                
        # Pre-process samples and labels # 0: classes (258), 1: instances (259)
        trainingVectors = trainingSet[:,0:256]
        trainingLabels = trainingSet[:,258:260]
        vecDim = trainingVectors.shape[1]

        testVectors = testSet[:,0:256]
        testLabels = testSet[:,258:260]
                
    #incClasses = random.sample(range(0,numClasses), numClasses)#
   
    #iRun = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    #iRun = np.array([2, 7, 1, 8, 0, 3, 9, 6, 4, 5])
    #iRun = np.array([6, 9, 5, 7, 1, 0, 3, 2, 4, 8])
    #iRun = np.array([4, 1, 6, 8, 7, 9, 5, 2, 0, 3])
    iRun = np.array([9, 1, 2, 7, 8, 6, 4, 0, 3, 5])

    if (trainFlag):
        
            trainWithReplay = 1  # memory replay flag
            pseudoSize = 5       # size of RNATs

            # Initiliaze networks
            myEpisodicGWR = EpisodicGWR()
            myEpisodicGWR.initNetwork(vecDim, numWeights=3, numClasses=10, numInstances=50)
            
            mySemanticGWR = SemanticGWR()
            mySemanticGWR.initNetwork(vecDim, numWeights=3, numClasses=10)
            
            ee = 1              # Number of training epochs per mini-batch
            iT = [0.3, 0.001]    # insertion thresholds
            lR = [0.3, 0.003]   # learning rates
            bP = 0.7            # beta parameter
            
            accMatrix = np.zeros((numClasses, numClasses+1, 2))
            
            for c in range(0, numClasses):
                ci = int(iRun[c])
                ti = int(iTr[ci,0])
                te = int(iTr[ci,1])
                
                print("-- Training class",c)
                
                if (c>0):
                    regulated=1
                else:
                    regulated=0
            
                myEpisodicGWR.train(trainingVectors[ti:te], trainingLabels[ti:te], ee, iT[0], bP, lR[0], lR[1], 1, 0)
                emBmuWeights, emBmuActivation, emBmuLabelClasses, emBmuLabelInstances = myEpisodicGWR.predict(trainingVectors[ti:te], 1)
                mySemanticGWR.train(emBmuWeights, emBmuLabelClasses, ee, iT[1], bP, lR[0], lR[1], 1, regulated)
                                    
                if (trainWithReplay):                    
                    if (c>0):
                        for i in range(0, replayVectors.shape[0]):
                            myEpisodicGWR.train(replayVectors[i], replayLabels[i], ee, iT[0], bP, lR[0], lR[1], 0, 0)
                            mySemanticGWR.train(replayVectors[i], replayLabels[i,:,0], ee, iT[1], bP, lR[0], lR[1], 0, 1)

                    replayVectors, replayLabels = myEpisodicGWR.replayData(pseudoSize)
                
                for s in range(0, c+1):
                    si = int(iRun[s])
                    tti = int(iTe[si,0])
                    tte = int(iTe[si,1])

                    emBmuWeights, emBmuActivation, emBmuLabelClasses, emBmuLabelInstances = myEpisodicGWR.predict(testVectors[tti:tte], 1)
                    emAccuracy = myEpisodicGWR.computeAccuracy(emBmuLabelInstances, testLabels[tti:tte,1])
                
                    smBmuWeights, smBmuActivation, smBmuLabelClasses = mySemanticGWR.predict(emBmuWeights, 1)
                    smAccuracy = mySemanticGWR.computeAccuracy(smBmuLabelClasses, testLabels[tti:tte,0])

                    accMatrix[c,s,0] = emAccuracy
                    accMatrix[c,s,1] = smAccuracy
                    
                for m in range(0, 2):
                    aC = 0
                    for u in range(0, s+1):
                        
                        aC += accMatrix[c,u,m]
                    accMatrix[c,numClasses,m] = (aC) / (s+1)
                    #print(accMatrix[c,numClasses,m])

                print(accMatrix[:,10])
                print(accMatrix[:,0])

#            plt.plot(accMatrix[:,0,0])
#            plt.plot(accMatrix[:,0,1])
#            
#            plt.plot(accMatrix[:,numClasses,0])
#            plt.plot(accMatrix[:,numClasses,1])
            