# -*- coding: utf-8 -*-
"""
Dual-memory Incremental learning with memory replay
@last-modified: 20 October 2018
@author: German I. Parisi (german.parisi@gmail.com)
Please cite this paper: Parisi, G.I., Tani, J., Weber, C., Wermter, S. (2018) Lifelong Learning of Spatiotemporal Representations with Dual-Memory Recurrent Self-Organization. arXiv:1805.10966
"""

from episodic_gwr import EpisodicGWR
from semantic_gwr import SemanticGWR
from core50 import CORe50
import numpy as np
#import csv
#import random

# Main ########################################################################

if __name__ == "__main__":
    
    dataFlag = 1             # Import dataset from file
    trainFlag = 1            # Train AGWR with imported dataset
    testFlag = 1             # Compute classification accuracy
        
    if (dataFlag):
        
        ds = CORe50()
        ds.loadData()
        
        print ("%s loaded." % ds.sName)
                
    if (trainFlag):        
        #incClasses = random.sample(range(0,numClasses), numClasses)#
        #iRun = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        #iRun = np.array([6, 9, 5, 7, 1, 0, 3, 2, 4, 8])
        #iRun = np.array([4, 1, 6, 8, 7, 9, 5, 2, 0, 3])
        #iRun = np.array([9, 1, 2, 7, 8, 6, 4, 0, 3, 5])
        iRun = np.array([2, 7, 1, 8, 0, 3, 9, 6, 4, 5])
        
        numWeights = [3, 3]  # size of temporal receptive field
        ee = 1               # Number of training epochs per mini-batch
        iT = [0.3, 0.001]    # insertion thresholds
        lR = [0.5, 0.005]    # learning rates
        bP = 0.7             # beta parameter
        trainWithReplay = 1  # memory replay flag
        pseudoSize = 5       # size of RNATs

        # Initiliaze networks
        myEpisodicGWR = EpisodicGWR()
        myEpisodicGWR.initNetwork(ds.vecDim, numWeights[0], ds.numClasses, ds.numInstances)
        
        mySemanticGWR = SemanticGWR()
        mySemanticGWR.initNetwork(ds.vecDim, numWeights[1], ds.numClasses)
        
        replayVectors = []
        replayLabels = []

        accMatrix = np.zeros((ds.numClasses, ds.numClasses+1, ds.numLabels))
        
        for c in range(0, ds.numClasses):
            ci = int(iRun[c])
            ti = int(ds.iTr[ci,0])
            te = int(ds.iTr[ci,1])
            
            print("- Training class %s" % c)
            
            regulated = (c > 0)
            
            replayFlag = (trainWithReplay) and (c > 0)
            
            if (replayFlag):
                replayVectors, replayLabels = myEpisodicGWR.replayData(pseudoSize)
        
            myEpisodicGWR.train(ds.trainingVectors[ti:te], ds.trainingLabels[ti:te], ee, iT[0], bP, lR[0], lR[1], 1, 0)
            emBmuWeights, emBmuActivation, emBmuLabelClasses, emBmuLabelInstances = myEpisodicGWR.predict(ds.trainingVectors[ti:te], 1)
            
            mySemanticGWR.train(emBmuWeights, emBmuLabelClasses, ee, iT[1], bP, lR[0], lR[1], 1, regulated)
                                
            if (replayFlag):
                for i in range(0, replayVectors.shape[0]):
                    myEpisodicGWR.train(replayVectors[i], replayLabels[i], 1, iT[0], bP, lR[0], lR[1], 0, 0)
                    mySemanticGWR.train(replayVectors[i], replayLabels[i,:,0], 1, iT[1], bP, lR[0], lR[1], 0, 1)

            for s in range(0, c+1):
                
                si = int(iRun[s])
                tti = int(ds.iTe[si, 0])
                tte = int(ds.iTe[si, 1])

                emBmuWeights, emBmuActivation, emBmuLabelClasses, emBmuLabelInstances = myEpisodicGWR.predict(ds.testVectors[tti:tte], 1)
                emAccuracy = myEpisodicGWR.computeAccuracy(emBmuLabelInstances, ds.testLabels[tti:tte,1])
                
                smBmuWeights, smBmuActivation, smBmuLabelClasses = mySemanticGWR.predict(emBmuWeights, 1)
                smAccuracy = mySemanticGWR.computeAccuracy(smBmuLabelClasses, ds.testLabels[tti:tte,0])

                accMatrix[c, s, 0] = emAccuracy
                accMatrix[c, s, 1] = smAccuracy
                
            for m in range(0, ds.numLabels):
                aC = 0
                for u in range(0, s+1):
                    aC += accMatrix[c, u, m]
                accMatrix[c, ds.numClasses, m] = (aC) / (s+1)

            print(accMatrix[:, 10])
            print(accMatrix[:, 0])

            #plt.plot(accMatrix[:,0,0])
            #plt.plot(accMatrix[:,0,1])
                      
            #plt.plot(accMatrix[:,numClasses,0])
            #plt.plot(accMatrix[:,numClasses,1])
        