# -*- coding: utf-8 -*-
"""
Dual-memory Batch learning
@last-modified: 23 October 2018
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
    dataFlag = 1            # Load dataset
    trainFlag = 1           # Train model
    testEpochs = 0          # Compute classification accuracy over epochs
    testContext = 1         # Test using temporal context

    if (dataFlag):
        
        ds = CORe50()
        ds.loadData()
        
        print ("%s loaded." % ds.sName)

    if (trainFlag):
        
        numWeights = [3, 3] # size of temporal receptive field
        ee = 35             # number of training epochs
        iT = [0.3, 0.001]   # insertion thresholds (Episodic and Semantic)
        lR = [0.5, 0.005]   # learning rates (BMU and neighbors)
        bP = 0.7            # beta parameter

        # Initiliaze networks
        myEpisodicGWR = EpisodicGWR()
        myEpisodicGWR.initNetwork(ds.vecDim, numWeights[0], ds.numClasses, ds.numInstances)
            
        mySemanticGWR = SemanticGWR()
        mySemanticGWR.initNetwork(ds.vecDim, numWeights[1], ds.numClasses)
        
        ef = ee if (testEpochs) else 1
        et = 1 if (testEpochs) else ee
        
        for e in range(0, ef):

            myEpisodicGWR.train(ds.trainingVectors, ds.trainingLabels, et, iT[0], bP, lR[0], lR[1], context=1, regulated=0)
            emBmuWeights, emBmuActivation, emBmuLabelClasses, emBmuLabelInstances = myEpisodicGWR.predict(ds.trainingVectors, 1)
            mySemanticGWR.train(emBmuWeights, emBmuLabelClasses, et, iT[1], bP, lR[0], lR[1], context=1, regulated=1)
            
            if (testEpochs) or (not testEpochs and e == ef):
        
                emBmuWeights, emBmuActivation, emBmuLabelClasses, emBmuLabelInstances = myEpisodicGWR.predict(ds.testVectors, testContext)
                emAccuracy = myEpisodicGWR.computeAccuracy(emBmuLabelInstances, ds.testLabels[:,1])
            
                smBmuWeights, smBmuActivation, smBmuLabelClasses = mySemanticGWR.predict(emBmuWeights, testContext)
                smAccuracy = mySemanticGWR.computeAccuracy(smBmuLabelClasses, ds.testLabels[:,0])
                
                print ("Epoch: %s, EM: %s, SM: %s" % ((e+1), emAccuracy, smAccuracy))
