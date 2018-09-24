# -*- coding: utf-8 -*-
"""
Growing Dual-Memory Networks (Batch Learning)

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
    dataFlag = 0            # Import dataset from file
    importFlag = 0          # Import saved network
    trainFlag = 1           # Train model
    testEpochs = 0          # Compute classification accuracy over epochs
    testContext = 1         # Use temporal context during test
    plotFlag = 0            # Plot 2D map
        
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

        # Initiliaze networks
        myEpisodicGWR = EpisodicGWR()
        myEpisodicGWR.initNetwork(vecDim, numWeights=3, numClasses=10, numInstances=50)
            
        mySemanticGWR = SemanticGWR()
        mySemanticGWR.initNetwork(vecDim, numWeights=3, numClasses=10)
            
        ee = 35             # Number of training epochs
        iT = [0.3, 0.001]   # insertion thresholds
        lR = [0.5, 0.005]   # learning rates
        bP = 0.7            # beta parameter
        
        if (testEpochs):
        
            for e in range(0, ee):  
                myEpisodicGWR.train(trainingVectors, trainingLabels, 1, iT[0], bP, lR[0], lR[1], context=1, regulated=0)
                emBmuWeights, emBmuActivation, emBmuLabelClasses, emBmuLabelInstances = myEpisodicGWR.predict(trainingVectors, 1)
                mySemanticGWR.train(emBmuWeights, emBmuLabelClasses, 1, iT[1], bP, lR[0], lR[1], context=1, regulated=1)
            
                emBmuWeights, emBmuActivation, emBmuLabelClasses, emBmuLabelInstances = myEpisodicGWR.predict(testVectors, testContext)
                emAccuracy = myEpisodicGWR.computeAccuracy(emBmuLabelInstances, testLabels[:,1])
            
                smBmuWeights, smBmuActivation, smBmuLabelClasses = mySemanticGWR.predict(emBmuWeights, testContext)
                smAccuracy = mySemanticGWR.computeAccuracy(smBmuLabelClasses, testLabels[:,0])
                
                print ("Epoch:", e+1, "Accuracy (EM, SM):", emAccuracy, smAccuracy)
        else:
            
            myEpisodicGWR.train(trainingVectors, trainingLabels, ee, iT[0], bP, lR[0], lR[1], context=1, regulated=0)
            emBmuWeights, emBmuActivation, emBmuLabelClasses, emBmuLabelInstances = myEpisodicGWR.predict(trainingVectors, 1)
            mySemanticGWR.train(emBmuWeights, emBmuLabelClasses, ee, iT[1], bP, lR[0], lR[1], context=1, regulated=1)
            
            emBmuWeights, emBmuActivation, emBmuLabelClasses, emBmuLabelInstances = myEpisodicGWR.predict(testVectors, testContext)
            emAccuracy = myEpisodicGWR.computeAccuracy(emBmuLabelInstances, testLabels[:,1])
        
            smBmuWeights, smBmuActivation, smBmuLabelClasses = mySemanticGWR.predict(emBmuWeights, testContext)
            smAccuracy = mySemanticGWR.computeAccuracy(smBmuLabelClasses, testLabels[:,0])
            
            print ("Accuracy (EM, SM):", emAccuracy, smAccuracy)