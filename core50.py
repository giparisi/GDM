# -*- coding: utf-8 -*-
"""
@last-modified: 20 October 2018
@author: German I. Parisi (german.parisi@gmail.com)
"""

import csv
import numpy as np

class CORe50:
    
    def loadFile(self, fn) -> np.ndarray:
        reader = csv.reader(open(fn,"r"),delimiter=',')
        x = list(reader)
        vec = np.array(x).astype('float')
        return  vec
    
    def divCategories(self, vectorSet) -> np.ndarray:
        c = 0
        segData = 0
        segIndex = 0
        ite = np.zeros((self.numClasses, self.numLabels))
        
        for i in range(0, vectorSet.shape[0]):
            if (vectorSet[i, self.categoryLabelIndex] != segData) or (i == vectorSet.shape[0]-1):                
                ite[c,0] = segIndex
                ite[c,1] = i
                c += 1
                segData = vectorSet[i, self.categoryLabelIndex]
                segIndex = i
                
        return ite
        
    def loadData(self) -> None:
        # numClasses and numInstances are set here for simplicity but the
        # model does not require these values to be fixed
        self.sName = 'CORe50'
        self.numClasses = 10
        self.numInstances = 50
        self.numLabels = 2
        self.vectorIndex  = [0, 256]
        self.categoryLabelIndex = 258
        self.instanceLabelIndex = 259
        
        trainingSet = self.loadFile("VGG16-5fps-training-classes.csv")
        testSet = self.loadFile("VGG16-5fps-test-classes.csv")

        self.iTr = self.divCategories(trainingSet)
        self.iTe = self.divCategories(testSet)
                
        # Pre-process samples and labels # 0 - classes (258), 1 - instances (259)
        self.trainingVectors = trainingSet[:, self.vectorIndex[0]:self.vectorIndex[1]]
        self.trainingLabels = trainingSet[:, self.categoryLabelIndex:(self.categoryLabelIndex+self.numLabels)]
        self.vecDim = self.trainingVectors.shape[1]

        self.testVectors = testSet[:, self.vectorIndex[0]:self.vectorIndex[1]]
        self.testLabels = testSet[:, self.categoryLabelIndex:(self.categoryLabelIndex+self.numLabels)]