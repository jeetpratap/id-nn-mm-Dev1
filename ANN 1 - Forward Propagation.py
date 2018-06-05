# -*- coding: utf-8 -*-
"""
Created on Thu Nov 09 17:37:37 2017

@author: JPRAT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.array(([3,5],[5,1],[10,2]),dtype=float)
y = np.array(([75],[82],[93]),dtype=float)
x
y


class Neural_Network(object):
    def __init__(self):
        # Define Hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3
        
        #Weights
        self.W1 = np.random.randn(self.inputLayerSize, \
                                  self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,\
                                  self.outputLayerSize)
        
    def forward(self, X):
        #Propogate inputs through network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat
    
    def sigmoid(self, z):
        #Apply Sigmoid function to scalar, vector or
        return 1/(1+np.exp(-z))
        
NN = Neural_Network()
yHat = NN.forward(x)
yHat




        
