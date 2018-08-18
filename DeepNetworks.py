import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as sk
import os.path
import sys
import subprocess
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import math
from sklearn.preprocessing import MinMaxScaler

class DeepNetworks:

    def __init__(self,dataObj, Spec, trainCycle):

        self.hist = Spec.modelHistory
        self.LSTM = Spec.LSTM
        self.LSTM_Layers = Spec.LSTM_Layers
        self.signalAmount = dataObj.data.shape[1]
        self.horizont = Spec.horizont
        self.interval = Spec.horizont
        self.trainCycle = trainCycle
        self.dataObj = dataObj
        self.scaling = Spec.scale
        self.train, self.test = dataObj.createSets()
        self.trainIn, self.trainOut = dataObj.createInputOutput(self.train, self.hist, self.horizont, self.interval)
        self.testIn, self.testOut = dataObj.createInputOutput(self.test, self.hist, self.horizont, self.interval)

    def modelling(self):

        # create network
        #toDo: choose between models
        self.model = Sequential()
        #self.model.add(LSTM(self.hist*4, input_shape = (1,self.hist*4), return_sequences=True))
        #self.model.add(LSTM(300, return_sequences=False))
        self.model.add(Dense(self.hist*self.signalAmount, input_shape = (1,self.hist*self.signalAmount)))
        if self.LSTM:
            self.model.add(LSTM(int(self.LSTM_Layers), return_sequences=True))
            self.model.add(LSTM(int(self.LSTM_Layers), return_sequences=True))
            self.model.add(LSTM(int(self.LSTM_Layers), return_sequences=True))
        #self.model.add(LSTM(self.horizont, return_sequences=True))
        self.model.add(Dense(self.horizont, init='uniform'))

        print(self.model.summary())

    def fitting(self):

        # (number of examples, number of timesteps, observations)

        self.trainIn= self.trainIn.reshape(self.trainIn.shape[0],1,self.trainIn.shape[1])
        self.testIn= self.testIn.reshape(self.testIn.shape[0],1,self.testIn.shape[1])
        self.trainOut= self.trainOut.reshape(self.trainOut.shape[0],1,self.trainOut.shape[1])
        self.testOut= self.testOut.reshape(self.testOut.shape[0],1,self.testOut.shape[1])       

        # Compile model
        self.model.compile(loss='mean_squared_error', optimizer='adam')           
        # Fit the model
        self.model.fit(self.trainIn,self.trainOut, nb_epoch= self.trainCycle, batch_size = 1)

    def predict(self):

        self.trainPred = list()
        self.testPred = list()

        # # history datapoints * # input series 
        boxLen = self.hist*self.dataObj.testPre.shape[1]

        for i in range(len(self.trainIn)):
            self.trainPred.append(self.model.predict(self.trainIn[i].reshape(1,1,boxLen)))
        for i in range(len(self.testIn)):
            self.testPred.append(self.model.predict(self.testIn[i].reshape(1,1,boxLen)))
            
        # invert predictions
        #Min,Max for inverse normalization 
        
        if self.scaling:
            Min1=min(np.array(self.dataObj.trainPre)[:,0])
            Min2=min(np.array(self.dataObj.testPre)[:,0])
            Max1=max(np.array(self.dataObj.trainPre)[:,0])
            Max2=max(np.array(self.dataObj.testPre)[:,0])
            self.trainPredN = np.array(self.trainPred)*(Max1-Min1) + Min1
            self.testPredN = np.array(self.testPred)*(Max2-Min2) + Min2
            self.trainOutN =  np.array(self.trainOut)*(Max1-Min1) + Min1
            self.testOutN = np.array(self.testOut)*(Max2-Min2) + Min2
        else:
            self.trainPredN = np.array(self.trainPred)
            self.testPredN = np.array(self.testPred)
            self.trainOutN =  np.array(self.trainOut)
            self.testOutN = np.array(self.testOut)
    

        return self.trainPredN,self.trainOutN, self.testPredN, self.testOutN