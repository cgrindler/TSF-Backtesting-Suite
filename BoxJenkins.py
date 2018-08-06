import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sklearn.metrics as sk
import os.path
import sys
import Specification
import subprocess
import math
from sklearn.preprocessing import MinMaxScaler


class BoxJenkins:

    def __init__(self, dataObj, Specs):
        self.order = Specs.order
        self.sorder = Specs.sorder
        self.ForecastCol = Specs.ForecastCol
        self.exog = None
        #toDO add support for exogenous variables
        self.dataIntervall = dataObj.data[dataObj.data.columns[int(self.ForecastCol)]]

        if not Specs.exogCol == "":
            self.exog = dataObj.data[dataObj.data.columns[int(Specs.exogCol)]]

        self.filterweight = Specs.filterweight

    def modelling(self):
        
        if self.filterweight is 0:
            mdlData = self.dataIntervall.astype(float)
            #mdlData.values = mdlData.values.astype(float)
        
        else:
            mdlData = sm.tsa.filters.hpfilter(self.dataIntervall,self.filterweight)[1];        
            
        self.mdl = sm.tsa.SARIMAX(mdlData, order=self.order, seasonal_order=self.sorder, enforce_stationarity=False,
                             enforce_invertibility=False,exog = self.exog)      

    def fitting(self, mdlName, folderPath, method="lbfgs"):
        
        mdlpath = folderPath+ "/"+mdlName

        print(mdlpath)
        if not os.path.isfile(mdlpath): #falls Modell noch nicht erstellt worden ist
            
            print("\nEstimation of Sarimax-Model "+str(self.order)+"x"+str(self.sorder)+"\n\n")
            
            try:
                self.fitted = self.mdl.fit(method=method,maxiter=1000) 
                
                ### algorithms/methods ###

                #- 'newton' for Newton-Raphson, 
                #- 'nm' for Nelder-Mead 
                #- 'bfgs' for Broyden-Fletcher-Goldfarb-Shanno (BFGS)
                #- 'lbfgs' for limited-memory BFGS with optional box constraints //default
                #- 'powell' for modified Powell's method
                #- 'cg' for conjugate gradient
                #- 'ncg' for Newton-conjugate gradient
                #- 'basinhopping' for global basin-hopping solver
            except:
              
                print("Could not estimate model parameters - try another SARIMAX model")
                os.system("pause")
                sys.exit()

            try:

                self.saveit(folderPath,mdlName)
            except:
                print("Saving the model caused problems - please debug the program")
                os.system("pause")
                sys.exit()                              
                #self.fit = None
                #self.fit = sm.load(path)
            
        else:
            self.fitted = sm.load(mdlpath)
   
        try: #no clue why you can't filter after saving in the same thread
            self.filt = self.mdl.filter(self.fitted.params)
        except:
            print("Model fitted and saved, the program will restart for results")
            python = sys.executable
            os.execl(python, python, * sys.argv)
        

    def predictDyn(self,nstep,n,hourOfDay,exog=None,anzahl=10):
        

        CastContainer = pd.Series()

        bis = int(anzahl) # number of n-step forecasts
        for x in range(0,0+bis): # für Lags in Prognosemodell wird ien 10 Tages Delay eingeführt
            
            #start = Season + Horizont + Horizont*i + delay
            
            startPr = n * nstep + nstep * x + hourOfDay
            if exog is None:
                CastContainer = CastContainer.append(self.filt.predict(start=startPr,end = startPr + nstep - 1,dynamic=True)) 
            else:    
                CastContainer = CastContainer.append(self.filt.predict(start=startPr,end = startPr + nstep - 1,dynamic=True, exog = exog.ix[startPr:startPr + nstep - 1])) 
        
        return CastContainer                                                                                      
    
    def predict1Step(self):
       

        self.predicted = self.filt.get_prediction()
        self.predicted.predicted_mean[self.predicted.predicted_mean < 0] = 0
        #self.predictdf = self.predict.predicted_mean.to_frame(name="Value")
        self.confint = self.predicted.conf_int()
        return self.predicted

                       
    def saveit(self, folderPath, mdlName):
        
        if not os.path.isdir(folderPath):
            try:                
                os.makedirs(folderPath)
                print("New Model-Folder created.")
            except: print("No Permissions. Please create Model-folder by your own.")

        self.fitted.save(folderPath+"/"+mdlName,remove_data=True)
        print("Model saved: " + str(os.path.isfile(folderPath+"/"+mdlName)))



