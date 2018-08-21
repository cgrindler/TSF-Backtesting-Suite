import numpy as np
import pandas as pd
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
        self.BackTest = Specs.BackTest
        self.dataIntervall = dataObj.data[dataObj.data.columns[int(self.ForecastCol)]]
        if not Specs.exogCol == "":

            self.exog = dataObj.data[dataObj.data.columns[int(Specs.exogCol)]]
        self.filterweight = Specs.filterweight

    def modelling(self):
        
        # for live operation there is a problem with the 
        # filtering approach: edges a wrongly calculated
        # in general filtering is tricky to implement for
        # operational use
          
        if self.filterweight is 0 or self.BackTest == False:
            mdlData = self.dataIntervall.astype(float)
        else:
            mdlData = sm.tsa.filters.hpfilter(self.dataIntervall,self.filterweight)[1];        
            
        self.mdl = sm.tsa.SARIMAX(mdlData, order=self.order, seasonal_order=self.sorder, enforce_stationarity=False,
                             enforce_invertibility=False,exog = self.exog)      

    def fitting(self, mdlName, folderPath, method="lbfgs"):
        
        mdlpath = folderPath+ "/"+mdlName
        sys.stdout.flush()
        
        if not os.path.isfile(mdlpath): 
            
            print("\nestimation of sarimax model "+str(self.order)+"x"+str(self.sorder)+"\n\n")
            sys.stdout.flush() 
            
            try:
                self.fitted = self.mdl.fit(method=method,maxiter=200)

                ### algorithms/methods ###

                #- 'newton' for Newton-Raphson, 
                #- 'nm' for Nelder-Mead 
                #- 'bfgs' for Broyden-Fletcher-Goldfarb-Shanno (BFGS)
                #- 'lbfgs' for limited-memory BFGS with optional box constraints //default
                #- 'powell' for modified Powell's method
                #- 'cg' for conjugate gradient
                #- 'ncg' for Newton-conjugate gradient
                #- 'basinhopping' for global basin-hopping solver

            except Exception as e:
                print("could not estimate model parameters:")
                print(e)
                sys.stdout.flush()
                return

            try:

                self.saveit(folderPath, mdlName)
                
            except Exception as e:
                print("saving the model caused problems - please debug the program")    
                print(e)  
                sys.stdout.flush()
                return            
                #self.fit = None
                #self.fit = sm.load(path)
            
        else:
            self.fitted = sm.load(mdlpath)
            print("model loaded: " +  mdlpath)

        try: #no clue why you can't filter after saving in the same thread
            self.filt = self.mdl.filter(self.fitted.params)
        except:
            print("model fitted and saved, the program will restart for results")
            python = sys.executable
            os.execl(python, python, * sys.argv)
        

    def predictDyn(self,nstep,n,delay,exog=None,anzahl=10):
        
        # calculates cyclically forecasts over 
        # given historical data bound to specific
        # step size

        CastContainer = pd.Series()
        # number of n-step forecasts
        bis = int(anzahl) 
        for x in range(0,0+bis): 
            
            #start[i]: season + horizont + horizont*i + delay
            startPr = n * nstep + nstep * x + delay
            if exog is None:
                CastContainer = CastContainer.append(self.filt.predict(start=startPr,end = startPr + nstep - 1,dynamic=True)) 
            else:    
                CastContainer = CastContainer.append(self.filt.predict(start=startPr,end = startPr + nstep - 1,dynamic=True, exog = exog.ix[startPr:startPr + nstep - 1])) 
        
        return CastContainer                                                                                      
    
    def predict1Step(self):

        # output of model estimation process
        self.predicted = self.filt.get_prediction()
        self.predicted.predicted_mean[self.predicted.predicted_mean < 0] = 0
        self.confint = self.predicted.conf_int()
        return self.predicted

    def predictOperative(self, step):

        # looks for incoming data length and calculates 
        # "out-of-sample" forecast behind last timestamp
        # only statsmodels => 0.9.0 supports real out-of-sample forecast
        # error with statsmodels 0.8.0
        # TBD

        start = len(self.dataIntervall) 
        try:
            return self.filt.predict(start=start, end=start+step-1, dynamic = True)
        except Exception as e:
            print("problems while generating forecast." + e)
            return None
 
    def saveit(self, folderPath, mdlName):
        
        if not os.path.isdir(folderPath):
            try:                
                os.makedirs(folderPath)
                print("new model folder created.")
            except: print("no permissions. please create model folder by your own.")

        self.fitted.save(folderPath+"/"+mdlName,remove_data=True)
        print("model saved: " + str(os.path.isfile(folderPath+"/"+mdlName)))



