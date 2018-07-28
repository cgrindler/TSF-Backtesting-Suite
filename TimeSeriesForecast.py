#!/usr/bin/python

# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import os.path
import sys
import Specification
import DataProcessing
import ErrorAnalysis

def generateModel(Specs,dataObj):
    
    if(Specs.Mode == "1"):

        import BoxJenkins
        modelObj = BoxJenkins.BoxJenkins(dataObj,Specs)

    elif(Specs.Mode == "2"):

        import DeepNetworks
        modelObj = DeepNetworks.DeepNetworks(dataObj, Specs, 10)

    modelObj.modelling()
    if(Specs.Mode == "1"):
        folderName=list(dataObj.data)[int(Specs.ForecastCol)] 
        folderPath = Specs.mdlPathBJ+"\\"+ folderName
        #if model exists, no parameter estimation is necessary
        modelObj.fitting(Specs.mdlName,folderPath) 
        
    if(Specs.Mode == "2"):
        modelObj.fitting()

    return modelObj

def BackTesting(Specs):

    #warning: Format YYYY-MM-DD  (~) > 1000x faster than DD.MM.YYYY     
    dataObj = DataProcessing.DataProcessing(Specs.datapath)
    modelObj = generateModel(Specs,dataObj)

    timeseries = dataObj.data[dataObj.data.columns[int(Specs.ForecastCol)]]
    timeseriesNF = dataObj.data["NoFilter"]
        
    print("\nPrediction with generated model"+str(Specs.order)+"x"+str(Specs.sorder)+"\n\n")


    if (Specs.Mode == "1"):

        ##### -1-Step Prediction- #####
        ax = plt.subplot(2,1,1)
        pred1 = modelObj.predict1Step()
        dataObj.visualize(ax, Specs.timeseriesName,'1-step Forecast', timeseriesNF, pred1.predicted_mean)
        F1 = ErrorAnalysis.ErrorAnalysis(timeseriesNF,modelObj.predicted.predicted_mean)
        one = "##### 1-step Prediction ##### \n"
        print(one)
        print(F1.criterias)
    
        print("\nCalculation of Sarimax-Model "+str(Specs.order)+"x"+str(Specs.sorder)+"\n\n")

        ##### -Multi-Step Prediction- #####
        ax1 = plt.subplot(2,1,2)
        Season = Specs.sorder[3]


        ForecastDyn = modelObj.predictDyn(nstep=Specs.horizont,n=Specs.sorder[0],hourOfDay=Specs.hourOfDay,anzahl=Specs.AnzahlPrognosen)            
    
        DataDyn = timeseriesNF.loc[ForecastDyn.index]
        dataObj.visualize(ax1, Specs.timeseriesName,str(Specs.horizont)+'-step Forecast', DataDyn,ForecastDyn)
        FDyn = ErrorAnalysis.ErrorAnalysis(DataDyn,ForecastDyn)
    
        multi = "\n##### "+str(Specs.horizont)+"-step Prediction ##### \n"
    
        print(multi)
        print(FDyn.criterias)


    else:
        trainPred, trainOut, testPred, testOut = modelObj.predict()
        Ftrain = ErrorAnalysis.ErrorAnalysis(trainOut.flatten(), trainPred.flatten())
        Ftest = ErrorAnalysis.ErrorAnalysis(testOut.flatten(), testPred.flatten())
        ax = plt.subplot(2,1,1)
        dataObj.visualize(ax, Specs.timeseriesName,str(Specs.horizont)+'-step Forecast',trainOut.flatten(), trainPred.flatten())
        ax1 = plt.subplot(2,1,2)
        multi = "\n##### "+str(Specs.horizont)+"-step Prediction ##### \n"
        dataObj.visualize(ax, Specs.timeseriesName,str(Specs.horizont)+'-step Forecast',testOut.flatten(), testPred.flatten())
        print(multi)
        print("Training:\n")
        print(Ftrain.criterias)
        print("\nTest:\n")
        print(Ftest.criterias)
                
    plt.show()

def Operate(Specs,currentData): 
	

    #warning: Format YYYY-MM-DD  (~) > 1000x faster than DD.MM.YYYY     
    dataObj = DataProcessing.DataProcessing(Specs.datapath)
    modelObj = generateModel(Specs,dataObj)

    #### -Live Multi - Step Prediction- #####   

    # 1.) predict with given (live) input csv file and saved model
    # 2.) generate output csv file 
    #TBD 

def main(Specs):
        
    if Specs.BackTest is True:
            
        print("(1): Backtesting\n\n")
        print("processing data from '"+Specs.location+"'...")
        BackTesting(Specs)
    else:
        print("(2): Live Forecast\n\n")
        liveObj = getLiveData() # TBD               
        Operate(Specs, liveObj)



if __name__ is not "__main__":
    
    Specs = Specification.Specification()        
    main(Specs)
else:	
    Specs = Specification.Specification()        
    Specs.BackTest=True
    main(Specs)
