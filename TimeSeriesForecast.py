#!/usr/bin/python

# -*- coding: utf-8 -*-

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
        folderPath = Specs.mdlPathBJ+"/"+ folderName
        #if model exists, no parameter estimation is necessary
        modelObj.fitting(Specs.mdlName,folderPath) 
        
    if(Specs.Mode == "2"):
        modelObj.fitting()

    return modelObj

def BackTesting(Specs):

    #warning: Format YYYY-MM-DD  (~) > 1000x faster than DD.MM.YYYY     
    dataObj = DataProcessing.DataProcessing(Specs.datapath)
    modelObj = generateModel(Specs,dataObj)

    folderName = list(dataObj.data)[int(Specs.ForecastCol)]
    timeseries = dataObj.data[dataObj.data.columns[int(Specs.ForecastCol)]]
    timeseriesNF = dataObj.data["NoFilter"]
        
    print("\nPrediction with generated model"+str(Specs.order)+"x"+str(Specs.sorder)+"\n\n")

    if(Specs.Mode == "1"):

        ##### -1-Step Prediction- #####
        pred1 = modelObj.predict1Step()
        F1 = ErrorAnalysis.ErrorAnalysis(timeseriesNF,modelObj.predicted.predicted_mean)
        one = "##### 1-step Prediction ##### \n"
        print(one)
        print(F1.criterias) 

        ##### -Multi-Step Prediction- #####
        Season = Specs.sorder[3]          
        ForecastDyn = modelObj.predictDyn(nstep=Specs.horizont,n=Specs.sorder[0],hourOfDay=Specs.hourOfDay,anzahl=Specs.AnzahlPrognosen)              
        DataDyn = timeseriesNF.loc[ForecastDyn.index]

        FDyn = ErrorAnalysis.ErrorAnalysis(DataDyn,ForecastDyn)
        multi = "\n##### "+str(Specs.horizont)+"-step Prediction ##### \n"
        print(multi)
        print(FDyn.criterias)

        try:
            import matplotlib.pyplot as plt
            ax = plt.subplot(2,1,1)
            dataObj.visualize(ax, Specs.timeseriesName,'1-step Forecast', timeseriesNF, pred1.predicted_mean)
            ax1 = plt.subplot(2,1,2)
            dataObj.visualize(ax1, Specs.timeseriesName,str(Specs.horizont)+'-step Forecast', DataDyn,ForecastDyn)
            plt.show()

        except:
            print("No visualization possible.")        
        
    else:
        print("\nPredictions with generated Neural Network")
        trainPred, trainOut, testPred, testOut = modelObj.predict()
        Ftrain = ErrorAnalysis.ErrorAnalysis(trainOut.flatten(), trainPred.flatten())
        Ftest = ErrorAnalysis.ErrorAnalysis(testOut.flatten(), testPred.flatten())
        
        try:
            import matplotlib.pyplot as plt
            ax = plt.subplot(2,1,1)
            dataObj.visualize(ax, Specs.timeseriesName,str(Specs.horizont)+'-step Forecast',trainOut.flatten(), trainPred.flatten())
            ax1 = plt.subplot(2,1,2)
            dataObj.visualize(ax, Specs.timeseriesName,str(Specs.horizont)+'-step Forecast',testOut.flatten(), testPred.flatten())
        except:
            print("No visualization possible.")
        
        print("Training:\n")
        print(Ftrain.criterias)
        print("\nTest:\n")
        print(Ftest.criterias)
        plt.show()     


def Operate(Specs): 
	

    dataObj = DataProcessing.DataProcessing(Specs.datapath)
    

    # TBD make sure that input data is valid etc.
    # TBD Error handling

    if not hasattr(dataObj, "data"):
        print("No valid data could be extracted.")
        return

    if Specs.Mode == "2":
        print("No Neural Network support at the moment.")
        return

    if not int(Specs.AnzahlPrognosen) == 1 or not Specs.AnzahlPrognosen:
        print("Corrected wrong configuration: Only realtime forecast is considered")
        Specs.AnzahlPrognosen == "1"


    import BoxJenkins
    folderName=list(dataObj.data)[int(Specs.ForecastCol)]  
    folderPath = Specs.mdlPathBJ+"/"+ folderName
    mdlObj = BoxJenkins.BoxJenkins(dataObj,Specs)
    
    #load model
    mdlObj.fitting(Specs.mdlName,folderPath)       
    
    Season = Specs.sorder[3]          
    ForecastDyn = mdlObj.predictDyn(nstep=Specs.horizont,n=Specs.sorder[0],hourOfDay=Specs.hourOfDay,anzahl=Specs.AnzahlPrognosen)              
    
    timeseriesNF = dataObj.data["NoFilter"]
    DataDyn = timeseriesNF.loc[ForecastDyn.index]   
    multi = "\n##### "+str(Specs.horizont)+"-step Prediction ##### \n"
    print(multi)
    FDyn = ErrorAnalysis.ErrorAnalysis(DataDyn,ForecastDyn)
    print(FDyn.criterias)

def main(Specs):
        
    if Specs.BackTest is True:
            
        print("(1): Backtesting\n\n")
        print("processing data from '"+Specs.location+"'...")
        BackTesting(Specs)
    else:
        print("(2): Live-Prediction: Docker Mode\n\n")		                       
        Operate(Specs)


Specs = Specification.Specification()     
if len(sys.argv) == 1 or sys.argv[1] != "--operate": 
    Specs.BackTest=True
else:
    Specs.BackTest=False

main(Specs)
