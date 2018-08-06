#!/usr/bin/python

# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import os.path
import sys
import Specification
import DataProcessing
import ErrorAnalysis


def BackTesting(Specs, csvObj=""):
                  
        #warning: Format YYYY-MM-DD  (~) > 1000x faster than DD.MM.YYYY     
        dataObj = DataProcessing.DataProcessing(Specs.datapath)

        if not hasattr(dataObj, "data"):
            return

        folderName=list(dataObj.data)[int(Specs.ForecastCol)]  

        timeseries = dataObj.data[dataObj.data.columns[int(Specs.ForecastCol)]]#[Specs.hourOfDay:]
        timeseriesNF = dataObj.data["NoFilter"]


        if(Specs.Mode == "1"):

            import BoxJenkins
            modelObj = BoxJenkins.BoxJenkins(dataObj,Specs)

        elif(Specs.Mode == "2"):

            import DeepNetworks
            modelObj = DeepNetworks.DeepNetworks(dataObj, Specs, 10)


        modelObj.modelling()
        

        if(Specs.Mode == "1"):
            folderPath = Specs.mdlPathBJ+"/"+ folderName
            #if model exists, no parameter estimation is necessary
            modelObj.fitting(Specs.mdlName,folderPath) 
            
        if(Specs.Mode == "2"):
            modelObj.fitting()
   

        if (Specs.Mode == "1"):

            print("\nPredictions with generated Sarimax-Model "+str(Specs.order)+"x"+str(Specs.sorder)+"\n\n")
            pred1 = modelObj.predict1Step()
            Season = Specs.sorder[3]          
            ForecastDyn = modelObj.predictDyn(nstep=Specs.horizont,n=Specs.sorder[0],hourOfDay=Specs.hourOfDay,anzahl=Specs.AnzahlPrognosen)              
            DataDyn = timeseriesNF.loc[ForecastDyn.index]
            
            one = "##### 1-step Prediction ##### \n"
            print(one)
            F1 = ErrorAnalysis.ErrorAnalysis(timeseriesNF,modelObj.predicted.predicted_mean)
            print(F1.criterias)
        
            multi = "\n##### "+str(Specs.horizont)+"-step Prediction ##### \n"
            print(multi)
            FDyn = ErrorAnalysis.ErrorAnalysis(DataDyn,ForecastDyn)
            print(FDyn.criterias)

            try:
                ##### -1-Step Prediction- #####
                ax = plt.subplot(2,1,1)
                dataObj.visualize(ax, Specs.timeseriesName,'1-step Forecast', timeseriesNF, pred1.predicted_mean)
                ##### -Multi-Step Prediction- #####
                ax1 = plt.subplot(2,1,2)
                dataObj.visualize(ax1, Specs.timeseriesName,str(Specs.horizont)+'-step Forecast', DataDyn,ForecastDyn)
            except:
                print("No visualization possible.")        
            
        else:
            print("\nPredictions with generated Neural Network")
            trainPred, trainOut, testPred, testOut = modelObj.predict()
            Ftrain = ErrorAnalysis.ErrorAnalysis(trainOut.flatten(), trainPred.flatten())
            Ftest = ErrorAnalysis.ErrorAnalysis(testOut.flatten(), testPred.flatten())
            
            try:
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
