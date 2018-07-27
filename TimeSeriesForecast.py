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
            folderPath = Specs.mdlPathBJ+"\\"+ folderName
            #if model exists, no parameter estimation is necessary
            modelObj.fitting(Specs.mdlName,folderPath) 
            
        if(Specs.Mode == "2"):
            modelObj.fitting()
 
        
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
	
    modelObj = BoxJenkins(Specs.order,Specs.sorder,currentData.data,Specs.filterweight)	
    mdl = modelObj.modelling()	
    modelObj.fitting(mdl,Specs.savepath)       
   
    #### -1-Step Prediction- #####   
    ax = plt.subplot(2,1,1)
    modelObj.predict1Step()
    #modelObj.visualize(ax, Specs.timeseriesName,'1-step Forecast', currentData.data)
    #F1 = ErrorAnalysis(currentData.data,modelObj.predict.predicted_mean)
    
    one = "##### 1-step Prediction ##### \n"
    print(one)
    print(modelObj.predict.predicted_mean)

    splitData = list()	
    if (len(sys.argv)>1):
        for x in range(0,int(sys.argv[1])):
            tmpstring =raw_input().split(';')
            
            date = tmpstring[0]
            value =  tmpstring[1].split(' ')[0]
            quality = tmpstring[1].split(' ')[1]
            splitData.append([date,float(value),quality])  
                         
    else: splitData.append(["123","456","100"]); print("##keine Echtzeitdaten##")
    
    predictSeries = pd.DataFrame(splitData,columns = ['date','data','quality'])
    predictSeries.index = predictSeries.date #pd.to_datetime(predictFrame.Datetime);
    return predictSeries

def main(Specs):
        
    if Specs.BackTest is True:
            
        print("(1): Backtesting\n\n")
        print("processing data from '"+Specs.location+"'...")
        BackTesting(Specs)
    else:
        print("(2): Live Forecast\n\n")
        sharpObj = getDatafromSharp()			                       
        Operate(Specs,sharpObj)



if __name__ is not "__main__":
    
    Specs = Specification.Specification()        
    main(Specs)
else:	
    print("Please run this Script from C# for Live-Prediction\n\n")
    Specs = Specification.Specification()        
    Specs.BackTest=True
    main(Specs)
