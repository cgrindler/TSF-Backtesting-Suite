#!/usr/bin/python

# -*- coding: utf-8 -*-

import os, time
import csv
from datetime import datetime
import os.path
import sys
import Specification
import DataProcessing
import ErrorAnalysis

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

def createCsv(forecasts, Specs):
    with open(Specs.output,'w') as file:
        file.write("TIMESTAMP;FORECAST\n")
        for time, value in  forecasts.items():
            file.write(str(time)+";"+str(value)+"\n")
        print("forecast created: \n\n", forecasts)
        #file.

def generateModel(Specs,dataObj):
    
    if(Specs.Mode == "1"):

        import BoxJenkins
        modelObj = BoxJenkins.BoxJenkins(dataObj,Specs)

    elif(Specs.Mode == "2"):

        import DeepNetworks
        modelObj = DeepNetworks.DeepNetworks(dataObj, Specs, 10)

    modelObj.modelling()

    if(Specs.Mode == "1"):
        attributeName = list(dataObj.data)[int(Specs.ForecastCol)]

        mdlPath =  Specs.mdlpath + "/" + attributeName

        print("model under " +  mdlPath)
        #if model exists, no parameter estimation is necessary
        modelObj.fitting(Specs.mdlName, mdlPath) 
        
    if(Specs.Mode == "2"):
        modelObj.fitting()

    return modelObj

def watchFile(Specs):


    class Handler(FileSystemEventHandler):

        def __init__(self,Specs):
            self.Specs = Specs
            #linux time
            self.old = time.mktime(datetime.now().timetuple())
        
        def on_any_event(self,event):
            
            try:
                if event.is_directory:
                    return None

                elif event.event_type == 'created':
                    # Take any action here when a file is first created.

                    print("Received created event - %s." % event.src_path)
                    sys.stdout.flush()
                    #workaround -> watchdog triggers same event twice :-/
                    new = os.stat(event.src_path).st_mtime
                    if (new - self.old) > 0.1:      
                        Specs.location = event.src_path.split("/")[-1].split(".")[0]
                        Specs.datapath = event.src_path
                        Operate(Specs)

                elif event.event_type == 'modified':
                    # Taken any action here when a file is modified.
                    print("Received modified event - %s." % event.src_path)
                    sys.stdout.flush()
                    new = os.stat(event.src_path).st_mtime
                    print(new-self.old)
                    if (new - self.old) > 0.1:      
                        Specs.location = event.src_path.split("/")[-1].split(".")[0]
                        Specs.datapath = event.src_path
                        Operate(Specs)
                    else:
                        print("Operation skipped.")
                
                print("old =" + str(self.old))
                self.old = new
                print("new =" + str(self.old))
                sys.stdout.flush()
                printWatching(Specs.watchDict)
                sys.stdout.flush()
            except Exception as e:
                print("problems during operation: " + str(e))
                sys.stdout.flush()


    watchPath = os.path.dirname(Specs.watchDict)
    printWatching(watchPath)

    observer = Observer()
    event_handler = Handler(Specs)
    observer.schedule(event_handler, watchPath,recursive=True)
    observer.start()
   
    try:
        while True:
            time.sleep(5)
    except:
        observer.stop()
        print("Error while watching" , watchPath)
        sys.stdout.flush()
    observer.join()

def printWatching(watchPath):
    print("----------------------------------------------------------")
    print("start watching folder ", watchPath)
    print(str(datetime.now()))
    print("forecast starts automatically if any changes are detected.")
    print("----------------------------------------------------------")
    sys.stdout.flush()

def BackTesting(Specs):

    #warning: Format YYYY-MM-DD  (~) > 1000x faster than DD.MM.YYYY     
    dataObj = DataProcessing.DataProcessing(Specs)
    modelObj = generateModel(Specs,dataObj)
    timeseriesNF = dataObj.data["NoFilter"]
        
    print("\nprediction with generated model"+str(Specs.order)+"x"+str(Specs.sorder)+"\n\n")

    if(Specs.Mode == "1"):

        ##### -1-Step Prediction- #####
        pred1 = modelObj.predict1Step()
        F1 = ErrorAnalysis.ErrorAnalysis(timeseriesNF,modelObj.predicted.predicted_mean)
        one = "##### 1-step Prediction ##### \n"
        print(one)
        sys.stdout.flush()
        print(F1.criterias) 
        sys.stdout.flush()

        ##### -Multi-Step Prediction- #####       
        ForecastDyn = modelObj.predictDyn(nstep=Specs.horizont,n=Specs.sorder[0],delay=Specs.delay,anzahl=Specs.AnzahlPrognosen)              
        DataDyn = timeseriesNF.loc[ForecastDyn.index]

        FDyn = ErrorAnalysis.ErrorAnalysis(DataDyn,ForecastDyn)
        multi = "\n##### "+str(Specs.horizont)+"-step Prediction ##### \n"
        print(multi)
        sys.stdout.flush()
        print(FDyn.criterias)
        sys.stdout.flush()

        try:
            import matplotlib.pyplot as plt
            ax = plt.subplot(2,1,1)
            dataObj.visualize(ax,'1-step Forecast', timeseriesNF, pred1.predicted_mean)
            ax1 = plt.subplot(2,1,2)
            dataObj.visualize(ax1,str(Specs.horizont)+'-step Forecast', DataDyn,ForecastDyn)
            plt.show()

        except Exception as e:
            print(e)
            print("No visualization possible.")        
        
    else:
        print("\npredictions with generated Neural Network")
        trainPred, trainOut, testPred, testOut = modelObj.predict()
        Ftrain = ErrorAnalysis.ErrorAnalysis(trainOut.flatten(), trainPred.flatten())
        Ftest = ErrorAnalysis.ErrorAnalysis(testOut.flatten(), testPred.flatten())
        
        try:
            import matplotlib.pyplot as plt
            ax = plt.subplot(2,1,1)
            dataObj.visualize(ax,str(Specs.horizont) + '-step Forecast',trainOut.flatten(), trainPred.flatten())
            ax1 = plt.subplot(2,1,2)
            dataObj.visualize(ax,str(Specs.horizont) + '-step Forecast',testOut.flatten(), testPred.flatten())
        except:
            print("No visualization possible.")
        
        print("Training:\n")
        print(Ftrain.criterias)
        sys.stdout.flush()
        print("\nTest:\n")
        print(Ftest.criterias)
        sys.stdout.flush()
        plt.show()     

def Operate(Specs): 

    dataObj = DataProcessing.DataProcessing(Specs)
    # TBD make sure that input data is valid etc.
    # TBD back test results against real data
    # TBD Error handling
    # TBD fix model paths
    #print("----------------------------------------------------------")
    #print("event forecast started.")
    #print(datetime.now())
    #print("----------------------------------------------------------")

    if not hasattr(dataObj, "data"):
        print("no valid data could be extracted.")
        return

    if Specs.Mode == "2":
        print("No Neural Network support at the moment.")
        return

    if not int(Specs.AnzahlPrognosen) == 1 or not Specs.AnzahlPrognosen:
        print("corrected wrong configuration: Only realtime forecast is considered")
        Specs.AnzahlPrognosen = 1

    print("input data: ", dataObj.datapath)
    
    #load model
    try:
        mdlObj = generateModel(Specs,dataObj)
    except Exception as e:
        print("problems while loading model: " + e)
    
    try: 
        ForecastDyn = mdlObj.predictOperative(Specs.horizont)
        timeseriesNF = dataObj.data["NoFilter"]
    except Exception as e:
        print("problems while prediction:\n" + e)

    try:
        print("creating csv.")
        createCsv(ForecastDyn, Specs)
        print("done")
    except Exception as e:
        print("no csv could be created:\n" + e)

    try:
        DataDyn = timeseriesNF[-Specs.horizont:]
        multi = "\n##### "+str(Specs.horizont)+"-step Prediction against last values ##### \n"
        print(multi)
        sys.stdout.flush()
        FDyn = ErrorAnalysis.ErrorAnalysis(DataDyn,ForecastDyn)
        print(FDyn.criterias)
        sys.stdout.flush()
    except Exception as e:
        print(e)

def main():
    
    if "--config" in sys.argv:
        Specs = Specification.Specification(sys.argv[sys.argv.index("--config")+1])
    else:
        Specs = Specification.Specification()

    if not "--operate" in sys.argv:  
        print("(1): Backtesting: Calculation and evaluating of statistical models\n\n")		
        sys.stdout.flush() 
        Specs.BackTest=True
        BackTesting(Specs)
    else:
        print("(2): Live-Prediction: Docker Mode\n\n")		
        sys.stdout.flush() 
        Specs.BackTest=False
        watchFile(Specs)

main()
