# -*- coding: utf-8 -*-
import getpass
import json
import os

class Specification:

    def __init__(self):
        
        currentdirectory = os.path.dirname(os.path.realpath(__file__))
        with open(currentdirectory + '/config.json') as myjson:
            data = json.load(myjson)


        self.BackTest = data["BackTest"]
        self.exogCol = data["exogCol"]   
        self.horizont = data["horizont"]
        self.order = (data["order"]["ar"],data["order"]["d"],data["order"]["ma"],);        
        self.sorder = (data["sorder"]["ar"],data["sorder"]["d"],data["sorder"]["ma"],data["sorder"]["s"])
        self.filterweight = data["filterweight"]
        self.hourOfDay = data["hourOfDay"]
        self.AnzahlPrognosen = data["nPrediction"]
        self.von=data["from"]
        self.bis=data["till"]
        self.ForecastCol = data["ForecastCol"]
        self.Mode = data["Mode"]
        self.modelHistory = data["modelHistory"]
        self.scale = data["scale"]
        ###### file paths ######
        self.datapath = list()
        self.LSTM = data["LSTM"]
        self.LSTM_Layers = data["LSTM_Layers"]
        self.sep = data["delimiter"]

        currentdirectory = os.path.dirname(os.path.realpath(__file__))
        self.datapath = data["datapath"]
        self.timeseriesName = self.datapath.split("/")[-2]
        self.location = self.datapath.split("/")[-1].split(".")[0]
        self.mdlName = "Mdl" + str(self.order[0]) + str(self.order[1]) + str(self.order[2]) + "S" + str(self.sorder[0])+str(self.sorder[1])+str(self.sorder[2])+str(self.sorder[3])+ "F" + str(self.filterweight)        
        self.mdlPathBJ = currentdirectory +"/Models/BoxJenkinsModels/"+self.timeseriesName+"/"+self.location
        self.mdlPathNN = currentdirectory +"/Models/NeuralNetworkModels/"+self.timeseriesName+"/"+self.location