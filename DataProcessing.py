import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.parser import parse
import Specification
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class DataProcessing:

    def __init__(self,Spec):

        self.Spec = Spec
        self.scaling = self.Spec.scale
        self.filterweight = self.Spec.filterweight
        self.datapath = self.Spec.datapath 

        # this class provides methods to prepare csv input data
        try:
            self.data = pd.read_csv(self.datapath, sep = self.Spec.sep)
            if self.Spec.BackTest != True:
                print("Input data to be used: ")
                print(self.data)
        except:
            print("\n\nCould not read input data "+self.datapath+". Please check if your input data is on the correct path specified or if you specified the correct delimiter.\n\n")
            return


        if (self.is_date(self.data[self.data.columns[0]][0])):
             self.data.index = self.data[self.data.columns[0]]
             self.data = self.data.drop(self.data.columns[0], axis = 1)
             self.ConvertFrame()
        else:
            try:
                self.data.index = self.generateIndex()
            except:
                print("Please specify correct time interval")

        self.data["NoFilter"] = self.data[self.data.columns[0]]
        self.data[self.data.columns[0]] = sm.tsa.filters.hpfilter(self.data[self.data.columns[0]],self.filterweight)[1]
        self.timeInterval(self.Spec.von,self.Spec.bis)
        self.replaceZeros()
       
    def ConvertFrame(self):

        self.data.index = pd.to_datetime(self.data.index) 
            
    def replaceZeros(self):
        
        for header in list(self.data):
            self.data[header][self.data[header] < 0] = 0

    def timeInterval(self,von,bis):

        self.data = self.data.ix[von:bis]

    def getSeries(self, data):

        return data.ix[:,"Value"]

    def createSets(self):

        #self.trainPre, self.testPre = train_test_split(self.data, test_size = 0.1, shuffle = False)

        
        cutPoint = round(len(self.data)/24*0.8)*24

        self.trainPre = self.data[0:cutPoint]
        self.testPre = self.data[cutPoint+1:-1]

        if self.scaling:
            scaler = MinMaxScaler(feature_range=(0, 1))
            return (scaler.fit_transform(self.trainPre),scaler.fit_transform(self.testPre))
        else:
            return(np.array(self.trainPre),np.array(self.testPre))

    def createInputOutput(self, tdata, hist, horizont, interval):

        #separate data into input/output samples  
        dataX1, dataY = [] , []

        # amount of forecasts <=> input/output training data
        for i in range(int((len(tdata)-hist-horizont)/interval)):
  
            sampleArr=np.zeros(hist*self.data.shape[1])
            sampleArr = tdata[i*interval:i*interval+hist].flatten() # *historic* values
            dataX1.append(sampleArr)
            dataY.append(tdata[((i*interval)+hist):((i*interval)+hist+horizont)][:,0]) # *future* values

        return np.array(dataX1), np.array(dataY)

    def generateIndex(self):
        #last element isn't considered ([0:-1])
        #toDo extend for days etc.
        return pd.date_range(self.Spec.von,self.Spec.bis, freq ='H')[:-1]

    def is_date(self, string):
        try: 
            parse(string)
            return True
        except:
            return False

    def visualize(self, ax, title, *data):
        
            import matplotlib.pyplot as plt
            #ax = plt.subplots()
            ax.set_title(title, loc='left')
            #ax.set_ylabel(labelname)

            if self.Spec.Mode == "1":
                data[0].plot(ax=ax, label='measured data', alpha=0.4, lw=5, use_index=False)
                data[1].plot(ax=ax, label=title, alpha=0.6, lw=3, use_index=False)
            else: #no DataFrame provided
                plt.plot(data[0], label='measured data', alpha=0.4, lw=5)
                plt.plot(data[1], label=title, alpha=0.6, lw=3)
                
            plt.legend(shadow=True, title="Legend", fancybox=True)

