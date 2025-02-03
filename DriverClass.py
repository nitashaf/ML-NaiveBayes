from DatasetClass import DataSet
from Dataset_Reader import Dataset_Reader
from PreProcess import PreProcess
from NaiveBayes import NaiveBayes
from Results import Results

#Main driver class calls all other classes  

class Driver:
    def __init__(self):
        self.DSNames = 'iris', 'glass', 'house-votes-84', 'soybean-small', 'breast-cancer-wisconsin'
    
    def main(self):
        #change the dataset number here from 0 to 4
        ds = DataSet(self.DSNames[1])
        ds_reader = Dataset_Reader()
        ds_reader.read_data_file(ds)
            
        pp = PreProcess(ds)
        #original data
        pp.split_data(ds)        
        nb = NaiveBayes()
        nb.mainFunction(ds)
            
        #noise data
        print("-----------------------------------")
        print("With Noise")
        print("-----------------------------------")
        pp.add_noise(ds)
        pp.split_data(ds)        
        nb = NaiveBayes()
        nb.mainFunction(ds)
    
#calling the driver class here.    
dr = Driver()
dr.main()    