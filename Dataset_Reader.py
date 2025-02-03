import pandas as pd

#Class reads the data file and the names file to load inti datase
#object
class Dataset_Reader:
    
    def __init__(self):
        self.Path = 'C:\\Users\\nitas\\Downloads\\Machine Learning 2024\\'
    
    #read the data from data file and fill the Datset class object
    def read_data_file(self, ds):
        df = pd.read_csv(self.Path+ds.name+'.data', delimiter=',', header=None)

        ds.set_org_data(df)
    
    
    #read name file, naming the features, just to fill
    #for results, not mendatory function
    def read_attributes(self, ds):
        
        start_string = 'Attribute Information'
        stop_string = 'Missing Attribute Values'
        isPrint = False
        
        #read section, where the file has information 
        #Attribute Information
        with open(Path+ds.name+'.names', 'r') as file:
            for line in file:
                if start_string in line:
                    isPrint = True
                    #continue
                    
                if stop_string in line:
                    isPrint = False
                    break
                        
                if isPrint:
                    print(line.strip())
                        