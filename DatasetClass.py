import pandas as pd

#this class is merely for encapsulation.
#this is how data of the dataset is stores in the objects 
#of this class.
class DataSet:
    #name of the dataset
    def __init__(self, name):
        self.name = name
    
    def set_feature(self, features):
        self.features = features
    
    def set_classes(self, classes):
        self.classes = classes
    
    def get_classes(self):
        return self.classes
        
    def set_org_data(self, org_data):
        self.org_data = org_data
        
    def get_org_data(self):
        return self.org_data
    
    # not writing all getters, setters, because
    # they are not required in python (i didnt know that before)