import pandas as pd

#Pre Process Class, for data cleaning, 
#checking for any abnormalities, pre processes the data 
import numpy as np
class PreProcess:
    
    def __init__(self, ds):
        # this function is only called for house-vote-84
        # dataset as this  has classes in the first column
        if(ds.name == 'house-votes-84'):
            #print("Here, means dataset name is: ",ds.name)
            self.move_class_column(ds)  
        self.set_Dataset_features(ds)
        #first make the org data discrete
        self.discretization(ds)
        #Now Shuffle the org data
        self.shuffle_data(ds)
        self.missing_values(ds)
    
    
    #this function sets the no of features and class names
    #to the Dataset class object
    def set_Dataset_features(self, ds):
        #set class names
        last_column_unique = ds.org_data.iloc[:, -1].unique()
        ds.set_classes(last_column_unique)
        #print(ds.get_classes())
        
        #set no of features, because one column is classes
        ds.feature_count = ds.org_data.shape[1] -1
        #print(ds.feature_count)
    
    
    #discretization function
    def discretization(self, ds):
        # This function is hardcoded accordign to each dataset. 
        # we will call this first on the original data, and then 
        # remaining all functions will remain same
        # if dataset is iris
        
        ds.org_bk = ds.org_data.copy()
        if(ds.name == 'iris'):
            ds.org_data_01 = ds.org_data.copy()
            num_bins = 10
            #all columns, because all features have continuous values
            for col in ds.org_data.columns[:-1]:
                #ds.org_data[col] = pd.qcut(ds.org_data[col], q=5, labels=['xsmall','small', 'Medium', 'Large', 'xLarge'])
                ds.org_data_01[col] = pd.cut(ds.org_data[col], bins=10, duplicates='drop')

            # Drop originalcolumns
        if(ds.name == 'glass'):
            num_bins = 5
            #first column is the discrete, 
            #all other attributes are continuous
            for col in ds.org_data.columns[1:-1]:
                ds.org_data[col] = pd.cut(ds.org_data[col], bins=10, duplicates='drop')
            
        #print(ds.org_data.head()) 
            
    #splitting data into 10 parts, Here we will
    #implement 10 fold cross validation
    def split_data(self,ds):
        
        # Size of each fold
        fold_size = len(ds.shufled_data) // 10        
        folds = []
        
        #9 of these folds will be  used for
        #training and 1 for testing
        for i in range(10):
            start = i * fold_size
            end = start + fold_size
            if i == 9:  
                folds.append(ds.shufled_data.iloc[start:])
            else:
                folds.append(ds.shufled_data.iloc[start:end])
        ds.ten_folds = folds

    
    #added the function to see, if we will need it for any dataset
    def one_hot_encoding(self, ds):
        pass
    
    #this step I added,because data was arranged in the classes
    # which means that test data was having only one class 
    def shuffle_data(self, ds):
        #shuffle the rows of data, so that classes are distributed equally
        #before splitting
        #just for testing I am not shuffling the data
        ds.shufled_data = ds.org_data.sample(frac=1).reset_index(drop=True)
        #ds.shufled_data = ds.org_data.copy()
        
    
    #this function handles missing values
    def missing_values(self, ds):      
        # Check if any missing values exist
        ds.shufled_data.replace('?', np.nan, inplace=True)
        has_missing_values = ds.shufled_data.isnull().values.any()
        missing_values_count = ds.shufled_data.isnull().sum().sum()
        
        #if more missing values, replace with mode.
        #selcting mode, because it will work with both
        #categorical and numeric data
        if (has_missing_values):
            print("This dataset has missing values")
            if missing_values_count > 1:
                # It has missing values  
                #print("This dataset has more missing values")
                rows_with_missing = ds.shufled_data[ds.shufled_data.isnull().any(axis=1)].index
                #print("Rows with missing values before filling:")
                #print(ds.shufled_data.loc[rows_with_missing])
    
                # Fill missing values for each column with its mode
                #using mode, because it will work for both categorical 
                #and numeric data
                if(ds.name == 'house-votes-84'):
                    #print("Inside the mode if")
                    for column in ds.shufled_data.columns[:-1]:
                        # Get mode of the column for dataset house-votes-84
                        # because this has the values y and n
                        mode_value = ds.shufled_data[column].mode()[0]
                        # Filling the missing values
                        ds.shufled_data[column].fillna(mode_value, inplace=True)
                if(ds.name == 'glass' or ds.name == 'soybean-small' or ds.name == 'breast-cancer-wisconsin'):
                    #print("Inside the median if")
                    for column in ds.shufled_data.columns[:-1]:
                        median_value = ds.shufled_data[column].median()
                        # Filling the missing values
                        ds.shufled_data[column].fillna(median_value, inplace=True)
    
                #print("Rows with missing values after filling:")
                #print(ds.shufled_data.loc[rows_with_missing])
            else:
                ds.shufled_data.dropna(inplace=True)
                print("Dropped the row")
    
    
    #this function handles the classses where class column is the first
    #this is kind of hardcoded thing for one dataset of voting
    def move_class_column(self, ds):
        # Remove the first column
        first_col = ds.org_data.pop(ds.org_data.columns[0])
        #append to last
        ds.org_data[len(ds.org_data.columns)] = first_col
        
   
    #add noise to the data
    def add_noise(self, ds):
        ds.noise_data = ds.shufled_data.copy()  

        # Step 1: Select 10% of the features (columns)
        #last column is the class
        total_columns = ds.noise_data.shape[1] -1
        num_columns = int(0.1 * total_columns)
        #select columns, except the last one, because that is class 
        noisy_columns = np.random.choice(ds.noise_data.columns[:-1], num_columns, replace=False)
        #print(noisy_columns)
        #print(ds.noise_data.head())

        # Step 2: Shuffle values within each selected feature
        for col in noisy_columns:
            # Shuffle the column values, just like previously, we shuffled 
            # row values
            #print(ds.noise_data[col].head())
            shuffled_values = ds.noise_data[col].sample(frac=1).values  
            ds.noise_data[col] = shuffled_values  
            #print(ds.noise_data[col].head())
            
            # Since all the remaing work is done on the
        # shuffled data hence it will be used
        ds.shufled_data = ds.noise_data.copy()
        #print(ds.shufled_data.head())