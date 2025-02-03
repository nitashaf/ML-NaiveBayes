from Results import Results


import pandas as pd
#this is the main class, where we implemented the
#formulas for both 
from collections import defaultdict,Counter
import numpy as np

class NaiveBayes:
    def __init__(self):
        self.function_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.class_probabilities = {}

    #train function with the naive bayes algorithm
    #using the same formula as mentioned in the description
    def train(self, ds):
        #calculate class size for each class
        N = len(ds.train_data)
        d = ds.feature_count
        
        #This is for storing values i.e {class:{Attr:{value: count}}}       
        for cls in ds.classes:
            
            #First Step: Seperating the data into classes 
            #print(cls)
            classes = ds.train_data.iloc[:, -1] == cls
            class_data = ds.train_data[classes]                       
            N_c = len(class_data)            
            #print(cls, ": ", class_count )
            
            #calculate the probability of each class
            self.class_probabilities[cls]  =  N_c / N
            #print("Class Propability ", self.class_probabilities[cls])

            #Second Step: for each clss, calculate the attribute counts and divide by d and N
            for col in class_data.columns[:-1]:
                #count the values of each attribute
                att_value = class_data[col].value_counts()
                #print(att_value)
                
                #calculate the actual equation
                for value, count in att_value.items():
                    self.function_dict[cls][col][value] = (count + 1) / (N_c + d)
                    
        #print(self.function_dict)
    
    #testign the model
    def test(self,ds):
        actual_classes = []
        predicteded_classes =[]
        
        for index, row in ds.test_data.iterrows():
            highest_prob = -np.inf
            predicted_class = None
            
            #for each row. predict a class, based on attributes 
            for cls in ds.classes:
                prior_prob = self.class_probabilities[cls]
                posterior_prob = prior_prob
            
                #except for the last column 
                for col, value in row[:-1].items():
                    if value in self.function_dict[cls][col]:
                        posterior_prob += self.function_dict[cls][col][value]
                    else:
                        posterior_prob +=  np.log(1 / (len(self.function_dict[cls][col]) + 1))
                        
                if posterior_prob > highest_prob:
                    highest_prob = posterior_prob
                    predicted_class = cls
                
            actual_classes.append(row.iloc[-1])
            predicteded_classes.append(predicted_class)
        
        #saving the result as dataframe 
        results = pd.DataFrame({
        'Actual': actual_classes,
        'Predicted': predicteded_classes})
        
        #print(results)
        return results
    
    #first prepare the test and train data from folds
    #then call train and test 10 times, and 
    #take the average to prepare the results
    def mainFunction(self, ds):
        
        #adding combined results for each fold
        combined_results = pd.DataFrame()
        #combined_results = pd.DataFrame()
        rs = Results()
        eval_result_dic = {}
        no_folds = 10
        
        #calling function just for 2 folds to make things simpler
        for i in range(no_folds):
            # one fold for testing
            print(f'Fold: {i}')
            ds.test_data = ds.ten_folds[i]
            # remaining 9 folds for training
            ds.train_data = pd.concat([ds.ten_folds[j] for j in range(10) if j != i])
            
            self.train(ds)
            results = self.test(ds)
            #print(results)
            
            #this is just for printing the full confusion matrix for report
            combined_results = pd.concat([combined_results, results], ignore_index=True)
            
            # calling the Results class function
            # for this fold confusion matrix will be created
            # and we will get the evaluation for this fold 
            eval_results = rs.main(results, ds)
            print(eval_results)
            eval_result_dic[i] = eval_results
        
        #this is just for report
        rs.print_Confusion_Matrix(combined_results, ds)
        #I am seperating it, because, for video i need the part before this line
        #once we have values for all folds, now take average of all folds
        ZOLoss_total = 0
        accuracy_total = 0
        percision_total = 0
        recall_total = 0 
        f1Score_total = 0
        for fold, values in eval_result_dic.items():
            ZOLoss, accuracy, percision, recall, f1Score = values
            ZOLoss_total += ZOLoss
            accuracy_total += accuracy
            percision_total += percision
            recall_total += recall
            f1Score_total += f1Score
        
        ZOLoss_avg = ZOLoss_total /no_folds
        accuracy_avg = accuracy_total /no_folds
        percision_avg = percision_total /no_folds
        recall_avg = recall_total/no_folds
        f1Score_avg = f1Score_total/no_folds
        
        print(ZOLoss_avg, accuracy_avg, percision_avg, recall_avg, f1Score_avg)