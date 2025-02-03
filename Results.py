import pandas as pd
import numpy as np
#This class is used to calculate results. 
class Results:
    
    def __init__(self):
        pass
    
    def main(self,results, ds):
        #this is the main function for this class
        #print("inside main function of Results class")
        #These are the results fone one fold
        conf_matrix_vals = self.confusion_matrix(results, ds)
        #call the loss function here for one fold values
        evaluations = self.LossFunctionMacro(ds, conf_matrix_vals)
        return evaluations
    
    
    def print_Confusion_Matrix(self, combinedResults, ds):
        conf_matrix_comb = pd.crosstab(combinedResults['Actual'], combinedResults['Predicted'], 
                                  rownames=['Actual'], colnames=['Predicted'], 
                                  dropna=False).reindex(index=ds.classes, columns=ds.classes, fill_value=0)
        print(conf_matrix_comb)
        
    #Confusion matrix calculated TP, TN, FP, and FN
    #Once we will have these information, we will then 
    #use it to calculate Accuracy, Recall and percision
    def confusion_matrix(self, result, ds):
            
        # This syntax for adding 0 where class is not being predicted 
        # help I have taken from Internet. but i forgot the site
        #This is creating the confusion matrix table

        conf_matrix = pd.crosstab(result['Actual'], result['Predicted'], 
                          rownames=['Actual'], colnames=['Predicted'], 
                          dropna=False).reindex(index=ds.classes, columns=ds.classes, fill_value=0)
        #print(conf_matrix)

        conf_matrix_dic = {}
        # I realized afterwards that we need to guage the performance of overall 
        # not the classes seperately, so calculating both
        TP_total =0
        FN_total =0
        FP_total =0
        TN_total =0

        for cls in ds.classes:
            list_values = []

            # When both actual and predicted classes match (True Positive)
            if cls in conf_matrix.index and cls in conf_matrix.columns:                
                TP = conf_matrix.at[cls, cls] 
            else:
                TP = 0
            list_values.append(TP)
            TP_total += TP

            # When the actual class is this class, All row values of this class - TP
            if cls in conf_matrix.index:
                FN = conf_matrix.loc[cls, :].sum() - TP
            else:
                FN = 0
            list_values.append(FN)
            FN_total += FN

            # When the predicted class is this class, Column values of this class - TP
            if cls in conf_matrix.columns:
                FP = conf_matrix.loc[:, cls].sum() - TP
            else:
                FP = 0
            list_values.append(FP)
            FP_total += FP


            # True negative is the TP Total - True negative of all other class
            #because then is when model correctly predicts negative of class
            #diagonal array will give all TPs
            diagonal = np.diag(conf_matrix)
            #print(diagonal)
            diagonal_sum = diagonal.sum()
            TN = diagonal_sum - TP
            list_values.append(TN)
            TN_total += TN

            conf_matrix_dic[cls] = list_values
            
        #end of class loop
        conf_matrix_values = [TP_total, FN_total, FP_total, TN_total]
        
        print("Confusion matrix values class wise:", conf_matrix_dic)
        Total = conf_matrix.sum().sum()
        #print(Total)        
        
        #return conf_matrix_values
        return conf_matrix_dic
        
    #Loss function usually us used while training the model, 
    #but here we will use it to calculate Accuracy, precision and Recall 
    #to gauge the overall performance of the model 
    def LossFunctionMicro(self, ds, conf_matrix_values):
        
        result_list = []
        #calcuate the evaluations
        #for values in conf_matrix_list:
        values = conf_matrix_values
        #print("For Class", cls)
        TP, FN, FP, TN = values  
        Total = (TP + FN + FP + TN) 
            
        #0/1 Loss is (FP + FN) / Total
        ZOLoss = (FP + FN) /Total
        #print("Zero One Loss Function is:", ZOLoss)
            
        #Accuracy
        accuracy = (TP + TN)/ Total 
        #print("Accuracy is:", accuracy)
            
        #Percision
        percision = TP/(TP + FP)
        #print("Percision is:", percision)
            
        #Recall
        recall = TP/(TP + FN)
        #print("Recall is:", recall)
            
        #F1Score 
        f1Score = 2*(percision * recall)/(percision + recall)
        #print("F1 Score is:", f1Score)
        
        result_list = [ZOLoss, accuracy, percision, recall, f1Score]
        return result_list
    
    #This loss function calculates the average per class
    def LossFunctionMacro(self, ds, conf_matrix_dic):
        
        result_list = []
        #calcuate the evaluations
        #for values in conf_matrix_list:
        total_ZOLoss = 0
        total_accuracy = 0
        total_percision = 0
        total_recall = 0
        total_f1Score = 0
        
        for cls, values in conf_matrix_dic.items():
        
            #print("For Class", cls)
            TP, FN, FP, TN = values  
            Total = (TP + FN + FP + TN) 

            #0/1 Loss is (FP + FN) / Total
            ZOLoss = (FP + FN) /Total
            #print("Zero One Loss Function is:", ZOLoss)
            total_ZOLoss += ZOLoss

            #Accuracy
            accuracy = (TP + TN)/ Total 
            #print("Accuracy is:", accuracy)
            total_accuracy += accuracy

            #Percision
            if (TP + FP) == 0:
                percision = 0
            else:    
                percision = TP/(TP + FP)
            #print("Percision is:", percision)
            total_percision += percision

            #Recall
            if (TP + FN) == 0:
                recall = 0
            else:
                recall = TP/(TP + FN)
            #print("Recall is:", recall)
            total_recall += recall

            #F1Score
            if(percision + recall) == 0:
                f1Score = 0
            else:
                f1Score = 2*(percision * recall)/(percision + recall)
            #print("F1 Score is:", f1Score)
            total_f1Score += f1Score
            
        #when we added evaluation for each class, now take average
        no_classes = len(ds.classes)
        #print(no_classes)
        ZOLoss_avg = total_ZOLoss / no_classes
        accuracy_avg = total_accuracy / no_classes
        percision_avg = total_percision / no_classes
        recall_avg = total_recall / no_classes
        f1Score_avg = total_f1Score / no_classes
        
        result_list = [ZOLoss_avg, accuracy_avg, percision_avg, recall_avg, f1Score_avg]    
        return result_list