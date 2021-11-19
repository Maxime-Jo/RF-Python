# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 23:55:16 2021

@author: tyler
"""

#Future Ideas:
    #1. Importance object that is computed using number of times a feature is featured in tree
    #2. Confusion matrix if y is boolean (w/ specificity and sensitivity)


import Error_Measures as EM

class Train_Output(EM.Error_Measures):
    
    def Output_Object(self, y, Train_Predictions, L_root_tree_building, L_records, L_train_pred, n_tree, num_feat):
        
        #Output Object of Train Function
        random_forest_train = []
        
        #Error Rate
        if y.dtype == 'bool':
            MSE = self.Missclasification(Train_Predictions, y)
            random_forest_train.append(MSE)
        else:
            Missclassification_Rate = self.MSE(Train_Predictions, y)
            random_forest_train.append(Missclassification_Rate)
        
        
        #Naming items in output object
        Forest = L_root_tree_building
        Node_assignments = L_records
        Tree_predictions = L_train_pred
        Trees = n_tree
        Features = num_feat
        
        #Appending to output object
        random_forest_train.append(Train_Predictions)
        random_forest_train.append(Forest)
        random_forest_train.append(Node_assignments)
        random_forest_train.append(Tree_predictions)
        random_forest_train.append(Trees)
        random_forest_train.append(Features)
        
        return random_forest_train