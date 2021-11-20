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
        
        #Error Rate
        if y.dtype == 'bool':
            error = self.Missclasification(Train_Predictions, y)
            err_type = "Missclassification Rate"
        else:
            error = self.MSE_Pred(Train_Predictions, y)
            err_type = "MSE"
        
        random_forest_train = {
            "Train_error" : error,
            "Train_error_type" : err_type,
            "Forest" : L_root_tree_building,
            "Node_assignments" : L_records,
            "Train_predictions" : L_train_pred,
            "nTrees" : n_tree,
            "nFeatures" : num_feat
        }
        
        return random_forest_train