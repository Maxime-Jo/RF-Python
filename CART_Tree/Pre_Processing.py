# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 18:59:03 2021

@author: tyler
"""
import numpy as np
import Missing_Value as MV
import Transform_Categorical as TC

class Pre_Processing(MV.Missing_Value, TC.CatTo):
    
    def __init__(self):
        MV.Missing_Value.__init__(self)
        self.cat_mapping = {}
        self.transform = None
    
    def Process_Train(self, X, y=None, num_method='mean', cat_method='mode', cat_column=None, k=5, transform='freq'): 
        
        self.transform = transform
        
        # Missing value imputation
        imp_X = self.impute(X, y, num_method, cat_method, cat_column, k)
        
        # Transform Categorical
        for c in self.cat_column:
            if self.transform == 'freq':    
                imp_X[:,c], self.cat_mapping[c] = self.FrequencyEncoding(imp_X[:,c])
            elif self.transform == 'target':
                imp_X[:,c], self.cat_mapping[c] = self.TargetEncoding(imp_X[:,c],y)
            elif self.transform == 'OneHot':
                if len(self.cat_col_map[c]) > 2 :
                    one_hot = self.OneHotEncoding(imp_X[:,c])
                    # add new columns to the end of array
                    imp_X = np.append(imp_X, one_hot, axis = 1)
                #else: imp_X[:,c] = tc.OneHotEncoding(imp_X[:,c])
            else: raise ValueError('Invalid transformation method:' + self.transform)
        
        # remove original categorical columns for One-Hot Encoding
        if self.transform == 'OneHot' and len(self.cat_col_map[c]) > 2: 
            imp_X = np.delete(imp_X, self.cat_column, axis = 1)
        return imp_X
    
    def Process_Test(self, X): 
       
        # Missing Values
        imp_X = self.impute(X, None, self.num_method, self.cat_method, self.cat_column, self.k)
        
        # Transform Categorical
        for c in self.cat_column:
            if self.transform in ['freq','target']:
                imp_X[:,c] = self.Encode_by_mapping(imp_X[:,c], self.cat_mapping[c])
            else: # One Hot
                if len(self.cat_col_map[c]) > 2 :
                    one_hot = self.OneHotEncoding(imp_X[:,c])
                    # add new columns to the end of array
                    imp_X = np.append(imp_X, one_hot, axis = 1)                    
                #else: imp_X[:,c] = tc.OneHotEncoding(imp_X[:,c])

        # remove original categorical columns for One-Hot Encoding
        if self.transform == 'OneHot' and len(self.cat_col_map[c]) > 2: 
            imp_X = np.delete(imp_X, self.cat_column, axis = 1)

        return imp_X
    
# import pandas as pd


# credit = pd.read_csv("CreditGame_TRAIN.csv")
# cr_test = pd.read_csv("CreditGame_TEST.csv")

# target_name = ["DEFAULT","PROFIT_LOSS"] # this dataset has 2 target columns

# # remove target, id and redundant columns
# X = credit.drop(columns = target_name + ["ID_TRAIN","TYP_FIN"])
# y = np.array(credit[target_name[1]])
 
# X_test = cr_test.drop(columns = ["ID_TEST","TYP_FIN"])


# pp = Pre_Processing()
# impX = pp.Process_Train(X, y, transform='OneHot')
# impT = pp.Process_Test(X_test)
