# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 18:59:03 2021

@author: tyler
"""
import Missing_Value as MV
import Transform_Categorical as TC

class Pre_Processing(TC.CatTo):
    
    def __init__(self):
        self.cat_mapping = None
        self.cat_col = None
        self.num_method = None
        self.cat_method = None
    
    def Process_Train(self, X, num_method, cat_method, cat_col=None): 
        
        self.cat_col = cat_col
        self.num_method = num_method
        self.cat_method = cat_method
        
        # Missing Values
        imp = MV.Missing_Value(X, self.cat_col)
        imp.impute(self.num_method, self.cat_method)
      
        imp_X = imp.data.transpose()

        # Transform Categorical
        for c in self.cat_col:
            imp_X[:,c], self.cat_mapping = self.FrequencyEncoding(imp_X[:,c])
        return imp_X
    
    def Process_Test(self, X): 
       
        # Missing Values
        imp = MV.Missing_Value(X, self.cat_col)
        imp.impute(self.num_method, self.cat_method)
      
        imp_X = imp.data.transpose()

        # Transform Categorical
        for c in self.cat_col:
            imp_X[:,c] = self.Encode_by_mapping(imp_X[:,c], self.cat_mapping)
        return imp_X