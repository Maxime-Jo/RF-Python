# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 18:59:03 2021

@author: tyler
"""

import Missing_Value as MV
import Transform_Categorical as TC

class Pre_Processing:
    
   def Process(self, X, cat_col): 
        # Missing Values
        imp = MV.Missing_Value(X, cat_col)
        imp.impute(num_method="mean", cat_method="mode")
      
        X = imp.data.transpose()

        # Transform Categorical
        CatTo = TC.CatTo        
        for c in cat_col:
            self.X[:,c], _ = CatTo.FrequencyEncoding(self.X[:,c])     # TO-DO "_"