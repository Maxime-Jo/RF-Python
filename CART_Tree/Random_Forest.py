import numpy as np
import Missing_Value as MV
import Transform_Categorical as TC
import Prediction as pdn

class Random_Forest:
    
    def __init__(self, X, y, cat_col):
        self.X = np.array(X)
        self.y = y
        
        # Missing Values
        imp = MV.Missing_Value(X, cat_col)
        imp.impute(num_method="mean", cat_method="mode")
      
        self.X = imp.data.transpose()

        # Transform Categorical
        CatTo = TC.CatTo        
        for c in cat_col:
            self.X[:,c], _ = CatTo.FrequencyEncoding(self.X[:,c])     # TO-DO "_"
        
        Pred = pdn.Prediction()
        Train_Predictions, Test_Predictions = Pred.Test_Prediction(self.X, self.y, self.X, self.y)    
