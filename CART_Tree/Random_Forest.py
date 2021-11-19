
import numpy as np
import Train
import Prediction as pdn
import Pre_Processing as PP
import Train_Output as TO

class Random_Forest(PP.process, pdn.Prediction):
    
    def __init__(self, X, y, cat_col):
        self.X = np.array(X)
        self.y = y
        self.cat_col = cat_col
        
        #Pre-process Data
        self.x = self.process(self.X, self.cat_col)
        
    
    def Fit(self, num_feat = 3, n_tree = 4, sample_n = None, min_bucket=5, max_size = 4):
        
        
        #Training Model
        train = Train.Train()
        L_records, L_root_tree_building, L_train_pred =  train.RF_Train(self.x, self.y, 
                                                                        sample_f = num_feat, 
                                                                        n_tree = n_tree, 
                                                                        sample_n = sample_n,
                                                                        min_bucket=min_bucket, 
                                                                        max_size = max_size)
        
        #Making Training Predictions
        Train_Predictions = self.Prediction(self.x, self.y,
                                            L_records, L_root_tree_building, L_train_pred)
        
        
        #Output Object
        Output = TO.Train_Output()
        Random_Forest_Train = Output.Output_Object(self.y, Train_Predictions, L_root_tree_building, L_records, 
                                                   L_train_pred, 
                                                   self.cat_col,
                                                   n_tree, num_feat)
        

        return Random_Forest_Train
        
    def Test_Prediction(self, train_object, X_test): #Remove cat_col
        
        #Pre-Process Data
        self.x = self.process(X_test, self.cat_col)
        
        #Idea is to load object from training function to predict
        L_records = train_object[1]
        L_root_tree_building = train_object[2]
        L_train_pred = train_object[3]
        
        Pred = pdn.Prediction()
        Test_Predictions = Pred.Prediction(X_test, self.y,
                                              L_records, L_root_tree_building, L_train_pred)
        
        
        return Test_Predictions
          
        
        




















