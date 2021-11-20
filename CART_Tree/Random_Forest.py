import Train as trn
import Prediction as pdn
import Pre_Processing as PP
import Train_Output as TO

class Random_Forest(PP.Pre_Processing, trn.Train, pdn.Prediction, TO.Train_Output):
    
    def __init__(self):
        #self.cat_col = None
        #self.cat_rec = None
        PP.Pre_Processing.__init__(self)
        trn.Train.__init__(self)
    
    def Fit(self, X, y, cat_col = None, num_feat = 3, n_tree = 4, sample_n = None, min_bucket=5, max_size = 4):
        
        #Pre-process Data
        x = self.Process_Train(X, num_method="mean", cat_method="mode", cat_col=cat_col)
        
        #Training Model
        L_records, L_root_tree_building, L_train_pred =  self.RF_Train(x, y, 
                                                                       num_feat = num_feat, 
                                                                       n_tree = n_tree, 
                                                                       sample_n = sample_n,
                                                                       min_bucket = min_bucket, 
                                                                       max_size = max_size,
                                                                       cores = 1)
   
        #Making Training Predictions
        Train_Predictions = self.Predict(x, y, L_records, L_root_tree_building, L_train_pred)
        
        #Output Object
        Random_Forest_Train = self.Output_Object(y, Train_Predictions, 
                                                 L_root_tree_building, L_records,
                                                 L_train_pred, n_tree, num_feat)
        
        return Random_Forest_Train
        
    def Test_Prediction(self, model, X_test, y):
        
        #Pre-Process Data
        x_test = self.Process_Test(X_test)
        
        #Idea is to load object from training function to predict
        L_records = model.get('Node_assignments')
        L_root_tree_building = model.get('Forest')
        L_train_pred = model.get('Train_predictions')
        
        Test_Predictions = self.Predict(x_test, y, L_records, L_root_tree_building, L_train_pred)

        return Test_Predictions
          
        
        




# from sklearn.datasets import load_boston
# X, y = load_boston(return_X_y=True)

# rf = Random_Forest()
# train_object = rf.Fit(X,y,[8], num_feat = 3, n_tree = 4, sample_n = None, min_bucket=5, max_size = 4)
# pred = rf.Test_Prediction(train_object, X)















