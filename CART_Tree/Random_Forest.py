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
    
    def Fit(self, X, y, cat_col = None, num_feat = 3, n_tree = 4, sample_n = 0.8, min_bucket=5, max_size = 4, strategy=None, bins = None, cores = 1):
        
        #Pre-process Data
        x = self.Process_Train(X, num_method="mean", cat_method="mode", cat_col=cat_col)
        
        #Training Model
        L_records, L_root_tree_building, L_train_pred =  self.RF_Train(x, y, 
                                                                       num_feat = num_feat, 
                                                                       n_tree = n_tree, 
                                                                       sample_n = sample_n,
                                                                       min_bucket = min_bucket, 
                                                                       max_size = max_size,
                                                                       strategy=strategy, 
                                                                       bins = bins,
                                                                       cores = cores)
   
        #Making Training Predictions
        Train_Predictions = self.Predict_y(x, L_records, L_root_tree_building, L_train_pred)
        
        #Output Object
        Random_Forest_Train = self.Output_Object(y, Train_Predictions, 
                                                 L_root_tree_building, L_records,
                                                 L_train_pred, n_tree, num_feat)
        
        return Random_Forest_Train
        
    def Predict(self, model, X_test): #Remove y
        
        #Pre-Process Data
        x_test = self.Process_Test(X_test)
        
        #Idea is to load object from training function to predict
        L_records = model.get('Node_assignments')
        L_root_tree_building = model.get('Forest')
        L_train_pred = model.get('Train_predictions')
        
        Test_Predictions = self.Predict_y(x_test, L_records, L_root_tree_building, L_train_pred)

        return Test_Predictions
          
        
##############################################################################      

#Loading sample data
# from sklearn.datasets import load_boston
# X, y = load_boston(return_X_y=True)

# #Load Random Forest Module
# rf = Random_Forest()

# #Train
# train_object = rf.Fit(X,y,[8], num_feat = 3, n_tree = 100, 
#                       sample_n = 0.8, min_bucket=5, max_size = 5, 
#                       strategy= None , bins = None, cores = 1)

# #Prediction
# pred = rf.Predict(train_object, X)

# print("Nb of trees: ", str(rf.counter_tree_visite))
# print("Nb of evaluated nodes: ", str(rf.counter_node_visite))
# print("Nb of evaluated feature: ", str(rf.counter_feature_visite))
# print("Nb of evaluated split: ", str(rf.counter_split_feature_visite))
# print("Nb of evaluated MoD: ", str(rf.counter_MoD))


# #Testing with Scikit learn
# from sklearn.ensemble import RandomForestRegressor
# regr = RandomForestRegressor(max_depth=4, random_state=0)
# sk_rf = regr.fit(X, y)
# sk_predict = sk_rf.predict(X)

# rf.MSE_Pred(pred, y)
# rf.MSE_Pred(sk_predict, y)

# # load dataset
# import pandas as pd
# import numpy as np

# credit = pd.read_csv("CreditGame_TRAIN.csv")

# target_name = ["DEFAULT","PROFIT_LOSS"] # this dataset has 2 target columns
# data = credit.drop(columns=target_name)

# # Pre-processing

# # mapping of categorical variables
# data['TYP_RES'] = data['TYP_RES'].map({'L':0, 'A':1, 'P':2})
# data['ST_EMPL'] = data['ST_EMPL'].map({'R':0, 'T':1, 'P':2})

# # 'TYP_FIN' column is not informative, only contains one value: 'AUTO'
# X = np.array(data.drop(columns=["ID_TRAIN","TYP_FIN"]))
# y = np.array(credit[target_name[1]])

# #Split train-test
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, random_state=42)

# # #Load Random Forest Module
# rf = Random_Forest()

# #Train
# train_object = rf.Fit(X_train,y_train,[7,8], num_feat = 3, n_tree = 100, 
#                       sample_n = 0.5, min_bucket=5, max_size = 5, 
#                       strategy= "bin" , bins = 15, cores = 1)

# #Prediction
# pred = rf.Predict(train_object, X_test, y_test)

# print("Nb of trees: ", str(rf.counter_tree_visite))
# print("Nb of evaluated nodes: ", str(rf.counter_node_visite))
# print("Nb of evaluated feature: ", str(rf.counter_feature_visite))
# print("Nb of evaluated split: ", str(rf.counter_split_feature_visite))
# print("Nb of evaluated MoD: ", str(rf.counter_MoD))


# #Testing with Scikit learn
# from sklearn.ensemble import RandomForestRegressor
# regr = RandomForestRegressor(max_depth=4, random_state=0)
# X_imp = rf.Process_Train(X_train, num_method="mean", cat_method="mode", cat_col=[7,8])
# sk_rf = regr.fit(X_imp, y_train)

# X_test_imp = rf.Process_Test(X_test)
# sk_predict = sk_rf.predict(X_test_imp)

# rf.MSE_Pred(pred, y_test)
# rf.MSE_Pred(sk_predict, y_test)














