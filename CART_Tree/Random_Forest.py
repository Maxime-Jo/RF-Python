import Train as trn
import Prediction as pdn
import Pre_Processing as PP
import Train_Output as TO
import time #sys, os, 

class Random_Forest(PP.Pre_Processing, trn.Train, pdn.Prediction, TO.Train_Output):
    
    def __init__(self):
        PP.Pre_Processing.__init__(self)
        trn.Train.__init__(self)
    
    def Fit(self, X, y=None, num_method='mean', cat_method='mode', cat_column = None,  k=5, transform='freq', 
            num_feat = 3, n_tree = 4, sample_n = 0.8, min_bucket=5, max_size = 4, 
            strategy=None, bins = None, cores = 1):

        start = time.time()
        #Pre-process Data
        x = self.Process_Train(X, y, num_method, cat_method, cat_column, k, transform)
        end = time.time()
        print("Pre-processing Elalpsed time: ",str(end - start), "Seconds")
        
        start = time.time()
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
        end = time.time()
        print("Training Elalpsed time: ",str(end - start), "Seconds")
   
        #Making Training Predictions
        Train_Predictions = self.Predict_y(x, L_records, L_root_tree_building, L_train_pred)
        
        #Output Object
        Random_Forest_Train = self.Output_Object(y, Train_Predictions, 
                                                 L_root_tree_building, L_records,
                                                 L_train_pred, n_tree, num_feat)
        
        return Random_Forest_Train
        
    def Predict(self, model, X_test, n_core=1):
        
        #Pre-Process Data
        x_test = self.Process_Test(X_test)
        
        #Idea is to load object from training function to predict
        L_records = model.get('Node_assignments')
        L_root_tree_building = model.get('Forest')
        L_train_pred = model.get('Train_predictions')

        
        Test_Predictions = self.Predict_y(x_test, L_records, L_root_tree_building, L_train_pred, n_core)

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






# # Load data
# from sklearn.datasets import load_breast_cancer
# X, y = load_breast_cancer(return_X_y=True)
# y = y.astype(bool)
# #Load Random Forest Module
# rf = Random_Forest()
# #Train
# train_object = rf.Fit(X,y,[], num_feat = int((X.shape[1]**0.5)), n_tree = 100, 
#                       sample_n = 0.8, min_bucket=5, max_size = 4, 
#                       strategy= "quant" , bins = 20, cores = 1)
# #Prediction
# pred = rf.Predict(train_object, X, cores = 1)
# pred[pred<=0.5] = 0
# pred[pred>0.5] = 1
# pred = pred.astype(bool)

# #Testing with Scikit learn
# from sklearn.ensemble import RandomForestClassifier
# regr = RandomForestClassifier(max_depth=4, random_state=0)
# sk_rf = regr.fit(X, y)
# sk_predict = sk_rf.predict(X)


# print(rf.Missclasification(pred, y))
# print(rf.Missclasification(sk_predict, y))


# cr_test = pd.read_csv("CreditGame_TEST.csv")
# X_test = cr_test.drop(columns = ["ID_TEST","TYP_FIN"])


# pp = Pre_Processing()
# impX = pp.Process_Train(X, y, transform='OneHot')
# impT = pp.Process_Test(X_test)




# load dataset
import pandas as pd
import numpy as np

credit = pd.read_csv("CreditGame_TRAIN.csv")
target_name = ["DEFAULT","PROFIT_LOSS"] # this dataset has 2 target columns

X = np.array(credit.drop(columns = target_name + ["ID_TRAIN","TYP_FIN"])) # remove target, id and redundant columns
y = np.array(credit[target_name[1]])

#Split train-test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

#Load Random Forest Module
rf = Random_Forest()
#Train
train_object = rf.Fit(X, y, num_method="mean", cat_method="mode", 
                      cat_column=None,  k=100,   transform="freq", 
                      num_feat = 5, n_tree = 4, 
                      sample_n = 0.8, min_bucket=5, max_size = 4,
                      strategy="bin" , bins = 20, cores = 1)

#Prediction
pred = rf.Predict(train_object, X_test, n_core=1)

print("Nb of trees: ", str(rf.counter_tree_visite))
print("Nb of evaluated nodes: ", str(rf.counter_node_visite))
print("Nb of evaluated feature: ", str(rf.counter_feature_visite))
print("Nb of evaluated split: ", str(rf.counter_split_feature_visite))
print("Nb of evaluated MoD: ", str(rf.counter_MoD))


#Testing with Scikit learn
from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(max_depth=4, random_state=0)
X_imp = rf.Process_Train(X_train, y, num_method="mean", cat_method="mode", cat_column=None, transform="freq")
sk_rf = regr.fit(X_imp, y_train)

X_test_imp = rf.Process_Test(X_test)
sk_predict = sk_rf.predict(X_test_imp)

print(rf.MSE_Pred(pred, y_test))
print(rf.MSE_Pred(sk_predict, y_test))



