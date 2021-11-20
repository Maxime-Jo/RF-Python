import pandas as pd
import numpy as np
import Random_Forest as RF


# load dataset
credit = pd.read_csv("CreditGame_TRAIN.csv")

target_name = ["DEFAULT","PROFIT_LOSS"] # this dataset has 2 target columns
data = credit.drop(columns=target_name)

# Pre-processing

# mapping of categorical variables
data['TYP_RES'] = data['TYP_RES'].map({'L':0, 'A':1, 'P':2})
data['ST_EMPL'] = data['ST_EMPL'].map({'R':0, 'T':1, 'P':2})

# 'TYP_FIN' column is not informative, only contains one value: 'AUTO'
X = np.array(data.drop(columns=["ID_TRAIN","TYP_FIN"]))
y = credit[target_name[0]]


rf = RF.Random_Forest()
rf_model = rf.Fit(X,y,cat_col=[7,8], num_feat = 3, n_tree = 4, sample_n = None, min_bucket=5, max_size = 4)
pred = rf.Test_Prediction(rf_model, X,y)

