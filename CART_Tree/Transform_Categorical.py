"""
Variable to test the class
"""

# from sklearn.datasets import load_boston
# X, y = load_boston(return_X_y=True)


# # create string feature
# feature = X[:,8].astype(str)


"""
The goal of this class is to transform any categorical vector into either one-hot-encoding matrix either in a continuous vector.

One-Hot-Encoding:
    if c represents the number of different classes,
    The one-hot-encoding is build c-1 columns for these classes.
    The first class is ignored.
    
Target Encoding:
    The idea of target encoding has been used by CATBoost a famous gradient boosting algorithm specialised for categorical variables.
    It is proposed as one-hot-encoding my consum too much memory when the number of category is large.
    Additionnally, one-hot-encoding does not encode any specific information regarding the category unlike embeddings
    (Word2Vec for the most famous in NLP).
    
    The goal of target encoding is to use the target information to encode the categories. The value of the category will then
    be equal to the mean of response for this category. We have added a prior to the mean in order to account for low frequency
    of certain category. The prior account for 30 elements but can be changed by the user. The prior will consider the overall
    mean of the response.
    
Frequency Encoding:
    The idea is similar to that of the target encoding. The frequency of the category is used instead of the response mean.
    
Input:
    - any vector: it is assumed to be a categorical variable
    - the format needs to be numpy array
    
Outputs:
    - one-hot-encoding: a numpy array containing c-1 columns where c represents the number of categories
    - target encoding or frequency encoding:
        - a continuous vector with the new categories embedding
        - a reference matrix to mappe the categories later (e.g., test data set)
"""



"""
The Class itself
"""
import numpy as np

class CatTo:
        
    def OneHotEncoding(self, feature):
        cat = np.unique(feature)                                        # list of the unique feature
        
        n = len(feature)                                                # length of the vector
        one_hot = np.array([feature])

        for c in cat[1:]:                                               # for all categories minus the first one!!
            
            one_hot_v = np.array([],dtype=int)                          # temporary array
            
            for i in range(0,n):                                        # visit element of the array                                               
                if feature[i] == c:                                     # one hot creation
                    one_hot_v = np.append(one_hot_v,1)
                else : one_hot_v = np.append(one_hot_v,0)
                
            one_hot = np.append(one_hot, [one_hot_v], axis = 0)         # create one hot vector
        
        one_hot = np.transpose(one_hot)
        one_hot = one_hot[:,1:].astype(int)
        
        return one_hot
    
    
    
    def TargetEncoding(self, feature, y, prior = 30):                   # https://maxhalford.github.io/blog/target-encoding/
        cat = np.unique(feature)
        y_mean = y.mean()
        
        trg_enc = np.array(feature)
        
        cat_rec = np.array([[0,0]])
        
        for c in cat: 
            sum_y = y[feature==c].sum()     # class stat
            count_y = len(y[feature==c])    # class stat
            
            cat_val = (sum_y+prior*y_mean)/(count_y+prior)      # encoding
            
            trg_enc[feature==c] = cat_val
            
            trg_enc = trg_enc.astype(float)
            
            cat_rec = np.concatenate((cat_rec, [[c,cat_val]]),0)
            
        cat_rec = cat_rec[1:,:]
            
        return trg_enc, cat_rec
    
    
    
    def FrequencyEncoding(self, feature, prior = 30):                   
        cat = np.unique(feature)
        
        n = len(feature)
        
        frq_enc = np.array(feature)
        
        cat_rec = np.array([[0,0]])
        
        for c in cat: 

            count = len(feature[feature==c])    # class stat
            
            cat_val = (count + prior)/(n + prior)    # encoding
            
            frq_enc[feature==c] = cat_val
            
            frq_enc = frq_enc.astype(float)
            
            cat_rec = np.concatenate((cat_rec, [[c,cat_val]]),0)
            
        cat_rec = cat_rec[1:,:]
            
        return frq_enc, cat_rec      
        
    def Encode_by_mapping(self, feature, cat_mapping):
        enc = np.array(feature)
        
        for i in range(len(cat_mapping)):
            enc[feature==cat_mapping[i,0]] = cat_mapping[i,1]
            enc = enc.astype(float)
             
        return enc
    

    
"""
Test the class output
"""
    
# CatTo = CatTo()

# test_OH = CatTo.OneHotEncoding(feature)
# test_Trg, test2_Trg = CatTo.TargetEncoding(feature,y)
# test_Frq, test2_Frq = CatTo.FrequencyEncoding(feature)

# test = CatTo.Encode_by_mapping(feature, test2_Trg)

#######
# k-fold for tree and leave?
# combine feature?
