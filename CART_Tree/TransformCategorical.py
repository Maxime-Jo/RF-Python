# Load datafrom sklearn.datasets import load_bostonX, y = load_boston(return_X_y=True)# create string featurefeature = X[:,8].astype(str)######## Fucntion start#######import numpy as npclass CatTo:            def OneHotEncoding(self, feature):        cat = np.unique(feature)                                        # list of the unique feature                n = len(feature)                                                # length of the vector        one_hot = np.array([feature])        for c in cat[1:]:                                               # for all categories minus the first one!!                        one_hot_v = np.array([],dtype=int)                          # temporary array                        for i in range(0,n):                                        # visit element of the array                                                               if feature[i] == c:                                     # one hot creation                    one_hot_v = np.append(one_hot_v,1)                else : one_hot_v = np.append(one_hot_v,0)                            one_hot = np.append(one_hot, [one_hot_v], axis = 0)         # create one hot vector                one_hot = np.transpose(one_hot)        one_hot = one_hot[:,1:].astype(int)                return one_hot                def TargetEncoding(self,feature ,y , prior = 30):                   # https://maxhalford.github.io/blog/target-encoding/        cat = np.unique(feature)        y_mean = y.mean()                trg_enc = np.array(feature)                cat_rec = np.array([[0,0]])                for c in cat:             sum_y = y[feature==c].sum()     # class stat            count_y = len(y[feature==c])    # class stat                        cat_val = (sum_y+prior*y_mean)/(count_y+prior)      # encoding                        trg_enc[feature==c] = cat_val                        trg_enc = trg_enc.astype(float)                        cat_rec = np.concatenate((cat_rec, [[c,cat_val]]),0)                    cat_rec = cat_rec[1:,:]                    return trg_enc, cat_rec                def FrequencyEncoding(self,feature , prior = 30):                           cat = np.unique(feature)                n = len(feature)                frq_enc = np.array(feature)                cat_rec = np.array([[0,0]])                for c in cat:             count = len(feature[feature==c])    # class stat                        cat_val = (count + prior)/(n + prior)    # encoding                        frq_enc[feature==c] = cat_val                        frq_enc = frq_enc.astype(float)                        cat_rec = np.concatenate((cat_rec, [[c,cat_val]]),0)                    cat_rec = cat_rec[1:,:]                    return frq_enc, cat_rec                           ######## Test#######    CatTo = CatTo()test_OH = CatTo.OneHotEncoding(feature)test_Trg, test2_Trg = CatTo.TargetEncoding(feature,y)test_Frq, test2_Frq = CatTo.FrequencyEncoding(feature)######## k-fold for tree and leave?# combine feature?