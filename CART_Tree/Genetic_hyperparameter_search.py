#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 10:24:56 2021

@author: maxime
"""

import math
import random

class Point:            # store information of one data point - coordinate of the point
    def __init__(self,x):
        self.data = x
        
    def distance(self,p):   # compute the distance between the point ifself and another point p
        d = 0.
        for i in range(len(p.data)):
            delta = self.data[i]-p.data[i]
            d += delta*delta
        d = math.sqrt(d)
        return d

class Solution:             # set of points with their corresponding clusters
    def __init__(self,priors, X_train, X_test, y_train, y_test):
        self.priors = priors
        self.prior_num_feat = priors[0]
        self.prior_n_tree = priors[1]
        self.prior_sample_n = priors[2]
        self.prior_min_bucket = priors[3]
        self.prior_max_size = priors[4]
        self.data = []
        self.sumdist = 0.
        self.X_y = [X_train, X_test, y_train, y_test]

    def quality(self,data):      # sum of distance for quality of clusters --> score we want to minimize
        self.data = data
        self.sumdist = 0.
        self.num_feat = int(round(self.data[0],0))
        self.n_tree = int(round(self.data[1],0))
        self.sample_n = round(self.data[2],2)
        self.min_bucket = int(round(self.data[3],0))
        self.max_size = int(round(self.data[4],0))
        
        rf = RF.Random_Forest()
        train_object = rf.Fit(self.X_y[0],self.X_y[2],[], num_feat = self.num_feat, n_tree = self.n_tree, 
                          sample_n = self.sample_n, min_bucket=self.min_bucket, max_size = self.max_size, 
                          strategy= None , bins = None, cores = 1)
        
        pred = rf.Predict(train_object, self.X_y[1])
        self.sumdist = rf.MSE_Pred(pred, self.X_y[3])
          
        return self.sumdist, self.data
    
    def record(self, data, sumdist):
        self.data = data
        self.sumdist = sumdist

    def random_solution(self): # assign cluster randomly
        self.data.clear()
        for prior in [self.prior_num_feat, self.prior_n_tree, self.prior_sample_n, self.prior_min_bucket, self.prior_max_size]:
            param = random.uniform(prior[0],prior[1]) 
            self.data.append(param)
        return self.data

    def set_data(self,data):    # copy data
        self.data = data

    def mutation(self,data,rate): #how many points are changing for the clusters
        self.data = data
        val,_ = self.quality(self.data)
        
        for iter in range(0,rate):
            i = random.choice(range(len(self.data)))
            bck = self.data[i]
            prior = self.priors[i]
            if i == 2:
                self.data[i] = round(random.uniform(prior[0],prior[1]),2)
            else:
                self.data[i] = int(round(random.uniform(prior[0],prior[1]),0))
            sumdist,_ = self.quality(self.data)
            if(sumdist < val):
                val = self.sumdist
            else:
               self.data[i] = bck       
               
        return val, self.data

    def crossover(self,data2): # 3 parameters: self, data2 and p: permutations
        for j in range(len(data2)): # CROSS OVER - before only aligning the solution
            if(random.choice([0,1]) == 0):
                self.data[j] = data2[j]
                
    def valid(self):
        for s in range(len(self.data)):
            if(self.data[s] > self.nb_class):
                print(s," data = ",self.data[s])
                return False
        return True
    def display(self):
        print(self.data)

class Population:
    def __init__(self):
        self.sols = []
        
    def addSolution(self,s):
        self.sols.append(s)
    
    def getData(self,i):
        if(i>= 0 and i<len(self.sols)):
            return self.sols[i].data
        else: 
            return None

    def setData(self,i,d):              # replace solution
        if(i>= 0 and i<len(self.sols)):
            self.sols[i].data = d
            
    def sort(self):                     # bubble sorte - for sort is n**2 but then will be sorted expect one element O(n)
        cont = True
        while cont:
            cont = False
            for i in range(len(self.sols)-1):
                if(self.sols[i].sumdist > self.sols[i+1].sumdist):
                    self.sols[i],self.sols[i+1] = self.sols[i+1],self.sols[i]
                    cont = True

    def randomGoodData(self):
        pos = random.randint(0,len(self.sols)//2)
        return self.sols[pos].data

    def randomBadData(self):
        c = len(self.sols)//2
        pos = random.randint(0,c)+c
        if(pos >= len(self.sols)): 
            pos=c
        return self.sols[pos].data

    def replaceRandomBadData(self,d):
        c = len(self.sols)//2
        pos = random.randint(0,c)+c
        if(pos >= len(self.sols)): 
            pos=c
        self.sols[pos] = d

        
    def bestQuality(self):              # among of the solution
        return self.sols[0].sumdist

    def bestData(self):
        return self.sols[0].data

    def valid(self):
        v = True
        for s in range(len(self.sols)):
            if(not self.sols[s].valid()):
                print(s," sol not valid")
                v = False
        return v

    def display(self):        
        for s in range(len(self.sols)):
            print(s," sol : ")
            self.sols[s].display()
            
            
            
            
import Random_Forest as RF
from sklearn.model_selection import train_test_split            
from sklearn.datasets import load_boston
from joblib import Parallel, delayed



X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


nb_sol = 15             # size of population
generations = 40        # nb of loops
mutation_rate = 2       # move 5 objects into another cluster to see if it improves


prior_num_feat = [5,13]
prior_n_tree = [50,150]
prior_sample_n = [0.5,1]
prior_min_bucket = [5,100]
prior_max_size = [2,30]

priors = [prior_num_feat,prior_n_tree,prior_sample_n,prior_min_bucket,prior_max_size]


class Hyperparameters_Search(Population):
    
    def Hyperparameters_Search(self, nb_sol, generations, mutation_rate,priors):
        
        print("##########################")
        print("initialise solutions")
        print("##########################")

        sol_to_quantify = []
        for i in range(0,nb_sol):
            s = Solution(priors, X_train, X_test, y_train, y_test)
            sol_to_quantify.append(s.random_solution())
        
        processed_list = Parallel(n_jobs=15)(delayed(s.quality)(i) for i in sol_to_quantify)
        
        for i in processed_list:
            s = Solution(priors, X_train, X_test, y_train, y_test)
            s.record(i[1],i[0])
            self.addSolution(s)
            self.sort()
        
        print("##########################")
        print("initialised solutions completed")
        print("##########################")
            
        for n in range(0,generations):
            print("##########################")
            print("start generation: ",str(n))
            sol_to_quantify = []
            for i in range(0,nb_sol):
                self.data1 = self.randomGoodData()
                self.data2 = self.randomBadData()
                tmp_sol = Solution(priors, X_train, X_test, y_train, y_test)
                tmp_sol.set_data(self.data1)
                tmp_sol.crossover(self.data2)
                sol_to_quantify.append(tmp_sol.data)
            
            processed_list = Parallel(n_jobs=15)(delayed(tmp_sol.mutation)(i, mutation_rate) for i in sol_to_quantify)
            
            for i in processed_list:
                s = Solution(priors, X_train, X_test, y_train, y_test)
                s.record(i[1],i[0])
                self.replaceRandomBadData(s)
                self.sort()
            print("##########################")
        
        
        
HS = Hyperparameters_Search()
HS.Hyperparameters_Search(nb_sol, generations, mutation_rate,priors)

for i in HS.sols:
    print(i.sumdist)
print(HS.sols[0].data)
print(HS.sols[1].data)

#HS.randomBadData().data

#HS.randomGoodData()
#HS.randomBadData()






