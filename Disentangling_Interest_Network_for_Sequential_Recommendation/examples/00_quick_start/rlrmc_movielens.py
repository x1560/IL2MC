#!/usr/bin/env python
# coding: utf-8

# *Copyright (c) Microsoft Corporation. All rights reserved.*
# 
# *Licensed under the MIT License.*

# ## Riemannian Low-rank Matrix Completion algorithm on Movielens dataset
# 
# Riemannian Low-rank Matrix Completion (RLRMC) is a matrix factorization based (vanilla) matrix completion algorithm that solves the optimization problem using Riemannian conjugate gradients algorithm (Absil et al., 2008). RLRMC is based on the works by Jawanpuria and Mishra (2018) and Mishra et al. (2013). 
# 
# The ratings matrix of movies (items) and users is modeled as a low-rank matrix. Let the number of movies be $d$ and the number of users be $T$. RLRMC algorithm assumes that the ratings matrix $M$ (of size $d\times T$) is partially known. The entry at $M(i,j)$ represents the rating given by the $j$-th user to the $i$-th movie. RLRMC learns matrix $M$ as $M=LR^\top$, where $L$ is a $d\times r$ matrix and $R$ is a $T\times r$ matrix. Here, $r$ is the rank hyper-parameter which needs to be provided to the RLRMC algorithm. Typically, it is assumed that $r\ll d,T$. The optimization problem is solved iteratively using the the Riemannian conjugate gradients algorithm. The Riemannian optimization framework generalizes a range of Euclidean first- and second-order algorithms such as conjugate gradients, trust-regions, among others, to Riemannian manifolds. A detailed exposition of the Riemannian optimization framework can be found in Absil et al. (2008). 
# 
# This notebook provides an example of how to utilize and evaluate RLRMC implementation in **recommenders**.

# In[1]:


import numpy as np
import sys
import time
import pandas as pd

from recommenders.datasets.python_splitters import python_random_split
from recommenders.datasets.python_splitters import python_stratified_split
from recommenders.datasets import movielens
from recommenders.models.rlrmc.RLRMCdataset import RLRMCdataset 
from recommenders.models.rlrmc.RLRMCalgorithm import RLRMCalgorithm 
# Pymanopt installation is required via
# pip install pymanopt 
from recommenders.evaluation.python_evaluation import (
    rmse, mae
)

# import logging

# %load_ext autoreload
# %autoreload 2


# In[2]:


print("Pandas version: {}".format(pd.__version__))
print("System version: {}".format(sys.version))


# Set the default parameters.
# 

# In[3]:


# Select Movielens data size: 100k, 1m, 10m, or 20m
MOVIELENS_DATA_SIZE = '10m'

# Model parameters

# rank of the model, a positive integer (usually small), required parameter
rank_parameter = 10
# regularization parameter multiplied to loss function, a positive number (usually small), required parameter
regularization_parameter = 0.001
# initialization option for the model, 'svd' employs singular value decomposition, optional parameter
initialization_flag = 'svd' #default is 'random'
# maximum number of iterations for the solver, a positive integer, optional parameter
maximum_iteration = 100 #optional, default is 100
# maximum time in seconds for the solver, a positive integer, optional parameter
maximum_time = 300#optional, default is 1000

# Verbosity of the intermediate results
verbosity=0 #optional parameter, valid values are 0,1,2, default is 0
# Whether to compute per iteration train RMSE (and test RMSE, if test data is given)
compute_iter_rmse=True #optional parameter, boolean value, default is False


# In[4]:


## Logging utilities. Please import 'logging' in order to use the following command. 
# logging.basicConfig(level=logging.INFO)


# ### 1. Download the MovieLens dataset
# 

# In[5]:



df = movielens.load_pandas_df(
    size=MOVIELENS_DATA_SIZE,
    header=["userID", "itemID", "rating", "timestamp"]
)


# ### 2. Split the data using the Spark chronological splitter provided in utilities

# In[6]:


## If both validation and test sets are required
# train, validation, test = python_random_split(df,[0.6, 0.2, 0.2])

## If validation set is not required
train, test = python_random_split(df,[0.8, 0.2])

## If test set is not required
# train, validation = python_random_split(df,[0.8, 0.2])

## If both validation and test sets are not required (i.e., the complete dataset is for training the model)
# train = df


# Generate an RLRMCdataset object from the data subsets.

# In[7]:


# data = RLRMCdataset(train=train, validation=validation, test=test)
data = RLRMCdataset(train=train, test=test) # No validation set
# data = RLRMCdataset(train=train, validation=validation) # No test set
# data = RLRMCdataset(train=train) # No validation or test set


# ### 3. Train the RLRMC model on the training data

# In[8]:


model = RLRMCalgorithm(rank = rank_parameter,
                       C = regularization_parameter,
                       model_param = data.model_param,
                       initialize_flag = initialization_flag,
                       maxiter=maximum_iteration,
                       max_time=maximum_time)


# In[9]:


start_time = time.time()

model.fit(data,verbosity=verbosity)

# fit_and_evaluate will compute RMSE on the validation set (if given) at every iteration
# model.fit_and_evaluate(data,verbosity=verbosity)

train_time = time.time() - start_time # train_time includes both model initialization and model training time. 

print("Took {} seconds for training.".format(train_time))


# ### 4. Obtain predictions from the RLRMC model on the test data

# In[10]:


## Obtain predictions on (userID,itemID) pairs (60586,54775) and (52681,36519) in Movielens 10m dataset
# output = model.predict([60586,52681],[54775,36519]) # Movielens 10m dataset

# Obtain prediction on the full test set
predictions_ndarr = model.predict(test['userID'].values,test['itemID'].values)


# ### 5. Evaluate how well RLRMC performs

# In[12]:


predictions_df = pd.DataFrame(data={"userID": test['userID'].values, "itemID":test['itemID'].values, "prediction":predictions_ndarr})

## Compute test RMSE 
eval_rmse = rmse(test, predictions_df)
## Compute test MAE 
eval_mae = mae(test, predictions_df)

print("RMSE:\t%f" % eval_rmse,
      "MAE:\t%f" % eval_mae, sep='\n')


# ### Reference
# [1] Pratik Jawanpuria and Bamdev Mishra. *A unified framework for structured low-rank matrix learning*. In International Conference on Machine Learning, 2018.
# 
# [2] Bamdev Mishra, Gilles Meyer, Francis Bach, and Rodolphe Sepulchre. *Low-rank optimization with trace norm penalty*. In SIAM Journal on Optimization 23(4):2124-2149, 2013.
# 
# [3] James Townsend, Niklas Koep, and Sebastian Weichwald. *Pymanopt: A Python Toolbox for Optimization on Manifolds using Automatic Differentiation*. In Journal of Machine Learning Research 17(137):1-5, 2016.
# 
# [4] P.-A. Absil, R. Mahony, and R. Sepulchre. *Optimization Algorithms on Matrix Manifolds*. Princeton University Press, Princeton, NJ, 2008.
# 
# [5] A. Edelman, T. Arias, and S. Smith. *The geometry of algo- rithms with orthogonality constraints*. SIAM Journal on Matrix Analysis and Applications, 20(2):303–353, 1998.
