#!/usr/bin/env python
# coding: utf-8

# # Geometry Aware Inductive Matrix Completion (GeoIMC)
# 
# GeoIMC is an inductive matrix completion algorithm based on the works by Jawanpuria et al. (2019)
# 
# Consider the case of MovieLens-100K (ML100K), Let $X \in R^{m \times d_1}, Z \in R^{n \times d_2} $ be the features of users and movies respectively. Let $M \in R^{m \times n}$, be the partially observed ratings matrix. GeoIMC models this matrix as $M = XUBV^TZ^T$, where $U \in R^{d_1 \times k}, V \in R^{d_2 \times k}, B \in R^{k \times k}$ are Orthogonal, Orthogonal, Symmetric Positive-Definite matrices respectively. This Optimization problem is solved by using Pymanopt.
# 
# 
# This notebook provides an example of how to utilize and evaluate GeoIMC implementation in **recommenders**
# 

# In[1]:


import tempfile
import zipfile
import pandas as pd
import numpy as np
import papermill as pm
import scrapbook as sb

from recommenders.datasets import movielens
from recommenders.models.geoimc.geoimc_data import ML_100K
from recommenders.models.geoimc.geoimc_algorithm import IMCProblem
from recommenders.models.geoimc.geoimc_predict import Inferer
from recommenders.evaluation.python_evaluation import (
    rmse, mae
)


# In[2]:


# Choose the MovieLens dataset
MOVIELENS_DATA_SIZE = '100k'
# Normalize user, item features
normalize = True
# Rank (k) of the model
rank = 300
# Regularization parameter
regularizer = 1e-3

# Parameters for algorithm convergence
max_iters = 150000
max_time = 1000
verbosity = 1


# ## 1. Download ML100K dataset and features

# In[3]:


# Create a directory to download ML100K
dp = tempfile.mkdtemp(suffix='-geoimc')
movielens.download_movielens(MOVIELENS_DATA_SIZE, f"{dp}/ml-100k.zip")
with zipfile.ZipFile(f"{dp}/ml-100k.zip", 'r') as z:
    z.extractall(dp)


# ## 2. Load the dataset using the example features provided in helpers
# 
# The features were generated using the same method as the work by Xin Dong et al. (2017)

# In[4]:


dataset = ML_100K(
    normalize=normalize,
    target_transform='binarize'
)


# In[5]:


dataset.load_data(f"{dp}/ml-100k/")


# In[6]:


print(f"""Characteristics:

              target: {dataset.training_data.data.shape}
              entities: {dataset.entities[0].shape}, {dataset.entities[1].shape}

              training: {dataset.training_data.get_data().data.shape}
              training_entities: {dataset.training_data.get_entity("row").shape}, {dataset.training_data.get_entity("col").shape}

              testing: {dataset.test_data.get_data().data.shape}
              test_entities: {dataset.test_data.get_entity("row").shape}, {dataset.test_data.get_entity("col").shape}
""")


# ## 3. Initialize the IMC problem

# In[7]:


np.random.seed(10)
prblm = IMCProblem(
    dataset.training_data,
    lambda1=regularizer,
    rank=rank
)


# In[8]:


# Solve the Optimization problem
prblm.solve(
    max_time,
    max_iters,
    verbosity
)


# In[9]:


# Initialize an inferer
inferer = Inferer(
    method='dot'
)


# In[10]:


# Predict using the parametrized matrices
predictions = inferer.infer(
    dataset.test_data,
    prblm.W
)


# In[11]:


# Prepare the test, predicted dataframes
user_ids = dataset.test_data.get_data().tocoo().row
item_ids = dataset.test_data.get_data().tocoo().col
test_df = pd.DataFrame(
    data={
        "userID": user_ids,
        "itemID": item_ids,
        "rating": dataset.test_data.get_data().data
    }
)
predictions_df = pd.DataFrame(
    data={
        "userID": user_ids,
        "itemID": item_ids,
        "prediction": [predictions[uid, iid] for uid, iid in list(zip(user_ids, item_ids))]
    }
)


# In[12]:


# Calculate RMSE
RMSE = rmse(
    test_df,
    predictions_df
)
# Calculate MAE
MAE = mae(
    test_df,
    predictions_df
)
print(f"""
RMSE: {RMSE}
MAE: {MAE}
""")


# In[ ]:


sb.glue("rmse", RMSE)
sb.glue("mae", MAE)


# ## References
# 
# [1] Pratik Jawanpuria, Arjun Balgovind, Anoop Kunchukuttan, Bamdev Mishra. _[Learning Multilingual Word Embeddings in Latent Metric Space: A Geometric Approach](https://www.mitpressjournals.org/doi/full/10.1162/tacl_a_00257)_. Transaction of the Association for Computational Linguistics (TACL), Volume 7, p.107-120, 2019.
# 
# [2] Xin Dong, Lei Yu, Zhonghuo Wu, Yuxia Sun, Lingfeng Yuan, Fangxi Zhang. [A Hybrid Collaborative Filtering Model withDeep Structure for Recommender Systems](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14676/13916).
# Proceedings of the Thirty-First AAAI Conference on Artificial Intelligence (AAAI-17), p.1309-1315, 2017.
