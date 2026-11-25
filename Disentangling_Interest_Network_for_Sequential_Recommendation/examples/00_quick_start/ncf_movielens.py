#!/usr/bin/env python
# coding: utf-8

# <i>Copyright (c) Microsoft Corporation. All rights reserved.</i>
# 
# <i>Licensed under the MIT License.</i>

# # Neural Collaborative Filtering on MovieLens dataset.
# 
# Neural Collaborative Filtering (NCF) is a well known recommendation algorithm that generalizes the matrix factorization problem with multi-layer perceptron. 
# 
# This notebook provides an example of how to utilize and evaluate NCF implementation in the `recommenders`. We use a smaller dataset in this example to run NCF efficiently with GPU acceleration on a [Data Science Virtual Machine](https://azure.microsoft.com/en-gb/services/virtual-machines/data-science-virtual-machines/).

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[3]:


import sys
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # only show error messages

from recommenders.utils.timer import Timer
from recommenders.models.ncf.ncf_singlenode import NCF
from recommenders.models.ncf.dataset import Dataset as NCFDataset
from recommenders.datasets import movielens
from recommenders.utils.notebook_utils import is_jupyter
from recommenders.datasets.python_splitters import python_chrono_split
from recommenders.evaluation.python_evaluation import (rmse, mae, rsquared, exp_var, map_at_k, ndcg_at_k, precision_at_k, 
                                                     recall_at_k, get_top_k_items)

print("System version: {}".format(sys.version))
print("Pandas version: {}".format(pd.__version__))
print("Tensorflow version: {}".format(tf.__version__))


# Set the default parameters.

# In[4]:


# top k items to recommend
TOP_K = 10

# Select MovieLens data size: 100k, 1m, 10m, or 20m
MOVIELENS_DATA_SIZE = '100k'

# Model parameters
EPOCHS = 50
BATCH_SIZE = 256

SEED = 42


# ### 1. Download the MovieLens dataset

# In[5]:


df = movielens.load_pandas_df(
    size=MOVIELENS_DATA_SIZE,
    header=["userID", "itemID", "rating", "timestamp"]
)


# ### 2. Split the data using the Spark chronological splitter provided in utilities

# In[6]:


train, test = python_chrono_split(df, 0.75)


# Filter out any users or items in the test set that do not appear in the training set.

# In[7]:


test = test[test["userID"].isin(train["userID"].unique())]
test = test[test["itemID"].isin(train["itemID"].unique())]


# Write datasets to csv files.

# In[8]:


train_file = "./train.csv"
test_file = "./test.csv"
train.to_csv(train_file, index=False)
test.to_csv(test_file, index=False)


# Generate an NCF dataset object from the data subsets.

# In[9]:


data = NCFDataset(train_file=train_file, test_file=test_file, seed=SEED)


# ### 3. Train the NCF model on the training data, and get the top-k recommendations for our testing data
# 
# NCF accepts implicit feedback and generates prospensity of items to be recommended to users in the scale of 0 to 1. A recommended item list can then be generated based on the scores. Note that this quickstart notebook is using a smaller number of epochs to reduce time for training. As a consequence, the model performance will be slighlty deteriorated. 

# In[ ]:


model = NCF (
    n_users=data.n_users, 
    n_items=data.n_items,
    model_type="NeuMF",
    n_factors=4,
    layer_sizes=[16,8,4],
    n_epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=1e-3,
    verbose=10,
    seed=SEED
)


# In[11]:


with Timer() as train_time:
    model.fit(data)

print("Took {} seconds for training.".format(train_time))


# In the movie recommendation use case scenario, seen movies are not recommended to the users.

# In[12]:


with Timer() as test_time:
    users, items, preds = [], [], []
    item = list(train.itemID.unique())
    for user in train.userID.unique():
        user = [user] * len(item) 
        users.extend(user)
        items.extend(item)
        preds.extend(list(model.predict(user, item, is_list=True)))

    all_predictions = pd.DataFrame(data={"userID": users, "itemID":items, "prediction":preds})

    merged = pd.merge(train, all_predictions, on=["userID", "itemID"], how="outer")
    all_predictions = merged[merged.rating.isnull()].drop('rating', axis=1)

print("Took {} seconds for prediction.".format(test_time))


# ### 4. Evaluate how well NCF performs

# The ranking metrics are used for evaluation.

# In[13]:


eval_map = map_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
eval_ndcg = ndcg_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
eval_precision = precision_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
eval_recall = recall_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)

print("MAP:\t%f" % eval_map,
      "NDCG:\t%f" % eval_ndcg,
      "Precision@K:\t%f" % eval_precision,
      "Recall@K:\t%f" % eval_recall, sep='\n')


# In[ ]:


if is_jupyter():
    # Record results with papermill for tests
    import papermill as pm
    import scrapbook as sb
    sb.glue("map", eval_map)
    sb.glue("ndcg", eval_ndcg)
    sb.glue("precision", eval_precision)
    sb.glue("recall", eval_recall)
    sb.glue("train_time", train_time.interval)
    sb.glue("test_time", test_time.interval)


# In[ ]:




