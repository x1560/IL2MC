#!/usr/bin/env python
# coding: utf-8

# <i>Copyright (c) Microsoft Corporation. All rights reserved.</i>
# 
# <i>Licensed under the MIT License.</i>

# ## FastAI Recommender
# 
# This notebook shows how to use the [FastAI](https://fast.ai) recommender which is using [Pytorch](https://pytorch.org/) under the hood. 

# In[1]:


# set the environment path to find Recommenders
from tempfile import TemporaryDirectory
import sys
import os
import pandas as pd
import numpy as np
import scrapbook as sb
import torch, fastai
from fastai.collab import collab_learner, CollabDataBunch, load_learner

from recommenders.utils.constants import (
    DEFAULT_USER_COL as USER, 
    DEFAULT_ITEM_COL as ITEM, 
    DEFAULT_RATING_COL as RATING, 
    DEFAULT_TIMESTAMP_COL as TIMESTAMP, 
    DEFAULT_PREDICTION_COL as PREDICTION
) 
from recommenders.utils.timer import Timer
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_stratified_split
from recommenders.models.fastai.fastai_utils import cartesian_product, score
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from recommenders.evaluation.python_evaluation import rmse, mae, rsquared, exp_var

print("System version: {}".format(sys.version))
print("Pandas version: {}".format(pd.__version__))
print("Fast AI version: {}".format(fastai.__version__))
print("Torch version: {}".format(torch.__version__))
print("Cuda Available: {}".format(torch.cuda.is_available()))
print("CuDNN Enabled: {}".format(torch.backends.cudnn.enabled))


# Defining some constants to refer to the different columns of our dataset.

# In[3]:


# top k items to recommend
TOP_K = 10

# Select MovieLens data size: 100k, 1m, 10m, or 20m
MOVIELENS_DATA_SIZE = '100k'

# Model parameters
N_FACTORS = 40
EPOCHS = 5


# In[4]:


ratings_df = movielens.load_pandas_df(
    size=MOVIELENS_DATA_SIZE,
    header=[USER,ITEM,RATING,TIMESTAMP]
)

# make sure the IDs are loaded as strings to better prevent confusion with embedding ids
ratings_df[USER] = ratings_df[USER].astype('str')
ratings_df[ITEM] = ratings_df[ITEM].astype('str')

ratings_df.head()


# In[5]:


# Split the dataset
train_valid_df, test_df = python_stratified_split(
    ratings_df, 
    ratio=0.75, 
    min_rating=1, 
    filter_by="item", 
    col_user=USER, 
    col_item=ITEM
)


# In[ ]:


# Remove "cold" users from test set  
test_df = test_df[test_df.userID.isin(train_valid_df.userID)]


# ## Training

# In[6]:


# fix random seeds to make sure our runs are reproducible
np.random.seed(101)
torch.manual_seed(101)
torch.cuda.manual_seed_all(101)


# In[7]:


with Timer() as preprocess_time:
    data = CollabDataBunch.from_df(train_valid_df, 
                                   user_name=USER, 
                                   item_name=ITEM, 
                                   rating_name=RATING, 
                                   valid_pct=0)


# In[8]:


data.show_batch()


# Now we will create a `collab_learner` for the data, which by default uses the [EmbeddingDotBias](https://docs.fast.ai/collab.html#EmbeddingDotBias) model. We will be using 40 latent factors. This will create an embedding for the users and the items that will map each of these to 40 floats as can be seen below. Note that the embedding parameters are not predefined, but are learned by the model.
# 
# Although ratings can only range from 1-5, we are setting the range of possible ratings to a range from 0 to 5.5 -- that will allow the model to predict values around 1 and 5, which improves accuracy. Lastly, we set a value for weight-decay for regularization.

# In[9]:


learn = collab_learner(data, n_factors=N_FACTORS, y_range=[0,5.5], wd=1e-1)
learn.model


# Now train the model for 5 epochs setting the maximal learning rate. The learner will reduce the learning rate with each epoch using cosine annealing.

# In[10]:


with Timer() as train_time:
    learn.fit_one_cycle(EPOCHS, max_lr=5e-3)

print("Took {} seconds for training.".format(train_time))


# Save the learner so it can be loaded back later for inferencing / generating recommendations

# In[ ]:


tmp = TemporaryDirectory()
model_path = os.path.join(tmp.name, "movielens_model.pkl")


# In[11]:


learn.export(model_path)


# ## Generating Recommendations
# 
# Load the learner from disk.

# In[12]:


learner = load_learner(tmp.name, "movielens_model.pkl")


# Get all users and items that the model knows

# In[13]:


total_users, total_items = learner.data.train_ds.x.classes.values()
total_items = total_items[1:]
total_users = total_users[1:]


# Get all users from the test set and remove any users that were know in the training set

# In[14]:


test_users = test_df[USER].unique()
test_users = np.intersect1d(test_users, total_users)


# Build the cartesian product of test set users and all items known to the model

# In[15]:


users_items = cartesian_product(np.array(test_users),np.array(total_items))
users_items = pd.DataFrame(users_items, columns=[USER,ITEM])


# 
# Lastly, remove the user/items combinations that are in the training set -- we don't want to propose a movie that the user has already watched.

# In[16]:


training_removed = pd.merge(users_items, train_valid_df.astype(str), on=[USER, ITEM], how='left')
training_removed = training_removed[training_removed[RATING].isna()][[USER, ITEM]]


# ### Score the model to find the top K recommendation

# In[17]:


with Timer() as test_time:
    top_k_scores = score(learner, 
                         test_df=training_removed,
                         user_col=USER, 
                         item_col=ITEM, 
                         prediction_col=PREDICTION)

print("Took {} seconds for {} predictions.".format(test_time, len(training_removed)))


# Calculate some metrics for our model

# In[18]:


eval_map = map_at_k(test_df, top_k_scores, col_user=USER, col_item=ITEM, 
                    col_rating=RATING, col_prediction=PREDICTION, 
                    relevancy_method="top_k", k=TOP_K)


# In[19]:


eval_ndcg = ndcg_at_k(test_df, top_k_scores, col_user=USER, col_item=ITEM, 
                      col_rating=RATING, col_prediction=PREDICTION, 
                      relevancy_method="top_k", k=TOP_K)


# In[20]:


eval_precision = precision_at_k(test_df, top_k_scores, col_user=USER, col_item=ITEM, 
                                col_rating=RATING, col_prediction=PREDICTION, 
                                relevancy_method="top_k", k=TOP_K)


# In[21]:


eval_recall = recall_at_k(test_df, top_k_scores, col_user=USER, col_item=ITEM, 
                          col_rating=RATING, col_prediction=PREDICTION, 
                          relevancy_method="top_k", k=TOP_K)


# In[22]:


print("Model:\t" + learn.__class__.__name__,
      "Top K:\t%d" % TOP_K,
      "MAP:\t%f" % eval_map,
      "NDCG:\t%f" % eval_ndcg,
      "Precision@K:\t%f" % eval_precision,
      "Recall@K:\t%f" % eval_recall, sep='\n')


# The above numbers are lower than [SAR](../sar_single_node_movielens.ipynb), but expected, since the model is explicitly trying to generalize the users and items to the latent factors. Next look at how well the model predicts how the user would rate the movie. Need to score `test_df` user-items only. 

# In[23]:


scores = score(learner, 
               test_df=test_df.copy(), 
               user_col=USER, 
               item_col=ITEM, 
               prediction_col=PREDICTION)


# Now calculate some regression metrics

# In[24]:


eval_r2 = rsquared(test_df, scores, col_user=USER, col_item=ITEM, col_rating=RATING, col_prediction=PREDICTION)
eval_rmse = rmse(test_df, scores, col_user=USER, col_item=ITEM, col_rating=RATING, col_prediction=PREDICTION)
eval_mae = mae(test_df, scores, col_user=USER, col_item=ITEM, col_rating=RATING, col_prediction=PREDICTION)
eval_exp_var = exp_var(test_df, scores, col_user=USER, col_item=ITEM, col_rating=RATING, col_prediction=PREDICTION)

print("Model:\t" + learn.__class__.__name__,
      "RMSE:\t%f" % eval_rmse,
      "MAE:\t%f" % eval_mae,
      "Explained variance:\t%f" % eval_exp_var,
      "R squared:\t%f" % eval_r2, sep='\n')


# That RMSE is actually quite good when compared to these benchmarks: https://www.librec.net/release/v1.3/example.html

# In[25]:


# Record results with papermill for tests
sb.glue("map", eval_map)
sb.glue("ndcg", eval_ndcg)
sb.glue("precision", eval_precision)
sb.glue("recall", eval_recall)
sb.glue("rmse", eval_rmse)
sb.glue("mae", eval_mae)
sb.glue("exp_var", eval_exp_var)
sb.glue("rsquared", eval_r2)
sb.glue("train_time", train_time.interval)
sb.glue("test_time", test_time.interval)


# In[ ]:


tmp.cleanup()

