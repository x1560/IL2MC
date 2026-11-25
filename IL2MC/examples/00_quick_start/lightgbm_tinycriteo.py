#!/usr/bin/env python
# coding: utf-8

# <i>Copyright (c) Microsoft Corporation. All rights reserved.</i>
# 
# <i>Licensed under the MIT License.</i>

# # LightGBM: A Highly Efficient Gradient Boosting Decision Tree
# This notebook will give you an example of how to train a LightGBM model to estimate click-through rates on an e-commerce advertisement. We will train a LightGBM based model on the Criteo dataset.
# 
# [LightGBM](https://github.com/Microsoft/LightGBM) is a gradient boosting framework that uses tree-based learning algorithms. It is designed to be distributed and efficient with the following advantages:
# * Fast training speed and high efficiency.
# * Low memory usage.
# * Great accuracy.
# * Support of parallel and GPU learning.
# * Capable of handling large-scale data.

# ## Global Settings and Imports

# In[9]:


import sys
import os
import numpy as np
import lightgbm as lgb
import papermill as pm
import scrapbook as sb
import pandas as pd
import category_encoders as ce
from tempfile import TemporaryDirectory
from sklearn.metrics import roc_auc_score, log_loss

import recommenders.models.lightgbm.lightgbm_utils as lgb_utils
import recommenders.datasets.criteo as criteo

print("System version: {}".format(sys.version))
print("LightGBM version: {}".format(lgb.__version__))


# ### Parameter Setting
# Let's set the main related parameters for LightGBM now. Basically, the task is a binary classification (predicting click or no click), so the objective function is set to binary logloss, and 'AUC' metric, is used as a metric which is less effected by imbalance in the classes of the dataset.
# 
# Generally, we can adjust the number of leaves (MAX_LEAF), the minimum number of data in each leaf (MIN_DATA), maximum number of trees (NUM_OF_TREES), the learning rate of trees (TREE_LEARNING_RATE) and EARLY_STOPPING_ROUNDS (to avoid overfitting) in the model to get better performance.
# 
# Besides, we can also adjust some other listed parameters to optimize the results. [In this link](https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst), a list of all the parameters is shown. Also, some advice on how to tune these parameters can be found [in this url](https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters-Tuning.rst). 

# In[2]:


MAX_LEAF = 64
MIN_DATA = 20
NUM_OF_TREES = 100
TREE_LEARNING_RATE = 0.15
EARLY_STOPPING_ROUNDS = 20
METRIC = "auc"
SIZE = "sample"


# In[3]:


params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'num_class': 1,
    'objective': "binary",
    'metric': METRIC,
    'num_leaves': MAX_LEAF,
    'min_data': MIN_DATA,
    'boost_from_average': True,
    #set it according to your cpu cores.
    'num_threads': 20,
    'feature_fraction': 0.8,
    'learning_rate': TREE_LEARNING_RATE,
}


# ## Data Preparation
# Here we use CSV format as the example data input. Our example data is a sample (about 100 thousand samples) from [Criteo dataset](https://www.kaggle.com/c/criteo-display-ad-challenge). The Criteo dataset is a well-known industry benchmarking dataset for developing CTR prediction models, and it's frequently adopted as evaluation dataset by research papers. The original dataset is too large for a lightweight demo, so we sample a small portion from it as a demo dataset.
# 
# Specifically, there are 39 columns of features in Criteo, where 13 columns are numerical features (I1-I13) and the other 26 columns are categorical features (C1-C26).

# In[4]:


nume_cols = ["I" + str(i) for i in range(1, 14)]
cate_cols = ["C" + str(i) for i in range(1, 27)]
label_col = "Label"

header = [label_col] + nume_cols + cate_cols
with TemporaryDirectory() as tmp:
    all_data = criteo.load_pandas_df(size=SIZE, local_cache_path=tmp, header=header)
display(all_data.head())


# First, we cut three sets (train_data (first 80%), valid_data (middle 10%) and test_data (last 10%)), cut from the original all data. <br>
# Notably, considering the Criteo is a kind of time-series streaming data, which is also very common in recommendation scenario, we split the data by its order.

# In[5]:


# split data to 3 sets    
length = len(all_data)
train_data = all_data.loc[:0.8*length-1]
valid_data = all_data.loc[0.8*length:0.9*length-1]
test_data = all_data.loc[0.9*length:]


# ## Basic Usage
# ### Ordinal Encoding
# Considering LightGBM could handle the low-frequency features and missing value by itself, for basic usage, we only encode the string-like categorical features by an ordinal encoder.

# In[11]:


ord_encoder = ce.ordinal.OrdinalEncoder(cols=cate_cols)

def encode_csv(df, encoder, label_col, typ='fit'):
    if typ == 'fit':
        df = encoder.fit_transform(df)
    else:
        df = encoder.transform(df)
    y = df[label_col].values
    del df[label_col]
    return df, y

train_x, train_y = encode_csv(train_data, ord_encoder, label_col)
valid_x, valid_y = encode_csv(valid_data, ord_encoder, label_col, 'transform')
test_x, test_y = encode_csv(test_data, ord_encoder, label_col, 'transform')

print('Train Data Shape: X: {trn_x_shape}; Y: {trn_y_shape}.\nValid Data Shape: X: {vld_x_shape}; Y: {vld_y_shape}.\nTest Data Shape: X: {tst_x_shape}; Y: {tst_y_shape}.\n'
      .format(trn_x_shape=train_x.shape,
              trn_y_shape=train_y.shape,
              vld_x_shape=valid_x.shape,
              vld_y_shape=valid_y.shape,
              tst_x_shape=test_x.shape,
              tst_y_shape=test_y.shape,))
train_x.head()


# ### Create model
# When both hyper-parameters and data are ready, we can create a model:

# In[7]:


lgb_train = lgb.Dataset(train_x, train_y.reshape(-1), params=params, categorical_feature=cate_cols)
lgb_valid = lgb.Dataset(valid_x, valid_y.reshape(-1), reference=lgb_train, categorical_feature=cate_cols)
lgb_test = lgb.Dataset(test_x, test_y.reshape(-1), reference=lgb_train, categorical_feature=cate_cols)
lgb_model = lgb.train(params,
                      lgb_train,
                      num_boost_round=NUM_OF_TREES,
                      early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                      valid_sets=lgb_valid,
                      categorical_feature=cate_cols)


# Now let's see what is the model's performance:

# In[12]:


test_preds = lgb_model.predict(test_x)
auc = roc_auc_score(np.asarray(test_y.reshape(-1)), np.asarray(test_preds))
logloss = log_loss(np.asarray(test_y.reshape(-1)), np.asarray(test_preds), eps=1e-12)
res_basic = {"auc": auc, "logloss": logloss}
print(res_basic)
sb.glue("res_basic", res_basic)


# <script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
# ## Optimized Usage
# ### Label-encoding and Binary-encoding
# Next, since LightGBM has a better capability in handling dense numerical features effectively, we try to convert all the categorical features in original data into numerical ones, by label-encoding [3] and binary-encoding [4]. Also due to the sequence property of Criteo, the label-encoding we adopted is executed one-by-one, which means we encode the samples in order, by the information of the previous samples before each sample (sequential label-encoding and sequential count-encoding). Besides, we also filter the low-frequency categorical features and fill the missing values by the mean of corresponding columns for the numerical features. (consulting `lgb_utils.NumEncoder`)
# 
# Specifically, in `lgb_utils.NumEncoder`, the main steps are as follows.
# * Firstly, we convert the low-frequency categorical features to `"LESS"` and the missing categorical features to `"UNK"`. 
# * Secondly, we convert the missing numerical features into the mean of corresponding columns. 
# * Thirdly, the string-like categorical features are ordinal encoded like the example shown in basic usage. 
# * And then, we target encode the categorical features in the samples order one-by-one. For each sample, we add the label and count information of its former samples into the data and produce new features. Formally, for $i=1,2,...,n$, we add $\frac{\sum\nolimits_{j=1}^{i-1} I(x_j=c) \cdot y}{\sum\nolimits_{j=1}^{i-1} I(x_j=c)}$ as a new label feature for current sample $x_i$, where $c$ is a category to encode in current sample, so $(i-1)$ is the number of former samples, and $I(\cdot)$ is the indicator function that check the former samples contain $c$ (whether $x_j=c$) or not. At the meantime, we also add the count frequency of $c$, which is $\frac{\sum\nolimits_{j=1}^{i-1} I(x_j=c)}{i-1}$, as a new count feature. 
# * Finally, based on the results of ordinal encoding, we add the binary encoding results as new columns into the data.
# 
# Note that the statistics used in the above process only updates when fitting the training set, while maintaining static when transforming the testing set because the label of test data should be considered as unknown.

# In[15]:


label_col = 'Label'
num_encoder = lgb_utils.NumEncoder(cate_cols, nume_cols, label_col)
train_x, train_y = num_encoder.fit_transform(train_data)
valid_x, valid_y = num_encoder.transform(valid_data)
test_x, test_y = num_encoder.transform(test_data)
del num_encoder
print('Train Data Shape: X: {trn_x_shape}; Y: {trn_y_shape}.\nValid Data Shape: X: {vld_x_shape}; Y: {vld_y_shape}.\nTest Data Shape: X: {tst_x_shape}; Y: {tst_y_shape}.\n'
      .format(trn_x_shape=train_x.shape,
              trn_y_shape=train_y.shape,
              vld_x_shape=valid_x.shape,
              vld_y_shape=valid_y.shape,
              tst_x_shape=test_x.shape,
              tst_y_shape=test_y.shape,))


# ### Training and Evaluation

# In[16]:


lgb_train = lgb.Dataset(train_x, train_y.reshape(-1), params=params)
lgb_valid = lgb.Dataset(valid_x, valid_y.reshape(-1), reference=lgb_train)
lgb_model = lgb.train(params,
                      lgb_train,
                      num_boost_round=NUM_OF_TREES,
                      early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                      valid_sets=lgb_valid)


# In[17]:


test_preds = lgb_model.predict(test_x)
auc = roc_auc_score(np.asarray(test_y.reshape(-1)), np.asarray(test_preds))
logloss = log_loss(np.asarray(test_y.reshape(-1)), np.asarray(test_preds), eps=1e-12)
res_optim = {"auc": auc, "logloss": logloss}
print(res_optim)
sb.glue("res_optim", res_optim)


# ## Model saving and loading
# Now we finish the basic training and testing for LightGBM, next let's try to save and reload the model, and then evaluate it again.

# In[18]:


with TemporaryDirectory() as tmp:
    save_file = os.path.join(tmp, r'finished.model')
    lgb_model.save_model(save_file)
    loaded_model = lgb.Booster(model_file=save_file)

# eval the performance again
test_preds = loaded_model.predict(test_x)

auc = roc_auc_score(np.asarray(test_y.reshape(-1)), np.asarray(test_preds))
logloss = log_loss(np.asarray(test_y.reshape(-1)), np.asarray(test_preds), eps=1e-12)
print({"auc": auc, "logloss": logloss})


# ## Additional Reading
# 
# \[1\] Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu. 2017. LightGBM: A highly efficient gradient boosting decision tree. In Advances in Neural Information Processing Systems. 3146–3154.<br>
# \[2\] The parameters of LightGBM: https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst <br>
# \[3\] Anna Veronika Dorogush, Vasily Ershov, and Andrey Gulin. 2018. CatBoost: gradient boosting with categorical features support. arXiv preprint arXiv:1810.11363 (2018).<br>
# \[4\] Scikit-learn. 2018. categorical_encoding. https://github.com/scikit-learn-contrib/categorical-encoding<br>
# 
