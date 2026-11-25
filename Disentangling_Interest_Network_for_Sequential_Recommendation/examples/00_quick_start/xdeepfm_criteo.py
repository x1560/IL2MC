#!/usr/bin/env python
# coding: utf-8

# <i>Copyright (c) Microsoft Corporation. All rights reserved.</i>
# 
# <i>Licensed under the MIT License.</i>

# # xDeepFM : the eXtreme Deep Factorization Machine 
# This notebook will give you a quick example of how to train an [xDeepFM model](https://arxiv.org/abs/1803.05170). 
# xDeepFM \[1\] is a deep learning-based model aims at capturing both lower- and higher-order feature interactions for precise recommender systems. Thus it can learn feature interactions more effectively and manual feature engineering effort can be substantially reduced. To summarize, xDeepFM has the following key properties:
# * It contains a component, named CIN, that learns feature interactions in an explicit fashion and in vector-wise level;
# * It contains a traditional DNN component that learns feature interactions in an implicit fashion and in bit-wise level.
# * The implementation makes this model quite configurable. We can enable different subsets of components by setting hyperparameters like `use_Linear_part`, `use_FM_part`, `use_CIN_part`, and `use_DNN_part`. For example, by enabling only the `use_Linear_part` and `use_FM_part`, we can get a classical FM model.
# 
# In this notebook, we test xDeepFM on two datasets: 1) a small synthetic dataset and 2) [Criteo dataset](http://labs.criteo.com/category/dataset)

# ## 0. Global Settings and Imports

# In[1]:


import sys
import os
import scrapbook as sb
from tempfile import TemporaryDirectory
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # only show error messages

from recommenders.utils.constants import SEED
from recommenders.models.deeprec.deeprec_utils import (
    download_deeprec_resources, prepare_hparams
)
from recommenders.models.deeprec.models.xDeepFM import XDeepFMModel
from recommenders.models.deeprec.io.iterator import FFMTextIterator

print("System version: {}".format(sys.version))
print("Tensorflow version: {}".format(tf.__version__))


# #### Parameters

# In[2]:


EPOCHS_FOR_SYNTHETIC_RUN = 15
EPOCHS_FOR_CRITEO_RUN = 10
BATCH_SIZE_SYNTHETIC = 128
BATCH_SIZE_CRITEO = 4096
RANDOM_SEED = SEED  # Set to None for non-deterministic result


# xDeepFM uses the FFM format as data input: `<label> <field_id>:<feature_id>:<feature_value>`  
# Each line represents an instance, `<label>` is a binary value with 1 meaning positive instance and 0 meaning negative instance. 
# Features are divided into fields. For example, user's gender is a field, it contains three possible values, i.e. male, female and unknown. Occupation can be another field, which contains many more possible values than the gender field. Both field index and feature index are starting from 1. <br>
# 
# ## 1. Synthetic data
# Now let's start with a small synthetic dataset. In this dataset, there are 10 fields, 1000 fefatures, and label is generated according to the result of a set of preset pair-wise feature interactions. 

# In[3]:


tmpdir = TemporaryDirectory()
data_path = tmpdir.name
yaml_file = os.path.join(data_path, r'xDeepFM.yaml')
train_file = os.path.join(data_path, r'synthetic_part_0')
valid_file = os.path.join(data_path, r'synthetic_part_1')
test_file = os.path.join(data_path, r'synthetic_part_2')
output_file = os.path.join(data_path, r'output.txt')

if not os.path.exists(yaml_file):
    download_deeprec_resources(r'https://recodatasets.z20.web.core.windows.net/deeprec/', data_path, 'xdeepfmresources.zip')


# #### 1.1 Prepare hyper-parameters
# prepare_hparams() will create a full set of hyper-parameters for model training, such as learning rate, feature number, and dropout ratio. We can put those parameters in a yaml file, or pass parameters as the function's parameters (which will overwrite yaml settings).

# In[4]:


hparams = prepare_hparams(yaml_file, 
                          FEATURE_COUNT=1000, 
                          FIELD_COUNT=10, 
                          cross_l2=0.0001, 
                          embed_l2=0.0001, 
                          learning_rate=0.001, 
                          epochs=EPOCHS_FOR_SYNTHETIC_RUN,
                          batch_size=BATCH_SIZE_SYNTHETIC)
print(hparams)


# #### 1.2 Create data loader
# Designate a data iterator for the model. xDeepFM uses FFMTextIterator. 

# In[5]:


input_creator = FFMTextIterator


# #### 1.3 Create model
# When both hyper-parameters and data iterator are ready, we can create a model:

# In[6]:


model = XDeepFMModel(hparams, input_creator, seed=RANDOM_SEED)

## sometimes we don't want to train a model from scratch
## then we can load a pre-trained model like this: 
#model.load_model(r'your_model_path')


# Now let's see what is the model's performance at this point (without starting training):

# In[7]:


print(model.run_eval(test_file))


# AUC=0.5 is a state of random guess. We can see that before training, the model behaves like random guessing.
# 
# #### 1.4 Train model
# Next we want to train the model on a training set, and check the performance on a validation dataset. Training the model is as simple as a function call:

# In[8]:


model.fit(train_file, valid_file)


# #### 1.5 Evaluate model
# 
# Again, let's see what is the model's performance now (after training):

# In[9]:


res_syn = model.run_eval(test_file)
print(res_syn)


# In[10]:


sb.glue("res_syn", res_syn)


# If we want to get the full prediction scores rather than evaluation metrics, we can do this:

# In[11]:


model.predict(test_file, output_file)


# ## 2. Criteo data 
# 
# Now we have successfully launched an experiment on a synthetic dataset. Next let's try something on a real world dataset, which is a small sample from [Criteo dataset](http://labs.criteo.com/category/dataset). Criteo dataset is a well known industry benchmarking dataset for developing CTR prediction models and it's frequently adopted as evaluation dataset by research papers. The original dataset is too large for a lightweight demo, so we sample a small portion from it as a demo dataset.

# In[12]:


print('demo with Criteo dataset')
hparams = prepare_hparams(yaml_file, 
                          FEATURE_COUNT=2300000, 
                          FIELD_COUNT=39, 
                          cross_l2=0.01, 
                          embed_l2=0.01, 
                          layer_l2=0.01,
                          learning_rate=0.002, 
                          batch_size=BATCH_SIZE_CRITEO, 
                          epochs=EPOCHS_FOR_CRITEO_RUN, 
                          cross_layer_sizes=[20, 10], 
                          init_value=0.1, 
                          layer_sizes=[20,20],
                          use_Linear_part=True, 
                          use_CIN_part=True, 
                          use_DNN_part=True)


# In[13]:


train_file = os.path.join(data_path, r'cretio_tiny_train')
valid_file = os.path.join(data_path, r'cretio_tiny_valid')
test_file = os.path.join(data_path, r'cretio_tiny_test')


# In[14]:


model = XDeepFMModel(hparams, FFMTextIterator, seed=RANDOM_SEED)


# In[15]:


# check the predictive performance before the model is trained
print(model.run_eval(test_file)) 


# In[16]:


model.fit(train_file, valid_file)


# In[17]:


# check the predictive performance after the model is trained
res_real = model.run_eval(test_file)
print(res_real)


# In[18]:


sb.glue("res_real", res_real)


# In[19]:


# Cleanup
tmpdir.cleanup()


# ## Reference
# \[1\] Lian, J., Zhou, X., Zhang, F., Chen, Z., Xie, X., & Sun, G. (2018). xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems. Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining, KDD 2018, London, UK, August 19-23, 2018.<br>
