#!/usr/bin/env python
# coding: utf-8

# <i>Copyright (c) Microsoft Corporation. All rights reserved.</i>
# 
# <i>Licensed under the MIT License.</i>

# # DKN : Deep Knowledge-Aware Network for News Recommendation
# 
# DKN \[1\] is a deep learning model which incorporates information from knowledge graph for better news recommendation. Specifically, DKN uses TransX \[2\] method for knowledge graph representation learning, then applies a CNN framework, named KCNN, to combine entity embedding with word embedding and generate a final embedding vector for a news article. CTR prediction is made via an attention-based neural scorer. 
# 
# ## Properties of DKN:
# 
# - DKN is a content-based deep model for CTR prediction rather than traditional ID-based collaborative filtering. 
# - It makes use of knowledge entities and common sense in news content via joint learning from semantic-level and knowledge-level representations of news articles.
# - DKN uses an attention module to dynamically calculate a user's aggregated historical representaition.
# 
# 
# ## Data format:
# 
# ### DKN takes several files as input as follows:
# 
# - **training / validation / test files**: each line in these files represents one instance. Impressionid is used to evaluate performance within an impression session, so it is only used when evaluating, you can set it to 0 for training data. The format is : <br> 
# `[label] [userid] [CandidateNews]%[impressionid] `<br> 
# e.g., `1 train_U1 N1%0` <br> 
# 
# - **user history file**: each line in this file represents a users' click history. You need to set `history_size` parameter in the config file, which is the max number of user's click history we use. We will automatically keep the last `history_size` number of user click history, if user's click history is more than `history_size`, and we will automatically pad with 0 if user's click history is less than `history_size`. the format is : <br> 
# `[Userid] [newsid1,newsid2...]`<br>
# e.g., `train_U1 N1,N2` <br> 
# 
# - **document feature file**: It contains the word and entity features for news articles. News articles are represented by aligned title words and title entities. To take a quick example, a news title may be: <i>"Trump to deliver State of the Union address next week"</i>, then the title words value may be `CandidateNews:34,45,334,23,12,987,3456,111,456,432` and the title entitie value may be: `entity:45,0,0,0,0,0,0,0,0,0`. Only the first value of entity vector is non-zero due to the word "Trump". The title value and entity value is hashed from 1 to `n` (where `n` is the number of distinct words or entities). Each feature length should be fixed at k (`doc_size` parameter), if the number of words in document is more than k, you should truncate the document to k words, and if the number of words in document is less than k, you should pad 0 to the end. 
# the format is like: <br> 
# `[Newsid] [w1,w2,w3...wk] [e1,e2,e3...ek]`
# 
# - **word embedding/entity embedding/ context embedding files**: These are `*.npy` files of pretrained embeddings. After loading, each file is a `[n+1,k]` two-dimensional matrix, n is the number of words(or entities) of their hash dictionary, k is dimension of the embedding, note that we keep embedding 0 for zero padding. 
# 
# In this experiment, we used GloVe\[4\] vectors to initialize the word embedding. We trained entity embedding using TransE\[2\] on knowledge graph and context embedding is the average of the entity's neighbors in the knowledge graph.<br>
# 
# ## MIND dataset
# 
# MIND dataset\[3\] is a large-scale English news dataset. It was collected from anonymized behavior logs of Microsoft News website. MIND contains 1,000,000 users, 161,013 news articles and 15,777,377 impression logs. Every news article contains rich textual content including title, abstract, body, category and entities. Each impression log contains the click events, non-clicked events and historical news click behaviors of this user before this impression.
# 
# In this notebook we are going to use a subset of MIND dataset, **MIND demo**. MIND demo contains 500 users, 9,432 news articles  and 6,134 impression logs. 
# 
# For this quick start notebook, we are providing directly all the necessary word embeddings, entity embeddings and context embedding files.

# ## Global settings and imports

# In[1]:


import sys
import os
from tempfile import TemporaryDirectory
import scrapbook as sb
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # only show error messages

from recommenders.models.deeprec.deeprec_utils import download_deeprec_resources, prepare_hparams
from recommenders.models.deeprec.models.dkn import DKN
from recommenders.models.deeprec.io.dkn_iterator import DKNTextIterator

print(f"System version: {sys.version}")
print(f"Tensorflow version: {tf.__version__}")


# ## Download and load data

# In[3]:


tmpdir = TemporaryDirectory()
data_path = os.path.join(tmpdir.name, "mind-demo-dkn")

yaml_file = os.path.join(data_path, r'dkn.yaml')
train_file = os.path.join(data_path, r'train_mind_demo.txt')
valid_file = os.path.join(data_path, r'valid_mind_demo.txt')
test_file = os.path.join(data_path, r'test_mind_demo.txt')
news_feature_file = os.path.join(data_path, r'doc_feature.txt')
user_history_file = os.path.join(data_path, r'user_history.txt')
wordEmb_file = os.path.join(data_path, r'word_embeddings_100.npy')
entityEmb_file = os.path.join(data_path, r'TransE_entity2vec_100.npy')
contextEmb_file = os.path.join(data_path, r'TransE_context2vec_100.npy')
if not os.path.exists(yaml_file):
    download_deeprec_resources(r'https://recodatasets.z20.web.core.windows.net/deeprec/', tmpdir.name, 'mind-demo-dkn.zip')
    


# ## Create hyper-parameters

# In[4]:


epochs = 10
history_size = 50
batch_size = 100


# In[5]:


hparams = prepare_hparams(yaml_file,
                          news_feature_file = news_feature_file,
                          user_history_file = user_history_file,
                          wordEmb_file=wordEmb_file,
                          entityEmb_file=entityEmb_file,
                          contextEmb_file=contextEmb_file,
                          epochs=epochs,
                          history_size=history_size,
                          batch_size=batch_size)
print(hparams)


# ## Train the DKN model

# In[6]:


model = DKN(hparams, DKNTextIterator)


# In[7]:


print(model.run_eval(valid_file))


# In[8]:


model.fit(train_file, valid_file)


# ## Evaluate the DKN model
# 
# Now we can check the performance on the test set:

# In[9]:


res = model.run_eval(test_file)
print(res)


# In[ ]:


sb.glue("res", res)


# ## References
# 
# \[1\] Wang, Hongwei, et al. "DKN: Deep Knowledge-Aware Network for News Recommendation." Proceedings of the 2018 World Wide Web Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 2018.<br>
# \[2\] Knowledge Graph Embeddings including TransE, TransH, TransR and PTransE. https://github.com/thunlp/KB2E <br>
# \[3\] Wu, Fangzhao, et al. "MIND: A Large-scale Dataset for News Recommendation" Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. https://msnews.github.io/competition.html <br>
# \[4\] GloVe: Global Vectors for Word Representation. https://nlp.stanford.edu/projects/glove/
