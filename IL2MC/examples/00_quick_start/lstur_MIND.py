#!/usr/bin/env python
# coding: utf-8

# <i>Copyright (c) Microsoft Corporation. All rights reserved.</i>
# 
# <i>Licensed under the MIT License.</i>

# # LSTUR: Neural News Recommendation with Long- and Short-term User Representations
# LSTUR \[1\] is a news recommendation approach capturing users' both long-term preferences and short-term interests. The core of LSTUR is a news encoder and a user encoder.  In the news encoder, we learn representations of news from their titles. In user encoder, we propose to learn long-term
# user representations from the embeddings of their IDs. In addition, we propose to learn short-term user representations from their recently browsed news via GRU network. Besides, we propose two methods to combine
# long-term and short-term user representations. The first one is using the long-term user representation to initialize the hidden state of the GRU network in short-term user representation. The second one is concatenating both
# long- and short-term user representations as a unified user vector.
# 
# ## Properties of LSTUR:
# - LSTUR captures users' both long-term and short term preference.
# - It uses embeddings of users' IDs to learn long-term user representations.
# - It uses users' recently browsed news via GRU network to learn short-term user representations.
# 
# ## Data format:
# For quicker training and evaluaiton, we sample MINDdemo dataset of 5k users from [MIND small dataset](https://msnews.github.io/). The MINDdemo dataset has the same file format as MINDsmall and MINDlarge. If you want to try experiments on MINDsmall and MINDlarge, please change the dowload source. Select the MIND_type parameter from ['large', 'small', 'demo'] to choose dataset.
#  
# **MINDdemo_train** is used for training, and **MINDdemo_dev** is used for evaluation. Training data and evaluation data are composed of a news file and a behaviors file. You can find more detailed data description in [MIND repo](https://github.com/msnews/msnews.github.io/blob/master/assets/doc/introduction.md)
# 
# ### news data
# This file contains news information including newsid, category, subcatgory, news title, news abstarct, news url and entities in news title, entities in news abstarct.
# One simple example: <br>
# 
# `N46466	lifestyle	lifestyleroyals	The Brands Queen Elizabeth, Prince Charles, and Prince Philip Swear By	Shop the notebooks, jackets, and more that the royals can't live without.	https://www.msn.com/en-us/lifestyle/lifestyleroyals/the-brands-queen-elizabeth,-prince-charles,-and-prince-philip-swear-by/ss-AAGH0ET?ocid=chopendata	[{"Label": "Prince Philip, Duke of Edinburgh", "Type": "P", "WikidataId": "Q80976", "Confidence": 1.0, "OccurrenceOffsets": [48], "SurfaceForms": ["Prince Philip"]}, {"Label": "Charles, Prince of Wales", "Type": "P", "WikidataId": "Q43274", "Confidence": 1.0, "OccurrenceOffsets": [28], "SurfaceForms": ["Prince Charles"]}, {"Label": "Elizabeth II", "Type": "P", "WikidataId": "Q9682", "Confidence": 0.97, "OccurrenceOffsets": [11], "SurfaceForms": ["Queen Elizabeth"]}]	[]`
# <br>
# 
# In general, each line in data file represents information of one piece of news: <br>
# 
# `[News ID] [Category] [Subcategory] [News Title] [News Abstrct] [News Url] [Entities in News Title] [Entities in News Abstract] ...`
# 
# <br>
# 
# We generate a word_dict file to tranform words in news title to word indexes, and a embedding matrix is initted from pretrained glove embeddings.
# 
# ### behaviors data
# One simple example: <br>
# `1	U82271	11/11/2019 3:28:58 PM	N3130 N11621 N12917 N4574 N12140 N9748	N13390-0 N7180-0 N20785-0 N6937-0 N15776-0 N25810-0 N20820-0 N6885-0 N27294-0 N18835-0 N16945-0 N7410-0 N23967-0 N22679-0 N20532-0 N26651-0 N22078-0 N4098-0 N16473-0 N13841-0 N15660-0 N25787-0 N2315-0 N1615-0 N9087-0 N23880-0 N3600-0 N24479-0 N22882-0 N26308-0 N13594-0 N2220-0 N28356-0 N17083-0 N21415-0 N18671-0 N9440-0 N17759-0 N10861-0 N21830-0 N8064-0 N5675-0 N15037-0 N26154-0 N15368-1 N481-0 N3256-0 N20663-0 N23940-0 N7654-0 N10729-0 N7090-0 N23596-0 N15901-0 N16348-0 N13645-0 N8124-0 N20094-0 N27774-0 N23011-0 N14832-0 N15971-0 N27729-0 N2167-0 N11186-0 N18390-0 N21328-0 N10992-0 N20122-0 N1958-0 N2004-0 N26156-0 N17632-0 N26146-0 N17322-0 N18403-0 N17397-0 N18215-0 N14475-0 N9781-0 N17958-0 N3370-0 N1127-0 N15525-0 N12657-0 N10537-0 N18224-0`
# <br>
# 
# In general, each line in data file represents one instance of an impression. The format is like: <br>
# 
# `[Impression ID] [User ID] [Impression Time] [User Click History] [Impression News]`
# 
# <br>
# 
# User Click History is the user historical clicked news before Impression Time. Impression News is the displayed news in an impression, which format is:<br>
# 
# `[News ID 1]-[label1] ... [News ID n]-[labeln]`
# 
# <br>
# Label represents whether the news is clicked by the user. All information of news in User Click History and Impression News can be found in news data file.

# ## Global settings and imports

# In[1]:


import sys
import os
import numpy as np
import zipfile
from tqdm import tqdm
import scrapbook as sb
from tempfile import TemporaryDirectory
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # only show error messages

from recommenders.models.deeprec.deeprec_utils import download_deeprec_resources 
from recommenders.models.newsrec.newsrec_utils import prepare_hparams
from recommenders.models.newsrec.models.lstur import LSTURModel
from recommenders.models.newsrec.io.mind_iterator import MINDIterator
from recommenders.models.newsrec.newsrec_utils import get_mind_data_set

print("System version: {}".format(sys.version))
print("Tensorflow version: {}".format(tf.__version__))


# # Prepare Parameters

# In[2]:


epochs = 5
seed = 40
batch_size = 32

# Options: demo, small, large
MIND_type = 'demo'


# ## Download and load data

# In[3]:


tmpdir = TemporaryDirectory()
data_path = tmpdir.name

train_news_file = os.path.join(data_path, 'train', r'news.tsv')
train_behaviors_file = os.path.join(data_path, 'train', r'behaviors.tsv')
valid_news_file = os.path.join(data_path, 'valid', r'news.tsv')
valid_behaviors_file = os.path.join(data_path, 'valid', r'behaviors.tsv')
wordEmb_file = os.path.join(data_path, "utils", "embedding.npy")
userDict_file = os.path.join(data_path, "utils", "uid2index.pkl")
wordDict_file = os.path.join(data_path, "utils", "word_dict.pkl")
yaml_file = os.path.join(data_path, "utils", r'lstur.yaml')

mind_url, mind_train_dataset, mind_dev_dataset, mind_utils = get_mind_data_set(MIND_type)

if not os.path.exists(train_news_file):
    download_deeprec_resources(mind_url, os.path.join(data_path, 'train'), mind_train_dataset)
    
if not os.path.exists(valid_news_file):
    download_deeprec_resources(mind_url,                                os.path.join(data_path, 'valid'), mind_dev_dataset)
if not os.path.exists(yaml_file):
    download_deeprec_resources(r'https://recodatasets.z20.web.core.windows.net/newsrec/',                                os.path.join(data_path, 'utils'), mind_utils)


# ## Create hyper-parameters

# In[4]:


hparams = prepare_hparams(yaml_file, 
                          wordEmb_file=wordEmb_file,
                          wordDict_file=wordDict_file, 
                          userDict_file=userDict_file,
                          batch_size=batch_size,
                          epochs=epochs)
print(hparams)


# In[5]:


iterator = MINDIterator


# ## Train the LSTUR model

# In[6]:


model = LSTURModel(hparams, iterator, seed=seed)


# In[7]:


print(model.run_eval(valid_news_file, valid_behaviors_file))


# In[8]:


get_ipython().run_cell_magic('time', '', 'model.fit(train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file)')


# In[9]:


get_ipython().run_cell_magic('time', '', 'res_syn = model.run_eval(valid_news_file, valid_behaviors_file)\nprint(res_syn)')


# In[ ]:


sb.glue("res_syn", res_syn)


# ## Save the model

# In[11]:


model_path = os.path.join(data_path, "model")
os.makedirs(model_path, exist_ok=True)

model.model.save_weights(os.path.join(model_path, "lstur_ckpt"))


# ## Output Prediction File
# This code segment is used to generate the prediction.zip file, which is in the same format in [MIND Competition Submission Tutorial](https://competitions.codalab.org/competitions/24122#learn_the_details-submission-guidelines).
# 
# Please change the `MIND_type` parameter to `large` if you want to submit your prediction to [MIND Competition](https://msnews.github.io/competition.html).

# In[12]:


group_impr_indexes, group_labels, group_preds = model.run_fast_eval(valid_news_file, valid_behaviors_file)


# In[13]:


with open(os.path.join(data_path, 'prediction.txt'), 'w') as f:
    for impr_index, preds in tqdm(zip(group_impr_indexes, group_preds)):
        impr_index += 1
        pred_rank = (np.argsort(np.argsort(preds)[::-1]) + 1).tolist()
        pred_rank = '[' + ','.join([str(i) for i in pred_rank]) + ']'
        f.write(' '.join([str(impr_index), pred_rank])+ '\n')


# In[14]:


f = zipfile.ZipFile(os.path.join(data_path, 'prediction.zip'), 'w', zipfile.ZIP_DEFLATED)
f.write(os.path.join(data_path, 'prediction.txt'), arcname='prediction.txt')
f.close()


# ## Reference
# \[1\] Mingxiao An, Fangzhao Wu, Chuhan Wu, Kun Zhang, Zheng Liu and Xing Xie: Neural News Recommendation with Long- and Short-term User Representations, ACL 2019<br>
# \[2\] Wu, Fangzhao, et al. "MIND: A Large-scale Dataset for News Recommendation" Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. https://msnews.github.io/competition.html <br>
# \[3\] GloVe: Global Vectors for Word Representation. https://nlp.stanford.edu/projects/glove/
