#!/usr/bin/env python
# coding: utf-8

# <i>Copyright (c) Microsoft Corporation. All rights reserved.</i>
# 
# <i>Licensed under the MIT License.</i>

# # TF-IDF Content-Based Recommendation on the COVID-19 Open Research Dataset
# This demonstrates a simple implementation of Term Frequency Inverse Document Frequency (TF-IDF) content-based recommendation on the [COVID-19 Open Research Dataset](https://azure.microsoft.com/en-us/services/open-datasets/catalog/covid-19-open-research/), hosted through Azure Open Datasets.
# 
# In this notebook, we will create a recommender which will return the top k recommended articles similar to any article of interest (query item) in the COVID-19 Open Research Dataset.

# In[1]:


import sys
# Import functions
from recommenders.datasets import covid_utils
from recommenders.models.tfidf.tfidf_utils import TfidfRecommender

# Print version
print("System version: {}".format(sys.version))


# ### 1. Load the dataset into a dataframe
# Let's begin by loading the metadata file for the dataset into a Pandas dataframe. This file contains metadata about each of the scientific articles included in the full dataset.

# In[2]:


# Specify container and metadata filename
container_name = 'covid19temp'
metadata_filename = 'metadata.csv'
sas_token = ''  # please see Azure Open Datasets notebook for SAS token

# Get metadata (may take around 1-2 min)
metadata = covid_utils.load_pandas_df(container_name=container_name, metadata_filename=metadata_filename, azure_storage_sas_token=sas_token)


# ### 2. Extract articles in the public domain
# The dataset contains articles using a variety of licenses. We will only be using articles that fall under the public domain ([cc0](https://creativecommons.org/publicdomain/zero/1.0/)).

# In[3]:


# View distribution of license types in the dataset
metadata['license'].value_counts().plot(kind='bar', title='License')


# In[4]:


# Extract metadata on public domain articles only
metadata_public = metadata.loc[metadata['license']=='cc0']

# Clean dataframe
metadata_public = covid_utils.clean_dataframe(metadata_public)


# Let's look at the top few rows of this dataframe which contains metadata on public domain articles.

# In[5]:


# Preview metadata for public domain articles
print('Number of articles in dataset: ' + str(len(metadata)))
print('Number of articles in dataset that fall under the public domain (cc0): ' + str(len(metadata_public)))
metadata_public.head()


# ### 3. Retrieve full article text
# Now that we have the metadata for the public domain articles as its own dataframe, let's retrieve the full text for each public domain scientific article.

# In[6]:


# Extract text from all public domain articles (may take 2-3 min)
all_text = covid_utils.get_public_domain_text(df=metadata_public, container_name=container_name, azure_storage_sas_token=sas_token)


# Notice that **all_text** is the same as **metadata_public** but now has an additional column called **full_text** which contains the full text for each respective article.

# In[7]:


# Preview
all_text.head()


# ### 4. Instantiate the recommender
# All functions for data preparation and recommendation are contained within the **TfidfRecommender** class we have imported. Prior to running these functions, we must create an object of this class.
# 
# Select one of the following tokenization methods to use in the model:
# 
# | tokenization_method | Description                                                                                                                      |
# |:--------------------|:---------------------------------------------------------------------------------------------------------------------------------|
# | 'none'              | No tokenization is applied. Each word is considered a token.                                                                     |
# | 'nltk'              | Simple stemming is applied using NLTK.                                                                                           |
# | 'bert'              | HuggingFace BERT word tokenization ('bert-base-cased') is applied.                                                               |
# | 'scibert'           | SciBERT word tokenization ('allenai/scibert_scivocab_cased') is applied.<br>This is recommended for scientific journal articles. |

# In[8]:


# Create the recommender object
recommender = TfidfRecommender(id_col='cord_uid', tokenization_method='scibert')


# ### 5. Prepare text for use in the TF-IDF recommender
# The raw text retrieved for each article requires basic cleaning prior to being used in the TF-IDF model.
# 
# Let's look at the full_text from the first article in our dataframe as an example.

# In[9]:


# Preview the first 1000 characters of the full scientific text from one example
print(all_text['full_text'][0][:1000])


# As seen above, there are some special characters (such as • ▲ ■ ≥ °) and punctuation which should be removed prior to using the text as input. Casing (capitalization) is preserved for [BERT-based tokenization methods](https://huggingface.co/transformers/model_doc/bert.html), but is removed for simple or no tokenization.
# 
# Let's join together the **title**, **abstract**, and **full_text** columns and clean them for future use in the TF-IDF model.

# In[10]:


# Assign columns to clean and combine
cols_to_clean = ['title','abstract','full_text']
clean_col = 'cleaned_text'
df_clean = recommender.clean_dataframe(all_text, cols_to_clean, clean_col)


# In[11]:


# Preview the dataframe with the cleaned text
df_clean.head()


# In[12]:


# Preview the first 1000 characters of the cleaned version of the previous example
print(df_clean[clean_col][0][:1000])


# Let's also tokenize the cleaned text for use in the TF-IDF model. The tokens are stored within our TfidfRecommender object.

# In[13]:


# Tokenize text with tokenization_method specified in class instantiation
tf, vectors_tokenized = recommender.tokenize_text(df_clean, text_col=clean_col)


# ### 6. Recommend articles using TF-IDF
# Let's now fit the recommender model to the processed data (tokens) and retrieve the top k recommended articles.
# 
# When creating our object, we specified k=5 so the `recommend_top_k_items` function will return the top 5 recommendations for each public domain article.

# In[14]:


# Fit the TF-IDF vectorizer
recommender.fit(tf, vectors_tokenized)

# Get recommendations
top_k_recommendations = recommender.recommend_top_k_items(df_clean, k=5)


# In our recommendation table, each row represents a single recommendation.
# 
# - **cord_uid** corresponds to the article that is being used to make recommendations from.
# - **rec_rank** contains the recommdation's rank (e.g., rank of 1 means top recommendation).
# - **rec_score** is the cosine similarity score between the query article and the recommended article.
# - **rec_cord_uid** corresponds to the recommended article.

# In[15]:


# Preview the recommendations
top_k_recommendations


# Optionally, we can access the full recommendation dictionary, which contains full ranked lists for each public domain article.

# In[16]:


# Optionally view full recommendation list
full_rec_list = recommender.recommendations

article_of_interest = 'ej795nks'
print('Number of recommended articles for ' + article_of_interest + ': ' + str(len(full_rec_list[article_of_interest])))


# Optionally, we can also view the tokens and stop words which were used in the recommender.

# In[17]:


# Optionally view tokens
tokens = recommender.get_tokens()

# Preview 10 tokens
print(list(tokens.keys())[:10])


# In[18]:


# Preview just the first 10 stop words sorted alphabetically
stop_words = list(recommender.get_stop_words())
stop_words.sort()
print(stop_words[:10])


# ### 7. Display top recommendations for article of interest
# Now that we have the recommendation table containing IDs for both query and recommended articles, we can easily return the full metadata for the top k recommendations for any given article.

# In[19]:


cols_to_keep = ['title','authors','journal','publish_time','url']
recommender.get_top_k_recommendations(metadata_public,article_of_interest,cols_to_keep)


# ### Conclusion
# In this notebook, we have demonstrated how to create a TF-IDF recommender to recommend the top k (in this case 5) articles similar in content to an article of interest (in this example, article with `cord_uid='ej795nks'`).
