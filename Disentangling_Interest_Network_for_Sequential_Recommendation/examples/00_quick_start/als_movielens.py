#!/usr/bin/env python
# coding: utf-8

# <i>Copyright (c) Microsoft Corporation. All rights reserved.</i>
# 
# <i>Licensed under the MIT License.</i>

# # Running ALS on MovieLens (PySpark)
# 
# Matrix factorization by [ALS](https://spark.apache.org/docs/latest/api/python/_modules/pyspark/ml/recommendation.html#ALS) (Alternating Least Squares) is a well known collaborative filtering algorithm.
# 
# This notebook provides an example of how to utilize and evaluate ALS PySpark ML (DataFrame-based API) implementation, meant for large-scale distributed datasets. We use a smaller dataset in this example to run ALS efficiently on multiple cores of a [Data Science Virtual Machine](https://azure.microsoft.com/en-gb/services/virtual-machines/data-science-virtual-machines/).

# **Note**: This notebook requires a PySpark environment to run properly. Please follow the steps in [SETUP.md](../../SETUP.md) to install the PySpark environment.

# In[1]:


# set the environment path to find Recommenders
import sys
import pyspark
from pyspark.ml.recommendation import ALS
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import StringType, FloatType, IntegerType, LongType
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from recommenders.utils.timer import Timer
from recommenders.datasets import movielens
from recommenders.utils.notebook_utils import is_jupyter
from recommenders.datasets.spark_splitters import spark_random_split
from recommenders.evaluation.spark_evaluation import SparkRatingEvaluation, SparkRankingEvaluation
from recommenders.utils.spark_utils import start_or_get_spark

print("System version: {}".format(sys.version))
print("Spark version: {}".format(pyspark.__version__))


# Set the default parameters.

# In[2]:


# top k items to recommend
TOP_K = 10

# Select MovieLens data size: 100k, 1m, 10m, or 20m
MOVIELENS_DATA_SIZE = '100k'

# Column names for the dataset
COL_USER = "UserId"
COL_ITEM = "MovieId"
COL_RATING = "Rating"
COL_TIMESTAMP = "Timestamp"


# ### 0. Set up Spark context
# 
# The following settings work well for debugging locally on VM - change when running on a cluster. We set up a giant single executor with many threads and specify memory cap. 

# In[3]:


# the following settings work well for debugging locally on VM - change when running on a cluster
# set up a giant single executor with many threads and specify memory cap
spark = start_or_get_spark("ALS PySpark", memory="16g")
spark.conf.set("spark.sql.analyzer.failAmbiguousSelfJoin", "false")


# ### 1. Download the MovieLens dataset

# In[4]:


# Note: The DataFrame-based API for ALS currently only supports integers for user and item ids.
schema = StructType(
    (
        StructField(COL_USER, IntegerType()),
        StructField(COL_ITEM, IntegerType()),
        StructField(COL_RATING, FloatType()),
        StructField(COL_TIMESTAMP, LongType()),
    )
)

data = movielens.load_spark_df(spark, size=MOVIELENS_DATA_SIZE, schema=schema)
data.show()


# ### 2. Split the data using the Spark random splitter provided in utilities

# In[5]:


train, test = spark_random_split(data, ratio=0.75, seed=123)
print ("N train", train.cache().count())
print ("N test", test.cache().count())


# ### 3. Train the ALS model on the training data, and get the top-k recommendations for our testing data
# 
# To predict movie ratings, we use the rating data in the training set as users' explicit feedback. The hyperparameters used in building the model are referenced from [here](http://mymedialite.net/examples/datasets.html). We do not constrain the latent factors (`nonnegative = False`) in order to allow for both positive and negative preferences towards movies.
# Timing will vary depending on the machine being used to train.

# In[6]:


header = {
    "userCol": COL_USER,
    "itemCol": COL_ITEM,
    "ratingCol": COL_RATING,
}


als = ALS(
    rank=10,
    maxIter=15,
    implicitPrefs=False,
    regParam=0.05,
    coldStartStrategy='drop',
    nonnegative=False,
    seed=42,
    **header
)


# In[7]:


with Timer() as train_time:
    model = als.fit(train)

print("Took {} seconds for training.".format(train_time.interval))


# In the movie recommendation use case, recommending movies that have been rated by the users do not make sense. Therefore, the rated movies are removed from the recommended items.
# 
# In order to achieve this, we recommend all movies to all users, and then remove the user-movie pairs that exist in the training dataset.

# In[8]:


with Timer() as test_time:

    # Get the cross join of all user-item pairs and score them.
    users = train.select(COL_USER).distinct()
    items = train.select(COL_ITEM).distinct()
    user_item = users.crossJoin(items)
    dfs_pred = model.transform(user_item)

    # Remove seen items.
    dfs_pred_exclude_train = dfs_pred.alias("pred").join(
        train.alias("train"),
        (dfs_pred[COL_USER] == train[COL_USER]) & (dfs_pred[COL_ITEM] == train[COL_ITEM]),
        how='outer'
    )

    top_all = dfs_pred_exclude_train.filter(dfs_pred_exclude_train[f"train.{COL_RATING}"].isNull())         .select('pred.' + COL_USER, 'pred.' + COL_ITEM, 'pred.' + "prediction")

    # In Spark, transformations are lazy evaluation
    # Use an action to force execute and measure the test time 
    top_all.cache().count()

print("Took {} seconds for prediction.".format(test_time.interval))


# In[9]:


top_all.show()


# ### 4. Evaluate how well ALS performs

# In[10]:


rank_eval = SparkRankingEvaluation(test, top_all, k = TOP_K, col_user=COL_USER, col_item=COL_ITEM, 
                                    col_rating=COL_RATING, col_prediction="prediction", 
                                    relevancy_method="top_k")


# In[11]:


print("Model:\tALS",
      "Top K:\t%d" % rank_eval.k,
      "MAP:\t%f" % rank_eval.map_at_k(),
      "NDCG:\t%f" % rank_eval.ndcg_at_k(),
      "Precision@K:\t%f" % rank_eval.precision_at_k(),
      "Recall@K:\t%f" % rank_eval.recall_at_k(), sep='\n')


# ### 5. Evaluate rating prediction

# In[12]:


# Generate predicted ratings.
prediction = model.transform(test)
prediction.cache().show()


# In[13]:


rating_eval = SparkRatingEvaluation(test, prediction, col_user=COL_USER, col_item=COL_ITEM, 
                                    col_rating=COL_RATING, col_prediction="prediction")

print("Model:\tALS rating prediction",
      "RMSE:\t%f" % rating_eval.rmse(),
      "MAE:\t%f" % rating_eval.mae(),
      "Explained variance:\t%f" % rating_eval.exp_var(),
      "R squared:\t%f" % rating_eval.rsquared(), sep='\n')


# In[14]:


if is_jupyter():
    # Record results with papermill for tests
    import papermill as pm
    import scrapbook as sb
    sb.glue("map", rank_eval.map_at_k())
    sb.glue("ndcg", rank_eval.ndcg_at_k())
    sb.glue("precision", rank_eval.precision_at_k())
    sb.glue("recall", rank_eval.recall_at_k())
    sb.glue("rmse", rating_eval.rmse())
    sb.glue("mae", rating_eval.mae())
    sb.glue("exp_var", rating_eval.exp_var())
    sb.glue("rsquared", rating_eval.rsquared())
    sb.glue("train_time", train_time.interval)
    sb.glue("test_time", test_time.interval)


# In[15]:


# cleanup spark instance
spark.stop()

