#!/usr/bin/env python
# coding: utf-8

# <i>Copyright (c) Microsoft Corporation. All rights reserved.</i>
# 
# <i>Licensed under the MIT License.</i>

# # Train SAR on MovieLens with Azure Machine Learning (Python, CPU)
# ---
# ## Introduction to Azure Machine Learning  
# The **[Azure Machine Learning service (AzureML)](https://docs.microsoft.com/azure/machine-learning/service/overview-what-is-azure-ml)** provides a cloud-based environment you can use to prep data, train, test, deploy, manage, and track machine learning models. By using Azure Machine Learning service, you can start training on your local machine and then scale out to the cloud. With many available compute targets, like [Azure Machine Learning Compute](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-set-up-training-targets#amlcompute) and [Azure Databricks](https://docs.microsoft.com/en-us/azure/azure-databricks/what-is-azure-databricks), and with [advanced hyperparameter tuning services](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-tune-hyperparameters), you can build better models faster by using the power of the cloud.
# 
# Data scientists and AI developers use the main [Azure Machine Learning Python SDK](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/intro?view=azure-ml-py) to build and run machine learning workflows with the Azure Machine Learning service. You can interact with the service in any Python environment, including Jupyter Notebooks or your favorite Python IDE. The Azure Machine Learning SDK allows you the choice of using local or cloud compute resources, while managing and maintaining the complete data science workflow from the cloud.
# ![AzureML Workflow](https://docs.microsoft.com/en-us/azure/machine-learning/service/media/overview-what-is-azure-ml/aml.png)
# 
# This notebook provides an example of how to utilize and evaluate the Simple Algorithm for Recommendation (SAR) algorithm using the Azure Machine Learning service. It takes the content of the [SAR quickstart notebook](sar_movielens.ipynb) and demonstrates how to use the power of the cloud to manage data, switch to powerful GPU machines, and monitor runs while training a model. 
# 
# See the hyperparameter tuning notebook for more advanced use cases with AzureML.
# 
# ### Advantages of using AzureML:
# - Manage cloud resources for monitoring, logging, and organizing your machine learning experiments.
# - Train models either locally or by using cloud resources, including GPU-accelerated model training.
# - Easy to scale out when dataset grows - by just creating and pointing to new compute target
# 
# ---
# ## Details of SAR
# <details>
#     <summary>Click to expand</summary>
#     
# SAR is a fast scalable adaptive algorithm for personalized recommendations based on user transaction history. It produces easily explainable / interpretable recommendations and handles "cold item" and "semi-cold user" scenarios. SAR is a kind of neighborhood based algorithm (as discussed in [Recommender Systems by Aggarwal](https://dl.acm.org/citation.cfm?id=2931100)) which is intended for ranking top items for each user. 
# 
# SAR recommends items that are most ***similar*** to the ones that the user already has an existing ***affinity*** for. Two items are ***similar*** if the users who have interacted with one item are also likely to have interacted with another. A user has an ***affinity*** to an item if they have interacted with it in the past.
# 
# ### Advantages of SAR:
# - High accuracy for an easy to train and deploy algorithm
# - Fast training, only requiring simple counting to construct matrices used at prediction time
# - Fast scoring, only involving multiplication of the similarity matric with an affinity vector
# 
# ### Notes to use SAR properly:
# - SAR does not use item or user features, so cannot handle cold-start use cases
# - SAR requires the creation of an $mxm$ dense matrix (where $m$ is the number of items). So memory consumption can be an issue with large numbers of items.
# - SAR is best used for ranking items per user, as the scale of predicted ratings may be different from the input range and will differ across users.
# For more details see the deep dive notebook on SAR here: [SAR Deep Dive Notebook](../02_model_collaborative_filtering/sar_deep_dive.ipynb)</details>
# ---
# ## Prerequisities
#    - **Azure Subscription**
#      - If you don’t have an Azure subscription, create a free account before you begin. Try the [free or paid version of Azure Machine Learning service today](https://azure.microsoft.com/en-us/free/services/machine-learning/).
#      - You get credits to spend on Azure services, which will easily cover the cost of running this example notebook. After they're used up, you can keep the account and use [free Azure services](https://azure.microsoft.com/en-us/free/). Your credit card is never charged unless you explicitly change your settings and ask to be charged. Or [activate MSDN subscriber benefits](https://azure.microsoft.com/en-us/pricing/member-offers/credit-for-visual-studio-subscribers/), which give you credits every month that you can use for paid Azure services.
# ---   

# In[29]:


# set the environment path to find Recommenders
import os
import shutil
import numpy as np
from tempfile import TemporaryDirectory

import azureml
from azureml.core import Workspace, Run, Experiment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.train.estimator import Estimator
from azureml.widgets import RunDetails

from recommenders.datasets import movielens

print("azureml.core version: {}".format(azureml.core.VERSION))


# In[26]:


# top k items to recommend
TOP_K = 10

# Select Movielens data size: 100k, 1m, 10m, or 20m
MOVIELENS_DATA_SIZE = '100k'


# ### Connect to an AzureML workspace
# 
# An [AzureML Workspace](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.workspace.workspace?view=azure-ml-py) is an Azure resource that organizes and coordinates the actions of many other Azure resources to assist in executing and sharing machine learning workflows. In particular, an Azure ML Workspace coordinates storage, databases, and compute resources providing added functionality for machine learning experimentation, deployment, inferencing, and the monitoring of deployed models.
# 
# The function below will get or create an AzureML Workspace and save the configuration to `aml_config/config.json`.
# 
# It defaults to use provided input parameters or environment variables for the Workspace configuration values. Otherwise, it will use an existing configuration file (either at `./aml_config/config.json` or a path specified by the config_path parameter).
# 
# Lastly, if the workspace does not exist, one will be created for you. See [this tutorial](https://docs.microsoft.com/en-us/azure/machine-learning/service/setup-create-workspace#portal) to locate information such as subscription id.

# In[ ]:


ws = Workspace.create(
    name="<WORKSPACE_NAME>",
    subscription_id="<SUBSCRIPTION_ID>",
    resource_group="<RESOURCE_GROUP>",
    location="<WORKSPACE_REGION>",
    exist_ok=True,
)


# ### Create a Temporary Directory
# This directory will house the data and scripts needed by the AzureML Workspace

# In[34]:


tmp_dir = TemporaryDirectory()


# ### Download dataset and upload to datastore
# 
# Every workspace comes with a default [datastore](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-access-data) (and you can register more) which is backed by the Azure blob storage account associated with the workspace. We can use it to transfer data from local to the cloud, and access it from the compute target.
# 
# The data files are uploaded into a directory named `data` at the root of the datastore.

# In[35]:


TARGET_DIR = 'movielens'

# download dataset
data = movielens.load_pandas_df(
    size=MOVIELENS_DATA_SIZE,
    header=['UserId','MovieId','Rating','Timestamp']
)

# upload dataset to workspace datastore
data_file_name = "movielens_" + MOVIELENS_DATA_SIZE + "_data.pkl"
data.to_pickle(os.path.join(tmp_dir.name, data_file_name))

ds = ws.get_default_datastore()
ds.upload(src_dir=tmp_dir.name, target_path=TARGET_DIR, overwrite=True, show_progress=False)


# ### Create or Attach Azure Machine Learning Compute 
# 
# We create a cpu cluster as our **remote compute target**. If a cluster with the same name already exists in your workspace, the script will load it instead. You can read [Set up compute targets for model training](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-set-up-training-targets) to learn more about setting up compute target on different locations. You can also create GPU machines when larger machines are necessary to train the model.
# 
# According to Azure [Pricing calculator](https://azure.microsoft.com/en-us/pricing/calculator/), with example VM size `STANDARD_D2_V2`, it costs a few dollars to run this notebook, which is well covered by Azure new subscription credit. For billing and pricing questions, please contact [Azure support](https://azure.microsoft.com/en-us/support/options/).
# 
# **Note**:
# - 10m and 20m dataset requires more capacity than `STANDARD_D2_V2`, such as `STANDARD_NC6` or `STANDARD_NC12`. See list of all available VM sizes [here](https://docs.microsoft.com/en-us/azure/templates/Microsoft.Compute/2018-10-01/virtualMachines?toc=%2Fen-us%2Fazure%2Fazure-resource-manager%2Ftoc.json&bc=%2Fen-us%2Fazure%2Fbread%2Ftoc.json#hardwareprofile-object).
# - As with other Azure services, there are limits on certain resources (e.g. AzureML Compute quota) associated with the Azure Machine Learning service. Please read [these instructions](https://docs.microsoft.com/en-us/azure/azure-supportability/resource-manager-core-quotas-request) on the default limits and how to request more quota.
# ---
# #### Learn more about Azure Machine Learning Compute
# <details>
#     <summary>Click to learn more about compute types</summary>
#     
# [Azure Machine Learning Compute](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-set-up-training-targets#amlcompute) is managed compute infrastructure that allows the user to easily create single to multi-node compute of the appropriate VM Family. It is created within your workspace region and is a resource that can be used by other users in your workspace. It autoscales by default to the max_nodes, when a job is submitted, and executes in a containerized environment packaging the dependencies as specified by the user.
# 
# Since it is managed compute, job scheduling and cluster management are handled internally by Azure Machine Learning service.
# 
# You can provision a persistent AzureML Compute resource by simply defining two parameters thanks to smart defaults. By default it autoscales from 0 nodes and provisions dedicated VMs to run your job in a container. This is useful when you want to continously re-use the same target, debug it between jobs or simply share the resource with other users of your workspace.
# 
# In addition to vm_size and max_nodes, you can specify:
# - **min_nodes**: Minimum nodes (default 0 nodes) to downscale to while running a job on AzureML Compute
# - **vm_priority**: Choose between 'dedicated' (default) and 'lowpriority' VMs when provisioning AzureML Compute. Low Priority VMs use Azure's excess capacity and are thus cheaper but risk your run being pre-empted
# - **idle_seconds_before_scaledown**: Idle time (default 120 seconds) to wait after run completion before auto-scaling to min_nodes
# - **vnet_resourcegroup_name**: Resource group of the existing VNet within which Azure MLCompute should be provisioned
# - **vnet_name**: Name of VNet
# - **subnet_name**: Name of SubNet within the VNet
# </details>
# ---

# In[6]:


# Remote compute (cluster) configuration. If you want to save the cost more, set these to small.
VM_SIZE = 'STANDARD_D2_V2'
# Cluster nodes
MIN_NODES = 0
MAX_NODES = 2

CLUSTER_NAME = 'cpucluster'

try:
    compute_target = ComputeTarget(workspace=ws, name=CLUSTER_NAME)
    print("Found existing compute target")
except:
    print("Creating a new compute target...")
    # Specify the configuration for the new cluster
    compute_config = AmlCompute.provisioning_configuration(
        vm_size=VM_SIZE,
        min_nodes=MIN_NODES,
        max_nodes=MAX_NODES
    )
    # Create the cluster with the specified name and configuration
    compute_target = ComputeTarget.create(ws, CLUSTER_NAME, compute_config)
    # Wait for the cluster to complete, show the output log
    compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)


# # Prepare training script
# ### 1. Create a directory
# Create a directory that will contain all the necessary code from your local machine that you will need access to on the remote resource. This includes the training script, and any additional files your training script depends on.

# In[36]:


SCRIPT_DIR = os.path.join(tmp_dir.name, 'movielens-sar')
os.makedirs(SCRIPT_DIR, exist_ok=True)
TRAIN_FILE = os.path.join(SCRIPT_DIR, 'train.py')


# ### 2. Create a training script
# To submit the job to the cluster, first create a training script. Run the following code to create the training script called `train.py` in temporary directory. This training adds a regularization rate to the training algorithm, so produces a slightly different model than the local version.
# 
# This code takes what is in the local quickstart and convert it to one single training script. We use run.log() to record parameters to the run. We will be able to review and compare these measures in the Azure Portal at a later time.

# In[37]:


get_ipython().run_cell_magic('writefile', '$TRAIN_FILE', '\nimport argparse\nimport os\nimport numpy as np\nimport pandas as pd\nimport itertools\nimport logging\n\nfrom azureml.core import Run\nfrom sklearn.externals import joblib\n\nfrom recommenders.utils.timer import Timer\nfrom recommenders.datasets import movielens\nfrom recommenders.datasets.python_splitters import python_stratified_split\nfrom recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k\nfrom recommenders.models.sar import SAR\n\n\nlogging.basicConfig(level=logging.DEBUG, \n                    format=\'%(asctime)s %(levelname)-8s %(message)s\')\n\n\nTARGET_DIR = \'movielens\'\nOUTPUT_FILE_NAME = \'outputs/movielens_sar_model.pkl\'\nMODEL_FILE_NAME = \'movielens_sar_model.pkl\'\n\n\n# get hold of the current run\nrun = Run.get_context()\n\n# let user feed in 2 parameters, the location of the data files (from datastore), and the regularization rate of the logistic regression model\nparser = argparse.ArgumentParser()\nparser.add_argument(\'--data-folder\', type=str, dest=\'data_folder\', help=\'data folder mounting point\')\nparser.add_argument(\'--data-file\', type=str, dest=\'data_file\', help=\'data file name\')\nparser.add_argument(\'--top-k\', type=int, dest=\'top_k\', default=10, help=\'top k items to recommend\')\nparser.add_argument(\'--data-size\', type=str, dest=\'data_size\', default=10, help=\'Movielens data size: 100k, 1m, 10m, or 20m\')\nargs = parser.parse_args()\n\n# set col names\nheader = {\n    "col_user": "UserId",\n    "col_item": "MovieId",\n    "col_rating": "Rating",\n    "col_timestamp": "Timestamp",\n}\n\n# read data\ndata_pickle_path = os.path.join(args.data_folder, args.data_file)\ndata = pd.read_pickle(path=data_pickle_path)\n\n# Log arguments to the run for tracking\nrun.log("top-k", args.top_k)\nrun.log("data-size", args.data_size)\n\n# split dataset into train and test\ntrain, test = python_stratified_split(data, ratio=0.75, col_user=header["col_user"], col_item=header["col_item"], seed=42)\n\n# instantiate the model\nmodel = SAR(\n    similarity_type="jaccard", \n    time_decay_coefficient=30, \n    time_now=None, \n    timedecay_formula=True, \n    **header\n)\n\n# train the SAR model\nwith Timer() as t:\n    model.fit(train)\n\nrun.log(name="Training time", value=t.interval)\n\n# predict top k items\nwith Timer() as t:\n    top_k = model.recommend_k_items(test, top_k=TOP_K, remove_seen=True)\n\nrun.log(name="Prediction time", value=t.interval)\n\n# compute evaluation metrics\neval_map = map_at_k(test, top_k, col_user="UserId", col_item="MovieId", \n                    col_rating="Rating", col_prediction="prediction", \n                    relevancy_method="top_k", k=args.top_k)\neval_ndcg = ndcg_at_k(test, top_k, col_user="UserId", col_item="MovieId", \n                      col_rating="Rating", col_prediction="prediction", \n                      relevancy_method="top_k", k=args.top_k)\neval_precision = precision_at_k(test, top_k, col_user="UserId", col_item="MovieId", \n                                col_rating="Rating", col_prediction="prediction", \n                                relevancy_method="top_k", k=args.top_k)\neval_recall = recall_at_k(test, top_k, col_user="UserId", col_item="MovieId", \n                          col_rating="Rating", col_prediction="prediction", \n                          relevancy_method="top_k", k=args.top_k)\n\nrun.log("map", eval_map)\nrun.log("ndcg", eval_ndcg)\nrun.log("precision", eval_precision)\nrun.log("recall", eval_recall)\n\n# automatic upload of everything in ./output folder doesn\'t work for very large model file\n# model file has to be saved to a temp location, then uploaded by upload_file function\njoblib.dump(value=model, filename=MODEL_FILE_NAME)\n\nrun.upload_file(OUTPUT_FILE_NAME, MODEL_FILE_NAME)')


# In[38]:


# copy dependent python files
UTILS_DIR = os.path.join(SCRIPT_DIR, 'recommenders')
if os.path.exists(UTILS_DIR):
    shutil.rmtree(UTILS_DIR)
shutil.copytree('../../recommenders/', UTILS_DIR)


# # Run training script
# ### 1. Create an estimator
# An [estimator](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-train-ml-models) object is used to submit the run. You can create and use a generic Estimator to submit a training script using any learning framework you choose (such as scikit-learn) you want to run on any compute target, whether it's your local machine, a single VM in Azure, or a GPU cluster in Azure. 
# 
# Create your estimator by running the following code to define:  
# * The name of the estimator object, `est`
# * The directory that contains your scripts. All the files in this directory are uploaded into the cluster nodes for execution. 
# * The compute target.  In this case you will use the AzureML Compute you created
# * The training script name, train.py
# * Parameters required from the training script 
# * Python packages needed for training
# * Connect to the data files in the datastore
# 
# In this tutorial, this target is AzureML Compute. All files in the script folder are uploaded into the cluster nodes for execution. `ds.as_mount()` mounts a datastore on the remote compute and returns the folder. See documentation [here](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-access-data#access-datastores-during-training).

# In[39]:


script_params = {
    '--data-folder': ds.as_mount(),
    '--data-file': 'movielens/' + data_file_name,
    '--top-k': TOP_K,
    '--data-size': MOVIELENS_DATA_SIZE
}

est = Estimator(source_directory=SCRIPT_DIR,
                script_params=script_params,
                compute_target=compute_target,
                entry_script='train.py',
                conda_packages=['pandas'],
                pip_packages=['sklearn', 'tqdm'])


# ### 2. Submit the job to the cluster
# An [experiment](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/intro?view=azure-ml-py#experiment) is a logical container in an AzureML Workspace. It hosts run records which can include run metrics and output artifacts from your experiments. We access an experiment from our AzureML workspace by name, which will be created if it doesn't exist.
# 
# Then, run the experiment by submitting the estimator object.

# In[40]:


# create experiment
EXPERIMENT_NAME = 'movielens-sar'
exp = Experiment(workspace=ws, name=EXPERIMENT_NAME)

run = exp.submit(config=est)


# 
# ### 3.  Monitor remote run
# 
# #### Jupyter widget
# 
# Jupyter widget can watch the progress of the run.  Like the run submission, the widget is asynchronous and provides live updates every 10-15 seconds until the job completes.

# In[41]:


RunDetails(run).show()


# ### 4. Viewing run results
# Azure Machine Learning stores all the details about the run in the Azure cloud. Let's access those details by retrieving a link to the run using the default run output. Clicking on the resulting link will take you to an interactive page.

# In[42]:


run


# Above cell should output similar table as below.
# ![Experiment submit output](https://recodatasets.z20.web.core.windows.net/images/aml_sar_output.jpg)
# After clicking "Link to Azure Portal", experiment run details tab looks like this with logged metrics.
# ![Azure Portal Experiment](https://recodatasets.z20.web.core.windows.net/images/aml_sar_workspace.jpg)

# In[43]:


# run below after run is complete, otherwise metrics is empty
metrics = run.get_metrics()
metrics


# # Deprovision compute resource
# To avoid unnecessary charges, if you created compute target that doesn't scale down to 0, make sure the compute target is deprovisioned after use.

# In[46]:


# delete () is used to deprovision and delete the AzureML Compute target. 
# do not run below before experiment completes

# compute_target.delete()

# deletion will take a few minutes. You can check progress in Azure Portal / Computing tab


# In[47]:


# clean up temporary directory
tmp_dir.cleanup()


# In[ ]:




