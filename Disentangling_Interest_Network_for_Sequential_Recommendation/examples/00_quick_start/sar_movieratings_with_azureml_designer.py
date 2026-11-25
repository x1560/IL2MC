#!/usr/bin/env python
# coding: utf-8

# <i>Copyright (c) Microsoft Corporation. All rights reserved.</i>
# 
# <i>Licensed under the MIT License.</i>

# # Quickstart to integrate Recommenders in AzureML Designer
# 
# This notebook shows how to integrate any algorithm in Recommenders library into AzureML Designer. 
# 
# [AzureML Designer](https://docs.microsoft.com/en-us/azure/machine-learning/concept-designer) lets you visually connect datasets and modules on an interactive canvas to create machine learning models. 
# 
# One of the features of AzureML Designer is that it is possible for developers to integrate any python library to make it available as a module/component. In this notebook are are going to show how to integrate [SAR](sar_movielens.ipynb) and several other modules in Designer.
# 
# **Note that custom module is renamed to component.**
# 
# ## Component implementation
# 
# The scenario that we are going to reproduce in Designer, as a reference example, is the content of the [SAR quickstart notebook](sar_movielens.ipynb). In it, we load a dataset, split it into train and test sets, train SAR algorithm, predict using the test set and compute several ranking metrics (precision at k, recall at k, MAP and nDCG).
# 
# For the pipeline that we want to create in Designer, we need to build the following modules:
# 
# - Stratified splitter
# - SAR training
# - SAR prediction
# - Precision at k
# - Recall at k
# - MAP
# - nDCG
# 
# The python code is defined with a python entry and a yaml file. All the python entries and yaml files for this pipeline can be found in [contrib/azureml_designer_modules](../../contrib/azureml_designer_modules).
# 
# 
# ### Define python entry
# 
# To illustrate how a python entry is defined we are going to explain the [precision at k entry](../../contrib/azureml_designer_modules/entries/precision_at_k_entry.py). A simplified version of the code is shown next:
# 
# ```python
# # Dependencies
# from azureml.studio.core.data_frame_schema import DataFrameSchema
# from azureml.studio.core.io.data_frame_directory import (
#     load_data_frame_from_directory,
#     save_data_frame_to_directory,
# )
# from recommenders.evaluation.python_evaluation import precision_at_k
# 
# # First, the input variables of precision_at_k are defined as argparse arguments
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--rating-true", help="True DataFrame.")
#     parser.add_argument("--rating-pred", help="Predicted DataFrame.")
#     parser.add_argument(
#         "--col-user", type=str, help="A string parameter with column name for user."
#     )
#     # ... more arguments
#     args, _ = parser.parse_known_args()
# 
#     # This module has two main inputs from the canvas, the true and predicted labels
#     # they are loaded into the runtime as a pandas DataFrame
#     rating_true = load_data_frame_from_directory(args.rating_true).data
#     rating_pred = load_data_frame_from_directory(args.rating_pred).data
# 
#     # The python function is instantiated and the computation is performed
#     eval_precision = precision_at_k(rating_true, rating_pred)
#     
#     # To output the result to Designer, we write it as a DataFrame
#     score_result = pd.DataFrame({"precision_at_k": [eval_precision]})
#     save_data_frame_to_directory(
#         args.score_result,
#         score_result,
#         schema=DataFrameSchema.data_frame_to_dict(score_result),
#     )
# ```
# 
# 
# ### Define component specification yaml
# 
# Once we have the python entry, we need to create the yaml file that will interact with Designer, [precision_at_k.yaml](../../recommenders/azureml/azureml_designer_modules/module_specs/precision_at_k.yaml).
# 
# ```yaml
# $schema: http://azureml/sdk-2-0/CommandComponent.json
# name: microsoft.com.cat.precision_at_k
# version: 1.1.1
# display_name: Precision at K
# type: CommandComponent
# description: 'Precision at K metric from Recommenders repo: https://github.com/Microsoft/Recommenders.'
# tags:
#   Recommenders:
#   Metrics:
# inputs:
#   rating_true:
#     type: AnyDirectory
#     description: True DataFrame.
#     optional: false
#   rating_pred:
#     type: AnyDirectory
#     description: Predicted DataFrame.
#     optional: false
#   user_column:
#     type: String
#     description: Column name of user IDs.
#     default: UserId
#     optional: false
#   item_column:
#     type: String
#     description: Column name of item IDs.
#     default: MovieId
#     optional: false
#   rating_column:
#     type: String
#     description: Column name of ratings.
#     default: Rating
#     optional: false
#   prediction_column:
#     type: String
#     description: Column name of predictions.
#     default: prediction
#     optional: false
#   relevancy_method:
#     type: String
#     description: method for determining relevancy ['top_k', 'by_threshold'].
#     default: top_k
#     optional: false
#   top_k:
#     type: Integer
#     description: Number of top k items per user.
#     default: 10
#     optional: false
#   threshold:
#     type: Float
#     description: Threshold of top items per user.
#     default: 10.0
#     optional: false
# outputs:
#   score:
#     type: AnyDirectory
#     description: Precision at k (min=0, max=1).
# code:
#   ../../
# command: >-
#   python contrib/azureml_designer_modules/entries/precision_at_k_entry.py
#   --rating-true {inputs.rating_true} --rating-pred {inputs.rating_pred} --col-user
#   {inputs.user_column} --col-item {inputs.item_column} --col-rating {inputs.rating_column}
#   --col-prediction {inputs.prediction_column} --relevancy-method {inputs.relevancy_method}
#   --k {inputs.top_k} --threshold {inputs.threshold} --score-result {outputs.score}
# environment:
#   conda:
#     conda_dependencies_file: contrib/azureml_designer_modules/module_specs/sar_conda.yaml
#   os: Linux
# 
# ```
# 
# In the yaml file we can see a number of sections. The heading defines attributes like name, version or description. In the section inputs, all inputs are defined. The two main dataframes have ports, which can be connected to other modules. The inputs without port appear in a canvas menu. The output is defined as a DataFrame as well. The last section, implementation, defines the conda environment, the associated python entry and the arguments to the python file.
# 
# 
# # Module Registration
# 
# Once the code is implemented, we need to register it as an AzureML Designer custom module. The registration can be performed following these simple steps:

# ## Register in studio UI
# 
# You can directly register a custom module/component in the studio UI.
# 
# Follow [this tutorial](https://aka.ms/component-tutorial1) to register the related components to your workspace.

# ## Register using CLI
# 
# ### CLI Installation
# 
# The first step is to install [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli) and Component CLI extension. Assuming that you have installed the Recommenders environment `reco_base` as explained in the [SETUP.md](../../SETUP.md).
# 

# In[ ]:


conda activate reco_base
pip install azure-cli


# In[ ]:


# Login
get_ipython().system('az login -o none')


# In[ ]:


# Uninstall azure-cli-ml (the `az ml` commands)
get_ipython().system('az extension remove -n azure-cli-ml ')

# Install remote version of azure-cli-ml (which includes `az ml component` commands)
get_ipython().system('az extension add --source https://azuremlsdktestpypi.blob.core.windows.net/wheels/modulesdkpreview/azure_cli_ml-0.1.0.29211468-py3-none-any.whl --pip-extra-index-urls https://azuremlsdktestpypi.azureedge.net/modulesdkpreview --yes --verbose')


# In[ ]:


get_ipython().system('az account set -s "Your subscription name"')
get_ipython().system('az ml folder attach -w "Your workspace name" -g "Your resource group name"')


# In[ ]:


import os
import tempfile
import shutil
import subprocess


# In[ ]:


# Regsiter components with spec via Azure CLI
root_path = os.path.abspath(os.path.join(os.getcwd(), "../../"))
specs_folder = os.path.join(root_path, "contrib/azureml_designer_modules/module_specs")
github_prefix = 'https://github.com/microsoft/recommenders/blob/main/recommenders/azureml/azureml_designer_modules/module_specs/'
specs = os.listdir(specs_folder)
for spec in specs:
    spec_path = github_prefix + spec
    print(f"Start to register component spec: {spec} ...")
    subprocess.run(f"az ml component create --file {spec_path}", shell=True)
    print(f"Done.")


# ## Running Recommenders in AzureML Designer
# 
# Once the modules are registered, they will appear in the canvas as the module `Recommenders`. There you will be able to create a pipeline like this:
# 
# ![img](https://raw.githubusercontent.com/Azure/AzureMachineLearningGallery/main/pipelines/sar-pipeline/sar-pipeline.png)
# 
# Now, thanks to AzureML Designer, users can compute the latest state of the art algorithms in recommendation systems without writing a line of python code.
# 
# ## References
# 
# 1. [AzureML Designer documentation](https://docs.microsoft.com/en-us/azure/machine-learning/concept-designer)
# 1. [Tutorial: Prediction of automobile price](https://aka.ms/designer-tutorial)
# 1. [Tutorial: Register component to workspace](https://aka.ms/component-tutorial1)
# 1. [Tutorial: Create a new component](https://aka.ms/component-tutorial2)
