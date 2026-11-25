

# <i>Copyright (c) Microsoft Corporation. All rights reserved.</i>
# 
# <i>Licensed under the MIT License.</i>

# # Sequential Recommender Quick Start
# 
# ### Example: SLi_Rec : Adaptive User Modeling with Long and Short-Term Preferences for Personailzed Recommendation
# Unlike a general recommender such as Matrix Factorization or xDeepFM (in the repo) which doesn't consider the order of the user's activities, sequential recommender systems take the sequence of the user behaviors as context and the goal is to predict the items that the user will interact in a short time (in an extreme case, the item that the user will interact next).
# 
# This notebook aims to give you a quick example of how to train a sequential model based on a public Amazon dataset. Currently, we can support NextItNet \[4\], GRU4Rec \[2\], Caser \[3\], A2SVD \[1\], SLi_Rec \[1\], and SUM \[5\]. Without loss of generality, this notebook takes [SLi_Rec model](https://www.microsoft.com/en-us/research/uploads/prod/2019/07/IJCAI19-ready_v1.pdf) for example.
# SLi_Rec \[1\] is a deep learning-based model aims at capturing both long and short-term user preferences for precise recommender systems. To summarize, SLi_Rec has the following key properties:
# 
# * It adopts the attentive "Asymmetric-SVD" paradigm for long-term modeling;
# * It takes both time irregularity and semantic irregularity into consideration by modifying the gating logic in LSTM.
# * It uses an attention mechanism to dynamic fuse the long-term component and short-term component.
# 
### this notebook, we test SLi_Rec on a subset of the public dataset: [Amazon_reviews](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_5.json.gz) and [Amazon_metadata](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Movies_and_TV.json.gz)
# 
# This notebook is tested under TF 2.6. 

# ## 0. Global Settings and Imports


import sys
import os
import logging
from tempfile import TemporaryDirectory
import numpy as np
import tensorflow.compat.v1 as tf
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


tf.get_logger().setLevel('ERROR') # only show error messages
sys.path.append("/home/lisong/SNA/IL2MC")
from recommenders.datasets import dium_amazon_reviews
from recommenders.utils.timer import Timer
from recommenders.utils.constants import SEED
from recommenders.models.deeprec.deeprec_utils import (
    prepare_hparams
)
from recommenders.datasets.amazon_reviews import download_and_extract, data_preprocessing
from recommenders.datasets.download_utils import maybe_download


# from recommenders.models.deeprec.models.sequential.sli_rec import SLI_RECModel as SeqModel
####  to use the other model, use one of the following lines:
# from recommenders.models.deeprec.models.sequential.asvd import A2SVDModel as SeqModel
# from recommenders.models.deeprec.models.sequential.caser import CaserModel as SeqModel
# from recommenders.models.deeprec.modelsscrapbook.sequential.gru4rec import GRU4RecModel as SeqModel


#from recommenders.models.deeprec.models.sequential.nextitnet import NextItNetModel

from recommenders.models.deeprec.io.sequential_iterator import SequentialIterator
from recommenders.models.deeprec.io.nextitnet_iterator import NextItNetIterator
from recommenders.models.deeprec.io.dium_iterator import DIUMIterator

if __name__ == '__main__':

    print("System version: {}".format(sys.version))
    print("Tensorflow version: {}".format(tf.__version__))
    ##  ATTENTION: change to the corresponding config file, e.g., caser.yaml for CaserModel, sum.yaml for SUMModel
    # model_sel="sum"
    #nohup python sequential_recsys_amazondataset.py > dium_nohway.log 2>&1 &
    model_sel="AILMN"#[gru4rec,dprnn4rec,sli_rec,sum,sli_rec_ssl]
    datatype="taobao" #[amazon,kuaishou,ml,yelp,taobao]

    if model_sel=="AILMN":
        from recommenders.models.deeprec.models .sequential.AILMN import DPRNN4RecModel as SeqModel
        yaml_file = '../../recommenders/models/deeprec/config/AILMN.yaml'
    if model_sel=="ILMN":
        from recommenders.models.deeprec.models .sequential.ILMN import DPRNN4RecModel as SeqModel
        yaml_file = '../../recommenders/models/deeprec/config/ILMN.yaml'
    if model_sel=="dprnn4rec":
        from recommenders.models.deeprec.models .sequential.DPRNN4Rec import DPRNN4RecModel as SeqModel
        yaml_file = '../../recommenders/models/deeprec/config/dprnn4rec.yaml'
    if model_sel=="sum":
        from recommenders.models.deeprec.models .sequential.sum import SUMModel as SeqModel
        yaml_file = '../../recommenders/models/deeprec/config/sum.yaml'
    if model_sel=="sumv3":
        # from recommenders.models.deeprec.models.sequential.dium import DIUMModel as SeqModel
        from recommenders.models.deeprec.models.sequential.sum import SUMModel as SeqModel
        yaml_file = '../../recommenders/models/deeprec/config/sumV3.yaml'
    if model_sel=="sumv4":
        from recommenders.models.deeprec.models.sequential.sum import SUMModel as SeqModel
        yaml_file = '../../recommenders/models/deeprec/config/sumV4.yaml'
    if model_sel == "sumv5":
        from recommenders.models.deeprec.models.sequential.sum import SUMModel as SeqModel
        yaml_file = '../../recommenders/models/deeprec/config/sumV5.yaml'
    if model_sel == "sumv6":
        from recommenders.models.deeprec.models.sequential.sum import SUMModel as SeqModel
        yaml_file = '../../recommenders/models/deeprec/config/sumV6.yaml'
    if model_sel == "sumv7":
        from recommenders.models.deeprec.models.sequential.sum import SUMModel as SeqModel
        yaml_file = '../../recommenders/models/deeprec/config/sumV7.yaml'
    if model_sel=="caser":
        from recommenders.models.deeprec.models.sequential.caser import CaserModel as SeqModel
        yaml_file = '../../recommenders/models/deeprec/config/caser.yaml'
    if model_sel == "sli_rec_ssl":
        from recommenders.models.deeprec.models.sequential.sli_rec_ssl import SLI_RECModel as SeqModel
        yaml_file = '../../recommenders/models/deeprec/config/sli_rec_ssl.yaml'
    if model_sel=="sli_rec":
        from recommenders.models.deeprec.models.sequential.sli_rec import SLI_RECModel as SeqModel
        yaml_file = '../../recommenders/models/deeprec/config/sli_rec.yaml'
    if model_sel=="asvd":
        from recommenders.models.deeprec.models.sequential.asvd import A2SVDModel as SeqModel
        yaml_file = '../../recommenders/models/deeprec/config/asvd.yaml'
    if model_sel=="nextitnet":
        from recommenders.models.deeprec.models.sequential.nextitnet import NextItNetModel as SeqModel
        yaml_file = '../../recommenders/models/deeprec/config/nextitnet.yaml'
    if model_sel=="gru4rec":
        from recommenders.models.deeprec.models.sequential.gru4rec import GRU4RecModel as SeqModel
        yaml_file = '../../recommenders/models/deeprec/config/gru4rec.yaml'

    # #### Parameters
    RANDOM_SEED = SEED  # Set None for non-deterministic result
    data_path = os.path.join("..", "..", "tests", "resources", "deeprec", "slirec")

    train_file = os.path.join(data_path, r'train_data')
    valid_file = os.path.join(data_path, r'valid_data')
    test_file = os.path.join(data_path, r'test_data')

    user_vocab = os.path.join(data_path, r'user_vocab.pkl')
    item_vocab = os.path.join(data_path, r'item_vocab.pkl')
    cate_vocab = os.path.join(data_path, r'category_vocab.pkl')
    catid2freq_vocab = os.path.join(data_path, r'catid2freq_vocab.pkl')

    if "wcy" in datatype:
        train_file=train_file+"_wcy"
        test_file = test_file + "_wcy"
        valid_file = valid_file + "_wcy"
        user_vocab = os.path.join(data_path, r'user_vocab_wcy.pkl')
        item_vocab = os.path.join(data_path, r'item_vocab_wcy.pkl')
        cate_vocab = os.path.join(data_path, r'category_vocab_wcy.pkl')
    # output_file = os.path.join(data_path, r'output.txt')
    itemidx2cat=os.path.join(data_path, r'itemidx2cat.pkl')

    hparams = prepare_hparams(yaml_file,
                              user_vocab=user_vocab,
                              item_vocab=item_vocab,
                              cate_vocab=cate_vocab,
                              catid2freq_vocab=catid2freq_vocab,
                              itemidx2cat=itemidx2cat
                              )

    if datatype=="amazon":
        reviews_name= "Clothing_Shoes_and_Jewelry.json"  # [Clothing_Shoes_and_Jewelry.json, reviews_Movies_and_TV_5.json,reviews_Electronics_10.json,reviews_CDs_and_Vinyl_5.json]
        meta_name= "meta_Clothing_Shoes_and_Jewelry.json"  # [meta_Clothing_Shoes_and_Jewelry.json ,meta_Movies_and_TV.json,	meta_Electronics.json,meta_CDs_and_Vinyl.json]
        filter_cate_name=["Clothing, Shoes & Jewelry"]
    elif "taobao" in datatype:
        reviews_name="taobao_user_behavior.csv"
        meta_name=""
        filter_cate_name = [None]
    elif datatype=="kuaishou":
        # reviews_name="kuaishou_small_matrix.csv"
        reviews_name="kuaishou_big_matrix.csv"
        meta_name="kuaishou_item_feat.csv"
        filter_cate_name = [None]
    elif datatype=="yelp":
        # reviews_name="kuaishou_small_matrix.csv"
        reviews_name = "yelp_academic_dataset_review.json"
        meta_name = "yelp_academic_dataset_business.json"
        filter_cate_name = [None]
    elif datatype=="ml":
        reviews_name = "movie.csv"
        meta_name = "movie_meta.csv"
        filter_cate_name = [None]
    reviews_file = os.path.join(data_path, reviews_name)
    meta_file = os.path.join(data_path, meta_name)

    sample_rate =1.0# sample a small item set for training and testing here for fast example

    input_files = [reviews_file, meta_file, train_file, valid_file, test_file, user_vocab, item_vocab, cate_vocab]

    input_creator = SequentialIterator
    if model_sel == "nextitnet":
        input_creator = NextItNetIterator
    if model_sel=="dium" and hparams.useMultiCate:
        input_creator = DIUMIterator
    train_num_ngs=hparams.train_num_ngs
    valid_num_ngs=hparams.valid_num_ngs
    test_num_ngs=hparams.test_num_ngs

    def dataPro():
        print("dataProcess dataset is: {}".format(datatype))
        if not os.path.exists(train_file):
            download_and_extract(reviews_name, reviews_file)
            download_and_extract(meta_name, meta_file)
            data_preprocessing(*input_files, sample_rate=sample_rate, valid_num_ngs=valid_num_ngs, test_num_ngs=test_num_ngs,dataset=datatype,itemidx2cat=itemidx2cat)
            #### uncomment this for the NextItNet model, because it does not need to unfold the user history
            # data_preprocessing(*input_files, sample_rate=sample_rate, valid_num_ngs=valid_num_ngs, test_num_ngs=test_num_ngs, is_history_expanding=False
        print("data_preprocessing begin...")
        if model_sel=="dium" and hparams.useMultiCate:
            dium_amazon_reviews.data_preprocessing(*input_files,catid2freq_vocab=catid2freq_vocab, sample_rate=sample_rate, valid_num_ngs=valid_num_ngs, test_num_ngs=test_num_ngs,hparams=hparams)
        else:
            data_preprocessing(*input_files, sample_rate=sample_rate, valid_num_ngs=valid_num_ngs,
                               test_num_ngs=test_num_ngs,dataset=datatype,itemidx2cat=itemidx2cat,filter_cate_name=filter_cate_name)


    def train(needRetrain=False):
        model = SeqModel(hparams, input_creator, seed=RANDOM_SEED)
        if needRetrain:
            path_best_trained = os.path.join(hparams.MODEL_DIR, "best_model")
            # path_best_trained = os.path.join("./tests/resources/deeprec/SUMV4_timeinterval/model/sum_model/", "best_model")
            if os.path.exists(path_best_trained+".index"):
                print('retrain: prepare loading saved model in {0}'.format(path_best_trained))
                model.load_model(path_best_trained)
            else:
                print('new train: can\'t load model in{}'.format(path_best_trained))

        with Timer() as train_time:
            model = model.fit(train_file, valid_file, valid_num_ngs=valid_num_ngs,eval_metric="group_auc")

        print('Time cost for training is {0:.2f} mins'.format(train_time.interval/60.0))
        # res_syn = model.run_eval(test_file, num_ngs=test_num_ngs)
        # print(res_syn)
        # sb.glue("res_syn", res_syn)
        # model = model.predict(test_file, output_file)
    def test(test_file,test_num_ngs):
        import os
        print(os.getcwd())
        model_best_trained = SeqModel(hparams, input_creator, seed=RANDOM_SEED)
        path_best_trained = os.path.join(hparams.MODEL_DIR, "best_model")
        print('loading saved model in {0}'.format(path_best_trained))
        model_best_trained.load_model(path_best_trained)
        res=model_best_trained.run_eval(filename=test_file, num_ngs=test_num_ngs)
        print(res)
    for param,val in hparams.values().items():
        print("{}: {}".format(param,val))
    # dataPro()
    print("model select is {}".format(model_sel))
    print("data set is: {}".format(datatype))
    train(needRetrain=True)
    # test(train_file,0)
    test(valid_file,valid_num_ngs)
    test(test_file,test_num_ngs)
    sys.exit(0)
