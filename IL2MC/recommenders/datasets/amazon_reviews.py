# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from collections import defaultdict
from datetime import datetime
import numpy as np
import pandas as pd
import gzip
import random
import logging
import _pickle as cPickle

from tqdm import tqdm

from recommenders.utils.constants import SEED
from recommenders.datasets.download_utils import maybe_download


random.seed(SEED)
logger = logging.getLogger()


def get_review_data(reviews_file):
    """Downloads amazon review data (only), prepares in the required format
    and stores in the same location

    Args:
        reviews_file (str): Filename for downloaded reviews dataset.
    """
    reviews_name = reviews_file.split("/")[-1]  # *.json (for url)
    download_and_extract(reviews_name, reviews_file)
    reviews_output = _reviews_preprocessing(reviews_file)
    return reviews_output



def filter_items_with_multiple_cids(record,itemidColName='iid',cateColName='category'):

    # item_cate = record[[itemidColName, cateColName]].drop_duplicates().groupby(itemidColName).count().reset_index().rename(columns={cateColName: 'count'})
    # items_with_single_cid = item_cate[item_cate['count'] == 1][itemidColName]
    #
    # record = pd.merge(record, items_with_single_cid, on=itemidColName)
    cateFreq = defaultdict(int)
    for cate in record[cateColName]:
        cateFreq[cate] += 1
    item_cates=defaultdict(list)
    for item,cate in zip(record[itemidColName],record[cateColName]):
        item_cates[item].append(cate)

    item_singleCate=dict()
    for item,cateList in item_cates.items():
        cateSelect = cateList[0]
        cateFreqMax = cateFreq[cateSelect]
        for c in cateList:
            if cateFreq[c] > cateFreqMax:
                cateFreqMax = cateFreq[c]
                cateSelect = c
        item_singleCate[item]=cateSelect
    cate_replace=[]
    for item in record[itemidColName]:
        cate_replace.append(item_singleCate[item])
    record.iloc[:, record.columns.get_loc(cateColName)] = cate_replace
    return record

def downsample(record, col, frac):

    sample_col = record[col].drop_duplicates().sample(frac=frac)

    record = record.merge(sample_col, on=col).reset_index(drop=True)

    return record


def filter_k_core(record, k_core, filtered_column, count_column):
    stat = record[[filtered_column, count_column]] \
        .groupby(filtered_column) \
        .count() \
        .reset_index() \
        .rename(index=str, columns={count_column: 'count'})

    stat = stat[stat['count'] >= k_core]

    record = record.merge(stat, on=filtered_column)
    record = record.drop(columns=['count'])

    return record

def _data_processing_byGlbTime(input_file,dataset="taobao"):
    global valid_test_split_ratio, train_valid_split_ratio, valid_test_split_time
    logger.info("start data processing...")
    dirs, _ = os.path.split(input_file)
    output_file = os.path.join(dirs, "preprocessed_output")

    f_input = open(input_file, "r")
    f_output = open(output_file, "w")
    if dataset=="tb":
        ## global time division: last day
        test_interval=24*3600
        user_touch_time = []
        for line in f_input:
            line = line.strip()
            time = int(line.split("\t")[3])
            user_touch_time.append(time)
        print("get user touch time completed")
        user_touch_time_sorted = sorted(user_touch_time)
        valid_test_split_time = user_touch_time_sorted[-1] - test_interval
        train_valid_split_time = user_touch_time_sorted[-1] - 2*test_interval

    elif dataset=="kuaishou" or dataset=="yelp"or dataset=="ml" or dataset=='amazon'or dataset=='taobao':
        if dataset=="taobao":
            valid_test_split_ratio=0.9
            train_valid_split_ratio=0.8
        if dataset=="kuaishou":
            valid_test_split_ratio=0.90
            train_valid_split_ratio=0.80
        if dataset=="yelp":
            valid_test_split_ratio = 0.90
            train_valid_split_ratio = 0.80
        if dataset=="ml":
            valid_test_split_ratio = 0.9
            train_valid_split_ratio = 0.8
        if dataset=="amazon":
            valid_test_split_ratio = 0.9
            train_valid_split_ratio = 0.8
        #train:[0,train_valid_split_ratio]  valid:[train_valid_split_ratio,valid_test_split_ratio]  test:[valid_test_split_ratio,1]
        user_touch_time = []
        for line in f_input:
            line = line.strip()
            time = int(float(line.split("\t")[3]))
            user_touch_time.append(time)
        print("get user touch time completed")  #
        user_touch_time_sorted = sorted(user_touch_time)
        valid_test_split_time = user_touch_time_sorted[int(len(user_touch_time_sorted)*valid_test_split_ratio)]
        train_valid_split_time = user_touch_time_sorted[int(len(user_touch_time_sorted)*train_valid_split_ratio)]

    f_input.seek(0)
    train_valid_testDataNum=[0,0,0]
    for line in f_input:
        line = line.strip()
        time = int(float(line.split("\t")[3])) # add
        if time < train_valid_split_time:
            f_output.write("train" + "\t" + line + "\n")
            train_valid_testDataNum[0]+=1
        elif train_valid_split_time <= time < valid_test_split_time:
            f_output.write("valid" + "\t" + line + "\n")
            train_valid_testDataNum[1] += 1
        else:
            f_output.write("test" + "\t" + line + "\n")
            train_valid_testDataNum[2] += 1
    print("_data_processing_byGlbTime, train:valid:test is {}:{}:{}".format(*train_valid_testDataNum))
    return output_file


def _data_generating_global(input_file, train_file, valid_file, test_file, min_sequence_list=[1,1,1],valid_test_sample_probability=0.2):
    """produce train, valid and test file from processed_output file
    Each user's behavior sequence will be unfolded and produce multiple lines in trian file.
    Like, user's behavior sequence: 12345, and this function will write into train file:
    1, 12, 123, 1234, 12345
    Add sampling with 1/10 train instances for long-range sequence dataset(kuaishou)
    """
    f_input = open(input_file, "r")
    f_train = open(train_file, "w")
    f_valid = open(valid_file, "w")
    f_test = open(test_file, "w")
    logger.info("data generating...")
    last_user_id = None
    for line in f_input:
        line_split = line.strip().split("\t")
        tfile = line_split[0]
        label = int(line_split[1])
        user_id = line_split[2]
        movie_id = line_split[3]
        date_time = line_split[4]
        category = line_split[5]

        if tfile == "train":
            fo = f_train
            sample_probability = 0
        elif tfile == "valid":
            fo = f_valid
            #  sample_probability = 0 # add
            sample_probability = round(np.random.uniform(0, 1), 1) # add
        elif tfile == "test":
            fo = f_test
            #  sample_probability = 0 # add
            sample_probability = round(np.random.uniform(0, 1), 1) # add
        if user_id != last_user_id:
            movie_id_list = []
            cate_list = []
            dt_list = []
        else:
            if 0 <= sample_probability < valid_test_sample_probability: # add: 1/5 probability
                history_clk_num = len(movie_id_list)
                cat_str = ""
                mid_str = ""
                dt_str = ""
                for c1 in cate_list:
                    cat_str += c1 + ","
                for mid in movie_id_list:
                    mid_str += mid + ","
                for dt_time in dt_list:
                    dt_str += dt_time + ","
                if len(cat_str) > 0:
                    cat_str = cat_str[:-1]#去掉多余逗号
                if len(mid_str) > 0:
                    mid_str = mid_str[:-1]#去掉多余逗号
                if len(dt_str) > 0:
                    dt_str = dt_str[:-1]#去掉多余逗号

                if (tfile=="train" and history_clk_num >= min_sequence_list[0])or \
                    (tfile=="valid" and history_clk_num >= min_sequence_list[1])or \
                    (tfile=="test" and history_clk_num >= min_sequence_list[2]):
                    fo.write(
                        line_split[1]
                        + "\t"
                        + user_id
                        + "\t"
                        + movie_id
                        + "\t"
                        + category
                        + "\t"
                        + date_time
                        + "\t"
                        + mid_str
                        + "\t"
                        + cat_str
                        + "\t"
                        + dt_str
                        + "\n"
                    )
            else:
                pass
        last_user_id = user_id
        if label:
            movie_id_list.append(movie_id)
            cate_list.append(category)
            dt_list.append(date_time)

def taobao_main(reviews_file):

    reviews = pd.read_csv(reviews_file, header=None, names=['uid', 'iid', 'category', 'behavior', 'ts'])
    reviews = reviews[reviews['behavior'] == 'pv']
    reviews = reviews.drop_duplicates(subset=['uid', 'iid'])
    reviews = filter_items_with_multiple_cids(reviews)#筛选出只有一个标签的物品
    # start_ts = int(datetime.timestamp(datetime(2017, 11, 25, 0, 0, 0)))
    # end_ts = int(datetime.timestamp(datetime(2017, 12, 3, 23, 59, 59)))
    start_ts = int(datetime.timestamp(datetime(2017, 11,25, 0, 0, 0)))
    end_ts = int(datetime.timestamp(datetime(2017, 12, 3, 23, 59, 59)))
    reviews = reviews[reviews['ts'] >= start_ts]
    reviews = reviews[reviews['ts'] <= end_ts]
    reviews = downsample(reviews, 'uid', 0.05)
    k_core = 10
    reviews = filter_k_core(reviews,k_core, 'iid', 'uid')
    reviews = filter_k_core(reviews, k_core, 'uid', 'iid')
    print("user num is {}".format(len(reviews['uid'].unique())))

    # columns = ["label", "user_id", "item_id", "timestamp", "cate_id"]
    # reviews = pd.read_csv(os.path.join(os.path.split(reviews_file)[0],"taobao_mutation.csv"), sep="\t", names=columns)
    # print("taobao len: {}".format(len(reviews)))
    business = reviews[['iid', 'category']].drop_duplicates()
    reviews = reviews[['uid', 'iid', 'ts']]

    dirs, _ = os.path.split(reviews_file)

    reviews_output = os.path.join(dirs, 'taobao_review_recommenders.csv')
    meta_output = os.path.join(dirs, 'taobao_business_recommenders.csv')

    reviews.to_csv(reviews_output, sep='\t', header=False, index=False)
    business.to_csv(meta_output, sep='\t', header=False, index=False)

    return reviews_output, meta_output

def preprocessing_kuaishou(reviews_file,meta_file):
    #user_id,video_id,play_duration,video_duration,time,date,timestamp,watch_ratio
    reviews = pd.read_csv(reviews_file,dtype=str)
    reviews=reviews[~reviews['timestamp'].isna()]
    reviews['watch_ratio'] = reviews['watch_ratio'].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    reviews['date'] = reviews['date'].astype(float)
    likeSheld=0.3

    reviews=reviews[ reviews['watch_ratio']>likeSheld ]
    reviews = reviews.drop_duplicates(subset=['user_id', 'video_id'])
    start_ts =20200820# best 202008 20
    end_ts =20200905
    reviews = reviews[reviews['date'] >= start_ts]
    reviews = reviews[reviews['date'] <= end_ts]
    k_core = 10
    reviews = filter_k_core(reviews, k_core, 'user_id', 'video_id')
    reviews = filter_k_core(reviews, k_core, 'video_id', 'user_id')
    reviews = reviews[['user_id', 'video_id', 'timestamp']]


    itemCate = pd.read_csv(meta_file,dtype=str).drop_duplicates()
    itemCate['feat'] = itemCate['feat'].astype(str)
    cateFreq = defaultdict(int)
    for cate in itemCate['feat']:
        cateList = eval(cate)
        for c in cateList:
            cateFreq[c] += 1

    def cateList2singleCate(cateList):
        cateList = eval(cateList)
        if isinstance(cateList, list):
            cateSelect = cateList[0]
            cateFreqMax = cateFreq[cateSelect]
            for c in cateList:
                if cateFreq[c] > cateFreqMax:
                    cateFreqMax = cateFreq[c]
                    cateSelect = c
            return cateSelect
        else:
            print("cateList is not list type\n")
            exit(0)

    itemCate['feat'] = itemCate['feat'].map(cateList2singleCate)


    dirs, _ = os.path.split(reviews_file)
    reviews_output = os.path.join(dirs, 'kuaishou_review_recommenders.csv')
    meta_output = os.path.join(dirs, 'kuaishou_itemCate_recommenders.csv')

    reviews.to_csv(reviews_output, sep='\t', header=False, index=False)
    itemCate.to_csv(meta_output, sep='\t', header=False, index=False)

    return reviews_output, meta_output

def preprocessing_ml(reviews_file,meta_file):
    #user_id,video_id,play_duration,video_duration,time,date,timestamp,watch_ratio
    reviews = pd.read_csv(reviews_file,sep="::",dtype=str, header=None, names=['uid', 'iid', 'rate', 'timestamp'])
    reviews=reviews[~reviews['timestamp'].isna()]
    reviews['rate'] = reviews['rate'].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    reviews['timestamp'] = reviews['timestamp'].astype(float)
    likeSheld=1.8
    reviews=reviews[ reviews['rate']>likeSheld ]

    reviews = reviews.drop_duplicates(subset=['uid', 'iid'])
    maxTime = max(reviews['timestamp'])
    minTime = maxTime - 3600 * 24 * 365 * 2.5
    reviews = reviews[(reviews['timestamp'] >= minTime) & (reviews['timestamp'] < maxTime)]
    # start_ts =20200825# best 202008 20
    # end_ts =20200905
    # reviews = reviews[reviews['date'] >= start_ts]
    # reviews = reviews[reviews['date'] <= end_ts]
    k_core = 10
    reviews = filter_k_core(reviews, k_core, 'uid', 'iid')
    reviews = filter_k_core(reviews, k_core, 'iid', 'uid')
    reviews = reviews[['uid', 'iid', 'timestamp']]

    itemCate =pd.read_csv(meta_file,sep="::",dtype=str, header=None, names=['iid', 'Title', 'Genres'],encoding='ISO-8859-1').drop_duplicates()
    itemCate['Genres'] = itemCate['Genres'].astype(str)
    cateFreq = defaultdict(int)
    for cate in itemCate['Genres']:
        cateList = cate.strip().split('|')
        for c in cateList:
            cateFreq[c] += 1
    print(cateFreq)
    def cateList2singleCate(cateList):
        cateList = cateList.strip().split('|')
        if isinstance(cateList, list):
            cateSelect = cateList[0]
            cateFreqMax = cateFreq[cateSelect]
            for c in cateList:
                if cateFreq[c] > cateFreqMax:
                    cateFreqMax = cateFreq[c]
                    cateSelect = c
            return cateSelect
        else:
            print("cateList is not list type\n")
            exit(0)
    print(itemCate)
    itemCate['Genres'] = itemCate['Genres'].apply(cateList2singleCate)
    print(itemCate)


    dirs, _ = os.path.split(reviews_file)
    reviews_output = os.path.join(dirs, 'ml_review_recommenders.csv')
    meta_output = os.path.join(dirs, 'ml_itemCate_recommenders.csv')

    reviews.to_csv(reviews_output, sep='\t', header=False, index=False)
    itemCate.to_csv(meta_output, sep='\t', header=False, index=False)

    return reviews_output, meta_output

def data_preprocessing(
    reviews_file,
    meta_file,
    train_file,
    valid_file,
    test_file,
    user_vocab,
    item_vocab,
    cate_vocab,
    sample_rate=0.01,
    valid_num_ngs=4,
    test_num_ngs=9,
    dataset='amazon',
    is_history_expanding=True,
    itemidx2cat=None,
    filter_cate_name=None
):
    """Create data for training, validation and testing from original dataset

    Args:
        reviews_file (str): Reviews dataset downloaded from former operations.
        meta_file (str): Meta dataset downloaded from former operations.
    """

    if dataset == 'taobao':
        reviews_output, meta_output = taobao_main(reviews_file)
    elif dataset == 'amazon':
        meta_output = _meta_preprocessing(meta_file,filter_cate_name=filter_cate_name)
        reviews_output = _reviews_preprocessing(reviews_file)
    elif dataset == 'kuaishou':
        reviews_output, meta_output = preprocessing_kuaishou(reviews_file,meta_file)
    elif dataset == 'yelp':
        reviews_output, meta_output= preprocessing_yelp(reviews_file,meta_file)
        # dirs, _ = os.path.split(reviews_file)
        # reviews_output = os.path.join(dirs, 'yelp_review_recommenders.csv')
        # meta_output = os.path.join(dirs, 'yelp_business_recommenders.csv')
    elif dataset=='ml':
        reviews_output, meta_output = preprocessing_ml(reviews_file, meta_file)

    instance_output = _create_instance(reviews_output, meta_output)
    _create_item2cate(instance_output)
    sampled_instance_file = _get_sampled_data(instance_output, sample_rate=sample_rate)

    if dataset in ['amazon','taobao','yelp','ml','kuaishou']:
        preprocessed_output = _data_processing_byGlbTime(sampled_instance_file,dataset=dataset)
    elif dataset in ['amazon']:
        preprocessed_output = _data_processing(sampled_instance_file)

    if is_history_expanding:
        if dataset in ['amazon','kuaishou','yelp','taobao','ml']:
            valid_test_sample_probability_dataset={'taobao':0.2,'kuaishou':0.2,'yelp':0.2,'ml':0.15,'amazon':0.2}
            _data_generating_global(preprocessed_output, train_file, valid_file, test_file,
                                    valid_test_sample_probability=valid_test_sample_probability_dataset[dataset],
                                    min_sequence_list=[20,20,20]
                                    )
        elif dataset in ['amazon']:
            _data_generating(preprocessed_output, train_file, valid_file, test_file)
    else:
        _data_generating_no_history_expanding(
            preprocessed_output, train_file, valid_file, test_file
        )

    _create_vocab(train_file, user_vocab, item_vocab, cate_vocab,itemidx2cat)
    _negative_sampling_offline(
        sampled_instance_file, valid_file, test_file, valid_num_ngs, test_num_ngs
    )


def _create_vocab(train_file, user_vocab, item_vocab, cate_vocab,itemidx2cat=None):

    f_train = open(train_file, "r")

    user_dict = {}
    item_dict = {}
    cat_dict = {}
    logger.info("vocab generating...")
    for line in f_train:
        arr = line.strip("\n").split("\t")
        uid = arr[1]
        mid = arr[2]
        cat = arr[3]
        mid_list = arr[5]
        cat_list = arr[6]
        if uid not in user_dict:
            user_dict[uid] = 0
        user_dict[uid] += 1
        if mid not in item_dict:
            item_dict[mid] = 0
        item_dict[mid] += 1
        if cat not in cat_dict:
            cat_dict[cat] = 0
        cat_dict[cat] += 1
        if len(mid_list) == 0:
            continue
        for m in mid_list.split(","):
            if m not in item_dict:
                item_dict[m] = 0
            item_dict[m] += 1
        for c in cat_list.split(","):
            if c not in cat_dict:
                cat_dict[c] = 0
            cat_dict[c] += 1

    sorted_user_dict = sorted(user_dict.items(), key=lambda x: x[1], reverse=True)
    sorted_item_dict = sorted(item_dict.items(), key=lambda x: x[1], reverse=True)
    sorted_cat_dict = sorted(cat_dict.items(), key=lambda x: x[1], reverse=True)

    uid_voc = {}
    index = 0
    for key, value in sorted_user_dict:
        uid_voc[key] = index
        index += 1
    global item2cate
    mid_voc = {}
    mid2cat={}
    mid_voc["default_mid"] = 0
    mid2cat[0]="default_cat"
    index = 1
    numHasCate=0
    numNotCat=0

    for key, value in sorted_item_dict:
        mid_voc[key] = index#itemName -> emdIdx
        if str(key) not in item2cate.keys():
            mid2cat[index]="default_cat"
            numNotCat+=1
            # print("item-cat:{}-{}".format(key, 'default_cat'))
        else:
            mid2cat[index]=item2cate[str(key)]#emdIdx->cate
            numHasCate+=1
            # print("item-cat:{}-{}".format(key, item2cate[str(key)]))
        index += 1
    print("create voc fin, num  of with cate/no cate={}:{}".format(numHasCate,numNotCat))
    cat_voc = {}
    cat_voc["default_cat"] = 0
    index = 1
    for key, value in sorted_cat_dict:
        cat_voc[key] = index
        index += 1

    cPickle.dump(uid_voc, open(user_vocab, "wb"))
    cPickle.dump(mid_voc, open(item_vocab, "wb"))
    cPickle.dump(cat_voc, open(cate_vocab, "wb"))
    if itemidx2cat is None:
        dirs, _ = os.path.split(user_vocab)
        itemidx2cat = os.path.join(dirs, "itemidx2cat.pkl")

    cPickle.dump(mid2cat,open(itemidx2cat,"wb"))

def _negative_sampling_offline(
    instance_input_file, valid_file, test_file, valid_neg_nums=4, test_neg_nums=49
):

    columns = ["label", "user_id", "item_id", "timestamp", "cate_id"]
    ns_df = pd.read_csv(instance_input_file, sep="\t", names=columns)
    items_with_popular = list(ns_df["item_id"].astype(str))
    global item2cate
    if valid_neg_nums>0:
        # valid negative sampling
        logger.info("start valid negative sampling")
        with open(valid_file, "r") as f:
            valid_lines = f.readlines()
        write_valid = open(valid_file, "w")
        for line in valid_lines:
            write_valid.write(line)
            words = line.strip().split("\t")
            positive_item = words[2]
            count = 0
            neg_items = set()
            while count < valid_neg_nums:
                neg_item = random.choice(items_with_popular)
                if neg_item == positive_item or neg_item in neg_items:
                    continue
                count += 1
                neg_items.add(neg_item)
                words[0] = "0"
                words[2] = str(neg_item)
                words[3] = str(item2cate[neg_item])

                write_valid.write("\t".join(words) + "\n")
    if test_neg_nums>0:
        # test negative sampling
        logger.info("start test negative sampling")
        with open(test_file, "r") as f:
            test_lines = f.readlines()
        write_test = open(test_file, "w")
        for line in test_lines:
            write_test.write(line)
            words = line.strip().split("\t")
            positive_item = words[2]
            count = 0
            neg_items = set()
            while count < test_neg_nums:
                neg_item = random.choice(items_with_popular)
                if neg_item == positive_item or neg_item in neg_items:
                    continue
                count += 1
                neg_items.add(neg_item)
                words[0] = "0"
                words[2] = str(neg_item)
                words[3] = str(item2cate[neg_item])
                write_test.write("\t".join(words) + "\n")


def _data_generating(input_file, train_file, valid_file, test_file, min_sequence=1):
    """produce train, valid and test file from processed_output file
    Each user's behavior sequence will be unfolded and produce multiple lines in trian file.
    Like, user's behavior sequence: 12345, and this function will write into train file:
    1, 12, 123, 1234, 12345
    """
    f_input = open(input_file, "r")
    f_train = open(train_file, "w")
    f_valid = open(valid_file, "w")
    f_test = open(test_file, "w")
    logger.info("data generating...")
    last_user_id = None
    for line in f_input:
        line_split = line.strip().split("\t")
        tfile = line_split[0]
        label = int(line_split[1])
        user_id = line_split[2]
        movie_id = line_split[3]
        date_time = line_split[4]
        category = line_split[5]

        if tfile == "train":
            fo = f_train
        elif tfile == "valid":
            fo = f_valid
        elif tfile == "test":
            fo = f_test
        if user_id != last_user_id:
            movie_id_list = []
            cate_list = []
            dt_list = []
        else:
            history_clk_num = len(movie_id_list)
            cat_str = ""
            mid_str = ""
            dt_str = ""
            for c1 in cate_list:
                cat_str += c1 + ","
            for mid in movie_id_list:
                mid_str += mid + ","
            for dt_time in dt_list:
                dt_str += dt_time + ","
            if len(cat_str) > 0:
                cat_str = cat_str[:-1]
            if len(mid_str) > 0:
                mid_str = mid_str[:-1]
            if len(dt_str) > 0:
                dt_str = dt_str[:-1]
            if history_clk_num >= min_sequence:
                fo.write(
                    line_split[1]
                    + "\t"
                    + user_id
                    + "\t"
                    + movie_id
                    + "\t"
                    + category
                    + "\t"
                    + date_time
                    + "\t"
                    + mid_str
                    + "\t"
                    + cat_str
                    + "\t"
                    + dt_str
                    + "\n"
                )
        last_user_id = user_id
        if label:
            movie_id_list.append(movie_id)
            cate_list.append(category)
            dt_list.append(date_time)


def _data_generating_no_history_expanding(
    input_file, train_file, valid_file, test_file, min_sequence=1
):
    """Produce train, valid and test file from processed_output file
    Each user's behavior sequence will only produce one line in train file.
    Like, user's behavior sequence: 12345, and this function will write into train file: 12345
    """
    f_input = open(input_file, "r")
    f_train = open(train_file, "w")
    f_valid = open(valid_file, "w")
    f_test = open(test_file, "w")
    logger.info("data generating...")

    last_user_id = None
    last_movie_id = None
    last_category = None
    last_datetime = None
    last_tfile = None
    for line in f_input:
        line_split = line.strip().split("\t")
        tfile = line_split[0]
        label = int(line_split[1])
        user_id = line_split[2]
        movie_id = line_split[3]
        date_time = line_split[4]
        category = line_split[5]

        if last_tfile == "train":
            fo = f_train
        elif last_tfile == "valid":
            fo = f_valid
        elif last_tfile == "test":
            fo = f_test
        if user_id != last_user_id or tfile == "valid" or tfile == "test":
            if last_user_id is not None:
                history_clk_num = len(
                    movie_id_list  # noqa: F821 undefined name 'movie_id_list'
                )
                cat_str = ""
                mid_str = ""
                dt_str = ""
                for c1 in cate_list[:-1]:  # noqa: F821 undefined name 'cate_list'
                    cat_str += c1 + ","
                for mid in movie_id_list[  # noqa: F821 undefined name 'movie_id_list'
                    :-1
                ]:
                    mid_str += mid + ","
                for dt_time in dt_list[:-1]:  # noqa: F821 undefined name 'dt_list'
                    dt_str += dt_time + ","
                if len(cat_str) > 0:
                    cat_str = cat_str[:-1]
                if len(mid_str) > 0:
                    mid_str = mid_str[:-1]
                if len(dt_str) > 0:
                    dt_str = dt_str[:-1]
                if history_clk_num > min_sequence:
                    fo.write(
                        line_split[1]
                        + "\t"
                        + last_user_id
                        + "\t"
                        + last_movie_id
                        + "\t"
                        + last_category
                        + "\t"
                        + last_datetime
                        + "\t"
                        + mid_str
                        + "\t"
                        + cat_str
                        + "\t"
                        + dt_str
                        + "\n"
                    )
            if tfile == "train" or last_user_id is None:
                movie_id_list = []
                cate_list = []
                dt_list = []
        last_user_id = user_id
        last_movie_id = movie_id
        last_category = category
        last_datetime = date_time
        last_tfile = tfile
        if label:
            movie_id_list.append(movie_id)
            cate_list.append(category)
            dt_list.append(date_time)


def _create_item2cate(instance_file):
    logger.info("creating item2cate dict")
    global item2cate
    instance_df = pd.read_csv(
        instance_file,
        sep="\t",
        names=["label", "user_id", "item_id", "timestamp", "cate_id"],
    )
    instance_df['item_id']=instance_df['item_id'].astype(str)
    instance_df['cate_id']=instance_df['cate_id'].astype(str)

    item2cate = instance_df.set_index("item_id")["cate_id"].to_dict()


def _get_sampled_data(instance_file, sample_rate):
    logger.info("getting sampled data...")
    global item2cate
    output_file = instance_file + "_" + str(sample_rate)
    columns = ["label", "user_id", "item_id", "timestamp", "cate_id"]
    ns_df = pd.read_csv(instance_file, sep="\t", names=columns)
    items_num = ns_df["user_id"].nunique()
    items_with_popular = set(list(ns_df["user_id"].unique()))
    items_sample, count = set(), 0
    if sample_rate<=0.7:
        while count < int(items_num * sample_rate):
            print(count/int(items_num * sample_rate),end=" ")
            random_item = random.choice(list(items_with_popular-items_sample))
            # if random_item not in items_sample:
            items_sample.add(random_item)
            count += 1
        ns_df_sample = ns_df[ns_df["user_id"].isin(items_sample)]
        ns_df_sample.to_csv(output_file, sep="\t", index=None, header=None)
    else:
        ns_df.to_csv(output_file, sep="\t", index=None, header=None)
    return output_file
def _meta_preprocessing(meta_readfile,itemidColName="asin",filter_cate_name=None):
    logger.info("start meta preprocessing...")
    print("start meta preprocessing...")
    meta_writefile = meta_readfile + "_output"
    return meta_writefile
    meta_r = open(meta_readfile, "r")
    meta_w = open(meta_writefile, "w")

    cnt = 0
    catKeyName=""
    catListis2D=True
    cateFreq=defaultdict(int)
    itemCateList=defaultdict(list)
    for line in tqdm(meta_r):
        # print(line)
        line_new = eval(line.replace("\n",""),{"true":True,"false":False,"null":None} )
        # print(line_new)
        if cnt==0:
            if "categories" in line_new.keys():
                catKeyName="categories"
            else:
                catKeyName="category"
            if isinstance(line_new[catKeyName][0], str):
                catListis2D = False
            else:
                catListis2D = True
            cnt+=1
        # print(line_new[catKeyName])
        cateList=[]
        if catListis2D:
            # meta_w.write(line_new[itemidColName] + "\t" + np.random.choice(line_new[catKeyName][0]) + "\n")
            meta_w.write(line_new[itemidColName]+"\n")

            for cates in line_new[catKeyName]:
                for c in cates:
                    if (filter_cate_name is None )or (c not in filter_cate_name):
                        cateList.append(c)
                        cateFreq[c]+=1
                        itemCateList[line_new[itemidColName]].append(c)

        else:
            # meta_w.write(line_new[itemidColName] + "\t" + np.random.choice(line_new[catKeyName]) + "\n")
            meta_w.write(line_new[itemidColName]+ "\n")
            for c in line_new[catKeyName]:
                if (filter_cate_name is None) or (c not in filter_cate_name):
                    cateList.append(c)
                    cateFreq[c] += 1
                    itemCateList[line_new[itemidColName]].append(c)
    meta_r.close()
    meta_w.close()
    item2singleCate=dict()
    for itemid,cates in itemCateList.items():
        select_cate=cates[0]
        maxFreq=cateFreq[cates[0]]
        for c in cates:
            if maxFreq<cateFreq[c]:
                select_cate=c
                maxFreq=cateFreq[c]
        item2singleCate[itemid]=select_cate
    metaDF=pd.read_csv(meta_writefile,header=None, names=['iid']).drop_duplicates()
    itemid_with_single_cate=[]
    print(metaDF)
    # print(item2singleCate)
    for itemid in metaDF['iid']:
        if itemid in item2singleCate.keys():
            itemid_with_single_cate.append(item2singleCate[itemid])
        else:
            itemid_with_single_cate.append("default_cat")

    metaDF['cate']=itemid_with_single_cate
    metaDF.to_csv(meta_writefile,sep='\t', header=False, index=False)
    return meta_writefile



def _reviews_preprocessing(reviews_readfile):
    logger.info("start reviews preprocessing...")
    reviews_writefile = reviews_readfile + "_output"
    reviews_r = open(reviews_readfile, "r")
    reviews_w = open(reviews_writefile, "w")
    for line in tqdm(reviews_r):
        line_new = eval(line.strip(),{"true":True,"false":False,"null":None} )
        reviews_w.write(
            str(line_new["reviewerID"])
            + "\t"
            + str(line_new["asin"])
            + "\t"
            + str(line_new["unixReviewTime"])
            + "\n"
        )
    reviews_r.close()
    reviews_w.close()

    reviews=pd.read_csv(reviews_writefile, sep="\t", names=["uid","iid","unixReviewTime"])
    maxTime=max(reviews['unixReviewTime'])
    minTime=maxTime-3600*24*365*3
    reviews=reviews[(reviews['unixReviewTime'] >= minTime) & (reviews['unixReviewTime'] < maxTime)]
    k_core = 10
    reviews = filter_k_core(reviews, k_core, 'iid', 'uid')
    reviews = filter_k_core(reviews, k_core, 'uid', 'iid')
    reviews_writefile_result = reviews_writefile + "_kcore"
    reviews.to_csv(reviews_writefile_result,sep='\t', header=False, index=False)

    return reviews_writefile_result


def preprocessing_yelp(reviews_readfile,meta_readfile):
    meta_r = open(meta_readfile, "r", encoding='UTF-8')
    cnt = 0
    catKeyName = ""
    itemCate=defaultdict(set)
    for line in tqdm(meta_r):
        line_new = eval(line.replace("\n", ""), {"true": True, "false": False, "null": None})
        if cnt == 0:
            if "categories" in line_new.keys():
                catKeyName = "categories"
            else:
                catKeyName = "category"
            cnt += 1
        if line_new[catKeyName] is not None:
            cates=line_new[catKeyName].split(", ")
            for c in cates:
                itemCate[line_new['business_id']].add(c)
    meta_r.close()
    itemWithSingleCate_Cate=dict()
    for item, cates in itemCate.items():
        # if len(cates)==1:
        #     itemWithSingleCate_Cate[item]=list(cates)[0]
        itemWithSingleCate_Cate[item]=list(cates)[0]
    logger.info("start reviews preprocessing...")
    reviews_writefile = reviews_readfile + "_output"
    if not os.path.exists(reviews_writefile):
        reviews_r = open(reviews_readfile, "r",encoding='utf-8')
        reviews=[]

        for line in tqdm(reviews_r):
            line_new = eval(line.strip(),{"true":True,"false":False,"null":None} )
            if str(line_new["business_id"]) in itemWithSingleCate_Cate.keys():
                reviews.append([str(line_new["user_id"]),
                                str(line_new["business_id"]),itemWithSingleCate_Cate[str(line_new["business_id"])],
                                str(line_new["stars"]),
                                datetime.timestamp(datetime.strptime(str(line_new["date"]), "%Y-%m-%d %H:%M:%S"))
                                ])
        reviews_r.close()
        reviews=pd.DataFrame(reviews,columns=['user_id', 'business_id','category','stars', 'date'])
        reviews.to_csv(reviews_writefile, header=False, index=False)
    else:
        reviews = pd.read_csv(reviews_writefile, header=None, names=['user_id', 'business_id','category','stars', 'date'])
    reviews=reviews.dropna()
    reviews['stars']=reviews['stars'].astype(float)
    reviews['date']=reviews['date'].astype(int)
    reviews = reviews[(reviews['date'] > 1226443721) & (reviews['date'] < 1576223149)]#avg len=25
    reviews = reviews[reviews['stars'] > 2.5 ]
    reviews = reviews.drop_duplicates(subset=['user_id', 'business_id'])


    # reviews = downsample(reviews, 'user_id', 0.5)
    k_core = 10
    reviews = filter_k_core(reviews, k_core, 'business_id', 'user_id')
    reviews = filter_k_core(reviews, k_core, 'user_id', 'business_id')

    business = reviews[['business_id', 'category']].drop_duplicates()
    reviews = reviews[['user_id', 'business_id', 'date']]
    print(reviews)
    dirs, _ = os.path.split(reviews_readfile)

    reviews_output = os.path.join(dirs, 'yelp_review_recommenders.csv')
    meta_output = os.path.join(dirs, 'yelp_business_recommenders.csv')

    reviews.to_csv(reviews_output, sep='\t', header=False, index=False)
    business.to_csv(meta_output, sep='\t', header=False, index=False)
    return reviews_output, meta_output

def _create_instance(reviews_file, meta_file):
    logger.info("start create instances...")
    dirs, _ = os.path.split(reviews_file)
    output_file = os.path.join(dirs, "instance_output")

    f_reviews = open(reviews_file, "r")
    user_dict = {}
    item_list = []
    for line in f_reviews:
        line = line.strip()
        reviews_things = line.split("\t")
        if reviews_things[0] not in user_dict:
            user_dict[reviews_things[0]] = []#reviews_things[0]==>uid
        user_dict[reviews_things[0]].append((line, float(reviews_things[-1])))
        item_list.append(reviews_things[1])

    f_meta = open(meta_file, "r")
    meta_dict = {}
    for line in f_meta:
        line = line.strip()
        meta_things = line.split("\t")
        if meta_things[0] not in meta_dict:
            meta_dict[meta_things[0]] = meta_things[1]

    f_output = open(output_file, "w")
    asinHasCate=dict()
    asinWithNoCate=dict()
    for user_behavior in user_dict:
        sorted_user_behavior = sorted(user_dict[user_behavior], key=lambda x: x[1])#按时间排序
        for line, _ in sorted_user_behavior:
            user_things = line.split("\t")
            asin = user_things[1]
            if asin in meta_dict:
                f_output.write("1" + "\t" + line + "\t" + meta_dict[asin] + "\n")
                asinHasCate[asin]=1
            else:
                f_output.write("1" + "\t" + line + "\t" + "default_cat" + "\n")
                asinWithNoCate[asin]=1
    print("=============the num of  asin has cate is {}, with no cate is {} =========\n".format(
        len(asinHasCate.keys()),len(asinWithNoCate.keys()) ) )
    f_reviews.close()
    f_meta.close()
    f_output.close()
    return output_file


def _data_processing(input_file):
    logger.info("start data processing...")
    dirs, _ = os.path.split(input_file)
    output_file = os.path.join(dirs, "preprocessed_output")

    f_input = open(input_file, "r")
    f_output = open(output_file, "w")
    user_count = {}
    for line in f_input:
        line = line.strip()
        user = line.split("\t")[1]
        if user not in user_count:
            user_count[user] = 0
        user_count[user] += 1
    f_input.seek(0)
    i = 0
    last_user = None
    for line in f_input:
        line = line.strip()
        user = line.split("\t")[1]
        if user == last_user:
            if i < user_count[user] - 2:
                f_output.write("train" + "\t" + line + "\n")
            elif i < user_count[user] - 1:
                f_output.write("valid" + "\t" + line + "\n")
            else:
                f_output.write("test" + "\t" + line + "\n")
        else:
            last_user = user
            i = 0
            if i < user_count[user] - 2:
                f_output.write("train" + "\t" + line + "\n")
            elif i < user_count[user] - 1:
                f_output.write("valid" + "\t" + line + "\n")
            else:
                f_output.write("test" + "\t" + line + "\n")
        i += 1
    return output_file


def download_and_extract(name, dest_path):
    """Downloads and extracts Amazon reviews and meta datafiles if they don’t already exist

    Args:
        name (str): Category of reviews.
        dest_path (str): File path for the downloaded file.

    Returns:
        str: File path for the extracted file.
    """
    dirs, _ = os.path.split(dest_path)
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    file_path = os.path.join(dirs, name)
    if not os.path.exists(file_path):
        _download_reviews(name, dest_path)
        _extract_reviews(file_path, dest_path)

    return file_path


def _download_reviews(name, dest_path):
    """Downloads Amazon reviews datafile.

    Args:
        name (str): Category of reviews
        dest_path (str): File path for the downloaded file
    """

    url = (
        "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"
        + name
        + ".gz"
    )

    dirs, file = os.path.split(dest_path)
    maybe_download(url, file + ".gz", work_directory=dirs)


def _extract_reviews(file_path, zip_path):
    """Extract Amazon reviews and meta datafiles from the raw zip files.

    To extract all files,
    use ZipFile's extractall(path) instead.

    Args:
        file_path (str): Destination path for datafile
        zip_path (str): zipfile path
    """
    with gzip.open(zip_path + ".gz", "rb") as zf, open(file_path, "wb") as f:
        shutil.copyfileobj(zf, f)
