import os
from collections import defaultdict
from datetime import datetime
import pandas as pd
import sys

from tqdm import tqdm

sys.path.append("../../")
from recommenders.datasets.amazon_reviews import _create_item2cate

'''
分析数据集的统计数据
'''

def amzDataAna():
    '''
    打印数据集的用户数、物品数、交互数、稠密度、平均长度
    Returns:

    '''
if __name__ == '__main__':
    dataPath="../../tests/resources/deeprec/slirec/"
    dataType="kuaishou"


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


    def downsample(record, col, frac):
        sample_col = record[col].drop_duplicates().sample(frac=frac)
        record = record.merge(sample_col, on=col).reset_index(drop=True)
        return record
    if dataType=="amazon":
        review_data_output=os.path.join(dataPath,"instance_output_1.0")#[label,userName,itemName,timestamp,cate]
        meta_data_output=os.path.join(dataPath,"meta_CDs_and_Vinyl.json_output")#[itemName,cate]
        reviewData=[]
        users,items=[],[]
        userItems=defaultdict(list)
        with open(review_data_output,"r") as reviewDataFile:
            lines=reviewDataFile.readlines()
            for line in lines:
                label,userName,itemName,timestamp,cate=line.strip().split('\t')
                users.append(userName)
                items.append(itemName)
                userItems[userName].append(itemName)
        sumLen=0
        for _,histbhv in userItems.items():
            sumLen+=len(histbhv)
        userNum=len(set(users))
        itemNum=len(set(items))
        interactionNum=len(users)
        print("userNum={}; itemNum={}".format( userNum, itemNum ))
        print("interaction Num={}".format(interactionNum))
        print("data sparsity={}".format(interactionNum/(userNum*itemNum)))
        print("Avg. len={}".format(sumLen/userNum))
    elif dataType=="taobao":
        def filter_items_with_multiple_cids(record, itemidColName='iid', cateColName='category'):
            item_cate = record[[itemidColName, cateColName]].drop_duplicates().groupby(
                itemidColName).count().reset_index().rename(columns={cateColName: 'count'})
            items_with_single_cid = item_cate[item_cate['count'] == 1][itemidColName]
            record = pd.merge(record, items_with_single_cid, on=itemidColName)
            return record


        reviews_file=review_data_output=os.path.join(dataPath,"taobao_user_behavior.csv")
        reviews = pd.read_csv(reviews_file, header=None, names=['uid', 'iid', 'category', 'behavior', 'ts'])
        reviews = reviews[reviews['behavior'] == 'pv']
        reviews = reviews.drop_duplicates(subset=['uid', 'iid'])
        reviews = filter_items_with_multiple_cids(reviews)  # 筛选出只有一个标签的物品
        reviews=reviews[reviews['ts']>0]
        maxTime,minTime=max(reviews['ts']),min(reviews['ts'])
        print("maxTime,minTime={}/{}".format(maxTime,minTime))

        start_ts = int(datetime.timestamp(datetime(2017, 11,1, 0, 0, 0)))
        end_ts = int(datetime.timestamp(datetime(2017, 12,3, 23, 59, 59)))
        reviews = reviews[reviews['ts'] >= start_ts]
        reviews = reviews[reviews['ts'] <= end_ts]
        user_len = reviews.groupby("uid")[['iid']].count()
        avgLen = sum(user_len['iid']) / len(user_len)
        print("avg bhv len is {} between {} and {}".format(avgLen,start_ts,end_ts))
        print("min timestamp is {} max is {} ".format(min(reviews['ts']), max(reviews['ts'])))


        orgreviews=reviews
        for i in tqdm(range(100)):
            # reviews = downsample(orgreviews, 'uid', 0.05)  # 保留0.05的用户 大概是1864名用户
            reviews=orgreviews
            k_core=40
            reviews = filter_k_core(reviews,2, 'iid', 'uid')
            reviews = filter_k_core(reviews, k_core, 'uid', 'iid')
            print("user num is {}".format(len(reviews['uid'].unique())))
            reviews = downsample(reviews, 'uid', 0.05)  # 保留0.05的用户 大概是1864名用户
            reviews = filter_k_core(reviews, k_core, 'uid', 'iid')
            print("user num is {}".format(len(reviews['uid'].unique())))
            user_len = reviews.groupby("uid")[['iid']].count()
            len_freq = defaultdict(int)
            for _, user_len_df in reviews.groupby("uid"):
                len_freq[user_len_df['iid'].count()] += 1
            print(len_freq)
            len_freq = sorted(len_freq.items())
            print(len_freq)
            try:
                avgLen = sum(user_len['iid']) / len(user_len)
            except:
                continue
            # userNum = len(reviews['uid'].unique())
            # itemNum = len(reviews['iid'].unique())
            # interactionNum = len(reviews)
            # print("userNum={}; itemNum={}".format(userNum, itemNum))
            # print("interaction Num={}".format(interactionNum))
            # print("data sparsity={}".format(interactionNum / (userNum * itemNum)))
            # print("avg bhv len is {} between after k_core\n\n".format(avgLen))
            maxTime, minTime = max(reviews['ts']), min(reviews['ts'])
            time_seq=sorted(reviews['ts'])
            for timeCol in range(1):
                test_interval=3*3600*timeCol
                # train_valid_split_time=maxTime-2*test_interval
                # valid_test_split_time=maxTime-test_interval
                train_valid_split_time=time_seq[int(len(time_seq)*0.7)]
                valid_test_split_time=time_seq[int(len(time_seq)*0.9)]

                trainDF=reviews[reviews['ts']<train_valid_split_time]
                validDF=reviews[(reviews['ts']>=train_valid_split_time) & (reviews['ts']<valid_test_split_time)]
                testDF=reviews[reviews['ts']>=valid_test_split_time]
                trainUser=set(trainDF['iid'].unique())
                validUser=set(validDF['iid'].unique())
                testUser=set(testDF['iid'].unique())
                train_valid_item_num=len(trainUser&(validUser))
                train_test_item_num = len(trainUser & (testUser))
                if train_test_item_num/len(testUser)<0.89 and len(trainDF)>2.5*len(validDF) and len(trainDF)<=20*len(validDF):
                    print("time control test_interval: ",test_interval)
                    reviews.to_csv(os.path.join(dataPath,"taobao_mutation.csv"), sep='\t', header=False, index=False)
                    print(len(trainUser), len(validUser), len(testUser),len(reviews))
                    print(len(trainDF),len(validDF),len(testDF))
                    print("succ train item  && with validItem num is {}".format(train_valid_item_num))
                    print("succ train item  && with testItem num is {}".format(train_test_item_num))
                else:
                    print(len(trainDF),len(validDF),len(testDF))
                    print(len(trainUser), len(validUser), len(testUser),len(reviews))
                    print("fail train item  && with testItem ratio is {}".format(train_test_item_num/len(testUser)))


        user_len = reviews.groupby("uid")[['iid']].count()
        avgLen = sum(user_len['iid']) / len(user_len)
        userNum = len(reviews['uid'].unique())
        itemNum = len(reviews['iid'].unique())
        interactionNum = len(reviews)
        print("userNum={}; itemNum={}".format(userNum, itemNum))
        print("interaction Num={}".format(interactionNum))
        print("data sparsity={}".format(interactionNum / (userNum * itemNum)))
        print("avg bhv len is {} between after downsample\n\n".format(avgLen))

        

        business = reviews[['iid', 'category']].drop_duplicates()
        reviews = reviews[['uid', 'iid', 'ts']]

        review_data_output = os.path.join(dataPath, "instance_output")  # [label,userName,itemName,timestamp,cate]
        # meta_data_output = os.path.join(dataPath, "meta_CDs_and_Vinyl.json_output")  # [itemName,cate]
        reviewData = []
        users, items = [], []
        userItems = defaultdict(list)
        userTimes= defaultdict(list)
        cateFreq=defaultdict(int)
        with open(review_data_output, "r") as reviewDataFile:
            lines = reviewDataFile.readlines()
            for line in lines:
                label, userName, itemName, timestamp, cate = line.strip().split('\t')
                cateFreq[cate]+=1
                users.append(userName)
                items.append(itemName)
                userItems[userName].append(itemName)
                userTimes[userName].append(timestamp)
        sumLen = 0
        for _, histbhv in userItems.items():
            sumLen += len(histbhv)
        userNum = len(set(users))
        itemNum = len(set(items))
        interactionNum = len(users)

        sumDiff=0
        cnt=0
        for _,time in userTimes.items():
            time=sorted(time)
            for i,j in zip(time[:-1],time[1:]):
                time_diff=float(j)-float(i)
                sumDiff+=time_diff
                cnt+=1
        print("userNum={}; itemNum={}".format(userNum, itemNum))
        print("interaction Num={}".format(interactionNum))
        print("data sparsity={}".format(interactionNum / (userNum * itemNum)))
        print("Avg. len={}".format(sumLen / userNum))
        print("avg time diff={}".format(sumDiff/cnt))

        sorted_cateFreqByFreq = sorted(cateFreq.items(), key=lambda item: -item[1])
        sum_freq = sum(cateFreq.values())
        sorted_cateFreqByFreq = [(c, round(f / sum_freq, 3)) for c, f in sorted_cateFreqByFreq]
        print(sorted_cateFreqByFreq)
        print(len(sorted_cateFreqByFreq))
        print([f for c, f in sorted_cateFreqByFreq])
    elif  dataType=="kuaishou":
        def _create_item2cate(instance_file):
            global item2cate
            instance_df = pd.read_csv(
                instance_file,
                sep="\t",
                names=["label", "user_id", "item_id", "timestamp", "cate_id"],
            )
            instance_df['item_id'] = instance_df['item_id'].astype(str)
            instance_df['cate_id'] = instance_df['cate_id'].astype(str)

            item2cate = instance_df.set_index("item_id")["cate_id"].to_dict()
            return item2cate

        reviewFile = os.path.join(dataPath, "./kuaishou_big_matrix.csv")  # [label,userName,itemName,timestamp,cate]
        metaFile=os.path.join(dataPath, "KuaiRec/data/item_feat.csv")  # [video_id,feat]
        instance_file=os.path.join(dataPath, "instance_output")

        kuaishouDF=pd.read_csv(reviewFile)
        kuaishouDF=kuaishouDF[~kuaishouDF['timestamp'].isna()]
        kuaishouDF['watch_ratio']=kuaishouDF['watch_ratio'].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        kuaishouDF['date'] = kuaishouDF['date'].astype(float)
        avgWatchRatio=sum(kuaishouDF['watch_ratio'])/len(kuaishouDF)
        print(avgWatchRatio)
        sheld=1
        disLike=kuaishouDF[kuaishouDF['watch_ratio']<sheld]
        like=kuaishouDF[kuaishouDF['watch_ratio']>=sheld]
        print("dislike/like is {}/{}".format(len(disLike),len(like)))
        kuaishouDF = like
        user_len=kuaishouDF.groupby("user_id")[['video_id']].count()
        avg=sum(user_len['video_id'])/len(user_len)
        print(avg)
        print("min timestamp is {} max is {} ".format(min(kuaishouDF['date']),max(kuaishouDF['date'])))

        #split data by date.  min timestamp is 20200705.0 max is 20200905.0
        # tbegin=20200825
        # tend=20200905
        # kuaishouDF=kuaishouDF[(kuaishouDF['date'] > tbegin)&( kuaishouDF['date'] < tend)]
        user_len=kuaishouDF.groupby("user_id")[['video_id']].count()
        avg = sum(user_len['video_id']) / len(user_len)
        print("the avg len of data after split is {} ".format(avg))
        userNum=len(kuaishouDF['user_id'].unique())
        itemNum=len(kuaishouDF['video_id'].unique())
        print("the user/item num of data after split is {}/{}".format(userNum,itemNum) )
        print("the sparsity of data after split is {}".format(len(kuaishouDF)/userNum/itemNum))


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
        k_core = 10
        reviews=kuaishouDF
        reviews = filter_k_core(reviews, k_core, 'user_id', 'video_id')
        reviews = filter_k_core(reviews, k_core, 'video_id', 'user_id')
        reviews = reviews[['user_id', 'video_id', 'timestamp']]
        reviews_timeseq=reviews.groupby('user_id')
        
        maxTime, minTime = max(reviews['timestamp']), min(reviews['timestamp'])
        time_seq=sorted(reviews['timestamp'])
        train_valid_split_time=time_seq[int(len(time_seq)*0.8)]
        valid_test_split_time=time_seq[int(len(time_seq)*0.9)]
        trainDF=reviews[reviews['timestamp']<train_valid_split_time]
        validDF=reviews[(reviews['timestamp']>=train_valid_split_time) & (reviews['timestamp']<valid_test_split_time)]
        testDF=reviews[reviews['timestamp']>=valid_test_split_time]
        trainUser=set(trainDF['video_id'].unique())   
        validUser=set(validDF['video_id'].unique())
        testUser=set(testDF['video_id'].unique())
        print(len(trainDF),len(validDF),len(testDF))
        print(len(trainUser),len(validUser),len(testUser))
        print("gru4rec train item  && with validItem num is {}".format(len(trainUser&(validUser))))
        print("gru4rec train item  && with testItem num is {}".format(len(trainUser&(testUser))))

        time_diff_sum=0
        time_diff_cnt=0
        for uid , uid_df in reviews_timeseq:
            time_seq=sorted(uid_df['timestamp'])
            for ti,tj in zip(time_seq[:-1],time_seq[1:]):
                time_diff_sum+=tj-ti
            time_diff_cnt+=len(time_seq)
        print("user bhv time diff is {}".format(time_diff_sum/time_diff_cnt))
        # kuaishouDF = downsample(kuaishouDF, 'user_id', 0.05)
        # userNum = len(kuaishouDF['user_id'].unique())
        # itemNum = len(kuaishouDF['video_id'].unique())
        # print("the user/item num of data after downsample is {}/{}".format(userNum, itemNum))
        # print("the sparsity of data after downsample is {}".format(userNum / itemNum))
        itemCate=pd.read_csv(metaFile)
        itemCate['feat']=itemCate['feat'].astype(str)
        cateFreq = defaultdict(int)
        for cate in itemCate['feat'].astype(str):
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
        print(itemCate)

        sorted_cateFreqByFreq=sorted(cateFreq.items(), key=lambda item: -item[1])
        sum_freq=sum(cateFreq.values())
        sorted_cateFreqByFreq=[(c,round(f/sum_freq,3))for c,f in sorted_cateFreqByFreq]
        print(sorted_cateFreqByFreq)
        print(len(sorted_cateFreqByFreq))
        print([f for c,f in sorted_cateFreqByFreq])
    elif dataType == "ml":
        reviewFile = os.path.join(dataPath, "movie.csv")  # [label,userName,itemName,timestamp,cate]
        metaFile = os.path.join(dataPath, "movie_meta.csv")  # [video_id,feat]
        instance_file = os.path.join(dataPath, "instance_output")

        reviews = pd.read_csv(reviewFile, sep="::", dtype=str, header=None, names=['uid', 'iid', 'rate', 'timestamp'])
        reviews = reviews[~reviews['timestamp'].isna()]
        reviews['rate'] = reviews['rate'].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        reviews['timestamp'] = reviews['timestamp'].astype(float)
        #统计分数分布
        rate_disb=reviews.groupby('rate')
        rate_freq=dict()
        for rate, rate_df in rate_disb:

            rate_freq[rate]=len(rate_df)
        print("ml 分数分布： \n",rate_freq)
        likeSheld=1.8
        reviews = reviews[reviews['rate'] > likeSheld]
        reviews = reviews.drop_duplicates(subset=['uid', 'iid'])
        # start_ts =20200825# best 202008 20
        # end_ts =20200905
        # reviews = reviews[reviews['date'] >= start_ts]
        # reviews = reviews[reviews['date'] <= end_ts]
        k_core = 10
        reviews = filter_k_core(reviews, k_core, 'uid', 'iid')
        reviews = filter_k_core(reviews, k_core, 'iid', 'uid')
        reviews = reviews[['uid', 'iid', 'timestamp']]

        itemCate = pd.read_csv(metaFile, sep="::", dtype=str, header=None,
                               names=['iid', 'Title', 'Genres'],encoding='ISO-8859-1').drop_duplicates()
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
        itemCate['Genres'] = itemCate['Genres'].map(cateList2singleCate)

        user_len = reviews.groupby("uid")[['iid']].count()
        avg = sum(user_len['iid']) / len(user_len)
        print(avg)
        print("min timestamp is {} max is {} ".format(min(reviews['timestamp']), max(reviews['timestamp'])))

        # split data by date.  min timestamp is 956703932.0 max is 1046454590.0
        tbegin = max(reviews['timestamp'])- 3600 * 24 * 365 * 2.5
        tend = max(reviews['timestamp'])#1046454590
        mlDF = reviews[(reviews['timestamp'] > tbegin) & (reviews['timestamp'] < tend)]
        user_len = mlDF.groupby("uid")[['iid']].count()
        avg = sum(user_len['iid']) / len(user_len)
        print("the avg len of data after split is {} ".format(avg))
        len_freq=defaultdict(int)
        for _,user_len_df in mlDF.groupby("uid"):
            len_freq[user_len_df['iid'].count()]+=1
        print(len_freq)
        len_freq=sorted(len_freq.items())
        print(len_freq)
        userNum = len(mlDF['uid'].unique())
        itemNum = len(mlDF['iid'].unique())
        print("the user/item num of data after split is {}/{}".format(userNum, itemNum))
        print("the sparsity of data after split is {}".format(len(mlDF) / userNum / itemNum))


        reviews_timeseq = reviews.groupby('uid')
        maxTime, minTime = max(reviews['timestamp']), min(reviews['timestamp'])
        time_seq = sorted(reviews['timestamp'])
        train_valid_split_time = time_seq[int(len(time_seq) * 0.8)]
        valid_test_split_time = time_seq[int(len(time_seq) * 0.9)]
        trainDF = reviews[reviews['timestamp'] < train_valid_split_time]
        validDF = reviews[
            (reviews['timestamp'] >= train_valid_split_time) & (reviews['timestamp'] < valid_test_split_time)]
        testDF = reviews[reviews['timestamp'] >= valid_test_split_time]
        trainUser = set(trainDF['iid'].unique())
        validUser = set(validDF['iid'].unique())
        testUser = set(testDF['iid'].unique())
        print(len(trainDF), len(validDF), len(testDF))
        print(len(trainUser), len(validUser), len(testUser))
        print("gru4rec train item  && with validItem num is {}".format(len(trainUser & (validUser))))
        print("gru4rec train item  && with testItem num is {}".format(len(trainUser & (testUser))))

        time_diff_sum = 0
        time_diff_cnt = 0
        for uid, uid_df in reviews_timeseq:
            time_seq = sorted(uid_df['timestamp'])
            for ti, tj in zip(time_seq[:-1], time_seq[1:]):
                time_diff_sum += tj - ti
            time_diff_cnt += len(time_seq)
        print("user bhv time diff is {}".format(time_diff_sum / time_diff_cnt))
    elif  dataType=="yelp":
        reviewFile = os.path.join(dataPath, "yelp_academic_dataset_review.csv")  # [label,userName,itemName,timestamp,cate]
        if not os.path.exists(reviewFile):
            reviews=[]
            with open(os.path.join(dataPath,"yelp_academic_dataset_review.json"),'r',encoding='utf-8') as file:
                for line in file:
                    line_new = eval(line.strip(), {"true": True, "false": False, "null": None})
                    reviews.append([str(line_new["user_id"]),
                                    str(line_new["business_id"]),
                                    str(line_new["stars"]),
                                    datetime.timestamp(
                                        datetime.strptime(str(line_new["date"]), "%Y-%m-%d %H:%M:%S"))
                                    ])
            reviewsDF=pd.DataFrame(reviews,columns=['user_id','item_id','stars','timestamp'])
            reviewsDF.to_csv(reviewFile,index=False)
        else:
            reviewsDF=pd.read_csv(reviewFile)
        reviewsDF['stars']=reviewsDF['stars'].astype('float')
        instance_df = pd.read_csv(
            os.path.join(dataPath, "instance_output"),
            sep="\t",
            names=["label", "user_id", "item_id", "timestamp", "cate_id"],
        )
        instance_df['item_id'] = instance_df['item_id'].astype(str)
        instance_df['cate_id'] = instance_df['cate_id'].astype(str)
        item2cate = instance_df.set_index("item_id")["cate_id"].to_dict()

        avgStars=sum(reviewsDF['stars'])/len(reviewsDF)
        print("avgStars={}".format(avgStars))
        sheld=2.9
        disLike=reviewsDF[reviewsDF['stars']<sheld]
        like=reviewsDF[reviewsDF['stars']>=sheld]
        print("dislike/like is {}/{}".format(len(disLike),len(like)))

        kuaishouDF =instance_df
        user_len=kuaishouDF.groupby("user_id")[['item_id']].count()
        avgLen=sum(user_len['item_id'])/len(user_len)
        print("avg bhv len is {}".format(avgLen))
        print("min timestamp is {} max is {} ".format(min(kuaishouDF['timestamp']),max(kuaishouDF['timestamp'])))

        #split data by date.  min timestamp is 1097547212.0 max is 1576223479.0
        tbegin=1376223149# 2019-1-1
        tend=1409850579# 2019-12-13 15:51:19
        '''
        
        [1109850579,1576223149] avglen=25
        [1109850579,1209850579] avglen=10
        [1209850579,1309850579] avglen=11
        [1309850579,1409850579] avglen=11
        
        [1409850579,1576223149] avglen=19
        [1459850579,1576223149] avglen=15
        [1409850579,1459850579] avglen=7
        
        [1109850579,1376223149] avglen=14
        [1376223149,1409850579] avglen=21
        '''
        # import  numpy as np
        # besti_j_avglen=[]
        # i_j_avglen = []
        # # for i in np.linspace(1109850579,1576223149,num=5):
        # #     for j in np.linspace(1109850579, 1576223149, num=5):
        # #         if i<j:
        # #             tmpDF=kuaishouDF[(kuaishouDF['timestamp'] > i)&( kuaishouDF['timestamp'] < j)]
        # #             user_len=tmpDF.groupby("user_id")[['item_id']].count()
        # #             avg = sum(user_len['item_id']) / len(user_len)
        # #             if avg>=25:
        # #                 print("the avg len of data after split is {} ".format(avg))
        # #                 besti_j_avglen.append([i,j,avg])
        # #             else:
        # #                 i_j_avglen.append([i,j,avg])
        # print("best begin end time ={}".format(besti_j_avglen))
        # print("normal begin end time={}".format(i_j_avglen))
        kuaishouDF = kuaishouDF[(kuaishouDF['timestamp'] >1226443721) & (kuaishouDF['timestamp'] < 1576223149)]
        userNum=len(kuaishouDF['user_id'].unique())
        itemNum=len(kuaishouDF['item_id'].unique())
        print("the user/item num of data after split is {}/{}".format(userNum,itemNum))
        print("the sparsity of data after split is {}".format(len(kuaishouDF)/userNum/itemNum))
        user_len = kuaishouDF.groupby("user_id")[['item_id']].count()
        avgLen = sum(user_len['item_id']) / len(user_len)
        print("avg bhv len is {}".format(avgLen))

        def downsample(record, col, frac):
            sample_col = record[col].drop_duplicates().sample(frac=frac)
            record = record.merge(sample_col, on=col).reset_index(drop=True)
            return record


        yelpDF=kuaishouDF
        yelpDF = downsample(yelpDF, 'user_id',1.0)
        userNum = len(yelpDF['user_id'].unique())
        itemNum = len(yelpDF['item_id'].unique())
        print("the user/item num of data after downsample is {}/{}".format(userNum, itemNum))
        print("interaction num is {}".format(len(yelpDF)))
        print("the sparsity of data after downsample is {}".format(len(yelpDF)/userNum/itemNum))
        user_len = yelpDF.groupby("user_id")[['item_id']].count()
        avgLen = sum(user_len['item_id']) / len(user_len)
        print("avg bhv len after downsample is {}".format(avgLen))
        reviews=yelpDF
        k_core = 10
        reviews = filter_k_core(reviews, k_core, 'item_id', 'user_id')
        reviews = filter_k_core(reviews, k_core, 'user_id', 'item_id')
        yelpDF=reviews
        userNum = len(yelpDF['user_id'].unique())
        itemNum = len(yelpDF['item_id'].unique())
        print("the user/item num of data after filter_k_core is {}/{}".format(userNum, itemNum))
        print("interaction num is {}".format(len(yelpDF)))
        print("the sparsity of data after filter_k_core is {}".format(len(yelpDF) / userNum / itemNum))
        user_len = yelpDF.groupby("user_id")[['item_id']].count()
        avgLen = sum(user_len['item_id']) / len(user_len)
        print("avg bhv len after filter_k_core is {}".format(avgLen))


        cateFreq = defaultdict(int)
        for c in item2cate.values():
            cateFreq[c] += 1

        sorted_cateFreqByFreq=sorted(cateFreq.items(), key=lambda item: -item[1])
        sum_freq=sum(cateFreq.values())
        sorted_cateFreqByFreq=[(c,round(f/sum_freq,3))for c,f in sorted_cateFreqByFreq]
        print(sorted_cateFreqByFreq)
        print(len(sorted_cateFreqByFreq))
        print([f for c,f in sorted_cateFreqByFreq])


    elif dataType == "coverage":
        import numpy as np

        N = 2180#类别数目
        M = 100#推荐列表大小
        probs_tabao=[0.085, 0.054, 0.048, 0.046, 0.046, 0.027, 0.024, 0.023, 0.021, 0.02, 0.018, 0.015, 0.014, 0.014, 0.013, 0.012, 0.01, 0.01, 0.01, 0.009, 0.009, 0.009, 0.009, 0.009, 0.008, 0.008, 0.007, 0.007, 0.007, 0.006, 0.006, 0.005, 0.005, 0.005, 0.005, 0.005, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]

        # probs = [0.16, 0.107, 0.088, 0.076, 0.072, 0.055, 0.054, 0.042, 0.04, 0.039, 0.037, 0.036, 0.029, 0.028, 0.021,
        #          0.019, 0.017, 0.017, 0.013, 0.013, 0.01, 0.008, 0.005, 0.003, 0.002, 0.002, 0.002, 0.002, 0.002, 0.001]
        # 将概率分布转化为numpy数组
        probs=probs_tabao
        probs = np.array(probs)
        # 计算不同颜色数的期望
        exp_M = np.sum([1-np.power(1-pi, M) for pi in probs])
        print(f"期望值为：{exp_M:.2f}")
        print("理论随机覆盖率为:{}".format(exp_M/M))#0.52 0.41
        # exp_M = np.sum([1 - np.power(1 - np.exp(), M) for i in range(len(probs))])
        #概率序列的幂律拟合
        probs_sim=[0.16*np.power(i+1,-0.55) for i in range(len(probs))]
        print("概率序列的拟合为:{}".format(probs_sim))

        exp_M = np.sum([1-np.power(1-pi, M) for pi in probs_sim])
        print(f"指数拟合概率的近似期望值为：{exp_M:.2f}")

