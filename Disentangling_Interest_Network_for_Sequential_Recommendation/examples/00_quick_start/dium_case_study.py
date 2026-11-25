from collections import defaultdict

import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import pandas as pd

import random

from tqdm import tqdm


def userTimeDecayFactorAna(userTimeDecayFactor):
    ax = plt.subplot(111)  # 1行1列的一张子图，即只有一张图。三个1分别表示1行、1列、第1个
    width = 10  # 宽度为10个单位
    hight = 3  # 高度为三个单位
    for factor in userTimeDecayFactor:
        # if factor<0:continue
        x = np.linspace(0, 30, 30)
        print(factor)
        y = np.exp(-x * factor)
        ax.plot(x, y, 'rD-')  # 对y1的数据作图，b表示蓝色，D表示菱形，-表示实直线
    # ax.plot(x, y2, 'r^-')  # 对y2的数据作图，r表示红色，^表示三角形，--表示破折线
    plt.show()
def randomcolor1():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color
def randomcolor(num=5):
    beauty_colors=["#9062DF","#EC59C0","#FF6B95","#FF966E","#FFC95B","#F9F871",
            "#64BAA9","#C4FCEF","#3CBEA9","#F234EB","#FF2CB0","#FF657B",
            "#FF9E56","#FFCF4F","#F9F871"]
    if num>len(beauty_colors):
        beauty_colors.extend([randomcolor1() for i in range(7+num-len(beauty_colors))])
    return np.random.choice(beauty_colors,replace=False,size=num)
def stackGroupBarplot(interestSemanticData=None):
    #interestSemanticData [T,Channel,K,2]==>[T,Channel,CNum]
    global bottomData, data
    if interestSemanticData is None:
        T,Channel,CNum=(5,4,7)
        Data=np.random.uniform(0,1,(T,Channel,CNum))
        Data=Data/np.sum(Data,axis=-1,keepdims=True)
        print(Data)
    else:
        T, Channel, CNum =interestSemanticData.shape
        Data=interestSemanticData
    colorOfC=randomcolor(num=CNum)
    fig, ax = plt.subplots()
    x_labels = ["t-{}".format(t) for t in range(T)]

    width=0.07
    xoffset=width*1.2
    for channel in range(Channel):
        #c=0
        bottomData=Data[:,channel,0]
        if channel==0:
            ax.bar(np.arange(T)+channel*xoffset, Data[:,channel,0], width=width, tick_label=x_labels,color=colorOfC[0],
                   label="p-{}".format(0))
        else:
            ax.bar(np.arange(T)+channel*xoffset, Data[:,channel,0], width=width, tick_label=x_labels,color=colorOfC[0])

        for c in range(1,CNum):
            data=Data[:,channel,c]
            if channel==0:
                ax.bar(np.arange(T)+channel*xoffset,data,width=width,tick_label=x_labels,
                   bottom=bottomData,color=colorOfC[c],label="p-{}".format(c))
            else:
                ax.bar(np.arange(T) + channel * xoffset, data, width=width, tick_label=x_labels,
                       bottom=bottomData, color=colorOfC[c])
            bottomData=bottomData+data
    # ax.legend()
    plt.savefig("stackGroupBarplot.jpg")
    plt.show()


def interestSenmanticClusterAna(userInterest, itemLookUp):
    userInterestL2norm = np.linalg.norm(userInterest, axis=-1, keepdims=True)  # [T,S,1]
    itemLookUpL2Norm = np.linalg.norm(itemLookUp, axis=-1, keepdims=True)  # [N,1]
    userInterest = userInterest / userInterestL2norm
    itemEmdNorm = itemLookUp / itemLookUpL2Norm

    userInterest = np.expand_dims(userInterest, axis=-2)  # [T,S,1,H]
    itemEmdNorm = np.expand_dims(np.expand_dims(itemEmdNorm, axis=0), axis=0)  # [1,1,N,H]
    simScore = np.sum(userInterest * itemEmdNorm, axis=-1)  # [T,S,N,H]=>[T,S,N]
    k=250
    timeLen = simScore.shape[0]
    channelNum = simScore.shape[1]
    # point:  id:[feaVec, clabel, simScore]
    clusterPoint = {}
    for t in [5]:
        for channel in np.random.choice(range(channelNum),replace=False,size=channelNum):
            sim = simScore[t, channel, :]
            order = np.argsort(sim)[::-1]  # 降序排序
            mins=999
            maxs=-999
            for id in order[:k]:
                mins=sim[id] if mins > sim[id] else mins
                maxs=sim[id] if sim[id]>maxs else maxs
            for id in order[:k]:
                if id not in clusterPoint:
                    clusterPoint[id] = []
                    clusterPoint[id].append(itemLookUp[id])
                    clusterPoint[id].append(channel)
                    clusterPoint[id].append((sim[id]-mins)/(maxs-mins))
                else:
                    channel, s = clusterPoint[id][1], clusterPoint[id][2]
                    if 2*s < (sim[id]-mins)/(maxs-mins):
                        clusterPoint[id][1] = channel
                        clusterPoint[id][2] = (sim[id]-mins)/(maxs-mins)
    label_scoreMax=np.zeros(shape=(channelNum,))
    label_scoreMin = np.ones(shape=(channelNum,))
    for id, feaVec_clabel_simScore in clusterPoint.items():
        clabel,simScore=feaVec_clabel_simScore[1],feaVec_clabel_simScore[2]
        if label_scoreMax[clabel]<simScore:
            label_scoreMax[clabel] = simScore
        if label_scoreMin[clabel]>simScore:
            label_scoreMin[clabel] =simScore

    ids, feaMat, clabels, simScores = [], [], [], []
    for id, feaVec_clabel_simScore in clusterPoint.items():
        ids.append(id)
        clabel, simScore = feaVec_clabel_simScore[1], feaVec_clabel_simScore[2]
        feaMat.append(feaVec_clabel_simScore[0])
        clabels.append(clabel)
        simScores.append(
                            (simScore-label_scoreMin[clabel])/
                         (label_scoreMax[clabel]-label_scoreMin[clabel])
                         )
    return feaMat, clabels, simScores

def pcaPlot(feaMat, clabels, simScores):
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2,n_iter=300).fit_transform(feaMat)
    labelNum=max(set(clabels))+1
    colors=randomcolor(labelNum)
    quart=0.15
    tsne_x,tsne_y=tsne[:,0],tsne[:,1]
    tsne_x_min,tsne_x_max=np.min(tsne_x),np.max(tsne_x)
    tsne_y_min,tsne_y_max=np.min(tsne_y),np.max(tsne_y)
    tsne_x_25Q=(tsne_x_max-tsne_x_min)*quart
    tsne_x_75Q = (tsne_x_max-tsne_x_min)*(1-quart)
    tsne_y_25Q=(tsne_y_max-tsne_y_min)*quart
    tsne_y_75Q = (tsne_y_max-tsne_y_min) * (1-quart)

    for idx,label_s in enumerate(zip(clabels,simScores)):
        label,s=label_s
        print(label_s)
        x,y=tsne[idx,0],tsne[idx,1]
        if x-tsne_x_min<tsne_x_25Q or x-tsne_x_min>tsne_x_75Q:
            continue
        if y-tsne_y_min<tsne_y_25Q or y-tsne_y_min>tsne_y_75Q:
            continue
        s=s if s<1 else 1
        s=s if s>0 else 0
        plt.scatter(x,y,c=colors[label],alpha=s)
    plt.savefig("interestSemanticClusterAna.jpg")
    plt.show()
if __name__ == '__main__':
    # testUser=[5,72,233]
    # testUser=[15795 , 860 ,6265, 37194]
    testUser = [860]
    #/home/lisong/SNA/Disentangling_Interest_Network_for_Sequential_Recommendation/examples/00_quick_start
    logPath="./expLog"
    with open(logPath + "/dicts.pkl", "rb") as dictFile:
        userDict, itemDict, cateDict=pickle.load(dictFile)
        uid2name={ id:name for name,id in userDict.items()}
        itemid2name={ id:name for name,id in itemDict.items()}
        cateid2name={id: name for name, id in cateDict.items()}

    with open(logPath+"/itemLookUp.pkl","rb") as itemLookUpFile:
        itemLookUp=pickle.load(itemLookUpFile)
    # stackGroupBarplot()
    # exit(0)

    def metaProcess(meta_readfile):
        meta_r = open(meta_readfile, "r")
        cate_freq = defaultdict(int)
        item2cate=defaultdict(list)
        for line in meta_r:
            line_new = eval(line)
            itemAsin=line_new["asin"]
            catesStrList = line_new["categories"][0]
            catesStrList = [cate.replace(",", "|") for cate in catesStrList]
            item2cate[itemAsin]=catesStrList
            for cate in catesStrList:
                cate_freq[cate] += 1
        return item2cate,cate_freq



    item2cate,cate_freq=metaProcess("/home/lisong/SNA/Disentangling_Interest_Network_for_Sequential_Recommendation/tests/resources/deeprec/slirec/meta_CDs_and_Vinyl.json")
    for uid in testUser:
        # print("=====test uid={}".format(uid))
        # with open(logPath+"/{}_expData.pkl".format(uid),"rb") as logFile:
        #     dataIdxOfUid, userInterest, userIS, userTimeToNow, userTimeDecayFactor=pickle.load(logFile)
        with open(logPath+"/{}_expData.pkl".format(uid),"rb") as logFile:
            dataIdxOfUid, userInterest, userIS=pickle.load(logFile)
        # userTimeDecayFactorAna(userTimeDecayFactor)
        # print(uid2name[uid])
        # print(len(np.exp(userTimeToNow)))
        # print(userTimeDecayFactor)#[S,]
        # print(userIS)  [T,S]
        def userISAna():
            ax = plt.subplot(111)  # 1行1列的一张子图，即只有一张图。三个1分别表示1行、1列、第1个
            width = 10  # 宽度为10个单位
            hight = 3  # 高度为三个单位
            # ax.axes.set_xlim(-0.5, width + 0.2)  # 设置横轴范围为-0.5到10.2
            # ax.axes.set_ylim(-0.5, hight + 0.2)  # 设置纵轴范围为-0.5到3.2
            for channel in range(4):
                IS=userIS[:,channel]
                ax.plot(range(len(IS)), IS, 'bD-')  # 对y1的数据作图，b表示蓝色，D表示菱形，-表示实直线
            # ax.plot(x, y2, 'r^-')  # 对y2的数据作图，r表示红色，^表示三角形，--表示破折线
            plt.title("{}".format(uid))
            plt.show()
        # userISAna()
        feaMat, clabels, simScores=interestSenmanticClusterAna(userInterest,itemLookUp)
        pcaPlot(feaMat, clabels, simScores)

        def interestSenmanticAna(userInterest,itemLookUp):
            #userInterest [T,S,H]
            #itemLookUp [N,H]
            userInterestL2norm=np.linalg.norm(userInterest,axis=-1,keepdims=True)#[T,S,1]
            itemLookUpL2Norm=np.linalg.norm(itemLookUp,axis=-1,keepdims=True)#[N,1]

            userInterest=userInterest/userInterestL2norm
            itemLookUp=itemLookUp/itemLookUpL2Norm

            userInterest=np.expand_dims(userInterest,axis=-2)#[T,S,1,H]
            itemLookUp=np.expand_dims(np.expand_dims(itemLookUp,axis=0),axis=0)#[1,1,N,H]
            simScore=np.sum(userInterest*itemLookUp,axis=-1)#[T,S,N,H]=>[T,S,N]
            k=300
            topKSimItemCate=[]
            cateSelect=defaultdict(int)
            timeLen=simScore.shape[0]
            channelNum=simScore.shape[1]
            for t in range(timeLen):
                Channel_t=[]
                for channel in range(channelNum):
                    Channel_t_c = []
                    sim=simScore[t,channel,:]
                    order=np.argsort(sim)[::-1]#降序排序
                    for i in order[:k]:
                        cates=item2cate[itemid2name[i]]
                        Channel_t_c.extend(cates)
                    Channel_t.append(Channel_t_c)
                topKSimItemCate.append(Channel_t)

            cate_t_channel_plot_score=[]
            cateSelect2DataIdx=0
            pltChannelNum=8
            for t in range(3,9):
                for channel in range(pltChannelNum):
                    channel_plot_cateData=dict()
                    itemCates=topKSimItemCate[t][channel]
                    # print("=========t={} channel={}=============".format(t, channel))
                    # print(itemCates)
                    channelCateFreq=defaultdict(int)
                    sum=0
                    for cate in itemCates:
                        channelCateFreq[cate] += 1/cate_freq[cate]
                        sum += 1 / cate_freq[cate]
                    for key in channelCateFreq.keys():
                        channelCateFreq[key]=channelCateFreq[key]/sum
                    channelCateFreq=[(c,f)for c,f in channelCateFreq.items()]
                    channelCateFreq=sorted(channelCateFreq,key=lambda x:x[1],reverse=True)
                    # print("check sunm={}".format(np.sum([s[1]  for s in channelCateFreq])))
                    sum=0
                    cnt=0
                    theseld=[1.5,200]
                    for cate,s in channelCateFreq:
                        sum+=s
                        cnt+=1
                        if cate not in cateSelect.keys():
                            cateSelect[cate] = cateSelect2DataIdx
                            cateSelect2DataIdx += 1
                        channel_plot_cateData[cateSelect[cate]]=s
                        if sum>theseld[0] or cnt>theseld[1]:
                            break

                    cate_t_channel_plot_score.append(channel_plot_cateData)
                    # print("time={},cate={},freq distrib:{}".format(t,channel,channelCateFreq))
            cateSelectNum=len(cateSelect.keys())
            interestSenmanticData=[]
            for idx,data in enumerate(cate_t_channel_plot_score):
                channel_t_c_data=np.zeros(shape=cateSelectNum)
                for dataIdx,score in data.items():
                    channel_t_c_data[dataIdx]=score
                interestSenmanticData.append(channel_t_c_data)
            # print(cate_t_channel_plot_score[3:10,0,:])
            interestSenmanticData=np.asarray(interestSenmanticData,dtype=np.float32).reshape((-1,pltChannelNum,cateSelectNum))
            return interestSenmanticData#[T,Channel,CNum]
        # interestSenmanticData=interestSenmanticAna(userInterest,itemLookUp)
        # stackGroupBarplot(interestSenmanticData)
        # userTimeDecayFactorAna()
