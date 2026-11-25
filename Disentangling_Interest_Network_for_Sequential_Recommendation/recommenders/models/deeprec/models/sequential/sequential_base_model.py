# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import cosine_similarity
import os
import abc
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from recommenders.models.deeprec.models.base_model import BaseModel
from recommenders.models.deeprec.deeprec_utils import cal_metric, load_dict


__all__ = ["SequentialBaseModel"]


class SequentialBaseModel(BaseModel):
    """Base class for sequential models"""

    def __init__(self, hparams, iterator_creator, graph=None, seed=None):
        """Initializing the model. Create common logics which are needed by all sequential models, such as loss function,
        parameter set.

        Args:
            hparams (HParams): A `HParams` object, hold the entire set of hyperparameters.
            iterator_creator (object): An iterator to load the data.
            graph (object): An optional graph.
            seed (int): Random seed.
        """
        self.hparams = hparams

        self.need_sample = hparams.need_sample
        self.train_num_ngs = hparams.train_num_ngs
        if self.train_num_ngs is None:
            raise ValueError(
                "Please confirm the number of negative samples for each positive instance."
            )
        self.min_seq_length = (
            hparams.min_seq_length if "min_seq_length" in hparams.values() else 1
        )
        self.hidden_size = (
            hparams.hidden_size if "hidden_size" in hparams.values() else None
        )
        self.graph = tf.Graph() if not graph else graph
        if "itemidx2cat" in self.hparams.values():
            itemidx2cat=load_dict(self.hparams.itemidx2cat)
            self.itemid2cat=np.zeros(shape=(len(itemidx2cat.keys()))).astype(np.str_)
            for itemidx,cat in itemidx2cat.items():
                self.itemid2cat[itemidx]=cat
        with self.graph.as_default():
            self.sequence_length = tf.compat.v1.placeholder(
                tf.int32, [None], name="sequence_length"
            )

        super().__init__(hparams, iterator_creator, graph=self.graph, seed=seed)

    @abc.abstractmethod
    def _build_seq_graph(self):
        """Subclass will implement this."""
        pass

    def _build_graph(self):
        """The main function to create sequential models.

        Returns:
            object: the prediction score make by the model.
        """
        hparams = self.hparams
        self.keep_prob_train = 1 - np.array(hparams.dropout)
        self.keep_prob_test = np.ones_like(hparams.dropout)

        with tf.compat.v1.variable_scope("sequential") as self.sequential_scope:
            self._build_embedding()
            self._lookup_from_embedding()
            model_output = self._build_seq_graph()
            logit = self._fcn_net(model_output, hparams.layer_sizes, scope="logit_fcn")
            self._add_norm()
            return logit

    def fit(
        self,
        train_file,
        valid_file,
        valid_num_ngs,
        eval_metric="group_auc",
    ):
        """Fit the model with `train_file`. Evaluate the model on `valid_file` per epoch to observe the training status.
        If `test_file` is not None, evaluate it too.

        Args:
            train_file (str): training data set.
            valid_file (str): validation set.
            valid_num_ngs (int): the number of negative instances with one positive instance in validation data.
            eval_metric (str): the metric that control early stopping. e.g. "auc", "group_auc", etc.

        Returns:
            object: An instance of self.
        """

        # check bad input.
        if not self.need_sample and self.train_num_ngs < 1:
            raise ValueError(
                "Please specify a positive integer of negative numbers for training without sampling needed."
            )
        if valid_num_ngs < 1:
            raise ValueError(
                "Please specify a positive integer of negative numbers for validation."
            )

        if self.need_sample and self.train_num_ngs < 1:
            self.train_num_ngs = 1

        if self.hparams.write_tfevents and self.hparams.SUMMARIES_DIR:
            if not os.path.exists(self.hparams.SUMMARIES_DIR):
                os.makedirs(self.hparams.SUMMARIES_DIR)

            self.writer = tf.compat.v1.summary.FileWriter(
                self.hparams.SUMMARIES_DIR, self.sess.graph
            )

        train_sess = self.sess
        eval_info = list()

        best_metric, self.best_epoch = 0, 0

        step = 0
        needTimeLine=False
        for epoch in range(1, self.hparams.epochs + 1):
            self.hparams.current_epoch = epoch
            epoch_loss = 0

            file_iterator = self.iterator.load_data_from_file(
                train_file,
                min_seq_length=self.min_seq_length,
                batch_num_ngs=self.train_num_ngs,
            )
            if "needTimeLineEpoch"  in self.hparams.values() and epoch==self.hparams.needTimeLineEpoch:
                needTimeLine=True

            for batch_data_input in tqdm(file_iterator):
                if batch_data_input:
                    step_result = self.train(train_sess, batch_data_input,needTimeLine=needTimeLine)
                    if needTimeLine:needTimeLine=False
                    (_, _, step_loss, step_data_loss, summary) = step_result
                    if self.hparams.write_tfevents and self.hparams.SUMMARIES_DIR:
                        self.writer.add_summary(summary, step)
                    epoch_loss += step_loss
                    step += 1
                    if step % self.hparams.show_step == 0:
                        print(
                            "step {0:d} , total_loss: {1:.4f}, data_loss: {2:.4f}".format(
                                step, step_loss, step_data_loss
                            )
                        )
            if self.hparams.save_model and self.hparams.MODEL_DIR:
                if not os.path.exists(self.hparams.MODEL_DIR):
                    os.makedirs(self.hparams.MODEL_DIR)
                checkpoint_path = self.saver.save(
                    sess=train_sess,
                    save_path=self.hparams.MODEL_DIR + "epoch_" + str(epoch),
                )

            valid_res = self.run_eval(valid_file, valid_num_ngs)
            print(
                "eval valid at epoch {0}: {1}".format(
                    epoch,
                    ",".join(
                        [
                            "" + str(key) + ":" + str(value)
                            for key, value in valid_res.items()
                        ]
                    ),
                )
            )
            eval_info.append((epoch, valid_res))

            progress = False
            early_stop = self.hparams.EARLY_STOP
            if valid_res[eval_metric] > best_metric:
                best_metric = valid_res[eval_metric]
                self.best_epoch = epoch
                progress = True
            else:
                if early_stop > 0 and epoch - self.best_epoch >= early_stop:
                    print("early stop at epoch {0}!".format(epoch))
                    break

            if self.hparams.save_model and self.hparams.MODEL_DIR:
                if not os.path.exists(self.hparams.MODEL_DIR):
                    os.makedirs(self.hparams.MODEL_DIR)
                if progress:
                    checkpoint_path = self.saver.save(  # noqa: F841
                        sess=train_sess,
                        save_path=os.path.join(self.hparams.MODEL_DIR, "best_model"),
                    )

        if self.hparams.write_tfevents:
            self.writer.close()

        print(eval_info)
        print("best epoch: {0}".format(self.best_epoch))
        return self
    def cal_metrics_basedOn_RecList(self,recItemListScore,posItem,histItems,item_lookup,metircs):
        res={}
        for metric in metircs:
            if metric.startswith("coverage"):
                coverage_list = [1, 2]
                ks = metric.split("@")
                if len(ks) > 1:
                    coverage_list = [int(token) for token in ks[1].split(";")]
                for k in coverage_list:

                    coverage_temp = np.mean(
                        [
                            self.Coverage(each_recItemListScore,each_histItems,item_lookup,k)
                            for each_recItemListScore,each_histItems in zip(recItemListScore,histItems)
                        ]
                    )
                    res["coverage@{0}".format(k)] = round(coverage_temp, 4)
            if metric.startswith("ILD"):
                ILD_list = [1, 2]
                ks = metric.split("@")
                if len(ks) > 1:
                    ILD_list = [int(token) for token in ks[1].split(";")]
                for k in ILD_list:
                    ILD_temp = np.mean(
                        [
                            self.ILD_score(each_recItemListScore,each_histItems,item_lookup, k)
                            for each_recItemListScore,each_histItems in zip(recItemListScore,histItems)
                        ]
                    )
                    res["ILD@{0}".format(k)] = round(ILD_temp, 4)
        return res
    def topKRec(self,recItemListScore,histItems,item_lookup,kVal):
        # disType:[disType="Euclidean","cosSim]
        k = min(np.shape(recItemListScore)[-1],kVal)
        dropHistItem = np.array(list(set(range(self.item_lookup.shape[0])) - set(histItems)))
        dropHistItemScore = recItemListScore[dropHistItem]
        order = np.argsort(dropHistItemScore)[::-1]  # 降序排序
        recItemListScore_top_k = np.take(dropHistItemScore, order[:k])
        # map2 org id
        itemLookUpIdxs_top_k = dropHistItem[order[:k]]
        itemEmd_top_k = item_lookup[itemLookUpIdxs_top_k]
        return itemLookUpIdxs_top_k,itemEmd_top_k
    def Coverage(self,recItemListScore,histItems,item_lookup,kVal):
        itemLookUpIdxs_top_k , _ =self.topKRec(recItemListScore,histItems,item_lookup,kVal)
        catesInRecList=self.itemid2cat[itemLookUpIdxs_top_k]
        # print("=====coveage: ====")
        # print(catesInRecList)
        # print(self.itemid2cat)
        catesInRecList=set(catesInRecList)
        # print(len(catesInRecList))
        # print(len(self.itemid2cat))
        cov=len(catesInRecList)/kVal
        return cov
    def ILD_score(self,recItemListScore,histItems,item_lookup,kVal):
        #[K,H]
        #cal intra-list dis
        #D_{i,j}=(item_{i}-item_{j})^2=item_{i}^2-2*item_{i}*item_{j}+item_{j}^2
        _ , itemEmd_top_k =self.topKRec(recItemListScore,histItems,item_lookup,kVal)
        ILD_use_scale="stdEul"#[init,meanDiv,z_score,normEul]

        if ILD_use_scale=="stdEul":
            miu_x=np.mean(item_lookup,axis=0)#[voc,H]=>[H,]
            std_x=np.std(item_lookup,axis=0,ddof=1)
            itemEmd_at_k=(itemEmd_top_k-miu_x)/std_x
            distances = pdist(itemEmd_at_k, metric='euclidean')
            # print("=====ILD_use_scale:stdEul ====")
            # print(len(distances))
            ILD = np.sum(distances)/len(distances)
            return ILD
        dis = 1-cosine_similarity(itemEmd_top_k, itemEmd_top_k)#[K,K]
        if ILD_use_scale=="init" or ILD_use_scale == "meanDiv":
            diag_mask=np.diag(np.diagonal(dis))
            ILD=np.sum(dis-diag_mask)*2/(kVal*(kVal-1))#initial
            if ILD_use_scale=="init": return ILD

        disPairInAllItem=1-cosine_similarity(item_lookup,item_lookup)#mean dis   dis[i,j] in all item set
        disPairInAllItem=disPairInAllItem-np.diag(np.diagonal(disPairInAllItem))#[K,K]
        miu=np.sum(disPairInAllItem)*2/(item_lookup.shape[0]*(item_lookup.shape[0]-1))#mean dis

        if ILD_use_scale == "meanDiv":
            return ILD/miu
        if ILD_use_scale=="z_score":
            sigma_sq=np.sum(  np.square(disPairInAllItem-miu)  *2/(item_lookup.shape[0]*(item_lookup.shape[0]-1)))
            sigma=np.sqrt(sigma_sq)
            dis=(dis-miu)/sigma
            diag_mask = np.diag(np.diagonal(dis))
            ILD = np.sum(dis - diag_mask) * 2 / (kVal * (kVal - 1))
            return ILD


    def run_eval(self, filename, num_ngs):
        """Evaluate the given file and returns some evaluation metrics.

        Args:
            filename (str): A file name that will be evaluated.
            num_ngs (int): The number of negative sampling for a positive instance.

        Returns:
            dict: A dictionary that contains evaluation metrics.
        """

        global item_lookup, cateIdOfAllItems_embedding
        load_sess = self.sess
        preds = []
        labels = []
        group_preds = []
        group_labels = []
        recItemListScores = []
        posItems, histItems = [], []

        short_mid_long_interval=(2.28,3.85)
        short_mid_long_interval_group=["short","mid","long"]
        short_mid_long_preds= [[],[],[]]
        short_mid_long_labels = [[],[],[]]
        short_mid_long_group_preds= [[], [], []]
        short_mid_long_group_labels = [[], [], []]
        # print(os.path.abspath("./uid_pred_list.txt"))
        uid_pred_list_file= open("./uid_pred_list.txt", 'w')

        group = num_ngs + 1
        item_lookup_with_cate_embedding=None
        for batch_data_input in tqdm(self.iterator.load_data_from_file(
            filename, min_seq_length=self.min_seq_length, batch_num_ngs=0
        )):
            if batch_data_input:
                if "other_metrics_basedOn_RecList" not in self.hparams.values() and "metrics_group_by_time_interval" not in self.hparams.values():
                    step_pred, step_labels = self.eval(load_sess, batch_data_input)
                    preds.extend(np.reshape(step_pred, -1))
                    labels.extend(np.reshape(step_labels, -1))
                    group_preds.extend(np.reshape(step_pred, (-1, group)))
                    group_labels.extend(np.reshape(step_labels, (-1, group)))
                if "other_metrics_basedOn_RecList" in self.hparams.values():
                    step_pred, step_labels, step_recItemListScore, step_posItem, step_histItems, item_lookup,cateIdOfAllItems_embedding = self.eval(
                        load_sess, batch_data_input)
                    preds.extend(np.reshape(step_pred, -1))
                    labels.extend(np.reshape(step_labels, -1))
                    # print("========sequential base model========== eval() pred & labels")
                    # print("======preds\n====",preds)
                    # print("======labels\n====",labels)
                    group_preds.extend(np.reshape(step_pred, (-1, group)))
                    group_labels.extend(np.reshape(step_labels, (-1, group)))

                    recItemListScores.extend(np.squeeze(step_recItemListScore))  # [B,vocNum,1]
                    posItems.extend(step_posItem)  # [B.]
                    histItems.extend(step_histItems)  # [B,T]
                if "metrics_group_by_time_interval" in self.hparams.values():
                    step_pred, step_labels,step_mask,step_time_to_now,step_last_diff,\
                    step_uid,step_time,step_target_item,step_beta_output,step_item_cate_history = self.eval(load_sess, batch_data_input)
                    preds.extend(np.reshape(step_pred, -1))
                    labels.extend(np.reshape(step_labels, -1))
                    group_preds.extend(np.reshape(step_pred, (-1, group)))
                    group_labels.extend(np.reshape(step_labels, (-1, group)))

                    short_mid_interval,mid_long_interval=short_mid_long_interval
                    for pred,label,last_diff in zip(np.reshape(step_pred, -1),np.reshape(step_labels, -1),step_last_diff):
                        if last_diff<short_mid_interval:#short diff
                            short_mid_long_preds[0].append(pred)
                            short_mid_long_labels[0].append(label)
                        elif last_diff<mid_long_interval:#mid time diff
                            short_mid_long_preds[1].append(pred)
                            short_mid_long_labels[1].append(label)
                        else:
                            short_mid_long_preds[2].append(pred)
                            short_mid_long_labels[2].append(label)

                    for group_pred,group_label,last_diff,uid,time,reclist,mutation_w,item_cate_history in zip(np.reshape(step_pred, (-1, group)),
                                                                np.reshape(step_labels, (-1, group)),
                                                                np.reshape(step_last_diff, (-1, group)),
                                                                np.reshape(step_uid, (-1, group)),
                                                                np.reshape(step_time, (-1, group)),
                                                                np.reshape(step_target_item, (-1, group)),
                                                                np.reshape(step_beta_output,(-1, group)),
                                                                np.reshape(step_item_cate_history,(-1, group,step_item_cate_history.shape[-1])),
                                                                ):
                        last_diff,uid,time,item_cate_history=last_diff[0],uid[0],time[0],item_cate_history[0,:]
                        uid_pred_list_file_line=str(uid)+"\t"+str(reclist[0])+"\t"+str(time)+"\t"+str(last_diff)+"\t"+\
                            ','.join(str(i) for i in reclist[1:]) +"\t"+\
                            ','.join(str(i) for i in group_pred) + "\t" + \
                            ','.join(str(i) for i in group_label) + "\t" + \
                            ','.join(str(i) for i in mutation_w) + "\n"
                        uid_pred_list_file.write(
                            uid_pred_list_file_line
                        )
                        if last_diff<short_mid_interval:#short diff
                            short_mid_long_group_preds[0].append(group_pred)
                            short_mid_long_group_labels[0].append(group_label)
                        elif last_diff<mid_long_interval:#mid time diff
                            short_mid_long_group_preds[1].append(group_pred)
                            short_mid_long_group_labels[1].append(group_label)
                        else:
                            short_mid_long_group_preds[2].append(group_pred)
                            short_mid_long_group_labels[2].append(group_label)



        print("==========start cal_metric=========\n")
        res = cal_metric(labels, preds, self.hparams.metrics)
        if "pairwise_metrics" in self.hparams.values():
            print("==========start pairwise_metrics=========\n")
            res_pairwise = cal_metric(
                group_labels, group_preds, self.hparams.pairwise_metrics
            )
            res.update(res_pairwise)


        if "other_metrics_basedOn_RecList" in self.hparams.values():
            print("==========start res_metrics_basedOn_RecList=========\n")
            if item_lookup_with_cate_embedding is None:
                item_lookup_with_cate_embedding = np.concatenate((item_lookup, cateIdOfAllItems_embedding), axis=-1)
            res_metrics_basedOn_RecList=\
                self.cal_metrics_basedOn_RecList(recItemListScores,posItems,histItems,item_lookup_with_cate_embedding,self.hparams.other_metrics_basedOn_RecList)

            res.update(res_metrics_basedOn_RecList)

        if "metrics_group_by_time_interval" in self.hparams.values():
            for idx,interval_group in enumerate(short_mid_long_interval_group):
                interval_group_metrics=cal_metric(short_mid_long_labels[idx], short_mid_long_preds[idx], self.hparams.metrics)
                interval_group_pair_metrics=cal_metric(short_mid_long_group_labels[idx],
                                                       short_mid_long_group_preds[idx],self.hparams.pairwise_metrics)

                key_suffix=interval_group
                def append_suffix_key(dictObj):
                    for orgKey in list(dictObj.keys()):
                        new_key = orgKey + key_suffix
                        dictObj[new_key] = dictObj.pop(orgKey)
                    return dictObj
                interval_group_metrics=append_suffix_key(interval_group_metrics)
                interval_group_pair_metrics=append_suffix_key(interval_group_pair_metrics)
                res.update(interval_group_metrics)
                res.update(interval_group_pair_metrics)
            print("short mid long interval group len is{}, {}, {}".format(len(short_mid_long_labels[0]),
                                                                          len(short_mid_long_labels[1]),
                                                                          len(short_mid_long_labels[2])))
        uid_pred_list_file.close()
        print("==========run eval end=========\n")
        return res

    def predict(self, infile_name, outfile_name):
        """Make predictions on the given data, and output predicted scores to a file.

        Args:
            infile_name (str): Input file name.
            outfile_name (str): Output file name.

        Returns:
            object: An instance of self.
        """

        load_sess = self.sess
        with tf.io.gfile.GFile(outfile_name, "w") as wt:
            for batch_data_input in self.iterator.load_data_from_file(
                infile_name, batch_num_ngs=0
            ):
                if batch_data_input:
                    step_pred = self.infer(load_sess, batch_data_input)
                    step_pred = np.reshape(step_pred, -1)
                    wt.write("\n".join(map(str, step_pred)))
                    wt.write("\n")
        return self

    def _build_embedding(self):
        """The field embedding layer. Initialization of embedding variables."""
        hparams = self.hparams
        self.user_vocab_length = len(load_dict(hparams.user_vocab))
        self.item_vocab_length = len(load_dict(hparams.item_vocab))
        self.cate_vocab_length = len(load_dict(hparams.cate_vocab))

        self.user_embedding_dim = hparams.user_embedding_dim
        self.item_embedding_dim = hparams.item_embedding_dim
        self.cate_embedding_dim = hparams.cate_embedding_dim

        with tf.compat.v1.variable_scope("embedding", initializer=self.initializer):
            self.user_lookup = tf.compat.v1.get_variable(
                name="user_embedding",
                shape=[self.user_vocab_length, self.user_embedding_dim],
                dtype=tf.float32,
            )
            self.item_lookup = tf.compat.v1.get_variable(
                name="item_embedding",
                shape=[self.item_vocab_length, self.item_embedding_dim],
                dtype=tf.float32,
            )
            self.cate_lookup = tf.compat.v1.get_variable(
                name="cate_embedding",
                shape=[self.cate_vocab_length, self.cate_embedding_dim],
                dtype=tf.float32,
            )
            if "time_stable_aware_fuse" in self.hparams.values() and self.hparams.time_stable_aware_fuse:
                self.time_stable_aware_fuse_exp_factor = tf.compat.v1.get_variable(
                    name="time_stable_aware_fuse_exp_factor",
                    shape=(),
                    initializer=tf.compat.v1.constant_initializer(
                        np.array([0.98]), dtype=np.float32
                    ),
                )
                # self.time_stable_aware_fuse_exp_factor_lookup = tf.compat.v1.get_variable(
                #     name="time_stable_aware_fuse_exp_factor_lookup",
                #     shape=[self.user_vocab_length,1],
                #     dtype=tf.float32,
                #     initializer=tf.ones_initializer()
                # )
            if "useMultiCate" in self.hparams.values() and self.hparams.useMultiCate:
                self.cate_weight_lookup = tf.compat.v1.get_variable(
                    name="cate_weight_embedding",
                    shape=[self.cate_vocab_length,1],
                    dtype=tf.float32,
                )

    def _lookup_from_embedding(self):
        """Lookup from embedding variables. A dropout layer follows lookup operations."""
        self.user_embedding = tf.nn.embedding_lookup(
            params=self.user_lookup, ids=self.iterator.users
        )
        # tf.compat.v1.summary.histogram("user_embedding_output", self.user_embedding)

        self.item_embedding = tf.compat.v1.nn.embedding_lookup(
            params=self.item_lookup, ids=self.iterator.items
        )
        self.item_history_embedding = tf.compat.v1.nn.embedding_lookup(
            params=self.item_lookup, ids=self.iterator.item_history
        )

        # tf.compat.v1.summary.histogram(
        #     "item_history_embedding_output", self.item_history_embedding
        # )


        self.cate_embedding = tf.compat.v1.nn.embedding_lookup(
            params=self.cate_lookup, ids=self.iterator.cates
        )#[B,]  OR [B,max_cate_num]

        self.cate_history_embedding = tf.compat.v1.nn.embedding_lookup(
            params=self.cate_lookup, ids=self.iterator.item_cate_history
        )  # [B,T,H]  OR [B,T,max_cate_num,H]

        self.cateIdOfAllItems_embedding=tf.compat.v1.nn.embedding_lookup(
            params=self.cate_lookup, ids=self.iterator.cateIdOfAllItems
        )  # [B,T,H]  OR [B,T,max_cate_num,H]

        # if "time_stable_aware_fuse" in  self.hparams.values() and self.hparams.time_stable_aware_fuse:
        #     self.time_stable_aware_fuse_exp_factor = tf.compat.v1.nn.embedding_lookup(
        #     params=self.time_stable_aware_fuse_exp_factor_lookup, ids=self.iterator.users
        # )  # [B,T,H]  OR [B,T,max_cate_num,H]
        #
        if "useMultiCate" in self.hparams.values() and self.hparams.useMultiCate:
            #self.cate_embedding ==> [B,max_cate_num,H]   mask [B,max_cate_num]

            # cateMask = self.iterator.cates
            # cond=tf.greater(cateMask,tf.zeros(tf.shape(cateMask),dtype=tf.int32))#  cate>0?
            # cateMask=tf.where(cond,tf.ones(tf.shape(cateMask)),tf.zeros(tf.shape(cateMask)) )
            # cateMask = tf.expand_dims(cateMask,-1)#mask [B,max_cate_num,1]
            # self.cate_embedding=tf.reduce_sum(self.cate_embedding*cateMask,-2)/tf.reduce_sum(cateMask,-2)
            # cate_history_mask=self.iterator.item_cate_history
            # cond_history = tf.greater(cate_history_mask, tf.zeros(tf.shape(cate_history_mask),dtype=tf.int32))  # cate>0?
            # cate_history_mask=tf.where(cond_history,tf.ones(tf.shape(cate_history_mask)),tf.zeros(tf.shape(cate_history_mask)))
            # cate_history_mask=tf.expand_dims(cate_history_mask,-1)#[B,T,max_cate_num,1]
            # self.cate_history_embedding=tf.reduce_sum(self.cate_history_embedding*cate_history_mask,-2)/tf.reduce_sum(cate_history_mask,-2)


            self.cateW = tf.compat.v1.nn.embedding_lookup(
                params=self.cate_weight_lookup, ids=self.iterator.cates
            )  # [B,1]  OR [B,max_cate_num,1]
            self.cateHistoryW = tf.compat.v1.nn.embedding_lookup(
                params=self.cate_weight_lookup, ids=self.iterator.item_cate_history
            )  # [B,T,1]  OR [B,T,max_cate_num,1]

            self.cate_embedding =tf.reduce_sum(self.cate_embedding*
                                               tf.expand_dims(tf.math.softmax(self.iterator.itemCateW,-1),-1)*
                                               self.cateW,
                                               -2)
            self.cate_history_embedding = tf.reduce_sum(self.cate_history_embedding*
                                                        tf.expand_dims(tf.math.softmax(self.iterator.itemHistoryCateW,axis=-1),-1)*
                                                        self.cateHistoryW,
                                                        -2)

            # self.cate_embedding =tf.reduce_sum(self.cate_embedding*tf.zeros_like(self.cate_embedding),-2)
            # self.cate_history_embedding = tf.reduce_sum(self.cate_history_embedding * tf.zeros_like(self.cate_history_embedding),-2)
            # tf.compat.v1.summary.histogram(
            #     "cate_history_embedding_output", self.cate_history_embedding
            # )

        involved_items = tf.concat(
            [
                tf.reshape(self.iterator.item_history, [-1]),
                tf.reshape(self.iterator.items, [-1]),
            ],
            -1,
        )
        self.involved_items, _ = tf.unique(involved_items)
        involved_item_embedding = tf.nn.embedding_lookup(
            params=self.item_lookup, ids=self.involved_items
        )
        self.embed_params.append(involved_item_embedding)

        involved_cates = tf.concat(
            [
                tf.reshape(self.iterator.item_cate_history, [-1]),
                tf.reshape(self.iterator.cates, [-1]),
            ],
            -1,
        )
        self.involved_cates, _ = tf.unique(involved_cates)
        involved_cate_embedding = tf.nn.embedding_lookup(
            params=self.cate_lookup, ids=self.involved_cates
        )
        self.embed_params.append(involved_cate_embedding)
        if "useCate" in self.hparams.values() and self.hparams.useCate:
            self.target_item_embedding = tf.concat(
                [self.item_embedding, self.cate_embedding], -1
            )
            self.item_history_embedding=tf.concat(
                [self.item_history_embedding, self.cate_history_embedding], -1
            )

        else:
            self.target_item_embedding = self.item_embedding
            #self.item_history_embedding=self.item_history_embedding

        # tf.compat.v1.summary.histogram(
        #     "target_item_embedding_output", self.target_item_embedding
        # )

    def _add_norm(self):
        """Regularization for embedding variables and other variables."""
        all_variables, embed_variables = (
            tf.compat.v1.trainable_variables(),
            tf.compat.v1.trainable_variables(
                self.sequential_scope._name + "/embedding"
            ),
        )
        layer_params = list(set(all_variables) - set(embed_variables))
        layer_params = [a for a in layer_params if "_no_reg" not in a.name]
        self.layer_params.extend(layer_params)
