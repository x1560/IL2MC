# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
import numpy as np
from recommenders.models.deeprec.models.sequential.sequential_base_model import (
    SequentialBaseModel,
)
from tensorflow.compat.v1.nn import dynamic_rnn
from recommenders.models.deeprec.models.sequential.rnn_cell_implement import (
    Time4LSTMCell,
)
import tensorflow_probability as tfp

__all__ = ["SLI_RECModel"]


class SLI_RECModel(SequentialBaseModel):
    """SLI Rec model

    :Citation:

        Z. Yu, J. Lian, A. Mahmoody, G. Liu and X. Xie, "Adaptive User Modeling with
        Long and Short-Term Preferences for Personailzed Recommendation", in Proceedings of
        the 28th International Joint Conferences on Artificial Intelligence, IJCAI’19,
        Pages 4213-4219, AAAI Press, 2019.
    """

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
            if "other_metrics_basedOn_RecList" in self.hparams.values():
                self.recItemListScore = self._fcn_net(self.recItemListScoreModelOut, hparams.layer_sizes, scope="logit_fcn",
                                                  reuse=True)

            self._add_norm()
            return logit

    def _build_seq_graph(self):
        """The main function to create sli_rec model.

        Returns:
            object: the output of sli_rec section.
        """
        hparams = self.hparams
        with tf.compat.v1.variable_scope("sli_rec"):
            hist_input = self.item_history_embedding
            self.mask = self.iterator.mask
            self.sequence_length = tf.reduce_sum(input_tensor=self.mask, axis=1)

            with tf.compat.v1.variable_scope("long_term_asvd"):
                att_outputs1 = self._attention(hist_input, hparams.attention_size)
                att_fea1 = tf.reduce_sum(input_tensor=att_outputs1, axis=1)
                tf.compat.v1.summary.histogram("att_fea1", att_fea1)

            item_history_embedding_new = tf.concat(
                [
                    self.item_history_embedding,
                    tf.expand_dims(self.iterator.time_from_first_action, -1),
                ],
                -1,
            )
            item_history_embedding_new = tf.concat(
                [
                    item_history_embedding_new,
                    tf.expand_dims(self.iterator.time_to_now, -1),
                ],
                -1,
            )
            with tf.compat.v1.variable_scope("rnn"):
                rnn_outputs, _ = dynamic_rnn(
                    Time4LSTMCell(hparams.hidden_size),
                    inputs=item_history_embedding_new,
                    sequence_length=self.sequence_length,
                    dtype=tf.float32,
                    scope="time4lstm",
                )
                tf.compat.v1.summary.histogram("LSTM_outputs", rnn_outputs)

            with tf.compat.v1.variable_scope("attention_fcn"):
                att_outputs2 = self._attention_fcn(
                    self.target_item_embedding, rnn_outputs
                )
                att_fea2 = tf.reduce_sum(input_tensor=att_outputs2, axis=1)
                tf.compat.v1.summary.histogram("att_fea2", att_fea2)

            # ensemble
            def top_k_around_median(x, k):
                # 计算中位数
                # k=10
                median = tfp.stats.percentile(x, 100.0, interpolation='nearest', axis=-1,
                                              keepdims=True)  # [B,1]
                print("median:{}".format(median.shape))
                # 计算每个元素与中位数的差值的绝对值
                abs_diff = tf.abs(x - median)  # [B,H]
                print("abs_diff:{}".format(abs_diff.shape))
                # 找到差值最小的K个元素的索引
                top_k = tf.argsort(abs_diff, direction='DESCENDING')[:, :k]  # [B,K]
                return top_k
            def ipPoolQuery(query, ipPool):
                '''

                Args:
                    query:  [,dim]
                    ipPool: [num_ip,dim]

                Returns:

                '''

                with tf.compat.v1.variable_scope("ipPoolQuery"):
                    query_size = query.shape[-1]
                    attention_w = tf.compat.v1.get_variable(name="ipPoolQueryattention_w", shape=[query_size, hparams.hidden_size],
                                                            )
                    attention_b = tf.compat.v1.get_variable(name="ipPoolQueryattention_b", shape=[hparams.hidden_size],
                                                            )
                    query_prj = tf.compat.v1.nn.xw_plus_b(query, attention_w, attention_b)
                # [B,D]  [poolNum,D]
                query_prj_expd = tf.expand_dims(query_prj, axis=1)  # [B,1,D]
                pool = tf.expand_dims(ipPool, axis=0)  # [1,poolNum,D]
                simMat = tf.matmul(query_prj, ipPool, transpose_b=True)  # [b,voc]
                median_sim_idx = top_k_around_median(simMat, 4)  # [B,K]
                median_sim_ip = tf.nn.embedding_lookup(
                    params=ipPool, ids=median_sim_idx
                )  # [B,K,H]
                # tf.reduce_sum(query_prj_expd*median_sim_ip,axis=-1,keepdims=True)#[B 1 D] [B K D]=>[B,K,1]

                att_logit = tf.nn.softmax(
                    tf.reduce_sum(query_prj_expd*median_sim_ip,axis=-1,keepdims=True)
                                          ,1)  # [B,K,1] along K
                collb_ip = tf.reduce_sum(median_sim_ip * att_logit, axis=1)  # [B,K,D]=>[B,D]
                return collb_ip
            def reconsItem(query,itemPool):
                '''
                Args:
                    query:  [B,dim]
                    ipPool: [num_ip,dim]
                Returns:
                '''
                itemDim=self.target_item_embedding.shape[-1]
                with tf.compat.v1.variable_scope("ipPoolQuery"):
                    query_size = query.shape[-1]
                    attention_w = tf.compat.v1.get_variable(name="itemPoolQueryattention_w",
                                                            shape=[query_size, itemDim],
                                                            )
                    attention_b = tf.compat.v1.get_variable(name="itemPoolQueryattention_b", shape=[itemDim],
                                                            )
                    query_prj = tf.compat.v1.nn.xw_plus_b(query, attention_w, attention_b)
                # [B,D]  [poolNum,D]
                query_prj_expd = tf.expand_dims(query_prj, axis=1)  # [B,1,D]
                simMat = tf.matmul(query_prj, itemPool, transpose_b=True)  # [b,voc]
                median_sim_idx = top_k_around_median(simMat, 4)  # [B,K]
                median_sim_ip = tf.nn.embedding_lookup(
                    params=itemPool, ids=median_sim_idx
                )  # [B,K,H]
                # tf.reduce_sum(query_prj_expd*median_sim_ip,axis=-1,keepdims=True)#[B 1 D] [B K D]=>[B,K,1]
                att_logit = tf.nn.softmax(
                    tf.reduce_sum(query_prj_expd * median_sim_ip, axis=-1, keepdims=True)
                    , 1)  # [B,K,1] along K
                collb_ip = tf.reduce_sum(median_sim_ip * att_logit, axis=1)  # [B,K,D]=>[B,D]
                return collb_ip
            if "item_ip_coll" in self.hparams.values() and self.hparams.item_ip_coll:
                states=rnn_outputs#[B,S,H]
                dropFirstHist = self.item_history_embedding[:, 1:, :]  # [B,T-1,H]
                query = dropFirstHist
                query_size = query.shape[-1]
                # self.hidden1_w = tf.compat.v1.get_variable(name="uitem_ip_coll_h1_w",
                #                                              shape=[query_size, 64],
                #                                              initializer=self.initializer, )
                # self.hidden1_b = tf.compat.v1.get_variable(name="item_ip_coll_h1_b", shape=[64],
                #                                              initializer=self.initializer, )
                #
                # self.hidden2_w = tf.compat.v1.get_variable(name="uitem_ip_coll_h2_w",
                #                                            shape=[64, self.hparams.hidden_size],
                #                                            initializer=self.initializer, )
                # self.hidden2_b = tf.compat.v1.get_variable(name="item_ip_coll_h2_b", shape=[self.hparams.hidden_size],
                #                                            initializer=self.initializer, )
                # hidden1=tf.compat.v1.nn.xw_plus_b(query,self.hidden1_w,self.hidden1_b)
                # ideal_state=tf.compat.v1.nn.xw_plus_b(hidden1,self.hidden2_w,self.hidden2_b)

                ideal_state=self._fcn_net(
                    tf.reshape(query,[-1,query_size]), hparams.att_fcn_layer_sizes, scope="idealStateEncoder",
                    _out_layer_size=self.hparams.hidden_size
                )
                ideal_state=tf.nn.tanh(ideal_state)
                if "reconstruct_item" in self.hparams.values() and self.hparams.reconstruct_item:
                    # self._decoder_hidden_1_w = tf.compat.v1.get_variable(name="_decoder_uitem_ip_coll_h1_w",
                    #                                                      shape=[self.hparams.hidden_size, 80],
                    #                                                      initializer=self.initializer, )
                    # self._decoder_hidden1_b = tf.compat.v1.get_variable(name="_decoder_item_ip_coll_h1_b", shape=[80],
                    #                                                     initializer=self.initializer, )
                    #
                    # self._decoder_hidden2_w = tf.compat.v1.get_variable(name="_decoder_uitem_ip_coll_h2_w",
                    #                                                     shape=[64, query_size],
                    #                                                     initializer=self.initializer, )
                    # self._decoder_hidden2_b = tf.compat.v1.get_variable(name="_decoder_item_ip_coll_h2_b",
                    #                                                     shape=[query_size],
                    #                                                     initializer=self.initializer, )
                    #
                    # decoder_hidden = tf.compat.v1.nn.xw_plus_b(ideal_state, self._decoder_hidden_1_w, self._decoder_hidden1_b)
                    # recons_item = tf.compat.v1.nn.xw_plus_b(decoder_hidden, self._decoder_hidden2_w, self._decoder_hidden2_b)
                    recons_item=self._fcn_net(
                        ideal_state, hparams.att_fcn_layer_sizes,
                        scope="idealStateDecoder",_out_layer_size=query_size
                    ) #[B*T,H]
                    recons_item=tf.nn.tanh(recons_item)

                    recons_item=tf.reshape(recons_item,tf.shape(query))
                    normalize_recons_item = tf.nn.l2_normalize(recons_item, -1)#[B,T-1,H]
                    normalize_org_item = tf.nn.l2_normalize(dropFirstHist, -1) #[B,T-1,H]
                    recons_item_ssl_loss = 1 - tf.reduce_sum(normalize_recons_item * normalize_org_item, -1)  # [B,T-1] 1-cosSim
                    recons_item_ssl_loss = recons_item_ssl_loss * tf.cast(self.iterator.mask[:, 1:], tf.float32)
                    self.recons_item_ssl_loss = tf.reduce_sum(recons_item_ssl_loss) / tf.reduce_sum(
                        tf.cast(self.iterator.mask[:, 1:], tf.float32))  # scalar loss
                    self.recons_item_ssl_loss = self.hparams.coll_muta_ip_ssl_regu * self.recons_item_ssl_loss


                states_t = states
                normalize_states_t = tf.nn.l2_normalize(states_t, -1)[:, :-1, :]  # [B,T-1,H]
                ideal_state = tf.reshape(ideal_state,tf.shape(normalize_states_t))
                normalize_ip = tf.nn.l2_normalize(ideal_state, -1)
                coll_muta_ip_ssl_loss = 1 - tf.reduce_sum(normalize_ip * normalize_states_t, -1)  # [B,T-1] 1-cosSim
                coll_muta_ip_ssl_loss = coll_muta_ip_ssl_loss * tf.cast(self.iterator.mask[:, 1:], tf.float32)
                self.coll_muta_ip_ssl_loss = tf.reduce_sum(coll_muta_ip_ssl_loss) / tf.reduce_sum(
                    tf.cast(self.iterator.mask[:, 1:], tf.float32))  # scalar loss
                self.coll_muta_ip_ssl_loss = self.hparams.coll_muta_ip_ssl_regu * self.coll_muta_ip_ssl_loss

            if "itemMutation" in self.hparams.values() and self.hparams.itemMutation:
                ipNum = self.hparams.ipNum
                self.ipPool = tf.compat.v1.get_variable(
                    name="ipPool", shape=[ipNum, self.hidden_size], initializer=self.initializer
                )
                if "reconstruct_item" in self.hparams.values() and self.hparams.reconstruct_item:
                    self.itemPool = tf.compat.v1.get_variable(
                        name="itemPool", shape=[ipNum, self.target_item_embedding.shape[-1]], initializer=self.initializer
                    )

                item_target = self.target_item_embedding  # [b,h]
                # ideal_state = ipPoolQuery(item_target,self.ipPool)
                ideal_state = tf.nn.tanh(self._fcn_net(
                    tf.reshape(item_target, [-1, query_size]), hparams.att_fcn_layer_sizes, scope="idealStateEncoder",
                    _out_layer_size=self.hparams.hidden_size,reuse=True
                ))

                with tf.compat.v1.name_scope("alpha"):
                    last_diff_idx = tf.cast(tf.reduce_sum(self.iterator.mask, keepdims=True, axis=-1),
                                            tf.int32) - 1  # [B,1]
                    row_idx = tf.expand_dims(tf.range(tf.shape(last_diff_idx)[0]), -1)
                    last_diff_idx = tf.concat([row_idx, last_diff_idx],
                                              axis=-1)  # [B,2]
                    last_diff = tf.gather_nd(self.iterator.time_to_now, last_diff_idx)  # [B,]
                    last_diff = tf.expand_dims(last_diff, -1)  # [B,1]
                    concat_all = tf.concat(
                        [
                            self.target_item_embedding,
                            att_fea1,
                            att_fea2,
                            last_diff,
                        ],
                        1,
                    )
                    last_hidden_nn_layer = concat_all
                    alpha_logit = self._fcn_net(
                        last_hidden_nn_layer, hparams.att_fcn_layer_sizes, scope="fcn_alpha"
                    )
                    alpha_output = tf.sigmoid(alpha_logit)
                    user_embed = att_fea1 * alpha_output + att_fea2 * (1.0 - alpha_output)

                    concat_all = tf.concat(
                        [
                            user_embed,
                            ideal_state,
                            user_embed * ideal_state,
                            last_diff,
                        ],
                        1,
                    )

                    last_hidden_nn_layer = concat_all
                    beta_logit = self._fcn_net(
                        last_hidden_nn_layer, hparams.att_fcn_layer_sizes, scope="fcn_beta"
                    )
                    beta_output = tf.sigmoid(beta_logit)
                    user_embed = user_embed + ideal_state * beta_output
                    if "reconstruct_item" in self.hparams.values() and self.hparams.reconstruct_item:
                        # next_item=reconsItem(item_target,self.itemPool)
                        next_item=tf.nn.tanh(self._fcn_net(
                        ideal_state, hparams.att_fcn_layer_sizes,
                        scope="idealStateDecoder",_out_layer_size=query_size,reuse=True)) #[B*T,H]
                        user_embed=tf.concat([user_embed,next_item,next_item*self.target_item_embedding],axis=-1)

                    # user_embed=concat_all
            else:
                with tf.compat.v1.name_scope("alpha"):
                    last_diff_idx = tf.cast(tf.reduce_sum(self.iterator.mask, keepdims=True, axis=-1),
                                            tf.int32) - 1  # [B,1]
                    row_idx = tf.expand_dims(tf.range(tf.shape(last_diff_idx)[0]), -1)
                    last_diff_idx = tf.concat([row_idx, last_diff_idx],
                                              axis=-1)  # [B,2]
                    last_diff = tf.gather_nd(self.iterator.time_to_now, last_diff_idx)  # [B,]
                    last_diff = tf.expand_dims(last_diff, -1)  # [B,1]
                    org_diff=tf.expand_dims(self.iterator.time_to_now[:, -1], -1)
                    concat_all = tf.concat(
                        [
                            self.target_item_embedding,
                            att_fea1,
                            att_fea2,
                            org_diff,
                        ],
                        1,
                    )
                    last_hidden_nn_layer = concat_all
                    alpha_logit = self._fcn_net(
                        last_hidden_nn_layer, hparams.att_fcn_layer_sizes, scope="fcn_alpha"
                    )
                    alpha_output = tf.sigmoid(alpha_logit)
                    user_embed = att_fea1 * alpha_output + att_fea2 * (1.0 - alpha_output)


                if "useShareIpPool" in hparams.values() and hparams.useShareIpPool:
                    ipNum = self.hparams.ipNum
                    self.ipPool = tf.compat.v1.get_variable(
                        name="ipPool", shape=[ipNum, self.hidden_size], initializer=self.initializer
                    )
                    if "ui_ip_coll" in self.hparams.values() and self.hparams.ui_ip_coll:
                        uesr_seq_expd = tf.expand_dims(self.user_embedding, axis=1)  # [B,1,H]
                        ui_cat = tf.concat([tf.tile(uesr_seq_expd, [1, tf.shape(self.item_history_embedding)[1], 1]),
                                            self.item_history_embedding], axis=-1)  # [B,T,H][B,T,H]
                        query = ui_cat
                        query_size = query.shape[-1]
                        self.attention_w = tf.compat.v1.get_variable(name="ui_ip_coll_w",
                                                                     shape=[query_size, hparams.hidden_size],
                                                                     initializer=self.initializer, )
                        self.attention_b = tf.compat.v1.get_variable(name="ui_ip_coll_b", shape=[hparams.hidden_size],
                                                                     initializer=self.initializer, )
                        realQuery = tf.compat.v1.nn.xw_plus_b(query, self.attention_w,
                                                              self.attention_b)  # [B,T,H]=>[B,T,h2]
                        # [B,D]  [poolNum,D]
                        pool = tf.expand_dims(self.ipPool, axis=0)  # [1,poolNum,D]
                        poolSoftBeta = 1000
                        att_logit = tf.nn.softmax(
                            poolSoftBeta * tf.matmul(realQuery, pool, transpose_b=True)
                            , axis=-1)  # [B,T,poolNum]
                        att_logit = tf.expand_dims(att_logit, -1)  # [B,T,poolNum,1]
                        pool = tf.expand_dims(pool, 0)  # [1,1,poolNUm,d]
                        coll_muta_ip = tf.reduce_sum(pool * att_logit, axis=-2)  # [B,t,poolNum,D]=>[B,t,D]
                        states_t = rnn_outputs
                        normalize_ip = tf.nn.l2_normalize(coll_muta_ip, -1)
                        normalize_states_t = tf.nn.l2_normalize(states_t, -1)
                        coll_muta_ip_ssl_loss = 1 - tf.reduce_sum(normalize_ip * normalize_states_t,
                                                                  -1)  # [B,T] 1-cosSim
                        coll_muta_ip_ssl_loss = coll_muta_ip_ssl_loss * tf.cast(self.iterator.mask, tf.float32)
                        self.coll_muta_ip_ssl_loss = tf.reduce_sum(coll_muta_ip_ssl_loss) / tf.reduce_sum(
                            tf.cast(self.iterator.mask, tf.float32))  # scalar loss
                        self.coll_muta_ip_ssl_loss = self.hparams.coll_muta_ip_ssl_regu * self.coll_muta_ip_ssl_loss
                        # self.coll_muta_ip_ssl_loss=tf.compat.v1.Print( self.coll_muta_ip_ssl_loss,[self.coll_muta_ip_ssl_loss],summarize=1,message="coll_muta_ip_ssl_loss")

                        normalize_ip = tf.nn.l2_normalize(self.ipPool, -1)  # [N,H]
                        ip_pair_sim = tf.matmul(normalize_ip, normalize_ip, transpose_b=True)  # [N,N]
                        ii_ip_sim = tf.linalg.diag_part(ip_pair_sim)  # [N]
                        ij_ip_sim = ip_pair_sim - tf.linalg.diag(ii_ip_sim)  # i_i val=>0, [N,N]
                        self.ipPool_sim = tf.reduce_mean(ij_ip_sim + 1) * 0.01
                if "userCollMutation" in self.hparams.values() and self.hparams.userCollMutation:
                    user_self = self.user_embedding#[b,h]

                    def top_k_around_median(x, k):
                        # 计算中位数
                        # k=10
                        median = tfp.stats.percentile(x, 100.0, interpolation='nearest', axis=-1, keepdims=True)  # [B,1]
                        print("median:{}".format(median.shape))
                        # 计算每个元素与中位数的差值的绝对值
                        abs_diff = tf.abs(x - median)  # [B,H]
                        print("abs_diff:{}".format(abs_diff.shape))
                        # 找到差值最小的K个元素的索引
                        top_k = tf.argsort(abs_diff, direction='DESCENDING')[:, :k]  # [B,K]

                        return top_k

                    allUserEmd = self.user_lookup#[VOC,H]
                    simMat = tf.matmul(user_self, allUserEmd, transpose_b=True)#[b,voc]
                    print(simMat.shape)
                    median_sim_idx = top_k_around_median(simMat, 4)# [B,K]
                    median_sim_user=tf.nn.embedding_lookup(
                        params=self.user_lookup, ids=median_sim_idx
                    )#[B,K,H]
                    other_user_w=tf.reduce_sum(tf.expand_dims(user_self,axis=1)*median_sim_user,axis=-1,keepdims=True)#[B,1,H]*[B,K,H]==>[B,K,1]
                    query=tf.tile(tf.expand_dims(self.target_item_embedding,1),[1,tf.shape(median_sim_user)[1],1])#[B,1,H]=>[B,k,H]
                    query=tf.concat([query,median_sim_user],axis=-1)
                    query_size = query.shape[-1]
                    # self.attention_w = tf.compat.v1.get_variable(name="ui_ip_coll_w",
                    #                                              shape=[query_size, hparams.hidden_size],
                    #                                              initializer=self.initializer, )
                    # self.attention_b = tf.compat.v1.get_variable(name="ui_ip_coll_b", shape=[hparams.hidden_size],
                    #                                              initializer=self.initializer, )
                    realQuery = tf.compat.v1.nn.xw_plus_b(query, self.attention_w, self.attention_b)  # [B,K,H]
                    # [B,D]  [poolNum,D]
                    pool = tf.expand_dims(self.ipPool, axis=0)  # [1,poolNum,D]
                    poolSoftBeta = 1000
                    att_logit = tf.nn.softmax(
                        poolSoftBeta * tf.matmul(realQuery, pool, transpose_b=True)
                        , axis=-1)  # [B,k,poolNum]
                    att_logit = tf.expand_dims(att_logit, -1)  # [B,K,poolNum,1]
                    pool = tf.expand_dims(pool, 0)  # [1,1,poolNUm,d]
                    coll_muta_ip = tf.reduce_sum(pool * att_logit, axis=-2)  # [B,K,poolNum,D]=>[B,K,D]
                    coll_muta_ip=other_user_w*coll_muta_ip#[B,k,1]*[B,k,D]
                    coll_muta_ip=tf.reduce_sum(coll_muta_ip,axis=1)#[B,D]

                    self_ip=tf.concat([user_self,self.target_item_embedding],axis=-1)#[B,H1+H2]
                    self_ip_mapping = tf.compat.v1.nn.xw_plus_b(self_ip, self.attention_w, self.attention_b)
                    self_ip_mapping = tf.expand_dims(self_ip_mapping, axis=1)  # [B,1,D]
                    pool = tf.expand_dims(self.ipPool, axis=0)  # [1,poolNum,D]
                    att_logit = tf.nn.softmax(
                        poolSoftBeta * tf.reduce_sum(self_ip_mapping * pool, axis=-1, keepdims=True)
                        , axis=-2)  # [B,poolNum,1]
                    self_ip = tf.reduce_sum(pool * att_logit, axis=1)  # [B,poolNum,D]=>[B,D]

                    self._alpha=attention_mat = tf.compat.v1.get_variable(
                        name="_alpha_no_reg",
                        shape=(),
                        initializer=tf.compat.v1.constant_initializer(
                            np.array([0.98]), dtype=np.float32
                        ),
                    )
                    fuse_alpha = tf.maximum(self._alpha, 0.15)
                    # fuse_alpha = tf.minimum(fuse_alpha, 1.0)
                    last_diff_idx=tf.cast(tf.reduce_sum(self.iterator.mask,keepdims=True,axis=-1),tf.int32)-1#[B,1]
                    row_idx=tf.expand_dims(tf.range(tf.shape(last_diff_idx)[0]), -1)

                    last_diff_idx = tf.concat([row_idx, last_diff_idx],
                                             axis=-1)  # [B,2]

                    last_diff=tf.gather_nd(self.iterator.time_to_now,last_diff_idx)#[B,]

                    last_diff=tf.expand_dims(last_diff,-1)#[B,1]
                    last_diff=tf.maximum(last_diff, 0.15)
                    # last_diff=tf.compat.v1.Print(last_diff,[tf.reduce_mean(last_diff)],summarize=1,message="last_diff")
                    fuse_alpha=tf.exp(-fuse_alpha*last_diff)#[B,1]*
                    # fuse_alpha=tf.compat.v1.Print(fuse_alpha,[tf.reduce_mean(fuse_alpha)],summarize=1,message="fuse_alpha")
                    user_embed = (user_embed + (coll_muta_ip+self_ip) * (1.0 - fuse_alpha) )/(2.0-fuse_alpha)
                if "time_stable_aware_fuse" in self.hparams.values() and self.hparams.time_stable_aware_fuse:
                    h_pool = ipPoolQuery(self.target_item_embedding, self.ipPool)  # [B,D]
                    with tf.compat.v1.name_scope("time_stable_aware_fuse_alpha"):
                        diff_mask = tf.cast(self.iterator.mask, dtype=tf.float32)  # [B,L]
                        time_diff_mean = tf.reduce_sum(self.iterator.time_diff, axis=-1, keepdims=True) / tf.reduce_sum(
                            diff_mask, axis=-1, keepdims=True)  # [B,1]
                        len_sub_1 = tf.reduce_sum(diff_mask, axis=-1, keepdims=True) - 1  # [B,1]
                        diff_sub_mean = diff_mask * time_diff_mean  # [B,L]
                        time_diff_var = tf.reduce_sum(tf.square(self.iterator.time_diff - diff_sub_mean), axis=-1,
                                                      keepdims=True) / len_sub_1  # [B,1]
                        # \sum(x-x_)@2/N-1
                        time_diff_var = tf.sqrt(time_diff_var)  # [B,]
                        # decayFactor=tf.maximum(self.time_stable_aware_fuse_exp_factor,0.8)#[B,1]
                        decayFactor = 1.0
                        factor = tf.exp(-time_diff_var * decayFactor)  # [B,1]
                        # alpha_output=tf.expand_dims(alpha_output,-1)
                        user_embed = (user_embed + h_pool * (1.0 - factor)) / (2.0 - factor)

            model_output = tf.concat([user_embed, self.target_item_embedding], 1)
            # tf.compat.v1.summary.histogram("model_output", model_output)
            if "other_metrics_basedOn_RecList" in self.hparams.values():
                item_lookup = tf.concat([self.item_lookup, self.cateIdOfAllItems_embedding], -1)  # [vocNum,H1+H2]
                allItemsEmd = tf.expand_dims(item_lookup, 0)  # [vocNum,H]==>[1,vocNum,H]
                user_embed = tf.expand_dims(user_embed, 1)  # [B,H]=>[B,1,H]
                user_embed = user_embed[::self.hparams.test_num_ngs + 1, :, :]
                vocNum = allItemsEmd.shape[1]
                user_embed = tf.tile(user_embed, [1, vocNum, 1])#[B,vocNum,H]
                allItemsEmd = tf.tile(allItemsEmd, [tf.shape(user_embed)[0], 1, 1])
                self.recItemListScoreModelOut = tf.concat([user_embed, allItemsEmd], -1)#[,32,h=40]

                # [B,vocNum,1]
            return model_output


    def _attention_fcn(self, query, user_embedding):
        """Apply attention by fully connected layers.

        Args:
            query (object): The embedding of target item which is regarded as a query in attention operations.
            user_embedding (object): The output of RNN layers which is regarded as user modeling.

        Returns:
            object: Weighted sum of user modeling.
        """
        hparams = self.hparams
        with tf.compat.v1.variable_scope("attention_fcn"):
            query_size = query.shape[1]
            boolean_mask = tf.equal(self.mask, tf.ones_like(self.mask))

            attention_mat = tf.compat.v1.get_variable(
                name="attention_mat",
                shape=[user_embedding.shape.as_list()[-1], query_size],
                initializer=self.initializer,
            )
            att_inputs = tf.tensordot(user_embedding, attention_mat, [[2], [0]])

            queries = tf.reshape(
                tf.tile(query, [1, att_inputs.shape[1]]), tf.shape(input=att_inputs)
            )
            last_hidden_nn_layer = tf.concat(
                [att_inputs, queries, att_inputs - queries, att_inputs * queries], -1
            )

            att_fnc_output = self._fcn_net(
                last_hidden_nn_layer, hparams.att_fcn_layer_sizes, scope="att_fcn"
            )
            # att_fnc_output=tf.compat.v1.Print(att_fnc_output,[tf.shape(att_fnc_output)],message="=====att_fnc_output:")

            att_fnc_output = tf.squeeze(att_fnc_output, -1)
            mask_paddings = tf.ones_like(att_fnc_output) * (-(2 ** 32) + 1)
            att_weights = tf.nn.softmax(
                tf.compat.v1.where(boolean_mask, att_fnc_output, mask_paddings),
                name="att_weights",
            )
            output = user_embedding * tf.expand_dims(att_weights, -1)
            return output
    def _compute_regular_loss(self):
        """Construct regular loss. Usually it's comprised of l1 and l2 norm.
        Users can designate which norm to be included via config file.

        Returns:
            object: Regular loss.
        """

        regular_loss = self._l2_loss() + self._l1_loss() + self._cross_l_loss()
        if "ui_ip_coll" in self.hparams.values() and self.hparams.ui_ip_coll:
            regular_loss = regular_loss +self.coll_muta_ip_ssl_loss+self.ipPool_sim
        if "item_ip_coll" in self.hparams.values() and self.hparams.item_ip_coll:
            regular_loss = regular_loss + self.coll_muta_ip_ssl_loss
        # if "reconstruct_item" in self.hparams.values() and self.hparams.reconstruct_item:
        #     regular_loss = regular_loss + self.recons_item_ssl_loss
        return tf.reduce_sum(input_tensor=regular_loss)