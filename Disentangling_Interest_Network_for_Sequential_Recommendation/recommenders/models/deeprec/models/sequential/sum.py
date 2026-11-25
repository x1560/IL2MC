# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.compat.v1.nn import dynamic_rnn
from tensorflow.python.ops import array_ops

from recommenders.models.deeprec.models.sequential.sequential_base_model import (
    SequentialBaseModel,
)
from recommenders.models.deeprec.models.sequential.sum_cells import (
    SUMCell,
    SUMV2Cell, SUMV3Cell, SUMV4Cell, SUMV5Cell, SUMV6Cell, SUMV7Cell, SUMV6TimeCell
)


class SUMModel(SequentialBaseModel):
    """Sequential User Matrix Model

    :Citation:

        Lian, J., Batal, I., Liu, Z., Soni, A., Kang, E. Y., Wang, Y., & Xie, X.,
        "Multi-Interest-Aware User Modeling for Large-Scale Sequential Recommendations", arXiv preprint arXiv:2102.09211, 2021.
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
        """The main function to create SUM model.

        Returns:
            object: The output of SUM section, which is a concatenation of user vector and target item vector.
        """
        hparams = self.hparams  # noqa: F841
        with tf.compat.v1.variable_scope("sum"):
            # if self.hparams.cell=="SUMV3":
            #     self.item_history_embedding = tf.concat(
            #         [self.item_history_embedding, self.cate_history_embedding], -1
            #     )
            self.history_embedding = self.item_history_embedding

            if self.hparams.useTimeInterval:
                if hparams.cell in ["SUMV6Time"]:
                    shiftTimeFromFirst = tf.concat(
                        [tf.zeros(shape=[tf.shape(self.iterator.time_from_first_action)[0], 1], dtype=tf.float32),
                         self.iterator.time_from_first_action[:, :-1]], -1)  # [0,..] [b,t]
                    self.history_embedding = tf.concat(
                        [
                            self.history_embedding,
                            tf.expand_dims(self.iterator.time_from_first_action, -1),
                            tf.expand_dims(self.iterator.time_to_now,-1)
                        ],
                        -1,
                    )
                else:
                    # [inp,ti-t1,ti-now]
                    shiftTimeInterval=tf.concat(  [tf.zeros(shape=[tf.shape(self.iterator.time_diff)[0],1],dtype=tf.float32),
                                                  self.iterator.time_diff[:,:-1] ],-1 )#[0,..]  b,t
                    self.history_embedding = tf.concat(
                        [
                            self.history_embedding,
                            tf.expand_dims(shiftTimeInterval, -1),
                        ],
                        -1,
                    )

            if "persInitState" in self.hparams.values() and self.hparams.persInitState:
                with tf.compat.v1.variable_scope('persInitStatelayer1'):
                    self.persInit1_w=tf.compat.v1.get_variable("persInit1_w", shape=[self.hparams.slots,self.hparams.hidden_size],
                                    initializer=self.initializer)#[S,D]
            if "personal_intst_act" in self.hparams.values() and self.hparams.personal_intst_act:
                self.personal_intst_act()
            cell = self._create_sumcell()
            self.cell = cell
            cell.model = self
            if hparams.cell in ["SUMV6", "SUMV7","SUMV6Time"]:
                dim_q = self.target_item_embedding.shape[-1]
                self.query_att_W = tf.compat.v1.get_variable(
                    name="query_att_W",
                    shape=[self.hidden_size, dim_q],
                    initializer=self.initializer,
                )

            final_state = self._build_sum(cell)

            # for _p in cell.parameter_set:
            #     tf.compat.v1.summary.histogram(_p.name, _p)
            # if hasattr(cell, "_alpha") and hasattr(cell._alpha, "name"):
            #     tf.compat.v1.summary.histogram(cell._alpha.name, cell._alpha)
            # if hasattr(cell, "_beta") and hasattr(cell._beta, "name"):
            #     tf.compat.v1.summary.histogram(cell._beta.name, cell._beta)
            if hparams.cell in ["SUMV6","SUMV7","SUMV6Time"]:

                final_state, att_weights = self._attention_query_by_state_uiMutation(
                    final_state, self.target_item_embedding
                )


            else:
                final_state, att_weights = self._attention_query_by_state(
                    final_state, self.target_item_embedding
                )

            model_output = tf.concat([final_state, self.target_item_embedding], 1)
            self.beta_output = tf.zeros_like(self.iterator.labels)

            if "other_metrics_basedOn_RecList" in self.hparams.values():
                item_lookup=tf.concat([self.item_lookup,self.cateIdOfAllItems_embedding],-1)#[vocNum,H1+H2]
                allItemsEmd = tf.expand_dims(item_lookup, 0)  # [vocNum,H]==>[1,vocNum,H]
                final_state = tf.expand_dims(final_state, 1)  # [B,H]=>[B,1,H]
                final_state = final_state[::self.hparams.test_num_ngs + 1, :, :]
                vocNum = allItemsEmd.shape[1]
                final_state = tf.tile(final_state, [1, vocNum, 1])
                allItemsEmd = tf.tile(allItemsEmd, [tf.shape(final_state)[0], 1, 1])
                self.recItemListScoreModelOut = tf.concat([final_state, allItemsEmd], -1)

                # [B,vocNum,1]
            # model_output = tf.concat([final_state, self.target_item_embedding,self.user_embedding], 1)
            # tf.compat.v1.summary.histogram("model_output", model_output)
        return model_output

    def eval(self, sess, feed_dict):
        """Evaluate the data in `feed_dict` with current model.

        Args:
            sess (object): The model session object.
            feed_dict (dict): Feed values for evaluation. This is a dictionary that maps graph elements to values.

        Returns:
            list: A list of evaluated results, including total loss value, data loss value, predicted scores, and ground-truth labels.
        """
        feed_dict[self.layer_keeps] = self.keep_prob_test
        feed_dict[self.is_train_stage] = False
        if "other_metrics_basedOn_RecList" not in self.hparams.values():
            return sess.run([self.pred, self.iterator.labels], feed_dict=feed_dict)
        else:
            return sess.run([self.pred, self.iterator.labels, self.recItemListScore, self.iterator.items,
                             self.iterator.item_history, self.item_lookup,self.cateIdOfAllItems_embedding], feed_dict=feed_dict)

    def getLongInterest(self,long_type="long_term_asvd"):
        '''
        attention sum,merge hist
        Returns:shortInterest
        '''
        if long_type=="long_term_asvd":
            with tf.compat.v1.variable_scope("long_term_asvd"):
                if self.hparams.useTimeInterval:
                    if self.hparams.cell in ["SUMV6Time"]:
                        history_embedding = self.history_embedding[:, :, :-2]  # [B,T,H]
                    else:
                        history_embedding=self.history_embedding[:,:,:-1]#[B,T,H+1]
                att_outputs1 = self._attention(history_embedding, self.hparams.attention_size)
                longInterest = tf.reduce_sum(input_tensor=att_outputs1, axis=1)
                # tf.compat.v1.summary.histogram("att_fea1", longInterest)
        elif long_type=="personal_intst_act":
            if "personal_intst_act" in self.hparams.values() and self.hparams.personal_intst_act:
                query=self.target_item_embedding
                self.long_inst_prj_w=tf.compat.v1.get_variable(
                    name="long_inst_prj_w", shape=[self.hidden_size, query.shape[-1]], initializer=self.initializer
                )
                att_weights = tf.nn.softmax(
                    tf.squeeze(
                        tf.matmul(
                            tf.tensordot(self.pers_actv_inst, self.long_inst_prj_w, axes=1),#[B,S,H]=> B S D
                            tf.expand_dims(query, -1),#[B,D,1]
                        ),
                        -1,
                    ),
                    -1,
                )
                # merge the memory states, the final shape is (BatchSize, HiddenSize)
                longInterest = tf.reduce_sum(
                    input_tensor=self.pers_actv_inst * tf.expand_dims(att_weights, -1), axis=1
                )  # [B,H]
        return longInterest
    def _attention_query_by_state_uiMutation(self, seq_output, query):

        att_weights = tf.constant(1.0, dtype=tf.float32)
        with tf.compat.v1.variable_scope("query_att"):
            if self.hparams.slots > 1:
                # reshape the memory states to (BatchSize, Slots, HiddenSize)
                memory_state = tf.reshape(
                    seq_output, [-1, self.hparams.slots, self.hidden_size]
                )  # [B,S,H]
                print("=======tf.compat.v1.variable_scope(query_att): query shape=======")
                print(query.shape)
                print("=======tf.compat.v1.variable_scope(query_att): query shape=======")
                if "itemMutation" in self.hparams.values() and self.hparams.itemMutation:
                    att_weights = tf.nn.softmax(
                        tf.squeeze(
                            tf.matmul(
                                tf.tensordot(memory_state, self.query_att_W, axes=1),
                                tf.expand_dims(query, -1),
                            ),
                            -1,
                        ),
                        -1,
                    )#[B,S]
                    # merge the memory states, the final shape is (BatchSize, HiddenSize)
                    att_res = tf.reduce_sum(
                        input_tensor=memory_state * tf.expand_dims(att_weights, -1), axis=1
                    )  # [B,H]

                    with tf.compat.v1.name_scope("itemMutation_alpha_fuse"):
                        last_diff_idx = tf.cast(tf.reduce_sum(self.iterator.mask, keepdims=True, axis=-1),
                                                tf.int32) - 1  # [B,1]
                        row_idx = tf.expand_dims(tf.range(tf.shape(last_diff_idx)[0]), -1)
                        last_diff_idx = tf.concat([row_idx, last_diff_idx],
                                                  axis=-1)  # [B,2]
                        last_diff = tf.gather_nd(self.iterator.time_to_now, last_diff_idx)  # [B,]
                        last_diff = tf.expand_dims(last_diff, -1)  # [B,1]

                        hidden1 = tf.compat.v1.nn.xw_plus_b(self.target_item_embedding, self.hidden1_w, self.hidden1_b)
                        ideal_state = tf.compat.v1.nn.xw_plus_b(hidden1, self.hidden2_w, self.hidden2_b)
                        if self.hparams.needLongInterest:
                            longInterest = self.getLongInterest()  # [B,H]
                            fuseWay="2mlp"
                            if fuseWay=="1mlp":
                                att_res=tf.concat(
                                    [
                                        att_res,
                                        longInterest,
                                        ideal_state,
                                        last_diff,
                                    ],
                                    1,
                                )
                            if fuseWay == "2mlp":
                                concat_all = tf.concat(
                                    [
                                        self.target_item_embedding,
                                        att_res,
                                        longInterest,
                                        last_diff,
                                    ],
                                    1,
                                )
                                last_hidden_nn_layer = concat_all
                                alpha_logit = self._fcn_net(
                                    last_hidden_nn_layer, self.hparams.att_fcn_layer_sizes, scope="sum_fcn_alpha_fuse_longShort"
                                )
                                alpha_output = tf.sigmoid(alpha_logit)
                                att_res = att_res * alpha_output + longInterest * (1.0 - alpha_output)

                                concat_all = tf.concat(
                                    [
                                        att_res,
                                        ideal_state,
                                        last_diff,
                                    ],
                                    1,
                                )
                                last_hidden_nn_layer = concat_all
                                beta_logit = self._fcn_net(
                                    last_hidden_nn_layer, self.hparams.att_fcn_layer_sizes, scope="sum_fcn_alpha_fuse_selfColl"
                                )
                                beta_output = tf.sigmoid(beta_logit)
                                att_res = (att_res + ideal_state * beta_output) / (1 + beta_output)
                        else:
                            concat_all = tf.concat(
                                [
                                    self.target_item_embedding,
                                    att_res,# [B,H]
                                    ideal_state,
                                    last_diff,
                                ],
                                1,
                            )
                            last_hidden_nn_layer = concat_all
                            alpha_logit = self._fcn_net(
                                last_hidden_nn_layer, [64, 40], scope="fcn_alpha_fuse"
                            )
                            alpha_output = tf.sigmoid(alpha_logit)  # [B,1]
                            att_res = (att_res + ideal_state * alpha_output)/(1+alpha_output)

                elif "userCollMutation" in self.hparams.values() and self.hparams.userCollMutation:
                    user_self = self.user_embedding  # [b,h]

                    allUserEmd = self.user_lookup  # [VOC,H]
                    simMat = tf.matmul(user_self, allUserEmd, transpose_b=True)  # [b,voc]
                    print(simMat.shape)
                    median_sim_idx = self.top_k_around_median(simMat, 4)  # [B,K]
                    median_sim_user = tf.nn.embedding_lookup(
                        params=self.user_lookup, ids=median_sim_idx
                    )  # [B,K,H]
                    other_user_w = tf.reduce_sum(tf.expand_dims(user_self, axis=1) * median_sim_user, axis=-1,
                                                 keepdims=True)  # [B,1,H]*[B,K,H]==>[B,K,1]
                    queryUI = tf.tile(tf.expand_dims(self.target_item_embedding, 1),
                                    [1, tf.shape(median_sim_user)[1], 1])  # [B,1,H]=>[B,k,H]
                    queryUI = tf.concat([queryUI, median_sim_user], axis=-1)
                    realQuery = tf.compat.v1.nn.xw_plus_b(queryUI, self.attention_w, self.attention_b)  # [B,K,H]
                    # [B,D]  [poolNum,D]
                    pool = tf.expand_dims(self.ipPool, axis=0)  # [1,poolNum,D]
                    poolSoftBeta = 1.0
                    att_logit = tf.nn.softmax(
                        poolSoftBeta * tf.matmul(realQuery, pool, transpose_b=True)
                        , axis=-1)  # [B,k,poolNum]
                    att_logit = tf.expand_dims(att_logit, -1)  # [B,K,poolNum,1]
                    pool = tf.expand_dims(pool, 0)  # [1,1,poolNUm,d]
                    coll_muta_ip = tf.reduce_sum(pool * att_logit, axis=-2)  # [B,K,poolNum,D]=>[B,K,D]
                    coll_muta_ip = other_user_w * coll_muta_ip  # [B,k,1]*[B,k,D]
                    coll_muta_ip = tf.reduce_sum(coll_muta_ip, axis=1,keepdims=True)  # [B,1,D]

                    self_ip = tf.concat([user_self, self.target_item_embedding], axis=-1)  # [B,H1+H2]
                    self_ip_mapping = tf.compat.v1.nn.xw_plus_b(self_ip, self.attention_w, self.attention_b)
                    self_ip_mapping = tf.expand_dims(self_ip_mapping, axis=1)  # [B,1,D]
                    pool = tf.expand_dims(self.ipPool, axis=0)  # [1,poolNum,D]
                    att_logit = tf.nn.softmax(
                        poolSoftBeta * tf.reduce_sum(self_ip_mapping * pool, axis=-1, keepdims=True)
                        , axis=-2)  # [B,poolNum,1]
                    self_ip = tf.reduce_sum(pool * att_logit, axis=1,keepdims=True)  # [B,poolNum,D]=>[B,1,D]
                    # print(tf.concat([memory_state,self_ip,coll_muta_ip],axis=-2).shape,"======emory_state,self_ip,coll_muta_ip=====")#(None, 8, 40)

                    memory_state_self_ip_coll_muta_ip=tf.concat([memory_state,self_ip,coll_muta_ip],axis=-2)
                    memory_state_self_ip_coll_muta_ip=memory_state

                    att_weights = tf.nn.softmax(
                        tf.squeeze(
                            tf.matmul(
                                tf.tensordot(memory_state_self_ip_coll_muta_ip, self.query_att_W, axes=1),
                                tf.expand_dims(query, -1),
                            ),
                            -1,
                        ),
                        -1,
                    )
                    # merge the memory states, the final shape is (BatchSize, HiddenSize)
                    att_res = tf.reduce_sum(
                        input_tensor=memory_state_self_ip_coll_muta_ip * tf.expand_dims(att_weights, -1), axis=1
                    )  # [B,H]

                    # memory_state_self_ip_coll_muta_ip = tf.concat([self_ip, coll_muta_ip], axis=-2)
                    # att_weights = tf.nn.softmax(
                    #     tf.squeeze(
                    #         tf.matmul(
                    #             tf.tensordot(memory_state_self_ip_coll_muta_ip, query_att_W, axes=1),
                    #             tf.expand_dims(query, -1),
                    #         ),
                    #         -1,
                    #     ),
                    #     -1,
                    # )
                    # att_res = tf.reduce_sum(
                    #     input_tensor=memory_state_self_ip_coll_muta_ip * tf.expand_dims(att_weights, -1), axis=1
                    # )  # [B,H]

                    fuse_alpha = tf.maximum(self.cell._alpha, 0.15)
                    # fuse_alpha = tf.minimum(fuse_alpha, 1.0)
                    last_diff_idx = tf.cast(tf.reduce_sum(self.iterator.mask, keepdims=True, axis=-1),
                                            tf.int32) - 1  # [B,1]
                    row_idx = tf.expand_dims(tf.range(tf.shape(last_diff_idx)[0]), -1)

                    last_diff_idx = tf.concat([row_idx, last_diff_idx],
                                              axis=-1)  # [B,2]

                    last_diff = tf.gather_nd(self.iterator.time_to_now, last_diff_idx)  # [B,]

                    last_diff = tf.expand_dims(last_diff, -1)  # [B,1]
                    last_diff = tf.maximum(last_diff, 0.15)
                    # last_diff=tf.compat.v1.Print(last_diff,[tf.reduce_mean(last_diff)],summarize=1,message="last_diff")
                    fuse_alpha = tf.exp(-fuse_alpha * last_diff)  # [B,1]*
                    # fuse_alpha=tf.compat.v1.Print(fuse_alpha,[tf.reduce_mean(fuse_alpha)],summarize=1,message="fuse_alpha")
                    # att_res = (att_res + (coll_muta_ip + self_ip) * (1.0 - fuse_alpha)) / (2.0 - fuse_alpha)
                    # att_res = fuse_alpha*att_res + (coll_muta_ip+self_ip) * (1.0 - fuse_alpha)
                    # att_res=tf.concat([att_res,(coll_muta_ip+self_ip) * (1.0 - fuse_alpha)],axis=-1)

                else:
                    # memory_state = tf.compat.v1.Print(memory_state, [memory_state[:10, :,:5]], summarize=200,
                    #                                  message="=====OUT QUERY memory_state=====\n")
                    att_weights = tf.squeeze(
                            tf.matmul(
                                tf.tensordot(memory_state, self.query_att_W, axes=1),
                                tf.expand_dims(query, -1),
                            ),
                            -1,
                        )#[B,S]
                    is_zero_vector = tf.reduce_all(tf.equal(memory_state,0), axis=2)#[B,S,H]=> B S
                    zero_state_padding=tf.ones_like(att_weights)* (-(2 ** 32) + 1)#[B,S]
                    att_weights=tf.compat.v1.where(is_zero_vector, zero_state_padding, att_weights)
                    att_weights=tf.nn.softmax(att_weights,axis=-1)# B S
                    # att_weights = tf.compat.v1.Print(att_weights, [att_weights[:20, :]], summarize=80,
                    #                                  message="=====OUT QUERY Att logit=====\n")

                    # merge the memory states, the final shape is (BatchSize, HiddenSize)

                    att_res = tf.reduce_sum(
                        input_tensor=memory_state * tf.expand_dims(att_weights, -1), axis=1
                    )  # [B,H]
                    # att_res = tf.compat.v1.Print(att_res, [att_res[:10, :5]], summarize=50,
                    #                              message="=====OUT QUERY overall state =====\n")
                    if self.hparams.needLongInterest:
                        longInterest = self.getLongInterest("personal_intst_act")  # [B,H]
                        att_res=tf.concat([att_res,longInterest,att_res*longInterest],axis=-1)

                if self.hparams.useTimeInterval:
                    if self.hparams.useperson_coll_diff_beta:
                        h_pool = self.cell.ipPoolQuery(query, self.ipPool)  # [B,D]
                        D_h_hat = att_res.shape[-1]
                        # comp JS divergence
                        person_coll_diff_beta = self.cell.JS(tf.nn.softmax(att_res, axis=-1),
                                                             tf.nn.softmax(h_pool, axis=-1))  # =>[B,1]
                        att_res = (att_res + h_pool * person_coll_diff_beta) / (1.0 + person_coll_diff_beta)

                    elif "time_stable_aware_fuse" in self.hparams.values() and self.hparams.time_stable_aware_fuse:
                        h_pool = self.cell.ipPoolQuery(query, self.ipPool)  # [B,D]
                        D_h_hat = att_res.shape[-1]
                        with tf.compat.v1.name_scope("alpha"):
                            diff_mask = tf.cast(self.iterator.mask, dtype=tf.float32)  # [B,L]
                            time_diff_mean = tf.reduce_sum(self.iterator.time_diff, axis=-1,
                                                           keepdims=True) / tf.reduce_sum(diff_mask, axis=-1,
                                                                                          keepdims=True)  # [B,1]
                            len_sub_1 = tf.reduce_sum(diff_mask, axis=-1, keepdims=True) - 1  # [B,1]
                            diff_sub_mean = diff_mask * time_diff_mean  # [B,L]
                            time_diff_var = tf.reduce_sum(tf.square(self.iterator.time_diff - diff_sub_mean), axis=-1,
                                                          keepdims=True) / len_sub_1  # [B,1]
                            # \sum(x-x_)@2/N-1
                            time_diff_var = tf.sqrt(time_diff_var)  # [B,]

                            decayFactor = tf.maximum(self.time_stable_aware_fuse_exp_factor, 0.15)  # [B,1]
                            # decayFactor = tf.minimum(decayFactor, 1.0)
                            # decayFactor=1.0
                            alpha_output = tf.exp(-time_diff_var * decayFactor)  # [B,1]
                            # alpha_output=tf.expand_dims(alpha_output,-1)

                            att_res = (att_res + h_pool * (1.0 - alpha_output)) / (2.0 - alpha_output)

                    elif "attention_fcn_alpha_fuse" in self.hparams.values() and self.hparams.attention_fcn_alpha_fuse:
                        h_pool = self.cell.ipPoolQuery(query, self.ipPool)  # [B,D]
                        D_h_hat = att_res.shape[-1]
                        last_diff_idx = tf.cast(tf.reduce_sum(self.iterator.mask, keepdims=True, axis=-1),
                                                tf.int32) - 1  # [B,1]
                        last_diff_idx = tf.stack(
                            [tf.expand_dims(tf.range(tf.shape(last_diff_idx)[0]), -1), last_diff_idx], axis=1)  # [B,2]
                        last_diff = tf.gather_nd(self.iterator.time_to_now, last_diff_idx)  # [B,]
                        last_diff = tf.expand_dims(last_diff, -1)  # [B,1]
                        last_diff = tf.maximum(last_diff, 0.15)  # [B,1]
                        with tf.compat.v1.name_scope("alpha"):
                            concat_all = tf.concat(
                                [
                                    self.target_item_embedding,
                                    att_res,
                                    h_pool,
                                    last_diff,
                                ],
                                1,
                            )
                            last_hidden_nn_layer = concat_all
                            alpha_logit = self._fcn_net(
                                last_hidden_nn_layer, self.hparams.att_fcn_layer_sizes, scope="fcn_alpha"
                            )
                            alpha_output = tf.sigmoid(alpha_logit)
                            att_res = att_res * alpha_output + h_pool * (1.0 - alpha_output)


            else:
                att_res = seq_output

        return att_res, att_weights

    def personal_intst_act(self):
        with tf.compat.v1.variable_scope("sum_cell_intst_actv"):
            hist_self_att=self._fcn_net(self.item_history_embedding,[64], scope="sum_cell_intst_actv_fcn")#[B,T,1]
            boolean_mask = tf.equal(self.iterator.mask, tf.ones_like(self.iterator.mask))#[B,T]
            mask_paddings = tf.ones_like(hist_self_att) * (-(2 ** 32) + 1)#[B,T,1]
            hist_self_att = tf.nn.softmax(
                tf.compat.v1.where(tf.expand_dims(boolean_mask,-1), hist_self_att, mask_paddings),
                name="att_weights",axis=1
            )#[B,T,1] along T axis
            virtual_item=tf.reduce_sum(hist_self_att*self.item_history_embedding,axis=1)#None,T, H=> B H
            print("virtual_item shape,", virtual_item.shape)
            ipNum = self.hparams.ipNum
            self.intstPool = tf.compat.v1.get_variable(
                name="intstPool", shape=[ipNum, self.hparams.hidden_size], initializer=self.initializer
            )
            # self.intstPool=tf.compat.v1.Print(self.intstPool,[self.intstPool[:10,:10]],summarize=100,message="=====interest pool=====\n")

            def intst_actv(query, ipPool):
                '''

                Args:
                    query:  [,dim]
                    ipPool: [num_ip,dim]

                Returns:

                '''
                query_prj_expd = tf.expand_dims(query, axis=1)  # [B,1,D]
                pool = tf.expand_dims(ipPool, axis=0)  # [1,poolNum,D]
                print("pool shape",ipPool.shape)
                simMat = tf.matmul(query, ipPool, transpose_b=True)  # [b,voc]
                print("simMat shape",simMat.shape)
                median_sim_idx = tf.argsort(simMat, direction='DESCENDING')[:,:self.hparams.slots]  # [B,K]
                print("median_sim_idx shape", median_sim_idx.shape)
                median_sim_ip = tf.nn.embedding_lookup(
                    params=ipPool, ids=median_sim_idx
                )  # [B,K,H]
                print("median_sim_ip shape",median_sim_ip.shape)
                # tf.reduce_sum(query_prj_expd*median_sim_ip,axis=-1,keepdims=True)#[B 1 D] [B K D]=>[B,K,1]
                att_logit = tf.reduce_sum(query_prj_expd*median_sim_ip,axis=-1,keepdims=True)# [B,K,1] along K
                att_logit= tf.nn.sigmoid(att_logit)# [B,K,1]
                collb_ip =median_sim_ip * att_logit  # [B,K,D]
                print("collb_ip shape",collb_ip.shape)
                return collb_ip

            self.pers_actv_inst=intst_actv(virtual_item,self.intstPool)# [B,K,D]

    def _attention_query_by_state(self, seq_output, query):
        """Merge a user's memory states conditioned by a query item.

        Params:
            seq_output: A flatten representation of SUM memory states for (a batch of) users
            query: (a batch of) target item candidates

        Returns:
            tf.Tensor, tf.Tensor: Merged user representation. Attention weights of each memory channel.
        """
        dim_q = query.shape[-1]
        att_weights = tf.constant(1.0, dtype=tf.float32)
        with tf.compat.v1.variable_scope("query_att"):
            if self.hparams.slots > 1:
                query_att_W = tf.compat.v1.get_variable(
                    name="query_att_W",
                    shape=[self.hidden_size, dim_q],
                    initializer=self.initializer,
                )
                # reshape the memory states to (BatchSize, Slots, HiddenSize)
                memory_state = tf.reshape(
                    seq_output, [-1, self.hparams.slots, self.hidden_size]
                )#[B,S,H]
                print("=======tf.compat.v1.variable_scope(query_att): query shape=======")
                print(query.shape)
                print("=======tf.compat.v1.variable_scope(query_att): query shape=======")
                att_weights = tf.nn.softmax(
                    tf.squeeze(
                        tf.matmul(
                            tf.tensordot(memory_state, query_att_W, axes=1),
                            tf.expand_dims(query, -1),
                        ),
                        -1,
                    ),
                    -1,
                )
                # merge the memory states, the final shape is (BatchSize, HiddenSize)
                att_res = tf.reduce_sum(
                    input_tensor=memory_state * tf.expand_dims(att_weights, -1), axis=1
                )#[B,H]



                if self.hparams.useTimeInterval:
                    if self.hparams.useperson_coll_diff_beta:
                        h_pool = self.cell.ipPoolQuery(query, self.ipPool)  # [B,D]
                        D_h_hat = att_res.shape[-1]
                        # comp JS divergence
                        person_coll_diff_beta = self.cell.JS(tf.nn.softmax(att_res, axis=-1),
                                                        tf.nn.softmax(h_pool, axis=-1))  # =>[B,1]
                        att_res=(att_res+h_pool*person_coll_diff_beta)/(1.0+person_coll_diff_beta)

                    elif "time_stable_aware_fuse" in self.hparams.values() and self.hparams.time_stable_aware_fuse:
                        h_pool = self.cell.ipPoolQuery(query, self.ipPool)  # [B,D]
                        D_h_hat = att_res.shape[-1]
                        with tf.compat.v1.name_scope("alpha"):
                            diff_mask=tf.cast(self.iterator.mask, dtype=tf.float32)#[B,L]
                            time_diff_mean = tf.reduce_sum(self.iterator.time_diff,axis=-1,keepdims=True)/tf.reduce_sum(diff_mask,axis=-1,keepdims=True)#[B,1]
                            len_sub_1=tf.reduce_sum(diff_mask,axis=-1,keepdims=True)-1# [B,1]
                            diff_sub_mean=diff_mask*time_diff_mean#[B,L]
                            time_diff_var=tf.reduce_sum(tf.square(self.iterator.time_diff-diff_sub_mean),axis=-1,keepdims=True)/len_sub_1#[B,1]
                            #\sum(x-x_)@2/N-1
                            time_diff_var=tf.sqrt(time_diff_var)#[B,]

                            decayFactor=tf.maximum(self.time_stable_aware_fuse_exp_factor,0.15)#[B,1]
                            # decayFactor = tf.minimum(decayFactor, 1.0)
                            # decayFactor=1.0
                            alpha_output=tf.exp(-time_diff_var*decayFactor)#[B,1]
                            # alpha_output=tf.expand_dims(alpha_output,-1)

                            att_res = (att_res + h_pool * (1.0 - alpha_output) )/(2.0-alpha_output)

                    elif "attention_fcn_alpha_fuse" in self.hparams.values() and self.hparams.attention_fcn_alpha_fuse:
                        h_pool = self.cell.ipPoolQuery(query, self.ipPool)  # [B,D]
                        D_h_hat = att_res.shape[-1]
                        last_diff_idx = tf.cast(tf.reduce_sum(self.iterator.mask,keepdims=True,axis=-1), tf.int32) - 1  # [B,1]
                        last_diff_idx = tf.stack([tf.expand_dims(tf.range(tf.shape(last_diff_idx)[0]),-1), last_diff_idx], axis=1)# [B,2]
                        last_diff = tf.gather_nd(self.iterator.time_to_now, last_diff_idx)  # [B,]
                        last_diff = tf.expand_dims(last_diff, -1)  # [B,1]
                        last_diff = tf.maximum(last_diff, 0.15)# [B,1]
                        with tf.compat.v1.name_scope("alpha"):
                            concat_all = tf.concat(
                                [
                                    self.target_item_embedding,
                                    att_res,
                                    h_pool,
                                    last_diff,
                                ],
                                1,
                            )
                            last_hidden_nn_layer = concat_all
                            alpha_logit = self._fcn_net(
                                last_hidden_nn_layer, self.hparams.att_fcn_layer_sizes, scope="fcn_alpha"
                            )
                            alpha_output = tf.sigmoid(alpha_logit)
                            att_res = att_res * alpha_output + h_pool * (1.0 - alpha_output)

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
                    def row_wise_softmax(x):
                        # 对每行的非零元素进行softmax
                        nonzero = tf.cast(tf.math.not_equal(x, 0), x.dtype)
                        e_x = tf.exp(x) * nonzero
                        row_sums = tf.reduce_sum(e_x, axis=1, keepdims=True)
                        return e_x / row_sums

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
                    realQuery = tf.compat.v1.nn.xw_plus_b(query, self.attention_w, self.attention_b)  # [B,K,H]
                    # [B,D]  [poolNum,D]
                    pool = tf.expand_dims(self.ipPool, axis=0)  # [1,poolNum,D]
                    poolSoftBeta = 1.0
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


                    fuse_alpha = tf.maximum(self.cell._alpha, 0.15)
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
                    att_res = (att_res + (coll_muta_ip+self_ip) * (1.0 - fuse_alpha) )/(2.0-fuse_alpha)
                    # att_res = fuse_alpha*att_res + (coll_muta_ip+self_ip) * (1.0 - fuse_alpha)
                    #att_res=tf.concat([att_res,(coll_muta_ip+self_ip) * (1.0 - fuse_alpha)],axis=-1)

                # if "useShareIpPool" in self.hparams.values() and self.hparams.useShareIpPool:
                #     self.ipPool=tf.compat.v1.Print(self.ipPool,[self.ipPool[:2,:10]],message="=======self.ipPool:\n",summarize=20)
                    # ip_pool = self.cell.ipPoolQuery(query, self.ipPool)  # [B,H]
                    # att_res = (att_res + ip_pool) / 2

            else:
                att_res = seq_output

        return att_res, att_weights

    def _create_sumcell(self):
        """Create a SUM cell
        Returns:
            object: An initialized SUM cell
        """
        hparams = self.hparams
        input_embedding_dim = self.history_embedding.shape[-1]#itemDim+cateDim
        if hparams.cell in ["SUMV6Time"]:
            input_params = [
                hparams.hidden_size * hparams.slots,
                hparams.slots,
                hparams.attention_size,
                input_embedding_dim,
            ]
        else:
            input_params = [
                hparams.hidden_size * hparams.slots + input_embedding_dim,
                hparams.slots,
                hparams.attention_size,
                input_embedding_dim,
            ]

        sumcells = {"SUM": SUMCell, "SUMV2": SUMV2Cell,"SUMV3": SUMV3Cell,"SUMV4": SUMV4Cell,"SUMV5":SUMV5Cell,"SUMV6":SUMV6Cell,"SUMV7":SUMV7Cell,
                    "SUMV6Time":SUMV6TimeCell}
        sumCell = sumcells[hparams.cell]
        res = None
        if hparams.cell in ["SUM", "SUMV2","SUMV3","SUMV4","SUMV5","SUMV6","SUMV7","SUMV6Time"]:
            if "useShareIpPool" in hparams.values() and hparams.useShareIpPool:
                ipNum = self.hparams.ipNum
                self.ipPool = tf.compat.v1.get_variable(
                    name="ipPool", shape=[ipNum, self.hidden_size], initializer=self.initializer
                )
                res = sumCell(*input_params,kernel_initializer=None,hparams=self.hparams,
                              layer_keeps=self.layer_keeps,ipPool=self.ipPool)
            elif "personal_intst_act" in self.hparams.values() and self.hparams.personal_intst_act:
                res = sumCell(*input_params, kernel_initializer=None, hparams=self.hparams,
                              layer_keeps=self.layer_keeps,readHeads=self.pers_actv_inst)
            else:
                res = sumCell(*input_params, kernel_initializer=None, hparams=self.hparams,
                              layer_keeps=self.layer_keeps)
        else:
            raise ValueError("ERROR! Cell type not support: {0}".format(hparams.cell))
        return res

    def top_k_around_median(self,x, k):
        # 计算中位数
        # k=10
        # median = tfp.stats.percentile(x, 100.0, interpolation='nearest', axis=-1,
        #                               keepdims=True)  # [B,1]
        # print("median:{}".format(median.shape))
        # # 计算每个元素与中位数的差值的绝对值
        # abs_diff = tf.abs(x - median)  # [B,H]
        # print("abs_diff:{}".format(abs_diff.shape))
        # # 找到差值最小的K个元素的索引
        top_k = tf.argsort(x, direction='DESCENDING')[:, :k]  # [B,K]
        print("top_k shape",top_k.shape)
        return top_k

    def _build_sum(self, cell):
        """Generate  user memory states from behavior sequence

        Args:
            object: An initialied SUM cell.

        Returns:
            object: A flatten representation of user memory states, in the shape of (BatchSize, SlotsNum x HiddenSize)
        """

        hparams = self.hparams
        # self._persInitStateW= tf.compat.v1.get_variable(
        #     name="_persInitStateW",
        #     shape=[hparams.user_embedding_dim, cell._num_units],
        #     initializer=self.initializer,
        # )#[userEmdDim,stateDim]


        with tf.compat.v1.variable_scope("sum_cell_init"):

            # if "persInitState" in self.hparams.values() and self.hparams.persInitState:
            #     with tf.compat.v1.variable_scope('persInitStatelayer1'):
            #         self.persInit1_w=tf.compat.v1.get_variable("persInit1_w", shape=[self.hparams.slots, cell._real_units],
            #                         initializer=self.initializer)#[S,D]

                    # self.persInit1_b=tf.compat.v1.get_variable("persInit1_b", shape=[cell._real_slots*cell._real_units],
                    #                 initializer=self.initializer)

                #
                # with tf.compat.v1.variable_scope('persInitStateoutput'):
                #     self.persInitOut_w=tf.compat.v1.get_variable("persInitOut_w", shape=[32, cell._num_units],
                #                     initializer=self.initializer)
                #     self.persInitOut_b=tf.compat.v1.get_variable("persInitOut_b", shape=[cell._num_units],
                #                     initializer=self.initializer)

            if "shareInitState" in self.hparams.values() and self.hparams.shareInitState:
                self.shareInitState = tf.compat.v1.get_variable(
                    name="shareInitState",
                    shape=[1,cell._num_units],
                    initializer=self.initializer,
                )  # [userEmdDim,stateDim]

        with tf.compat.v1.variable_scope("sum"):
            if self.hparams.history_cat_target:
                realHistMask=tf.concat([self.iterator.mask,
                                        tf.zeros(shape=[tf.shape(self.iterator.mask)[0],1],dtype=tf.int32)],-1)#[None,max_seq_length]  #[B,T+1]
                self.mask = tf.concat([tf.ones(shape=[tf.shape(self.iterator.mask)[0],1],dtype=tf.int32),
                                       self.iterator.mask],-1)#[None,max_seq_length]  #[B,T+1]
                self.sequence_length = tf.reduce_sum(input_tensor=self.mask, axis=1)#[None,]
                target_item_embedding_with_time_diff=tf.expand_dims(
                        tf.concat([self.target_item_embedding,
                                   tf.expand_dims(self.iterator.time_diff[:,-1],-1)
                                   ],-1)
                    ,1)#[B,1,D+1]
                rum_outputs, final_state = dynamic_rnn(
                    cell,
                    inputs=tf.concat([self.history_embedding,target_item_embedding_with_time_diff],1),
                    dtype=tf.float32,
                    sequence_length=self.sequence_length,
                    scope="sum",
                    initial_state=cell.zero_state(
                        tf.shape(input=self.history_embedding)[0], tf.float32
                    ),
                    # initial_state=tf.matmul(self.user_embedding)
                )
                #get final state
                final_state_out_mask=tf.cast(self.mask-realHistMask,dtype=tf.float32)#[B,T+1]
                query_state_hat=rum_outputs[:,:,:hparams.hidden_size]*tf.expand_dims(final_state_out_mask,-1)#[b,t+1,d]
                final_state=tf.reduce_sum(query_state_hat,axis=1)#[B,D]

                #get aux loss
                realHistMask=tf.cast(realHistMask,dtype=tf.float32)#[B,T+1]
                self.time_interval_evo_loss=tf.reduce_sum(rum_outputs[:,:,-2]*realHistMask)#[B,T]==>1
                self.person_coll_diff_loss= tf.reduce_sum(rum_outputs[:, :, -1] * realHistMask)#[B,T]=>1
            else:

                if "persInitState" in self.hparams.values() and self.hparams.persInitState:
                    # initState = tf.matmul(self.user_embedding, self._persInitStateW)
                    # self.initState=tf.nn.elu(tf.compat.v1.nn.xw_plus_b(self.user_embedding,self.persInit1_w,self.persInit1_b), name='initState')
                    #[B,H]==>[B,S*H]   inSize]
                    if self.hparams.useTimeInterval:
                        if self.hparams.cell in ["SUMV6Time"]:
                            history_embedding = self.history_embedding[:, :, :-2]  # [B,T,H]
                        else:
                            history_embedding=self.history_embedding[:,:,:-1]#[B,T,H]
                    else:
                        history_embedding=self.history_embedding
                    initStateScore = tf.matmul(history_embedding, self.persInit1_w,
                                               transpose_b=True)  # [B,T,H] [S,H]=>[B,T,S]
                    mask = (tf.cast(self.iterator.mask,
                                    dtype=tf.float32) - 1) * 1e6  # [B,T]  [1,0]=>[0,-1]*1e6   [0,-1e6]
                    initStateScore = initStateScore + tf.expand_dims(mask, axis=-1)  # [B,T,S]*[B,T,1]
                    initStateScore = tf.nn.softmax(initStateScore, axis=-1)  # [B,T,S] along T
                    initStateScore = tf.expand_dims(initStateScore, -1)  # [B,T,S,1]
                    history_embedding = tf.expand_dims(history_embedding, axis=-2)  # [B,T,1,H]
                    initState = tf.reduce_sum(history_embedding * initStateScore, axis=1)  # [B,T,S,H]#=>[B,S,H]
                    self.initState = tf.reshape(initState, [-1, self.hparams.slots * cell._real_units])
                    if self.hparams.cell in ["SUMV6Time"]:
                        initZeros = tf.zeros(
                            shape=[tf.shape(self.initState)[0], cell.state_size - cell._real_units * cell._real_slots])
                        initState = tf.concat([self.initState,initZeros], axis=-1)
                    elif cell.state_size-cell._real_units*cell._real_slots >0:
                        initZeros=tf.zeros(shape=[tf.shape(self.initState)[0],cell.state_size-cell._real_units*cell._real_slots])
                        initState=tf.concat([self.initState,initZeros],axis=-1)
                    else:
                        initState=self.initState
                    # initState=tf.compat.v1.Print(initState,[ tf.reshape(initState[0,:-self.history_embedding.shape[-1]],
                    #                                                     shape=[self.hparams.slots,self.hparams.hidden_size])[:,:6]],
                    #                              summarize=36,message="initState")

                    initStateNorm=tf.nn.l2_normalize(tf.reshape(self.initState,[-1,cell._real_slots,cell._real_units]),-1)#[B,S,H]
                    initState_pair_sim = tf.matmul(initStateNorm, initStateNorm, transpose_b=True)  # [B,S,S]
                    ii_initState_sim = tf.linalg.diag_part(initState_pair_sim)  # [B,S]
                    ij_initState_sim = initState_pair_sim - tf.linalg.diag(ii_initState_sim)  # i_i val=>0,  [B,S,S]
                    initState_regu_l = tf.reduce_mean(ij_initState_sim) # [B,S,S]==> scalar,
                    # factor=b*t*n*(n-1)
                    self.disInitState=initState_regu_l
                elif "shareInitState" in self.hparams.values() and self.hparams.shareInitState:
                    initState = tf.tile(
                        self.shareInitState, [tf.shape(input=self.history_embedding)[0], 1]
                    )  # [B,numUnits]
                else:
                    initState=cell.zero_state(
                        tf.shape(input=self.history_embedding)[0], tf.float32
                    )

                self.mask = self.iterator.mask  # [None,max_seq_length]
                self.sequence_length = tf.reduce_sum(input_tensor=self.mask, axis=1)  # [None,]
                rum_outputs, final_state = dynamic_rnn(
                    cell,
                    inputs=self.history_embedding,
                    dtype=tf.float32,
                    sequence_length=self.sequence_length,
                    scope="sum",
                    initial_state=initState
                )
                if self.hparams.cell in ["SUMV6Time"]:
                    final_state = array_ops.slice(final_state, [0, cell._num_units], [-1, cell._num_units])  # [b,S*h]

                if self.hparams.useTimeInterval and self.hparams.useperson_coll_diff_beta:
                    self.person_coll_diff_loss= tf.reduce_sum(rum_outputs[:, :, 0] * tf.cast(self.mask,tf.float32))#[B,T]=>1
                final_state = final_state[:, : hparams.slots * hparams.hidden_size]

                if "item_ip_coll" in self.hparams.values() and self.hparams.item_ip_coll:
                    with tf.compat.v1.variable_scope("itemIpCollVar"):
                        seq_len=self.history_embedding.shape[1]#T
                        dropFirstHist = self.item_history_embedding[:, 1:, :]  # [B,T-1,D]
                        states = tf.reshape(rum_outputs[:, :, : hparams.slots * hparams.hidden_size],
                                            [tf.shape(dropFirstHist)[0], seq_len, cell._real_slots, cell._real_units])  # [B,T,S*H]=>[..,S,H]
                        states = tf.matmul(states, self.query_att_W)[:, :-1, :, :]  # [B,T,S,H]=>[B,T-1,S,D]
                        print("=====states===shape==: {}".format(states.shape))
                        print("=====dropFirstHist===shape==: {}".format(tf.expand_dims(dropFirstHist,-2).shape))

                    with tf.compat.v1.variable_scope("inner_prd"):
                        inner_prd=states* tf.expand_dims(dropFirstHist,-2)#[B,T-1,S,D]*[B,T-1,1,D]=>[B,T-1,S,D]
                    with tf.compat.v1.variable_scope("att_weights"):
                        att_weights=tf.nn.softmax(tf.reduce_sum(inner_prd,axis=-1,keepdims=True),axis=-2)#[B,T-1,S,D]=>[B,T-1,S,1] along slot_num
                        states_t = tf.reduce_sum(
                            input_tensor=states * att_weights, axis=-2
                        )  # [B,T-1,S,D]=>[B,T-1,D]

                    query = dropFirstHist
                    query_size = query.shape[-1]
                    self.hidden1_w = tf.compat.v1.get_variable(name="uitem_ip_coll_h1_w",
                                                               shape=[query_size, 64],
                                                               initializer=self.initializer, )
                    self.hidden1_b = tf.compat.v1.get_variable(name="item_ip_coll_h1_b", shape=[64],
                                                               initializer=self.initializer, )

                    self.hidden2_w = tf.compat.v1.get_variable(name="uitem_ip_coll_h2_w",
                                                               shape=[64, self.hparams.hidden_size],
                                                               initializer=self.initializer, )
                    self.hidden2_b = tf.compat.v1.get_variable(name="item_ip_coll_h2_b",
                                                               shape=[self.hparams.hidden_size],
                                                               initializer=self.initializer, )
                    hidden1 = tf.compat.v1.nn.xw_plus_b(query, self.hidden1_w, self.hidden1_b)
                    ideal_state = tf.compat.v1.nn.xw_plus_b(hidden1, self.hidden2_w, self.hidden2_b)

                    normalize_ip = tf.nn.l2_normalize(ideal_state, -1)
                    normalize_states_t = tf.nn.l2_normalize(states_t, -1)  # [B,T-1,H]
                    coll_muta_ip_ssl_loss = 1 - tf.reduce_sum(normalize_ip * normalize_states_t, -1)  # [B,T-1] 1-cosSim
                    coll_muta_ip_ssl_loss = coll_muta_ip_ssl_loss * tf.cast(self.iterator.mask[:, 1:], tf.float32)
                    self.coll_muta_ip_ssl_loss = tf.reduce_sum(coll_muta_ip_ssl_loss) / tf.reduce_sum(
                        tf.cast(self.iterator.mask[:, 1:], tf.float32))  # scalar loss
                    self.coll_muta_ip_ssl_loss = self.hparams.coll_muta_ip_ssl_regu * self.coll_muta_ip_ssl_loss

                if "interestDisReg" in self.hparams.values() and self.hparams.interestDisReg:
                    states = rum_outputs[:, :, : hparams.slots * hparams.hidden_size]  # [B,T,H*S]
                    st_sp=tf.shape(states)
                    self.interestDisRegLoss=self._ip_reg_l(tf.reshape(states,[-1,st_sp[1],hparams.slots,hparams.hidden_size]))#[B,T,S,H]
                    # self.interestDisRegLoss = tf.compat.v1.Print(self.interestDisRegLoss, [self.interestDisRegLoss], "self.interestDisRegLoss===: \n")
                if self.hparams.cell in ["SUMV6","SUMV2"]:
                    if "time_diff_sum" in self.hparams.values() and self.hparams.time_diff_sum:
                        states=rum_outputs[:, :,: hparams.slots * hparams.hidden_size]#[B,T,H*S]
                        factor = tf.maximum(self.cell._alpha, 0.15)
                        # factor = tf.minimum(factor, 1.0)
                        sumFactor=tf.exp(-factor*self.iterator.time_to_now)*tf.cast(self.iterator.mask,dtype=tf.float32)#[B,T]
                        sumFactor=sumFactor/tf.reduce_sum(sumFactor,axis=-1,keepdims=True)#[B,T]/[B,1]
                        sumState=states*tf.expand_dims(sumFactor,-1)#[B,T,h*s]*[B,T,1]
                        final_state=tf.reduce_sum(sumState,axis=1)+final_state#[B,T,s*h]
                    if "ui_ip_coll" in self.hparams.values() and self.hparams.ui_ip_coll:
                        uesr_seq_expd=tf.expand_dims(self.user_embedding,axis=1)#[B,1,H]
                        dropFirstHist=self.item_history_embedding[:,1:,:]#[B,T,H]

                        ui_cat=tf.concat([tf.tile(uesr_seq_expd,[1,tf.shape(dropFirstHist)[1],1]),
                                          dropFirstHist],axis=-1)#[B,T-1,H],[B,T-1,H]
                        query=ui_cat
                        query_size = query.shape[-1]
                        self.attention_w = tf.compat.v1.get_variable(name="ui_ip_coll_w",
                                                                shape=[query_size, hparams.hidden_size],
                                                                initializer=self.initializer, )
                        self.attention_b = tf.compat.v1.get_variable(name="ui_ip_coll_b", shape=[hparams.hidden_size],
                                                                initializer=self.initializer, )
                        realQuery = tf.compat.v1.nn.xw_plus_b(query, self.attention_w, self.attention_b)#[B,T-1,H]=>[B,T-1,h2]
                        # [B,D]  [poolNum,D]
                        pool = tf.expand_dims(self.ipPool, axis=0)  # [1,poolNum,D]
                        poolSoftBeta = 1.0
                        att_logit = tf.nn.softmax(
                            poolSoftBeta*tf.matmul(realQuery,pool,transpose_b=True)
                            , axis=-1)  # [B,T-1,poolNum]
                        att_logit=tf.expand_dims(att_logit,-1)#[B,T-1,poolNum,1]
                        pool=tf.expand_dims(pool,0)#[1,1,poolNUm,d]
                        coll_muta_ip = tf.reduce_sum(pool * att_logit, axis=-2)  # [B,t-1,poolNum,D]=>[B,t-1,D]

                        states_t=tf.reduce_sum(tf.reshape(states,shape=[-1,tf.shape(states)[1],hparams.slots,hparams.hidden_size]),
                                               axis=-2)#[B,T,H]a

                        normalize_ip = tf.nn.l2_normalize(coll_muta_ip, -1)
                        normalize_states_t = tf.nn.l2_normalize(states_t, -1)[:,:-1,:]#[B,T-1,H]

                        coll_muta_ip_ssl_loss=1-tf.reduce_sum(normalize_ip*normalize_states_t,-1)#[B,T-1] 1-cosSim
                        coll_muta_ip_ssl_loss=coll_muta_ip_ssl_loss*tf.cast(self.iterator.mask[:,1:],tf.float32)
                        self.coll_muta_ip_ssl_loss=tf.reduce_sum(coll_muta_ip_ssl_loss)/tf.reduce_sum(tf.cast(self.iterator.mask[:,1:],tf.float32))#scalar loss
                        self.coll_muta_ip_ssl_loss=self.hparams.coll_muta_ip_ssl_regu*self.coll_muta_ip_ssl_loss
                        self.coll_muta_ip_ssl_loss=tf.compat.v1.Print( self.coll_muta_ip_ssl_loss,[self.coll_muta_ip_ssl_loss],summarize=1,message="coll_muta_ip_ssl_loss")
                        self.ipPool=tf.compat.v1.Print(self.ipPool,[self.ipPool[:5,:5]],summarize=25,message="====ipPool===: ")
                        normalize_ip = tf.nn.l2_normalize(self.ipPool, -1)#[N,H]
                        ip_pair_sim = tf.matmul(normalize_ip, normalize_ip, transpose_b=True)  # [N,N]
                        ii_ip_sim = tf.linalg.diag_part(ip_pair_sim)  # [N]
                        ij_ip_sim = ip_pair_sim - tf.linalg.diag(ii_ip_sim)  # i_i val=>0, [N,N]
                        self.ipPool_sim= tf.reduce_mean(ij_ip_sim+1)*0.01
                        # self.ipPool_sim=tf.compat.v1.Print(self.ipPool_sim,[self.ipPool_sim],summarize=1,message="====ipPool_sim===: ")
                        # self.ipPool = tf.compat.v1.Print(self.ipPool, [self.ipPool[:4,:4]], summarize=16,
                        #                                      message="====ipPool===: ")
                        # factor=b*t*n*(n-1)

        # self.heads = cell.heads
        # self.alpha = cell._alpha
        # self.beta = cell._beta
        # tf.compat.v1.summary.histogram("SUM_outputs", rum_outputs)
        return final_state

    def _ip_reg_l(self,ip_states):
        # ip_states:[B,T,S,H]
        if self.hparams.interestDisReg=="mutInfo" :
            normalize_ip = tf.nn.l2_normalize(ip_states, -1)
            ip_pair_sim = tf.matmul(normalize_ip, normalize_ip, transpose_b=True)  # [B,T,S,S]
            temprature = 1.1
            ip_pair_sim = tf.math.exp(ip_pair_sim / temprature)
            ii_ip_sim = tf.linalg.diag_part(ip_pair_sim)  # [B,T,S]
            ij_ip_sim = ip_pair_sim - tf.linalg.diag(ii_ip_sim)  # i_i val=>0,  [B,T,S,S]
            ij_ip_sim_sum = tf.reduce_sum(ij_ip_sim, -1)  # [B,T,S]
            ip_regu_l = tf.reduce_sum(-tf.math.log(ii_ip_sim / ij_ip_sim_sum))/self.hparams.slots  # [B,T,S]==> scalar
        elif self.hparams.interestDisReg == "disSim":
            normalize_ip = tf.nn.l2_normalize(ip_states, -1)
            ip_pair_sim = tf.matmul(normalize_ip, normalize_ip, transpose_b=True)  # [B,T,S,S]
            ii_ip_sim = tf.linalg.diag_part(ip_pair_sim)  # [B,T,S]
            ij_ip_sim = ip_pair_sim - tf.linalg.diag(ii_ip_sim)  # i_i val=>0,  [B,T,S,S]
            ij_ip_sim=ij_ip_sim*tf.expand_dims(tf.expand_dims(tf.cast(self.iterator.mask,tf.float32),-1),-1)#[b,t,1,1]
            ip_regu_l = tf.reduce_sum(ij_ip_sim)/(tf.reduce_sum(tf.cast(self.iterator.mask,tf.float32))*self.hparams.slots*(self.hparams.slots-1))  # [B,T,S,S]==> scalar,
            #factor=b*t*n*(n-1)
        else:
            return 0.0
        return self.hparams.ip_diff_regu*ip_regu_l

    def _compute_regular_loss(self):
        """Construct regular loss. Usually it's comprised of l1 and l2 norm.
        Users can designate which norm to be included via config file.

        Returns:
            object: Regular loss.
        """

        regular_loss = self._l2_loss() + self._l1_loss() + self._cross_l_loss()
        if self.hparams.useTimeInterval and self.hparams.useperson_coll_diff_beta:
            # self.person_coll_diff_loss=tf.compat.v1.Print(self.person_coll_diff_loss,[self.person_coll_diff_loss],message="====self.person_coll_diff_loss\n")
            regular_loss = regular_loss + self.hparams.person_coll_diff_regu*self.person_coll_diff_loss
                           # +self.hparams.time_interval_evo_regu*self.time_interval_evo_loss
        if "interestDisReg" in self.hparams.values() and self.hparams.interestDisReg:
            regular_loss=regular_loss+ self.interestDisRegLoss
        if "ui_ip_coll" in self.hparams.values() and self.hparams.ui_ip_coll:
            regular_loss = regular_loss + self.coll_muta_ip_ssl_loss+self.ipPool_sim
        if "item_ip_coll" in self.hparams.values() and self.hparams.item_ip_coll:
            regular_loss = regular_loss + self.coll_muta_ip_ssl_loss
        if "persInitState" in self.hparams.values() and self.hparams.persInitState:
            regular_loss=regular_loss+self.disInitState
        return tf.reduce_sum(input_tensor=regular_loss)