# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
# from keras.layers.legacy_rnn.rnn_cell_impl import GRUCell, LSTMCell
from keras.legacy_tf_layers.core import dense
from tensorflow.python.keras.layers.legacy_rnn.rnn_cell_impl import LSTMCell, GRUCell

from recommenders.models.deeprec.models.sequential.sequential_base_model import (
    SequentialBaseModel,
)
import tensorflow_probability as tfp
from tensorflow.compat.v1.nn import dynamic_rnn
import numpy as np
__all__ = ["GRU4RecModel"]


class GRU4RecModel(SequentialBaseModel):
    """GRU4Rec Model

    :Citation:

        B. Hidasi, A. Karatzoglou, L. Baltrunas, D. Tikk, "Session-based Recommendations
        with Recurrent Neural Networks", ICLR (Poster), 2016.
    """

    def _build_seq_graph(self):
        """The main function to create GRU4Rec model.

        Returns:
            object:the output of GRU4Rec section.
        """
        with tf.compat.v1.variable_scope("gru4rec"):
            if "useShareIpPool" in self.hparams.values() and self.hparams.useShareIpPool:
                ipNum = self.hparams.ipNum
                self.ipPool = tf.compat.v1.get_variable(
                    name="ipPool", shape=[ipNum, self.hidden_size], initializer=self.initializer
                )
            # final_state = self._build_lstm()
            final_state = self._build_gru()
            model_output = tf.concat([final_state, self.target_item_embedding], 1)
            # tf.compat.v1.summary.histogram("model_output", model_output)
            return model_output
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
            if "userEmdAuxLoss" in self.hparams.values() and self.hparams.userEmdAuxLoss:
                item_lookup = tf.concat([self.item_lookup, self.cateIdOfAllItems_embedding], -1)  # [vocNum,H1+H2]
                userEmd = tf.concat([tf.zeros(shape=[tf.shape(self.user_embedding)[0],self.hparams.hidden_size]),
                                                self.user_embedding], axis=-1)#[B,H1+H2]

                userEmd=tf.tile(tf.expand_dims(userEmd,axis=1),[1,tf.shape(self.history_embedding)[1],1])#[B,T,H1+H2]

                posHistItem=self.history_embedding#[B,T,H]
                negHistItem=tf.compat.v1.random_uniform(shape=[tf.shape(posHistItem)[0],posHistItem.shape[1]], minval=1, maxval=self.item_lookup.shape[0], dtype=tf.float32)
                negHistItem = tf.cast(negHistItem, dtype=tf.int32)
                negHistItem=tf.nn.embedding_lookup(
                    params=item_lookup, ids=negHistItem
                )  # [B,T,H]
                posNegHistItem=tf.concat([tf.expand_dims(posHistItem,-2),tf.expand_dims(negHistItem,-2)],axis=-2)#[B,T,2,H]
                userEmd=tf.tile(tf.expand_dims(userEmd,axis=-2),[1,1,2,1])#[B,T,2,H1+H2]
                userEmdPosNegHistItem=tf.concat([userEmd,posNegHistItem],axis=-1)

                recHistItemScore = self._fcn_net(userEmdPosNegHistItem, self.hparams.layer_sizes,
                                                      scope="logit_fcn",
                                                      reuse=True)#[B,T,2,1]
                recHistItemScore=tf.squeeze(recHistItemScore)#[B,T,2]

                poslabelHistScore=tf.cast(self.iterator.mask,tf.float32)#[B,T]
                posNegMask = tf.tile(tf.expand_dims(poslabelHistScore,-1),[1,1,2])#[B,T,1]=>[B,T,2]
                neglabelHistScore=tf.zeros_like(poslabelHistScore)#[B,T]
                posNegLabelHistScore=tf.concat([tf.expand_dims(poslabelHistScore,-1),tf.expand_dims(neglabelHistScore,-1)],axis=-1)#[B,T,2]
                recScore=recHistItemScore*posNegMask
                self.userEmdAuxLoss = tf.reduce_mean(
                    input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=tf.reshape(recScore, [-1]),
                        labels=tf.reshape(posNegLabelHistScore, [-1]),
                    )
                )
            if "other_metrics_basedOn_RecList" in self.hparams.values():
                self.recItemListScore = self._fcn_net(self.recItemListScoreModelOut, hparams.layer_sizes, scope="logit_fcn",
                                                  reuse=True)

            self._add_norm()
            return logit
    def _build_lstm(self):
        """Apply an LSTM for modeling.

        Returns:
            object: The output of LSTM section.
        """
        with tf.compat.v1.name_scope("lstm"):
            self.mask = self.iterator.mask
            self.sequence_length = tf.reduce_sum(input_tensor=self.mask, axis=1)
            self.history_embedding = tf.concat(
                [self.item_history_embedding, self.cate_history_embedding], 2
            )
            rnn_outputs, final_state = dynamic_rnn(
                LSTMCell(self.hidden_size),
                inputs=self.history_embedding,
                sequence_length=self.sequence_length,
                dtype=tf.float32,
                scope="lstm",
            )
            tf.compat.v1.summary.histogram("LSTM_outputs", rnn_outputs)
            return final_state[1]

    def _build_gru(self):
        """Apply a GRU for modeling.

        Returns:
            object: The output of GRU section.
        """
        with tf.compat.v1.name_scope("gru"):
            self.mask = self.iterator.mask
            self.sequence_length = tf.reduce_sum(input_tensor=self.mask, axis=1)
            self.history_embedding = self.item_history_embedding
            rnn_outputs, final_state = dynamic_rnn(
                GRUCell(self.hidden_size),
                inputs=self.history_embedding,
                sequence_length=self.sequence_length,
                dtype=tf.float32,
                scope="gru",
            )
            if "ui_ip_coll" in self.hparams.values() and self.hparams.ui_ip_coll:
                states=rnn_outputs#[B,S,H]
                uesr_seq_expd = tf.expand_dims(self.user_embedding, axis=1)  # [B,1,H]
                dropFirstHist = self.item_history_embedding[:, 1:, :]  # [B,T,H]

                ui_cat = tf.concat([tf.tile(uesr_seq_expd, [1, tf.shape(dropFirstHist)[1], 1]),
                                    dropFirstHist], axis=-1)  # [B,T-1,H],[B,T-1,H]
                query = ui_cat
                query_size = query.shape[-1]

                self.hidden1_w = tf.compat.v1.get_variable(name="uitem_ip_coll_h1_w",
                                                           shape=[query_size, 64],
                                                           initializer=self.initializer, )
                self.hidden1_b = tf.compat.v1.get_variable(name="item_ip_coll_h1_b", shape=[64],
                                                           initializer=self.initializer, )

                self.hidden2_w = tf.compat.v1.get_variable(name="uitem_ip_coll_h2_w",
                                                           shape=[64, self.hparams.hidden_size],
                                                           initializer=self.initializer, )
                self.hidden2_b = tf.compat.v1.get_variable(name="item_ip_coll_h2_b", shape=[self.hparams.hidden_size],
                                                           initializer=self.initializer, )
                hidden1 = tf.compat.v1.nn.xw_plus_b(query, self.hidden1_w, self.hidden1_b)
                coll_muta_ip = tf.compat.v1.nn.xw_plus_b(hidden1, self.hidden2_w, self.hidden2_b)

                states_t = states
                normalize_ip = tf.nn.l2_normalize(coll_muta_ip, -1)
                normalize_states_t = tf.nn.l2_normalize(states_t, -1)[:, :-1, :]  # [B,T-1,H]
                coll_muta_ip_ssl_loss = 1 - tf.reduce_sum(normalize_ip * normalize_states_t, -1)  # [B,T-1] 1-cosSim
                coll_muta_ip_ssl_loss = coll_muta_ip_ssl_loss * tf.cast(self.iterator.mask[:, 1:], tf.float32)
                self.coll_muta_ip_ssl_loss = tf.reduce_sum(coll_muta_ip_ssl_loss) / tf.reduce_sum(
                    tf.cast(self.iterator.mask[:, 1:], tf.float32))  # scalar loss
                self.coll_muta_ip_ssl_loss = self.hparams.coll_muta_ip_ssl_regu * self.coll_muta_ip_ssl_loss

            if "item_ip_coll" in self.hparams.values() and self.hparams.item_ip_coll:
                states=rnn_outputs#[B,S,H]
                dropFirstHist = self.item_history_embedding[:, 1:, :]  # [B,T-1,H]
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
                self.hidden2_b = tf.compat.v1.get_variable(name="item_ip_coll_h2_b", shape=[self.hparams.hidden_size],
                                                           initializer=self.initializer, )

                self._decoder_hidden_1_w = tf.compat.v1.get_variable(name="_decoder_uitem_ip_coll_h1_w",
                                                           shape=[self.hparams.hidden_size, 64],
                                                           initializer=self.initializer, )
                self._decoder_hidden1_b = tf.compat.v1.get_variable(name="_decoder_item_ip_coll_h1_b", shape=[64],
                                                           initializer=self.initializer, )

                self._decoder_hidden2_w = tf.compat.v1.get_variable(name="_decoder_uitem_ip_coll_h2_w",
                                                           shape=[64, query_size],
                                                           initializer=self.initializer, )
                self._decoder_hidden2_b = tf.compat.v1.get_variable(name="_decoder_item_ip_coll_h2_b", shape=[query_size],
                                                           initializer=self.initializer, )
                hidden1=tf.compat.v1.nn.xw_plus_b(query,self.hidden1_w,self.hidden1_b)
                ideal_state=tf.compat.v1.nn.xw_plus_b(hidden1,self.hidden2_w,self.hidden2_b)
                if "reconstruct_item" in self.hparams.values() and self.hparams.reconstruct_item:
                    decoder_hidden = tf.compat.v1.nn.xw_plus_b(ideal_state, self._decoder_hidden_1_w, self._decoder_hidden1_b)
                    recons_item = tf.compat.v1.nn.xw_plus_b(decoder_hidden, self._decoder_hidden2_w, self._decoder_hidden2_b)
                    normalize_recons_item = tf.nn.l2_normalize(recons_item, -1)#[B,T-1,H]
                    normalize_org_item = tf.nn.l2_normalize(dropFirstHist, -1) #[B,T-1,H]
                    recons_item_ssl_loss = 1 - tf.reduce_sum(normalize_recons_item * normalize_org_item, -1)  # [B,T-1] 1-cosSim
                    recons_item_ssl_loss = recons_item_ssl_loss * tf.cast(self.iterator.mask[:, 1:], tf.float32)
                    self.recons_item_ssl_loss = tf.reduce_sum(recons_item_ssl_loss) / tf.reduce_sum(
                        tf.cast(self.iterator.mask[:, 1:], tf.float32))  # scalar loss
                    self.recons_item_ssl_loss = self.hparams.coll_muta_ip_ssl_regu * self.recons_item_ssl_loss

                states_t = states
                normalize_ip = tf.nn.l2_normalize(ideal_state, -1)
                normalize_states_t = tf.nn.l2_normalize(states_t, -1)[:, :-1, :]  # [B,T-1,H]
                coll_muta_ip_ssl_loss = 1 - tf.reduce_sum(normalize_ip * normalize_states_t, -1)  # [B,T-1] 1-cosSim
                coll_muta_ip_ssl_loss = coll_muta_ip_ssl_loss * tf.cast(self.iterator.mask[:, 1:], tf.float32)
                self.coll_muta_ip_ssl_loss = tf.reduce_sum(coll_muta_ip_ssl_loss) / tf.reduce_sum(
                    tf.cast(self.iterator.mask[:, 1:], tf.float32))  # scalar loss
                self.coll_muta_ip_ssl_loss = self.hparams.coll_muta_ip_ssl_regu * self.coll_muta_ip_ssl_loss
            if "itemMutation" in self.hparams.values() and self.hparams.itemMutation:
                item_target = self.target_item_embedding  # [b,h]

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

                with tf.compat.v1.name_scope("alpha"):
                    last_diff_idx = tf.cast(tf.reduce_sum(self.iterator.mask, keepdims=True, axis=-1),
                                            tf.int32) - 1  # [B,1]
                    row_idx = tf.expand_dims(tf.range(tf.shape(last_diff_idx)[0]), -1)
                    last_diff_idx = tf.concat([row_idx, last_diff_idx],
                                              axis=-1)  # [B,2]
                    last_diff = tf.gather_nd(self.iterator.time_to_now, last_diff_idx)  # [B,]
                    last_diff = tf.expand_dims(last_diff, -1)  # [B,1]

                    hidden1 = tf.compat.v1.nn.xw_plus_b(self.target_item_embedding, self.hidden1_w, self.hidden1_b)
                    ideal_state = tf.compat.v1.nn.xw_plus_b(hidden1, self.hidden2_w, self.hidden2_b)
                    concat_all = tf.concat(
                        [
                            final_state,
                            ideal_state,
                            last_diff,
                        ],
                        1,
                    )
                    last_hidden_nn_layer = concat_all
                    alpha_logit = self._fcn_net(
                        last_hidden_nn_layer, [64,40], scope="fcn_alpha"
                    )
                    alpha_output = tf.sigmoid(alpha_logit)
                    final_state = final_state + ideal_state * alpha_output

            # tf.compat.v1.summary.histogram("GRU_outputs", rnn_outputs)
            if "other_metrics_basedOn_RecList" in self.hparams.values():
                item_lookup = tf.concat([self.item_lookup, self.cateIdOfAllItems_embedding], -1)  # [vocNum,H1+H2]
                allItemsEmd = tf.expand_dims(item_lookup, 0)  # [vocNum,H]==>[1,vocNum,H]
                user_embed = tf.expand_dims(final_state, 1)  # [B,H]=>[B,1,H]
                user_embed = user_embed[::self.hparams.test_num_ngs + 1, :, :]
                vocNum = allItemsEmd.shape[1]
                user_embed = tf.tile(user_embed, [1, vocNum, 1])#[B,vocNum,H]
                allItemsEmd = tf.tile(allItemsEmd, [tf.shape(user_embed)[0], 1, 1])
                self.recItemListScoreModelOut = tf.concat([user_embed, allItemsEmd], -1)#[,32,h=40]

            return final_state

    def _compute_regular_loss(self):
        """Construct regular loss. Usually it's comprised of l1 and l2 norm.
        Users can designate which norm to be included via config file.

        Returns:
            object: Regular loss.
        """

        regular_loss = self._l2_loss() + self._l1_loss() + self._cross_l_loss()
        if "ui_ip_coll" in self.hparams.values() and self.hparams.ui_ip_coll:
            regular_loss = regular_loss + self.coll_muta_ip_ssl_loss
        if "item_ip_coll" in self.hparams.values() and self.hparams.item_ip_coll:
            regular_loss = regular_loss + self.coll_muta_ip_ssl_loss
        if "reconstruct_item" in self.hparams.values() and self.hparams.reconstruct_item:
            regular_loss = regular_loss + self.recons_item_ssl_loss
        if "userEmdAuxLoss" in self.hparams.values() and self.hparams.userEmdAuxLoss:
            regular_loss = regular_loss +self.userEmdAuxLoss
        return tf.reduce_sum(input_tensor=regular_loss)