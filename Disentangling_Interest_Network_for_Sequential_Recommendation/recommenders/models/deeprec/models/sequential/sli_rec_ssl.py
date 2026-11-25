# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
import numpy as np

from recommenders.models.deeprec.deeprec_utils import load_dict
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
    def _build_embedding(self):
        """The field embedding layer. Initialization of embedding variables."""
        super(SLI_RECModel, self)._build_embedding()
        hparams = self.hparams
        self.item_vocab_length = len(load_dict(hparams.item_vocab))
        self.item_embedding_dim = hparams.item_embedding_dim
        with tf.compat.v1.variable_scope("embedding", initializer=self.initializer):
            self.item_mutation_lookup = tf.compat.v1.get_variable(
                name="item_mutation_embedding",
                shape=[self.item_vocab_length, hparams.item_embedding_dim],
                dtype=tf.float32,
            )
            self.cate_mutation_lookup = tf.compat.v1.get_variable(
                name="cate_mutation_embedding",
                shape=[self.cate_vocab_length, self.cate_embedding_dim],
                dtype=tf.float32,
            )

    def _lookup_from_embedding(self):
        """Lookup from embedding variables. A dropout layer follows lookup operations.
        """
        super(SLI_RECModel, self)._lookup_from_embedding()

        self.item_mutation_embedding = tf.nn.embedding_lookup(
            self.item_mutation_lookup, self.iterator.items
        )
        self.padd_ssl_state=tf.concat([tf.nn.embedding_lookup(
            self.item_mutation_lookup, tf.zeros_like(self.iterator.items)
        ),tf.nn.embedding_lookup(
            self.cate_mutation_lookup, tf.zeros_like(self.iterator.cates)
        )],axis=-1)

        self.cate_mutation_embedding = tf.compat.v1.nn.embedding_lookup(
            params=self.cate_mutation_lookup, ids=self.iterator.cates
        )  # [B,]  OR [B,max_cate_num]
        self.embed_params.append(self.item_mutation_embedding)
        self.embed_params.append(self.cate_mutation_embedding)
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
                if "itemMutation" in self.hparams.values() and self.hparams.itemMutation:
                    ideal_state,self.coll_muta_ip_ssl_loss=self.item_mutation_ssl(user_embed)
                    #
                    # fuse_way="alpha_sum"# alpha_sum concat
                    # if fuse_way=="alpha_sum":
                    #     concat_all = tf.concat(
                    #         [
                    #             user_embed,
                    #             ideal_state,
                    #             user_embed*ideal_state,
                    #             user_embed+ideal_state,
                    #             last_diff,
                    #         ],
                    #         1,
                    #     )
                    #     last_hidden_nn_layer = concat_all
                    #     beta_logit = self._fcn_net(
                    #         last_hidden_nn_layer, hparams.att_fcn_layer_sizes, scope="fcn_beta"
                    #     )
                    #     beta_output = tf.sigmoid(beta_logit)
                    #     user_embed=(user_embed+beta_output*ideal_state)/(1+beta_output)
                    #
                    #     #padding item ssl
                    #     concat_all = tf.concat(
                    #         [
                    #             user_embed,
                    #             self.padd_ssl_state,
                    #             user_embed * self.padd_ssl_state,
                    #             user_embed + self.padd_ssl_state,
                    #             last_diff,
                    #         ],
                    #         1,
                    #     )
                    #     last_hidden_nn_layer = concat_all
                    #     gama_logit = self._fcn_net(
                    #         last_hidden_nn_layer, hparams.att_fcn_layer_sizes, scope="fcn_beta",reuse=True
                    #     )
                    #     gama_output = tf.sigmoid(gama_logit)#[B,]
                    #     self.pad_state_ssl_loss=self.hparams.coll_muta_ip_ssl_regu*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(gama_output), logits=gama_output))
                    self.pad_state_ssl_loss=0.0
                    # elif fuse_way=="concat":
                    #     user_embed=tf.concat([user_embed,ideal_state,user_embed*ideal_state,ideal_state+user_embed], 1)
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

    def item_mutation_ssl(self,user_embed):
        #user_emd [b, h]
        pos_negItem=self.target_item_embedding#[B,H]
        pos_negItem=tf.reshape(pos_negItem,[-1,1+self.iterator.batch_num_ngs,pos_negItem.shape[-1]])
        posItem_embedding=pos_negItem[:,0,:]#[B/1+negNum,H]
        negItem_embedding=pos_negItem[:,1:,:]#[B/1+negNum,negNum,H]
        # query_size=pos_negItem.shape[-1]
        # ideal_state = self._fcn_net(
        #     tf.reshape(pos_negItem, [-1, query_size]), [80,40],activations=["relu","relu"], scope="idealStateEncoder",
        #     _out_layer_size=self.hparams.hidden_size
        # )
        ideal_state=tf.concat([self.item_mutation_embedding,self.cate_mutation_embedding],axis=-1)#[B,H1+H2]
        print("ideal_state shape",ideal_state.shape)
        normalize_ideal_state = tf.nn.l2_normalize(ideal_state, -1)#[B,H]
        normalize_pred_user_emd = tf.nn.l2_normalize(user_embed, -1)#[B,H]
        ssl_way="bpr"
        if ssl_way=="bpr":
            normalize_ideal_state=tf.reshape(normalize_ideal_state,[-1,1+self.iterator.batch_num_ngs,pos_negItem.shape[-1]])
            #[B,1+num_negs,H]
            normalize_pred_user_emd=tf.reshape(normalize_pred_user_emd,[-1,1+self.iterator.batch_num_ngs,pos_negItem.shape[-1]])[:,0,:]
            # [B,H]
            pos_normalize_ideal_state=tf.expand_dims(normalize_ideal_state[:,0,:],axis=1)#[B,1,H]
            neg_normalize_ideal_state=normalize_ideal_state[:,1:,:]#[B,num_negs,H]
            normalize_pred_user_emd=tf.expand_dims(normalize_pred_user_emd,axis=1)#[B,1,H]
            neg_anchor_inner_pair=normalize_pred_user_emd*neg_normalize_ideal_state#[B,num_negs,H]
            pos_anchor_inner_pair=normalize_pred_user_emd*pos_normalize_ideal_state#[B,num_negs,H]
            coll_muta_ip_ssl_loss=self.hparams.coll_muta_ip_ssl_regu*tf.reduce_mean(tf.nn.softplus(tf.reduce_sum(
                neg_anchor_inner_pair-pos_anchor_inner_pair,axis=-1
            )))
        elif ssl_way=="easy_equal":
            pos_neg_mask=tf.cast(self.iterator.labels,dtype=tf.float32)#[1,0,0]
            cosSim_pair=tf.reduce_sum(normalize_ideal_state * normalize_pred_user_emd, -1)  # [B,]
            revs_cossim=cosSim_pair*(-2.0)*pos_neg_mask+cosSim_pair#[-cosSim,cosSim] # [B,]
            coll_muta_ip_ssl_loss = self.hparams.coll_muta_ip_ssl_regu*tf.reduce_mean(revs_cossim)
        return ideal_state,coll_muta_ip_ssl_loss
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
        if "item_ip_coll" in self.hparams.values() and self.hparams.item_ip_coll:
            regular_loss = regular_loss + self.coll_muta_ip_ssl_loss+self.pad_state_ssl_loss
        if "reconstruct_item" in self.hparams.values() and self.hparams.reconstruct_item:
            regular_loss = regular_loss + self.recons_item_ssl_loss
        return tf.reduce_sum(input_tensor=regular_loss)