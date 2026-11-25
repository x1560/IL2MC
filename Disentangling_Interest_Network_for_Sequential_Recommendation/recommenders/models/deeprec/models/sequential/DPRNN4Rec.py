# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
# from keras.layers.legacy_rnn.rnn_cell_impl import GRUCell, LSTMCell
from keras.legacy_tf_layers.core import dense
from tensorflow.python.keras.layers.legacy_rnn.rnn_cell_impl import LSTMCell, GRUCell
from tensorflow.python.ops import variable_scope as vs

from recommenders.models.deeprec.models.sequential.rnn_cell_implement import Time4LSTMCell
from recommenders.models.deeprec.models.sequential.sequential_base_model import (
    SequentialBaseModel,
)
import tensorflow_probability as tfp
from tensorflow.compat.v1.nn import dynamic_rnn
import numpy as np
__all__ = ["DPRNN4RecModel"]


class DPRNN4RecModel(SequentialBaseModel):
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
            # final_state = self._build_lstm()
            final_state = self._build_dprnn()
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
            return final_state[1]

    def _build_dprnn_block(self,input,hop=2,scope_block="dprnn_block"):
        '''

        Args:
            input: [B,T,H]
            out: [B,T,H]
            mask:[B,T,H]
        Returns:
        '''
        input=tf.concat(
                [
                    input,
                    tf.expand_dims(self.iterator.time_from_first_action, -1),
                    tf.expand_dims(self.iterator.time_to_now, -1),
                ],
                -1,
            )

        org_batch=tf.shape(input)[0]
        org_max_T=input.shape[1]
        org_hidden_size=input.shape[-1]
        padding_num=hop-org_max_T%hop
        pad_max_T=org_max_T+padding_num
        input=tf.concat([input,tf.zeros(shape=[org_batch,padding_num,org_hidden_size])],axis=1)#[B,T+padNum,H]
        time_from_first_action_pad=tf.concat([self.iterator.time_from_first_action,tf.zeros(shape=[org_batch,padding_num])],axis=-1)
        time_to_now_pad=tf.concat([self.iterator.time_to_now,tf.zeros(shape=[org_batch,padding_num])],axis=-1)

        mask_pad=tf.concat([self.iterator.mask,tf.zeros(shape=[org_batch,padding_num],dtype=tf.int32)],axis=1)#[B,T+padNum]
        history_embedding_step_hop = tf.reshape(input,[-1, hop, input.shape[-1]])  # [b*t/hop,hop,dim]
        mask_step_hop_intra = tf.reshape(mask_pad, [-1, hop])  # [b*t/hop,hop]
        rnn_outputs_step_hop_intra, final_state_step_hop = dynamic_rnn(
            Time4LSTMCell(self.hidden_size),
            inputs=history_embedding_step_hop,
            sequence_length=tf.reduce_sum(input_tensor=mask_step_hop_intra, axis=1),
            dtype=tf.float32,
            scope="gru_step_hop_intra_"+scope_block,  # 子序列内部
        )  # [B*T/2,2,H]
        rnn_outputs_step_hop_intra = tf.reshape(rnn_outputs_step_hop_intra, [org_batch,pad_max_T,self.hidden_size])  # [B,T,H]
        needFC=False
        if needFC:
            layer1_w_intra = vs.get_variable(
                "layer1_w_intra"+scope_block, shape=[rnn_outputs_step_hop_intra.shape[-1],rnn_outputs_step_hop_intra.shape[-1]],dtype=tf.float32
            )
            layer1_b_intra = vs.get_variable(
                "layer1_b_intra"+scope_block, shape=[rnn_outputs_step_hop_intra.shape[-1]],dtype=tf.float32
            )
            layer2_w_intra = vs.get_variable(
                "layer2_w_intra"+scope_block, shape=[rnn_outputs_step_hop_intra.shape[-1],rnn_outputs_step_hop_intra.shape[-1]],dtype=tf.float32
            )
            layer2_b_intra = vs.get_variable(
                "layer2_b_intra"+scope_block, shape=[rnn_outputs_step_hop_intra.shape[-1]],dtype=tf.float32
            )
            rnn_outputs_step_hop_intra=tf.nn.relu(tf.compat.v1.nn.xw_plus_b(rnn_outputs_step_hop_intra,layer1_w_intra,layer1_b_intra))
            # rnn_outputs_step_hop_intra=tf.compat.v1.nn.xw_plus_b(rnn_outputs_step_hop_intra,layer2_w_intra,layer2_b_intra)

        rnn_outputs_step_hop_intra_org = rnn_outputs_step_hop_intra  # [B,T,H]
        print("rnn_outputs_step_hop_intra_org,",rnn_outputs_step_hop_intra_org.shape)
        rnn_outputs_step_hop_intra = tf.concat([rnn_outputs_step_hop_intra,tf.expand_dims(time_from_first_action_pad,-1)
                                              ,tf.expand_dims(time_to_now_pad,-1)], -1)  # [B,T,H+2]
        rnn_outputs_step_hop_intra = tf.reshape(rnn_outputs_step_hop_intra, [org_batch,pad_max_T//hop,hop,rnn_outputs_step_hop_intra.shape[-1]])

        mask_step_hop_inter = tf.reshape(mask_pad, [org_batch,pad_max_T//hop,hop])  # [B,T//2,2]

        rnn_outputs_step_hop_intra = tf.transpose(rnn_outputs_step_hop_intra, perm=[0, 2, 1, 3])  # [B,2,T/2,H]
        mask_step_hop_inter = tf.transpose(mask_step_hop_inter, perm=[0, 2, 1])  # [B,2,T/2]

        rnn_outputs_step_hop_intra = tf.reshape(rnn_outputs_step_hop_intra, [-1,pad_max_T//hop,
                                                                         rnn_outputs_step_hop_intra.shape[
                                                                             -1]])  # [B*2,T/2,H]
        mask_step_hop_inter = tf.reshape(mask_step_hop_inter, [-1,pad_max_T//hop])  # [B*2,T/2]

        rnn_outputs_step_hop_inter, final_state_step_hop_inter = dynamic_rnn(
            Time4LSTMCell(self.hidden_size),
            inputs=rnn_outputs_step_hop_intra,
            sequence_length=tf.reduce_sum(input_tensor=mask_step_hop_inter, axis=1),
            dtype=tf.float32,
            scope="gru_step_hop_inter_"+scope_block,  # 子序列之间
        )  # [B*2,T/2,H]

        dprnnOut = tf.keras.layers.LayerNormalization(axis=-1)(rnn_outputs_step_hop_inter)

        dprnnOut = tf.reshape(dprnnOut, [-1,hop, pad_max_T//hop,
                                         dprnnOut.shape[
                                             -1]])  # [B,2,T/2,H]
        dprnnOut = tf.transpose(dprnnOut, perm=[0, 2, 1, 3])  # [B,T/2,2,H]
        dprnnOut = tf.reshape(dprnnOut,[org_batch,pad_max_T,self.hidden_size])  # [B,T,H]
        dprnnOut = dprnnOut + rnn_outputs_step_hop_intra_org
        dprnnOut=dprnnOut[:,:org_max_T,:]
        print("dprnnOut.shape", dprnnOut.shape)
        return  dprnnOut
    def _build_dprnn(self):
        """Apply a GRU for modeling.

        Returns:
            object: The output of GRU section.
        """
        with tf.compat.v1.name_scope("dprnn"):
            self.mask = self.iterator.mask
            self.sequence_length = tf.reduce_sum(input_tensor=self.mask, axis=1)
            with tf.compat.v1.variable_scope("long_term_asvd"):
                att_outputs1 = self._attention(self.item_history_embedding, self.hparams.attention_size)
                att_fea1 = tf.reduce_sum(input_tensor=att_outputs1, axis=1)

            self.history_embedding = self.item_history_embedding#[B,T,H]
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
            # #step=1
            with tf.compat.v1.variable_scope("rnn"):
                rnn_outputs_hop_1, _ = dynamic_rnn(
                    Time4LSTMCell(self.hparams.hidden_size),
                    inputs=item_history_embedding_new,
                    sequence_length=self.sequence_length,
                    dtype=tf.float32,
                    scope="time4lstm",
                )#[B,T,H]
            #step=
            blockCombineWay='stack'
            if blockCombineWay=='stack':
                dprnnOut_1=rnn_outputs_hop_1
                dprnnOut_2=self._build_dprnn_block(input=self.item_history_embedding,hop=2,scope_block="hop_2")#[B,T,H]
                dprnnOut_4=self._build_dprnn_block(input=dprnnOut_2,hop=4,scope_block="hop_4")#[B,T,H]
                dprnnOut_8 = self._build_dprnn_block(input=dprnnOut_4, hop=8, scope_block="hop_8")  # [B,T,H]
            elif blockCombineWay=='same_level':
                dprnnOut_1 = rnn_outputs_hop_1
                dprnnOut_2 = self._build_dprnn_block(input=self.item_history_embedding, hop=2,scope_block="hop_2")  # [B,T,H]
                dprnnOut_4 = self._build_dprnn_block(input=self.item_history_embedding, hop=4, scope_block="hop_4")  # [B,T,H]
                dprnnOut_8 = self._build_dprnn_block(input=self.item_history_embedding, hop=8, scope_block="hop_8")  # [B,T,H]
            fuseWay="concat"#sum concat
            if fuseWay=="sum":
                dprnnOut=dprnnOut_1+dprnnOut_2+dprnnOut_4
            elif fuseWay=="concat":
                dprnnOut=tf.concat([dprnnOut_1,dprnnOut_2,dprnnOut_4,dprnnOut_8],axis=-1)
            with tf.compat.v1.variable_scope("attention_fcn"):
                att_outputs2 = self._attention_fcn(
                    self.target_item_embedding, dprnnOut
                )
                att_fea2 = tf.reduce_sum(input_tensor=att_outputs2, axis=1)#[B,H]

            with tf.compat.v1.name_scope("alpha"):
                last_diff_idx = tf.cast(tf.reduce_sum(self.iterator.mask, keepdims=True, axis=-1),
                                        tf.int32) - 1  # [B,1]
                row_idx = tf.expand_dims(tf.range(tf.shape(last_diff_idx)[0]), -1)
                last_diff_idx = tf.concat([row_idx, last_diff_idx],
                                          axis=-1)  # [B,2]
                last_diff = tf.gather_nd(self.iterator.time_to_now, last_diff_idx)  # [B,]
                last_diff = tf.expand_dims(last_diff, -1)  # [B,1]
                org_diff = tf.expand_dims(self.iterator.time_to_now[:, -1], -1)
                concat_all = tf.concat(
                    [
                        self.target_item_embedding,
                        att_fea1,
                        att_fea2,
                        last_diff,
                    ],
                    1,
                )
                print("concat_all.shape", concat_all.shape)
                print("att_fea1.shape", att_fea1.shape)
                print("att_fea2.shape", att_fea2.shape)

                last_hidden_nn_layer = concat_all
                alpha_logit = self._fcn_net(
                    last_hidden_nn_layer, self.hparams.att_fcn_layer_sizes, scope="fcn_alpha"
                )
                alpha_output = tf.sigmoid(alpha_logit)
                dprnn_user_embed = tf.concat([att_fea1 * alpha_output , att_fea2 * (1.0 - alpha_output)],axis=-1)


                print("user_embed.shape", dprnn_user_embed.shape)
            if "other_metrics_basedOn_RecList" in self.hparams.values():
                item_lookup = tf.concat([self.item_lookup, self.cateIdOfAllItems_embedding], -1)  # [vocNum,H1+H2]
                allItemsEmd = tf.expand_dims(item_lookup, 0)  # [vocNum,H]==>[1,vocNum,H]
                user_embed = tf.expand_dims(dprnn_user_embed, 1)  # [B,H]=>[B,1,H]
                user_embed = user_embed[::self.hparams.test_num_ngs + 1, :, :]
                vocNum = allItemsEmd.shape[1]
                user_embed = tf.tile(user_embed, [1, vocNum, 1])#[B,vocNum,H]
                allItemsEmd = tf.tile(allItemsEmd, [tf.shape(user_embed)[0], 1, 1])
                self.recItemListScoreModelOut = tf.concat([user_embed, allItemsEmd], -1)#[,32,h=40]

            return dprnn_user_embed
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

        return tf.reduce_sum(input_tensor=regular_loss)