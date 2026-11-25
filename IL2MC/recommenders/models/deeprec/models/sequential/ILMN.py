# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
# from keras.layers.legacy_rnn.rnn_cell_impl import GRUCell, LSTMCell
from keras.legacy_tf_layers.core import dense
from tensorflow.python.keras.layers.legacy_rnn.rnn_cell_impl import LSTMCell, GRUCell
from tensorflow.python.ops import variable_scope as vs

from recommenders.models.deeprec.deeprec_utils import load_dict
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

    def _build_train_opt(self):
        """Construct gradient descent based optimization step
        In this step, we provide gradient clipping option. Sometimes we what to clip the gradients
        when their absolute values are too large to avoid gradient explosion.

        Returns:
            object: An operation that applies the specified optimization step.
        """
        train_step = self._train_opt()
        trainable_vars = tf.compat.v1.trainable_variables()
        unfreeze_keys=self.hparams.unfreeze_keys
        unfreeze_var_list=[]
        for trainable_var in trainable_vars:
            for unfreeze_key in unfreeze_keys:
                if unfreeze_key in trainable_var.name:
                    unfreeze_var_list.append(trainable_var)
        if len(unfreeze_var_list)==0:
            unfreeze_var_list=trainable_vars
        unfreeze_var_name_list=[t.name for t in unfreeze_var_list]
        print("\n=====unfreeze_var_name_list=====\n",unfreeze_var_name_list)
        gradients, variables = zip(*train_step.compute_gradients(self.loss,var_list=unfreeze_var_list))
        if self.hparams.is_clip_norm:
            gradients = [
                None
                if gradient is None
                else tf.clip_by_norm(gradient, self.hparams.max_grad_norm)
                for gradient in gradients
            ]
        return train_step.apply_gradients(zip(gradients, variables))
    def _build_embedding(self):
        """The field embedding layer. Initialization of embedding variables."""
        super(DPRNN4RecModel, self)._build_embedding()
        hparams = self.hparams
        self.item_vocab_length = len(load_dict(hparams.item_vocab))
        self.item_embedding_dim = hparams.item_embedding_dim
        if "itemMutation" in self.hparams.values() and self.hparams.itemMutation:
            with tf.compat.v1.variable_scope("embedding", initializer=self.initializer):
                self.item_mutation_lookup = tf.compat.v1.get_variable(
                    name="item_mutation_embedding",
                    shape=[self.item_vocab_length, (hparams.step_L+2)*hparams.item_embedding_dim],
                    dtype=tf.float32,
                )
                self.cate_mutation_lookup = tf.compat.v1.get_variable(
                    name="cate_mutation_embedding",
                    shape=[self.cate_vocab_length, (hparams.step_L+2)*self.cate_embedding_dim],
                    dtype=tf.float32,
                )

    def _lookup_from_embedding(self):
        """Lookup from embedding variables. A dropout layer follows lookup operations.
        """
        super(DPRNN4RecModel, self)._lookup_from_embedding()
        if "itemMutation" in self.hparams.values() and self.hparams.itemMutation:
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

    def _build_seq_graph(self):
        """The main function to create GRU4Rec model.

        Returns:
            object:the output of GRU4Rec section.
        """
        with tf.compat.v1.variable_scope("gru4rec"):
            # final_state = self._build_lstm()
            final_state = self._build_dprnn()
            model_output = tf.concat([final_state, self.target_item_embedding], 1)
            print("model out shape",model_output.shape)
            # tf.compat.v1.summary.histogram("model_output", model_output)
            return model_output
    def ssl_mutation_mlp(self):
        ssl_way="BCE"
        if ssl_way=="BCE":
            pos_ideal_state=tf.reshape(self.ideal_state,[-1,1+self.iterator.batch_num_ngs,self.ideal_state.shape[-1]])
            pos_ideal_state=tf.tile(tf.expand_dims(pos_ideal_state[:,0,:],axis=1),[1,1+self.iterator.batch_num_ngs,1])
            pos_ideal_state=tf.reshape(pos_ideal_state,[-1,self.ideal_state.shape[-1]])
            self._fcn_net(tf.concat([pos_ideal_state,self.target_item_embedding],axis=-1), self.hparams.layer_sizes, scope="logit_fcn",reuse=True)
            group = self.iterator.batch_num_ngs + 1
            logits = tf.reshape(self.logit, (-1, group))
            labels = tf.reshape(self.iterator.labels, (-1, group))
            softmax_pred = tf.nn.softmax(logits, axis=-1)
            boolean_mask = tf.equal(labels, tf.ones_like(labels))
            mask_paddings = tf.ones_like(softmax_pred)
            pos_softmax = tf.compat.v1.where(boolean_mask, softmax_pred, mask_paddings)
            coll_muta_ip_ssl_loss = -tf.cast(group,tf.float32) * tf.reduce_mean(input_tensor=tf.math.log(pos_softmax))
            self.coll_muta_ip_ssl_loss+=self.hparams.coll_muta_ip_ssl_regu*coll_muta_ip_ssl_loss
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

            # self.ssl_mutation_mlp()

            if "other_metrics_basedOn_RecList" in self.hparams.values():
                self.recItemListScore = self._fcn_net(self.recItemListScoreModelOut, hparams.layer_sizes, scope="logit_fcn",
                                                  reuse=True)

            self._add_norm()
            return logit

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
        if "needInterEvo" in self.hparams.values() and (not self.hparams.needInterEvo):
            dprnnOut=rnn_outputs_step_hop_intra_org
        else:
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
                dprnnOut_list=[]
                dprnnOut_list.append(rnn_outputs_hop_1)
                last_out=None
                for step in range(1,self.hparams.step_L+1):#0 1 2 L-1 L==>1 2 4 2^L-1 2^L
                    if step==1:
                        dprnnOut_step_i=self._build_dprnn_block(input=self.item_history_embedding,hop=2**step,scope_block="hop_{}".format(2**step))#[B,T,H]
                    else:
                        dprnnOut_step_i=self._build_dprnn_block(input=last_out,hop=2**step,scope_block="hop_{}".format(2**step))#[B,T,H]
                    last_out = dprnnOut_step_i
                    dprnnOut_list.append(last_out)

            elif blockCombineWay=='same_level':
                dprnnOut_1 = rnn_outputs_hop_1
                dprnnOut_2 = self._build_dprnn_block(input=self.item_history_embedding, hop=2,scope_block="hop_2")  # [B,T,H]
                dprnnOut_4 = self._build_dprnn_block(input=self.item_history_embedding, hop=4, scope_block="hop_4")  # [B,T,H]
                dprnnOut_8 = self._build_dprnn_block(input=self.item_history_embedding, hop=8, scope_block="hop_8")  # [B,T,H]
            fuseWay="concat"#sum concat
            if fuseWay=="sum":
                dprnnOut=dprnnOut_1+dprnnOut_2+dprnnOut_4
            elif fuseWay=="concat":
                # dprnnOut=tf.concat([dprnnOut_1,dprnnOut_2,dprnnOut_4,dprnnOut_8],axis=-1)
                if self.hparams.needMultiScaleLocality:
                    dprnnOut=tf.concat(dprnnOut_list,axis=-1)#L=3, [1,2,4,8]  [B,L,4*H]
                else:
                    lower_information_mask=tf.zeros_like(dprnnOut_list[-2])
                    lower_information_mask=tf.tile(lower_information_mask,[1,1,self.hparams.step_L-1])
                    dprnnOut=tf.concat([lower_information_mask,dprnnOut_list[-1],
                                        tf.zeros_like(dprnnOut_list[-2])],axis=-1)
            with tf.compat.v1.variable_scope("attention_fcn"):
                att_outputs2 = self._attention_fcn(
                    self.target_item_embedding, dprnnOut
                )#[B,T,step_L+1,H]
                att_fea2 = tf.reduce_sum(input_tensor=att_outputs2, axis=1)#[B,step_L+1,H]

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
                dprnn_user_embed = tf.concat([att_fea1 * alpha_output, att_fea2 * (1.0 - alpha_output)], axis=-1)
                # dprnn_user_embed = att_fea1 * alpha_output+ att_fea2 * (1.0 - alpha_output)
                # dprnn_user_embed = tf.concat([att_fea1 * alpha_output , att_fea2 * (1.0 - alpha_output)],axis=-1)
                if "itemMutation" in self.hparams.values() and self.hparams.itemMutation:
                    with tf.compat.v1.name_scope("itemMutation"):
                        # ideal_state, self.coll_muta_ip_ssl_loss = self.item_recons_mutation_ssl(dprnn_user_embed)
                        self.ideal_state, self.coll_muta_ip_ssl_loss = self.item_mutation_ssl(dprnn_user_embed)
                        # ideal_state, self.coll_muta_ip_ssl_loss = self.seqInterest_mutation_ssl(dprnn_user_embed)
                        self.pad_state_ssl_loss=0
                        # return dprnn_user_embed
                        fuse_way = "alpha_sum"  # alpha_sum concat
                        concat_all = tf.concat(
                            [
                                dprnn_user_embed,
                                self.ideal_state,
                                self.target_item_embedding,
                                # ideal_state*self.target_item_embedding,
                                dprnn_user_embed * self.ideal_state,
                                dprnn_user_embed + self.ideal_state,
                                last_diff,
                            ],
                            1,
                        )
                        last_hidden_nn_layer = concat_all
                        beta_logit = self._fcn_net(
                            last_hidden_nn_layer, self.hparams.att_fcn_layer_sizes, scope="mutation_beta"
                        )
                        beta_output = tf.sigmoid(beta_logit)
                        print("beta_output shape",beta_output.shape)
                        cold_boolean_mask = tf.equal(self.iterator.items, tf.zeros_like(self.iterator.items))  # [B,]
                        cold_boolean_mask=tf.expand_dims(cold_boolean_mask,axis=-1)# [B,1]
                        muta_cold_alleviate_factor = tf.compat.v1.where(cold_boolean_mask,
                                                                             tf.zeros_like(beta_output,
                                                                                           dtype=tf.float32),
                                                                             beta_output)#[B,1]
                        # muta_cold_alleviate_factor=tf.compat.v1.Print(muta_cold_alleviate_factor,[muta_cold_alleviate_factor[:20,0]],
                        #                                               summarize=20,message="\n=====muta_cold_alleviate_factor======")
                        # dprnn_user_embed_org=dprnn_user_embed
                        if fuse_way=="concat":
                            dprnn_user_embed=tf.concat([dprnn_user_embed,muta_cold_alleviate_factor * self.ideal_state],axis=-1)
                        else:
                            # beta_output = tf.compat.v1.Print(beta_output,[beta_output[:20, 0]],
                            #                                                 summarize=20,message="\n=====beta_output======")
                            dprnn_user_embed = (dprnn_user_embed + beta_output * self.ideal_state) / (1 + beta_output)#[B,H]
                            # dprnn_user_embed = (1-beta_output)*dprnn_user_embed + beta_output * ideal_state
                        # padding item ssl
                        print("dprnn_user_embed shape",dprnn_user_embed.shape)
                        # concat_all = tf.concat(
                        #     [
                        #         dprnn_user_embed_org,
                        #         self.padd_ssl_state,
                        #         dprnn_user_embed_org * self.padd_ssl_state,
                        #         dprnn_user_embed_org + self.padd_ssl_state,
                        #         last_diff,
                        #     ],
                        #     1,
                        # )
                        #
                        # last_hidden_nn_layer = concat_all
                        # gama_logit = self._fcn_net(
                        #     last_hidden_nn_layer, self.hparams.att_fcn_layer_sizes, scope="fcn_beta", reuse=True
                        # )
                        # gama_output = tf.sigmoid(gama_logit)  # [B,]
                        # self.pad_state_ssl_loss = self.hparams.coll_muta_ip_ssl_regu * tf.reduce_mean(
                        #     tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(gama_output),
                        #                                             logits=gama_output))
                        self.pad_state_ssl_loss=0.0
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

    def item_mutation_ssl(self,user_embed):
        #user_emd [b, h]
        pos_negItem=self.target_item_embedding#[B,H]
        pos_negItem=tf.reshape(pos_negItem,[-1,1+self.iterator.batch_num_ngs,pos_negItem.shape[-1]])
        posItem_embedding=pos_negItem[:,0,:]#[B/1+negNum,H]
        negItem_embedding=pos_negItem[:,1:,:]#[B/1+negNum,negNum,H]
        query_size=pos_negItem.shape[-1]
        user_embed_size=user_embed.shape[-1]#[B,(L+2)*H]
        # ipNum = self.hparams.ipNum
        # self.ipPool = tf.compat.v1.get_variable(
        #     name="mutation_ipPool", shape=[ipNum, (self.hparams.step_L+2)*self.hparams.hidden_size], initializer=self.initializer
        # )
        # query_prj = self._fcn_net(
        #     tf.reshape(pos_negItem, [-1, query_size]), [80],activations=["tanh"], scope="mutation_query_prj",
        #     _out_layer_size=(self.hparams.step_L+2)*self.hparams.hidden_size
        # )
        # # [B,D]
        # query_prj=tf.nn.tanh(query_prj)
        # query_prj_expd = tf.expand_dims(query_prj, axis=1)  # [B,1,D]
        # simMat = tf.matmul(query_prj, self.ipPool, transpose_b=True)  # [b,voc]
        # topk_sim_idx = tf.argsort(simMat, direction='DESCENDING')[:, :self.hparams.topk_sim_k_val]  # [B,K]
        # topk_sim_ip = tf.nn.embedding_lookup(
        #     params=self.ipPool, ids=topk_sim_idx
        # )  # [B,K,H]
        # tf.reduce_sum(query_prj_expd*median_sim_ip,axis=-1,keepdims=True)#[B 1 D] [B K D]=>[B,K,1]
        # att_logit = tf.nn.softmax(
        #     tf.reduce_sum(query_prj_expd * topk_sim_ip, axis=-1, keepdims=True)
        #     , 1)  # [B,K,1] along K
        # ideal_state = tf.reduce_sum(topk_sim_ip * att_logit, axis=1)  # [B,K,D]=>[B,D]
        ideal_state=tf.concat([self.item_mutation_embedding,self.cate_mutation_embedding],axis=-1)#[B,H1+H2]

        print("ideal_state shape",ideal_state.shape)
        normalize_ideal_state = tf.nn.l2_normalize(ideal_state, -1)#[B,H]
        # normalize_pred_user_emd = tf.nn.l2_normalize(user_embed, -1)#[B,H]
        normalize_pred_user_emd = tf.nn.l2_normalize(user_embed, -1)#[B,H]
        ssl_way="bpr"
        if ssl_way=="bpr":
            normalize_ideal_state=tf.reshape(normalize_ideal_state,[-1,1+self.iterator.batch_num_ngs,normalize_ideal_state.shape[-1]])
            #[B/1+numneg,1+num_negs,H]
            normalize_pred_user_emd=tf.reshape(normalize_pred_user_emd,[-1,1+self.iterator.batch_num_ngs,normalize_pred_user_emd.shape[-1]])[:,0,:]
            # [B/1+numneg,H]
            pos_normalize_ideal_state=tf.expand_dims(normalize_ideal_state[:,0,:],axis=1)#[B/1+numneg,1,H]
            neg_normalize_ideal_state=normalize_ideal_state[:,1:,:]#[B/1+numneg,num_negs,H]
            normalize_pred_user_emd=tf.expand_dims(normalize_pred_user_emd,axis=1)#[B/1+numneg,1,H]
            neg_anchor_inner_pair=normalize_pred_user_emd*neg_normalize_ideal_state#[B,num_negs,H]
            pos_anchor_inner_pair=normalize_pred_user_emd*pos_normalize_ideal_state#[B,num_negs,H]

            boolean_mask = tf.equal(self.iterator.items, tf.zeros_like(self.iterator.items))#[B,]
            coll_muta_cold_alleviate_factor=tf.compat.v1.where(boolean_mask, tf.zeros_like(self.iterator.items,dtype=tf.float32), self.hparams.coll_muta_ip_ssl_regu*
                               tf.ones_like(self.iterator.items,dtype=tf.float32))
            print("neg_anchor_inner_pair shape",neg_anchor_inner_pair.shape)
            print("pos_anchor_inner_pair shape",pos_anchor_inner_pair.shape)
            coll_muta_cold_alleviate_factor=tf.reshape(coll_muta_cold_alleviate_factor,
                                                       [-1,1+self.iterator.batch_num_ngs])[:,0]#[B/(1+numneg),]
            neg_pos_anchor_inner_pair = tf.reduce_sum(
                neg_anchor_inner_pair - tf.tile(pos_anchor_inner_pair, [1, self.iterator.batch_num_ngs, 1]), axis=-1
            )  # [B/(1+numneg),negNum]
            coll_muta_ip_ssl_loss=tf.reduce_mean(
                tf.nn.softplus(neg_pos_anchor_inner_pair),axis=-1)#[B,]
            coll_muta_ip_ssl_loss=coll_muta_cold_alleviate_factor*coll_muta_ip_ssl_loss
            coll_muta_ip_ssl_loss=tf.reduce_mean(coll_muta_ip_ssl_loss)

        elif ssl_way=="easy_equal":
            pos_neg_mask=tf.cast(self.iterator.labels,dtype=tf.float32)#[1,0,0]
            cosSim_pair=tf.reduce_sum(normalize_ideal_state * normalize_pred_user_emd, -1)  # [B,]
            revs_cossim=cosSim_pair*(-2.0)*pos_neg_mask+cosSim_pair#[-cosSim,cosSim] # [B,]
            coll_muta_ip_ssl_loss = self.hparams.coll_muta_ip_ssl_regu*tf.reduce_mean(revs_cossim)
        return ideal_state,coll_muta_ip_ssl_loss
    def item_recons_mutation_ssl(self,user_embed):
        #user_emd [b, h]
        pos_negItem=self.target_item_embedding#[B,itemDim]
        pos_negItem=tf.reshape(pos_negItem,[-1,1+self.iterator.batch_num_ngs,pos_negItem.shape[-1]])
        #[B/(1+negNum),(1+negNum),itemDim]
        posItem_embedding=pos_negItem[:,0,:]#[B/1+negNum,itemDim]
        negItem_embedding=pos_negItem[:,1:,:]#[B/1+negNum,negNum,itemDim]
        query_size=pos_negItem.shape[-1]
        user_embed_size=user_embed.shape[-1]#[B,(L+2)*H]
        ipNum = self.hparams.ipNum
        self.ipPool = tf.compat.v1.get_variable(
            name="mutation_ipPool", shape=[ipNum, query_size],
            initializer=self.initializer
        )#[voc,itemDim]
        query_prj = self._fcn_net(
            user_embed, [80],activations=["tanh"], scope="mutation_query_prj",
            _out_layer_size=query_size
        )
        # [B,(L+2)*D]==>[B,itemDim]
        query_prj=tf.nn.tanh(query_prj)# [B,itemDim]
        query_prj_expd = tf.expand_dims(query_prj, axis=1)  # [B,1,itemDim]
        simMat = tf.matmul(query_prj, self.ipPool, transpose_b=True)  # [B,voc]
        topk_sim_idx = tf.argsort(simMat, direction='DESCENDING')[:, :self.hparams.topk_sim_k_val]  # [B,K]
        topk_sim_ip = tf.nn.embedding_lookup(
            params=self.ipPool, ids=topk_sim_idx
        )  # [B,K,itemDim]
        att_logit = tf.nn.softmax(
            tf.reduce_sum(query_prj_expd * topk_sim_ip, axis=-1, keepdims=True)
            , 1)  # [B,K,1] along K
        ideal_item = tf.reduce_sum(topk_sim_ip * att_logit, axis=1)  # [B,K,itemDim]=>[B,itemDim]
        print("ideal_state shape",ideal_item.shape)
        normalize_ideal_item = tf.nn.l2_normalize(ideal_item, -1)#[B,itemDim]
        normalize_pos_item=tf.nn.l2_normalize(posItem_embedding, -1)#[B/(1+negNum),H]
        print("normalize_ideal_item.shape",normalize_ideal_item.shape)
        print("normalize_pos_item.shape",normalize_pos_item.shape)
        ssl_way="bpr"
        if ssl_way=="bpr":
            normalize_ideal_state=tf.reshape(normalize_ideal_item,[-1,1+self.iterator.batch_num_ngs,pos_negItem.shape[-1]])
            #[B/(1+numneg),1+num_negs,itemdim]
            print("normalize_ideal_state shape",normalize_ideal_state.shape)
            pos_normalize_ideal_state=tf.expand_dims(normalize_ideal_state[:,0,:],axis=1)#[B/(1+numneg),1,H]
            neg_normalize_ideal_state=normalize_ideal_state[:,1:self.iterator.batch_num_ngs+1,:]#[B/(1+numneg),num_negs,H]
            print("neg_normalize_ideal_state shape",neg_normalize_ideal_state.shape)
            normalize_label_emd=tf.expand_dims(normalize_pos_item,axis=1)#[B/(1+numneg),1,H]
            neg_anchor_inner_pair=normalize_label_emd*neg_normalize_ideal_state#[B/(1+numneg),num_negs,H]
            pos_anchor_inner_pair=normalize_label_emd*pos_normalize_ideal_state#[B/(1+numneg),num_negs,H]

            boolean_mask = tf.equal(self.iterator.items, tf.zeros_like(self.iterator.items))#[B,]
            coll_muta_cold_alleviate_factor=tf.compat.v1.where(boolean_mask, tf.zeros_like(self.iterator.items,dtype=tf.float32),
                                                               self.hparams.coll_muta_ip_ssl_regu*
                               tf.ones_like(self.iterator.items,dtype=tf.float32))#[B,]
            coll_muta_cold_alleviate_factor=tf.reshape(coll_muta_cold_alleviate_factor,[-1,1+self.iterator.batch_num_ngs])[:,0]#[B/(1+numneg),]
            print("neg_anchor_inner_pair shape", neg_anchor_inner_pair.shape)
            print("pos_anchor_inner_pair shape", pos_anchor_inner_pair.shape)
            neg_pos_anchor_inner_pair=tf.reduce_sum(
                neg_anchor_inner_pair-tf.tile(pos_anchor_inner_pair,[1,self.iterator.batch_num_ngs,1]),axis=-1
            )#[B/(1+numneg),negNum]
            print("coll_muta_cold_alleviate_factor shape",coll_muta_cold_alleviate_factor.shape)

            coll_muta_ip_ssl_loss=tf.reduce_mean(tf.nn.softplus(neg_pos_anchor_inner_pair),axis=-1)#[B/(1+numneg),]
            print("coll_muta_ip_ssl_loss shape",coll_muta_ip_ssl_loss.shape)
            # coll_muta_ip_ssl_loss=self.hparams.coll_muta_ip_ssl_regu*coll_muta_ip_ssl_loss
            coll_muta_ip_ssl_loss=coll_muta_cold_alleviate_factor*coll_muta_ip_ssl_loss

            coll_muta_ip_ssl_loss=tf.reduce_mean(coll_muta_ip_ssl_loss)
        # elif ssl_way=="easy_equal":
        #     pos_neg_mask=tf.cast(self.iterator.labels,dtype=tf.float32)#[1,0,0]
        #     cosSim_pair=tf.reduce_sum(normalize_ideal_state * normalize_pred_user_emd, -1)  # [B,]
        #     revs_cossim=cosSim_pair*(-2.0)*pos_neg_mask+cosSim_pair#[-cosSim,cosSim] # [B,]
        #     coll_muta_ip_ssl_loss = self.hparams.coll_muta_ip_ssl_regu*tf.reduce_mean(revs_cossim)
        return ideal_item,coll_muta_ip_ssl_loss
    def seqInterest_mutation_ssl(self,user_embed):
        #user_emd [b, h]
        pos_negItem=self.target_item_embedding#[B,itemDim]
        pos_negItem=tf.reshape(pos_negItem,[-1,1+self.iterator.batch_num_ngs,pos_negItem.shape[-1]])
        #[B/(1+negNum),(1+negNum),itemDim]
        posItem_embedding=pos_negItem[:,0,:]#[B/1+negNum,itemDim]
        negItem_embedding=pos_negItem[:,1:,:]#[B/1+negNum,negNum,itemDim]
        query_size=pos_negItem.shape[-1]
        user_embed_size=user_embed.shape[-1]#[B,(L+2)*H]
        ipNum = self.hparams.ipNum
        self.ipPool = tf.compat.v1.get_variable(
            name="mutation_ipPool", shape=[ipNum, user_embed_size],
            initializer=self.initializer
        )#[voc,itemDim]
        # query_prj = self._fcn_net(
        #     user_embed, [80],activations=["tanh"], scope="mutation_query_prj",
        #     _out_layer_size=user_embed_size
        # )
        # query_prj=tf.nn.tanh(query_prj)# [B,itemDim]
        self.ipPool=tf.compat.v1.Print(self.ipPool,[self.ipPool[:6,:6]],summarize=36,message="\n======ipPool====")
        query_prj=user_embed#[B,(L+2)*hiddensize]
        # [B,(L+2)*D]==>[B,itemDim]
        query_prj_expd = tf.expand_dims(query_prj, axis=1)  # [B,1,(L+2)*hiddensize]
        simMat = tf.matmul(query_prj, self.ipPool, transpose_b=True)  # [B,voc]
        topk_sim_idx = tf.argsort(simMat, direction='DESCENDING')[:, :self.hparams.topk_sim_k_val]  # [B,K]
        topk_sim_idx=tf.compat.v1.Print(topk_sim_idx,[topk_sim_idx[:21:5,:16]],summarize=64,message="topk_sim_idx")
        topk_sim_ip = tf.nn.embedding_lookup(
            params=self.ipPool, ids=topk_sim_idx
        )  # [B,K,itemDim]
        att_logit = tf.nn.softmax(
            tf.reduce_sum(query_prj_expd * topk_sim_ip, axis=-1, keepdims=True)
            , 1)  # [B,K,1] along K
        ideal_state = tf.reduce_sum(topk_sim_ip * att_logit, axis=1)  # [B,K,itemDim]=>[B,(L+2)*hiddensize]
        print("ideal_state shape",ideal_state.shape)
        self.ideal_state=ideal_state
        ssl_way="BCE"
        if ssl_way=="BCE":
            coll_muta_ip_ssl_loss=0.0
        if ssl_way=="bpr":
            normalize_ideal_state = tf.nn.l2_normalize(ideal_state, -1)  # [B,itemDim]
            normalize_pos_item = tf.nn.l2_normalize(posItem_embedding, -1)  # [B/(1+negNum),H]
            print("normalize_ideal_item.shape", normalize_ideal_state.shape)
            print("normalize_pos_item.shape", normalize_pos_item.shape)
            normalize_ideal_state=tf.reshape(normalize_ideal_state,[-1,1+self.iterator.batch_num_ngs,pos_negItem.shape[-1]])
            #[B/(1+numneg),1+num_negs,itemdim]
            print("normalize_ideal_state shape",normalize_ideal_state.shape)
            pos_normalize_ideal_state=tf.expand_dims(normalize_ideal_state[:,0,:],axis=1)#[B/(1+numneg),1,H]
            neg_normalize_ideal_state=normalize_ideal_state[:,1:self.iterator.batch_num_ngs+1,:]#[B/(1+numneg),num_negs,H]
            print("neg_normalize_ideal_state shape",neg_normalize_ideal_state.shape)
            normalize_label_emd=tf.expand_dims(normalize_pos_item,axis=1)#[B/(1+numneg),1,H]
            neg_anchor_inner_pair=normalize_label_emd*neg_normalize_ideal_state#[B/(1+numneg),num_negs,H]
            pos_anchor_inner_pair=normalize_label_emd*pos_normalize_ideal_state#[B/(1+numneg),num_negs,H]

            boolean_mask = tf.equal(self.iterator.items, tf.zeros_like(self.iterator.items))#[B,]
            coll_muta_cold_alleviate_factor=tf.compat.v1.where(boolean_mask, tf.zeros_like(self.iterator.items,dtype=tf.float32),
                                                               self.hparams.coll_muta_ip_ssl_regu*
                               tf.ones_like(self.iterator.items,dtype=tf.float32))#[B,]
            coll_muta_cold_alleviate_factor=tf.reshape(coll_muta_cold_alleviate_factor,[-1,1+self.iterator.batch_num_ngs])[:,0]#[B/(1+numneg),]
            print("neg_anchor_inner_pair shape", neg_anchor_inner_pair.shape)
            print("pos_anchor_inner_pair shape", pos_anchor_inner_pair.shape)
            neg_pos_anchor_inner_pair=tf.reduce_sum(
                neg_anchor_inner_pair-tf.tile(pos_anchor_inner_pair,[1,self.iterator.batch_num_ngs,1]),axis=-1
            )#[B/(1+numneg),negNum]
            print("coll_muta_cold_alleviate_factor shape",coll_muta_cold_alleviate_factor.shape)

            coll_muta_ip_ssl_loss=tf.reduce_mean(tf.nn.softplus(neg_pos_anchor_inner_pair),axis=-1)#[B/(1+numneg),]
            print("coll_muta_ip_ssl_loss shape",coll_muta_ip_ssl_loss.shape)
            # coll_muta_ip_ssl_loss=self.hparams.coll_muta_ip_ssl_regu*coll_muta_ip_ssl_loss
            coll_muta_ip_ssl_loss=coll_muta_cold_alleviate_factor*coll_muta_ip_ssl_loss

            coll_muta_ip_ssl_loss=tf.reduce_mean(coll_muta_ip_ssl_loss)
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
    def _multi_scale_fuse(self, query, user_embedding):
        """Apply attention by fully connected layers.

        Args:
            query (object): The embedding of target item which is regarded as a query in attention operations.
                #[B,H]
            user_embedding (object): The output of RNN layers which is regarded as user modeling.
                #[B,T,(L+1)*H]
        Returns:
            object: Weighted sum of user modeling.
        """
        hparams = self.hparams
        with tf.compat.v1.variable_scope("_multi_scale_fuse_attention_fcn"):
            query_size = query.shape[1]
            boolean_mask = tf.equal(self.mask, tf.ones_like(self.mask))#[B,T]
            boolean_mask=tf.tile(tf.expand_dims(boolean_mask,-1),[1,1,self.hparams.step_L+1])#[B,T,step_L+1]
            user_embedding=tf.reshape(user_embedding,[-1,self.mask.shape[-1],self.hparams.step_L+1,
                                                      self.hparams.hidden_size])#[B,T,step_L+1,H]
            attention_mat = tf.compat.v1.get_variable(
                name="attention_mat",
                shape=[user_embedding.shape.as_list()[-1], query_size],
                initializer=self.initializer,
            )
            att_inputs = tf.tensordot(user_embedding, attention_mat, [[-1], [0]])#[B,T,step_L+1,D]

            queries = tf.reshape(
                tf.tile(query, [1, att_inputs.shape[1]*att_inputs.shape[2]]), tf.shape(input=att_inputs)
            )#[B,T,step_L+1,D]

            last_hidden_nn_layer = tf.concat(
                [att_inputs, queries, att_inputs - queries, att_inputs * queries], -1
            )

            att_fnc_output = self._fcn_net(
                last_hidden_nn_layer, hparams.att_fcn_layer_sizes, scope="att_fcn"
            )#[B,T,step_L+1,1]
            # att_fnc_output=tf.compat.v1.Print(att_fnc_output,[tf.shape(att_fnc_output)],message="=====att_fnc_output:")

            att_fnc_output = tf.squeeze(att_fnc_output, -1)#[B,T,step_L+1]
            mask_paddings = tf.ones_like(att_fnc_output) * (-(2 ** 32) + 1)
            att_weights=tf.compat.v1.where(boolean_mask, att_fnc_output, mask_paddings)#[B,T,step_L+1]
            att_weights = tf.nn.softmax(
                tf.reshape(att_weights,[-1,att_weights.shape[1]*att_weights.shape[2]]),#[B,T*(step_L+1)]
                name="att_weights",axis=-1
            )
            user_embedding=tf.reshape(user_embedding,[-1,self.mask.shape[-1]*(self.hparams.step_L+1),self.hparams.hidden_size])
            print("userembedding shape ",user_embedding.shape)
            output =  user_embedding* tf.expand_dims(att_weights, -1)#[B,T*(step_L+1),H]#[B,(T*step_L+1),1]
            print("output.shape ",output.shape)
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
        return tf.reduce_sum(input_tensor=regular_loss)