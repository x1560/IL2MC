# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
import pickle
from tensorflow.compat.v1.nn import dynamic_rnn
import numpy as np
from recommenders.models.deeprec.deeprec_utils import cal_metric
from recommenders.models.deeprec.models.sequential.rnn_cell_implement import Time4LSTMCell
from recommenders.models.deeprec.models.sequential.sequential_base_model import (
    SequentialBaseModel,
)
from recommenders.models.deeprec.models.sequential.dium_cells import (
    DIUMCell, DIUM2WayEvoCellV2, DIUMCellIP_MERGE,
)
from sklearn.metrics.pairwise import cosine_similarity


class DIUMModel(SequentialBaseModel):
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
        with tf.compat.v1.variable_scope("dium"):
            self.mask = self.iterator.mask  # [None,max_seq_length]
            self.sequence_length = tf.reduce_sum(input_tensor=self.mask, axis=1)  # [None,]
            if self.hparams.useCate:
                self.history_embedding = tf.concat(
                    [self.item_history_embedding, self.cate_history_embedding], 2
                )#[B,T,H]
            else:
                self.history_embedding = self.item_history_embedding
                self.target_item_embedding = self.item_embedding

            if self.hparams.needLongInterest or self.hparams.persInitState:
                self.longInterest = self.getLongInterest()

            if self.hparams.useTimeSliRec:
                #[inp,ti-t1,ti-now]
                self.history_embedding = tf.concat(
                    [
                        self.history_embedding,
                        tf.expand_dims(self.iterator.time_from_first_action, -1),
                        tf.expand_dims(self.iterator.time_to_now, -1),
                    ],
                    -1,
                )
                # seqEmbedding=self.seqEncoding()
                # self.history_embedding = tf.concat(
                #     [
                #         seqEmbedding,
                #         tf.expand_dims(self.iterator.time_from_first_action, -1),
                #         tf.expand_dims(self.iterator.time_to_now, -1),
                #     ],
                #     -1,
                # )
            elif self.hparams.useTimeBochner:
                self.time_history=tf.expand_dims(self.iterator.time_history, -1)#[B,T,1]
                self.time_embedding_bochner_weight = tf.compat.v1.get_variable(
                    name="time_embedding_bochner_weight",
                    shape=[hparams.time_embedding_dim],
                    initializer=self.initializer,
                )  # [timeEmdDim]
                self.time_embedding_maping_weight=tf.compat.v1.get_variable(
                    name="time_embedding_maping_weight",
                    shape=[2*hparams.time_embedding_dim,hparams.time_embedding_dim],
                    initializer=self.initializer,
                )  # [timeEmdDim,]
                self.time_embedding_maping_bias=tf.compat.v1.get_variable(
                    name="time_embedding_maping_bias",
                    shape=[hparams.time_embedding_dim],
                    initializer=self.initializer,
                )  # [timeEmdDim,]
                expdTimeWeight=tf.expand_dims(tf.expand_dims(self.time_embedding_bochner_weight,0),0)#[1,1,D]
                time_cos_part=tf.math.cos(self.time_history*expdTimeWeight)#[B,T,D]
                time_sin_part=tf.math.sin(self.time_history*expdTimeWeight)#[B,T,D]
                self.time_embedding_bochner=tf.concat([time_cos_part,time_sin_part],-1)/tf.math.sqrt(float(hparams.time_embedding_dim))#[B,T,2*D]

                self.time_embedding_bochner=tf.compat.v1.nn.xw_plus_b(
                    self.time_embedding_bochner, self.time_embedding_maping_weight, self.time_embedding_maping_bias
                )  # [B,h]
                self.history_embedding=tf.concat([self.history_embedding, self.time_embedding_bochner], -1)

                time_embedding_bochner_weight_expd=tf.expand_dims(self.time_embedding_bochner_weight, 0)#[1,D]
                self.target_item_time=tf.expand_dims(self.iterator.time,-1)#[B,1]
                # tf.math.cos(self.target_item_time * time_embedding_bochner_weight_expd)  # [B,D]
                # tf.math.sin(self.target_item_time * time_embedding_bochner_weight_expd)  # [B,D]
                self.target_item_time_embedding=tf.concat([
                    tf.math.cos(self.target_item_time * time_embedding_bochner_weight_expd),
                    tf.math.sin(self.target_item_time * time_embedding_bochner_weight_expd)
                ],-1)/tf.math.sqrt(float(hparams.time_embedding_dim))#[B,2*D]
                self.target_item_time_embedding =tf.compat.v1.nn.xw_plus_b(
                    self.target_item_time_embedding,self.time_embedding_maping_weight,self.time_embedding_maping_bias
                )
                self.target_item_embedding = tf.concat(
                    [self.target_item_embedding, self.target_item_time_embedding], -1
                )  # [B,H]

                # self.time_diff=tf.expand_dims(self.iterator.time_diff,-1)# time_diff[0]<= time[1]-time[0]     [B]
                # self.time_to_now=tf.expand_dims(self.iterator.time_to_now,-1)


                # self.history_embedding=tf.concat([self.history_embedding,tf.tile(self.time_diff,[1,self.history_embedding.shape[1],1])],2)#[B,T,H]=>[B,T,H+1]
                # self.history_embedding = tf.concat([self.history_embedding,tf.tile(tf.expand_dims(self.history_user_time_decay_factor,1) , [1,self.history_embedding.shape[1],1])],2)
                # #[B,T,H+1]==>[B,T,H+1+K]
                # self.target_item_embedding = tf.concat([self.target_item_embedding,self.history_user_time_decay_factor],1)#[B,H]==>[B,H+K]

            cell = self._create_diumcell()
            self.cell = cell
            cell.model = self


            final_state,rum_out_state_seq = self._build_dium(cell)
            if self.hparams.useInterestDiffReg:
                self.ip_diff_l=self._ip_reg_l( tf.reshape(rum_out_state_seq[:,:,:self.hparams.slots * self.hparams.hidden_size],
                                                     [-1,rum_out_state_seq.shape[1], self.hparams.slots, self.hidden_size]) )#[B,T,S,H]
            if self.hparams.use_delta_is_ssl_loss:
                self.delta_is_ssl_loss = tf.reduce_mean(rum_out_state_seq[:, :, -1:])
                self.delta_is_ssl_loss = self.hparams.is_ssl_regu * self.delta_is_ssl_loss
            if self.hparams.use_next_it_aux_loss:
                negSample=self.item_history_embedding[::-1,1:,:]#[B T-1 D]
                negSample=negSample[:,::-1,:]#[B T-1 D]
                negLabel=tf.zeros(shape=[tf.shape(negSample)[0],tf.shape(negSample)[1]],dtype=tf.float32)#[B]
                pos=self.item_history_embedding[:,1:,:]#[B,T-1,D]
                posLabel=tf.ones(shape=[tf.shape(pos)[0],tf.shape(pos)[1]],dtype=tf.float32)
                # posLabel=tf.compat.v1.Print(posLabel,[tf.shape(posLabel)],message="======posLabel:\n")

                # ip_t=rum_out_state_seq[:, :-1, :self.hparams.hidden_size]#[B,T-1,D]
                testItemEmd=tf.concat([negSample,pos],0)#[2*B,T-1,D]
                ip_t = tf.tile(rum_out_state_seq[:, :-1, :self.hparams.hidden_size],[2,1,1])#[2*B,T-1,D]
                aux_net_fea=tf.concat([testItemEmd,ip_t,testItemEmd*ip_t],-1)#[2*B,T-1,D+D+D]

                # aux_net_fea=tf.compat.v1.Print(aux_net_fea,[tf.shape(aux_net_fea)],message="======aux_net_fea:\n")
                label=tf.concat([negLabel,posLabel],-1)
                logit = self._fcn_net(tf.reshape(aux_net_fea,[-1,tf.shape(aux_net_fea)[-1]]),
                                      hparams.next_it_aux_layer_sizes,first_layer_size=ip_t.get_shape().as_list()[-1]*3,scope="next_it_aux_layer")
                # logit=tf.compat.v1.Print(logit,[tf.shape(logit)],message="======logit:\n")
                # label=tf.compat.v1.Print(label,[tf.shape(label)],message="======label:\n")
                self.next_it_aux_loss = self.hparams.next_it_aux_regu*tf.reduce_mean(
                    input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=tf.math.sigmoid(tf.reshape(logit,[-1])),
                        labels=tf.reshape(label,[-1]),
                    )
                )
            for _p in cell.parameter_set:
                tf.compat.v1.summary.histogram(_p.name, _p)
            if hasattr(cell, "_alpha") and hasattr(cell._alpha, "name"):
                tf.compat.v1.summary.histogram(cell._alpha.name, cell._alpha)
            if hasattr(cell, "_beta") and hasattr(cell._beta, "name"):
                tf.compat.v1.summary.histogram(cell._beta.name, cell._beta)

            final_state, att_weights = self._attention_query_by_state(
                final_state, self.target_item_embedding,rum_out_state_seq
            )

            if self.hparams.needLongInterest:
                if self.hparams.sliGating:
                    with tf.compat.v1.variable_scope("attention_fcn"):
                        att_outputs2 = self._attention_fcn(
                            self.target_item_embedding, rum_out_state_seq
                        )

                        att_fea2 = tf.reduce_sum(input_tensor=att_outputs2, axis=1)
                        tf.compat.v1.summary.histogram("att_fea2", att_fea2)
                    # ensemble
                    with tf.compat.v1.name_scope("gating"):
                        concat_all = tf.concat(
                            [
                                self.target_item_embedding,
                                self.longInterest,
                                att_fea2,
                                tf.expand_dims(self.iterator.time_to_now[:, -1], -1),
                            ],
                            1,
                        )
                        last_hidden_nn_layer = concat_all
                        alpha_logit = self._fcn_net(
                            last_hidden_nn_layer, hparams.att_fcn_layer_sizes, scope="fcn_alpha"
                        )
                        alpha_output = tf.sigmoid(alpha_logit)

                        model_output = tf.concat([final_state,self.longInterest * alpha_output , att_fea2 * (1.0 - alpha_output),self.target_item_embedding],1)
                else:
                    model_output = tf.concat([final_state, self.target_item_embedding,self.longInterest], 1)

            else:
                model_output = tf.concat([final_state, self.target_item_embedding], 1)

            tf.compat.v1.summary.histogram("model_output", model_output)

            if "other_metrics_basedOn_RecList" in self.hparams.values():
                allItemsEmd=tf.expand_dims(self.item_lookup,0)#[vocNum,H]==>[1,vocNum,H]

                final_state=tf.expand_dims(final_state,1)#[B,H]=>[B,1,H]
                longInterest=tf.expand_dims(self.longInterest, 1)#[B,H]=>[B,1,H]

                final_state=final_state[::self.hparams.valid_test_num_ngs+1,:,:]
                longInterest=longInterest[::self.hparams.valid_test_num_ngs+1,:,:]
                vocNum=allItemsEmd.shape[1]

                final_state=tf.tile(final_state,[1,vocNum,1])
                longInterest=tf.tile(longInterest,[1,vocNum,1])
                allItemsEmd = final_state+allItemsEmd-final_state

                self.recItemListScoreModelOut = tf.concat([final_state,allItemsEmd, longInterest], -1)
                #[B,vocNum,1]

            return model_output
                #final state [B,H]

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
            att_fnc_output = tf.squeeze(att_fnc_output, -1)
            mask_paddings = tf.ones_like(att_fnc_output) * (-(2 ** 32) + 1)
            att_weights = tf.nn.softmax(
                tf.compat.v1.where(boolean_mask, att_fnc_output, mask_paddings),
                name="att_weights",
            )
            output = user_embedding * tf.expand_dims(att_weights, -1)
            return output

    def getLongInterest(self):
        '''
        attention sum,merge hist
        Returns:shortInterest
        '''
        if self.hparams.itemIPSim:
            history_embedding_l2norm=tf.nn.l2_normalize(self.history_embedding,axis=-1)#[B,T,H]
            target_embedding_l2norm=tf.expand_dims(tf.nn.l2_normalize(self.target_item_embedding,axis=-1),1)#[B,1,H]
            attScore=history_embedding_l2norm*target_embedding_l2norm+(1e-7)#[B,T,H]
        else:
            attScore=self.history_embedding*tf.expand_dims(self.target_item_embedding,1)#[B,T,H*B,1,H]==>[B,T,H]

        attScore=tf.nn.softmax(tf.reduce_sum(attScore,axis=-1,keepdims=True),1)#[B,T,1]
        longInterest=attScore*self.history_embedding#[B,T,1]*[B,T,H]=>[b t h]
        longInterest=tf.reduce_sum(longInterest,1)#[B,H]
        return longInterest
    def _attention_query_by_state(self, seq_output, query,rum_out_state_seq):
        """Merge a user's memory states conditioned by a query item.

        Params:
            seq_output: A flatten representation of dium memory states for (a batch of) users
            query: (a batch of) target item candidates shape=[B,itemDim+cateDim]

        Returns:
            tf.Tensor, tf.Tensor: Merged user representation. Attention weights of each memory channel.
        """

        dim_q = query.shape[-1]
        att_weights = tf.constant(1.0, dtype=tf.float32)
        with tf.compat.v1.variable_scope("query_att"):
            if self.hparams.slots > 1:
                # reshape the memory states to (BatchSize, Slots, HiddenSize)
                if "2way" not in self.hparams.cell:
                    if self.hparams.use_delta_is_ssl_loss:
                        ip_state=tf.reshape(seq_output[:,:-1], [-1, self.hparams.slots, self.hidden_size])
                    else:
                        ip_state = tf.reshape(seq_output, [-1, self.hparams.slots, self.hidden_size])
                    if self.hparams.useSquash:
                        is_att_weights = self.cell.squash(ip_state)
                else:
                    ip_state = tf.reshape(
                        seq_output[:, :self.hparams.slots * self.hparams.hidden_size],
                        [-1, self.hparams.slots, self.hidden_size]
                    )
                    if self.hparams.useSquash:
                        is_state = tf.reshape(
                            seq_output[:,self.hparams.slots * self.hparams.hidden_size:], [-1, self.hparams.slots, self.hidden_size]
                        )#[B,D]==>[B,h] [b,s,h]
                        # squash IS
                        is_att_weights = self.cell.squashIS(is_state)

                if self.hparams.itemIPSim=="cosSim":
                    # cos sim
                    norm_inputs = tf.expand_dims(tf.nn.l2_normalize(query, axis=-1), 1)  # [B,1,H]
                    norm_ip_state = tf.nn.l2_normalize(ip_state, axis=-1)  # [B,S,H]
                    ip_att_logit_mat = tf.reduce_sum(norm_inputs * norm_ip_state, axis=-1,keepdims=True)  # [B,S,H]=>[B,S,1]
                elif self.hparams.itemIPSim=="dotProduct":
                    ip_att_logit_mat=tf.reduce_sum(tf.expand_dims(query,1)*ip_state, axis=-1,keepdims=True)#[B,S,1]

                ip_att_weights=tf.nn.softmax(self.beta*ip_att_logit_mat,axis=1)#[B,S,1] along S

                if self.hparams.expAnalyse:
                    #get pos User info
                    self.expUser=self.iterator.users[::self.hparams.valid_test_num_ngs+1]#pos User id
                    self.expUserP=tf.reshape(rum_out_state_seq[::self.hparams.valid_test_num_ngs+1,:,:self.hparams.slots * self.hparams.hidden_size],
                                             [-1,rum_out_state_seq.shape[1],self.hparams.slots, self.hidden_size])
                    self.exptUserIS=self.cell.squash(self.expUserP)#[B,T,S,1]
                    if self.hparams.useTimeBochner:
                        # self.expUserTime_to_now=self.iterator.time_to_now[::self.hparams.valid_test_num_ngs+1]
                        self.expUserTime = self.iterator.time_history[::self.hparams.valid_test_num_ngs + 1]
                if self.hparams.useSquash:
                    if self.hparams.jointIPISAggType=="accuracy":
                        if self.hparams.useISIPAttGatingFactor:
                            # IPI_ISI_Att = ((tf.square(self.ISIPAttGatingFactor)+1)*ip_read_att_weights * is_att_weights)/\
                            #               ( tf.square(self.ISIPAttGatingFactor)   *(ip_read_att_weights + is_att_weights))
                            # [B,S,1]
                            jointIPISAtt = self.cell.ISIPAttGatingFactor * ip_att_weights + (
                                        1 - self.cell.ISIPAttGatingFactor) * is_att_weights
                            jointIPISAtt= tf.nn.softmax(jointIPISAtt, axis=1)
                        else:
                            jointIPISAtt=ip_att_weights*is_att_weights# [B,S,1]

                    elif self.hparams.jointIPISAggType=="diversity":
                        is_att_weights = tf.nn.softmax( is_att_weights, axis=1)
                        jointIPISAtt = tf.maximum(ip_att_weights,is_att_weights)# [B,S,1]
                    elif self.hparams.jointIPISAggType=="serendipity":
                        second2lastMask=tf.concat( [ self.mask[:,1:],
                                                    tf.zeros([tf.shape(self.mask)[0],1],dtype=tf.int32) ]
                                                   ,-1)
                        second2lastMask=self.mask-second2lastMask
                        second2lastMask=tf.concat([ second2lastMask[:,1:],
                                                   tf.zeros([tf.shape(self.mask)[0],1],dtype=tf.int32)
                                                    ]
                                                  ,-1)
                        second2lastMask=tf.cast(tf.expand_dims(second2lastMask,-1),tf.float32)#[B,T,1]
                        if self.hparams.use_delta_is_ssl_loss:
                            second2lastIS = tf.reduce_sum(rum_out_state_seq[:,:,:-1]*second2lastMask,axis=1)
                        else:
                            second2lastIS = tf.reduce_sum(rum_out_state_seq[:,:, :]*second2lastMask,axis=1)#[B,D]

                        second2lastIS = tf.reshape(
                            second2lastIS,
                            [-1, self.hparams.slots, self.hidden_size]
                        )
                        second2lastIs_att_weights = self.cell.squash(second2lastIS)
                        firstDiff=is_att_weights-second2lastIs_att_weights
                        firstDiff = tf.nn.softmax(firstDiff, 1)
                        jointIPISAtt = ip_att_weights * firstDiff
                    # jointIPISAtt= tf.nn.softmax(self.beta * jointIPISAtt, axis=1)
                    att_res = tf.reduce_sum(
                        input_tensor=jointIPISAtt * ip_state, axis=1
                    )  # [B,H]
                else:
                    att_res = tf.reduce_sum(
                        input_tensor=ip_att_weights * ip_state, axis=1
                    )  # [B,H]
            else:
                att_res = seq_output
        # att_res = tf.compat.v1.Print(att_res, [att_res[:2,:15]], message="\n=======att_res=======\n")
        return att_res, att_weights

    def _create_diumcell(self):
        """Create a dium cell

        Returns:
            object: An initialized dium cell
        """
        hparams = self.hparams
        input_embedding_dim = self.hparams.item_embedding_dim#itemDim+cateDim
        if hparams.useCate:input_embedding_dim+=hparams.cate_embedding_dim
        if hparams.useTimeBochner:input_embedding_dim+=2*hparams.time_embedding_dim
        if hparams.useTimeSliRec:input_embedding_dim+=2
        if "2way" in self.hparams.cell:
            input_params = [
                2*hparams.hidden_size * hparams.slots + input_embedding_dim,#IP+IS+last inp
                hparams.slots,
                hparams.attention_size,
                input_embedding_dim,
            ]
        else:
            ssl_dim=1 if hparams.use_delta_is_ssl_loss else 0
            input_params = [
                hparams.hidden_size * hparams.slots +ssl_dim,  # IP+IS+last inp
                hparams.slots,
                hparams.attention_size,
                input_embedding_dim,
            ]
        diumcells = {"DIUM": DIUMCell,"DIUM2WayEvoCellV2":DIUM2WayEvoCellV2,"DIUMCellIP_MERGE":DIUMCellIP_MERGE}
        diumCell = diumcells[hparams.cell]
        res = None
        if hparams.cell in ["DIUM","DIUM2WayEvoCellV2","DIUMCellIP_MERGE"]:
            res = diumCell(*input_params,_beta=hparams.beta,kernel_initializer=None,hparams=hparams)
        else:
            raise ValueError("ERROR! Cell type not support: {0}".format(hparams.cell))
        return res
    def seqEncoding(self):
        with tf.compat.v1.variable_scope("seqEncoding"):
            rnn_outputs, _ = dynamic_rnn(
                Time4LSTMCell(self.hparams.hidden_size),
                inputs=self.history_embedding,
                sequence_length=self.sequence_length,
                dtype=tf.float32,
                scope="time4lstm",
            )
            tf.compat.v1.summary.histogram("LSTM_outputs", rnn_outputs)
        return rnn_outputs#[B,T,H] states
    def _build_dium(self, cell):

        """Generate  user memory states from behavior sequence

        Args:
            object: An initialied dium cell.

        Returns:
            object: A flatten representation of user memory states, in the shape of (BatchSize, SlotsNum x HiddenSize)
        """
        # self.history_embedding=tf.concat([self.history_embedding,tf.tile(self.time_diff,[1,self.history_embedding.shape[1],1])],2)#[B,T,H]=>[B,T,H+1]
        # self.history_embedding = tf.concat([self.history_embedding,tf.tile(tf.expand_dims(self.history_user_time_decay_factor,1) , [1,self.history_embedding.shape[1],1])],2)
        # #[B,T,H+1]==>[B,T,H+1+K]
        # self.target_item_embedding = tf.concat([self.target_item_embedding,self.history_user_time_decay_factor],1)#[B,H]==>[B,H+K]
        hparams = self.hparams
        if self.hparams.persInitState:
            self._persInitStateW= tf.compat.v1.get_variable(
                name="_persInitStateW",
                shape=[self.target_item_embedding.shape[-1], cell._num_units],
                initializer=self.initializer,
            )#[userEmdDim,stateDim]
        if self.hparams.shareInitState:
            self.shareInitState = tf.compat.v1.get_variable(
                name="shareInitState",
                shape=[1,cell._num_units],
                initializer=self.initializer,
            )  # [userEmdDim,stateDim]
        with tf.compat.v1.variable_scope("dium"):
            print(tf.shape(self.target_item_embedding))
            if self.hparams.persInitState:
                initState=tf.matmul(self.longInterest,self._persInitStateW)
            elif self.hparams.shareInitState:
                initState=tf.tile(
                    self.shareInitState,[tf.shape(input=self.history_embedding)[0],1]
                )#[B,numUnits]
            else:
                # initState=cell.zero_state(
                #     tf.shape(input=self.history_embedding)[0], tf.float32
                # )
                initState=None
            with tf.compat.v1.variable_scope("diumDynamicRNN"):
                rum_out_state_seq, final_state = dynamic_rnn(
                    cell,
                    inputs=self.history_embedding,
                    dtype=tf.float32,
                    sequence_length=self.sequence_length,
                    scope="diumCell",
                    initial_state=initState
                )
                tf.compat.v1.summary.histogram("LSTM_outputs", rum_out_state_seq)

            if "2way" in self.hparams.cell:
                final_state = final_state[:, : 2*hparams.slots * hparams.hidden_size]
                rum_out_state_seq=rum_out_state_seq[:,:,: 2*hparams.slots * hparams.hidden_size]

            self.alpha = cell._alpha
            self.beta = cell._beta
            tf.compat.v1.summary.histogram("DIUM_outputs", rum_out_state_seq)

        return final_state,rum_out_state_seq

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
            ip_regu_l = tf.reduce_sum(ij_ip_sim)*2/((self.hparams.slots)*(self.hparams.slots-1))  # [B,T,S,S]==> scalar
        else:
            return None
        return self.hparams.ip_diff_regu*ip_regu_l

    def _compute_regular_loss(self):
        """Construct regular loss. Usually it's comprised of l1 and l2 norm.
        Users can designate which norm to be included via config file.

        Returns:
            object: Regular loss.
        """

        regular_loss = self._l2_loss() + self._l1_loss() + self._cross_l_loss()
        if self.hparams.useInterestDiffReg:
            regular_loss=regular_loss+self.ip_diff_l
        if self.hparams.use_delta_is_ssl_loss:
            regular_loss = regular_loss + self.delta_is_ssl_loss
        if self.hparams.use_next_it_aux_loss:
            regular_loss = regular_loss + self.next_it_aux_loss
        # regular_loss = self._l2_loss() + self._l1_loss() + self._cross_l_loss()+self.ip_diff_l

        return tf.reduce_sum(input_tensor=regular_loss)
    def run_eval(self, filename, num_ngs):
        """Evaluate the given file and returns some evaluation metrics.

        Args:
            filename (str): A file name that will be evaluated.
            num_ngs (int): The number of negative sampling for a positive instance.

        Returns:
            dict: A dictionary that contains evaluation metrics.
        """

        global item_lookup,userdict,itemdict
        load_sess = self.sess
        preds = []
        labels = []
        group_preds = []
        group_labels = []
        recItemListScores=[]
        posItems, histItems=[],[]

        expUser,expUserP, expUserIS, expUserTime =[],[],[],[]

        group = num_ngs + 1

        for batch_data_input in self.iterator.load_data_from_file(
            filename, min_seq_length=self.min_seq_length, batch_num_ngs=0
        ):
            if "other_metrics_basedOn_RecList" not in self.hparams.values():
                if not self.hparams.expAnalyse:
                    if batch_data_input:
                        step_pred, step_labels = self.eval(load_sess, batch_data_input)
                        preds.extend(np.reshape(step_pred, -1))
                        labels.extend(np.reshape(step_labels, -1))
                        group_preds.extend(np.reshape(step_pred, (-1, group)))
                        group_labels.extend(np.reshape(step_labels, (-1, group)))
                elif self.hparams.useTimeBochner:
                    if batch_data_input:
                        step_pred, step_labels, \
                        item_lookup,\
                        step_expUser,\
                        step_expUserP,\
                        step_expUserIS,\
                        step_expUserTime_from_first= self.eval(load_sess, batch_data_input)

                        preds.extend(np.reshape(step_pred, -1))
                        labels.extend(np.reshape(step_labels, -1))
                        group_preds.extend(np.reshape(step_pred, (-1, group)))
                        group_labels.extend(np.reshape(step_labels, (-1, group)))
                        # print(step_expUser.shape,step_expUserP.shape,step_expUserIS.shape,step_expUserTime_to_now.shape,step_expUserTimeDecayFactor.shape)
                        #(8,) (8, 25, 4, 64) (8, 25, 4, 1) (8, 25) (8, 4)  B=8
                        for expSum,stepExpInfo in zip([expUser,     expUserP,     expUserIS,     step_expUserTime_from_first     ],
                                                      [step_expUser,step_expUserP,step_expUserIS,step_expUserTime_from_first]):
                            expSum.extend(stepExpInfo)
                else:
                    if batch_data_input:
                        step_pred, step_labels, \
                        item_lookup,\
                        step_expUser,\
                        step_expUserP,\
                        step_expUserIS=self.eval(load_sess, batch_data_input)

                        preds.extend(np.reshape(step_pred, -1))
                        labels.extend(np.reshape(step_labels, -1))
                        group_preds.extend(np.reshape(step_pred, (-1, group)))
                        group_labels.extend(np.reshape(step_labels, (-1, group)))
                        # print(step_expUser.shape,step_expUserP.shape,step_expUserIS.shape,step_expUserTime_to_now.shape,step_expUserTimeDecayFactor.shape)
                        #(8,) (8, 25, 4, 64) (8, 25, 4, 1) (8, 25) (8, 4)  B=8
                        for expSum,stepExpInfo in zip([expUser,     expUserP,     expUserIS,],
                                                      [step_expUser,step_expUserP,step_expUserIS]):
                            expSum.extend(stepExpInfo)

            else:
                if batch_data_input:
                    # batch_pos_data_input = dict()
                    # for feedKey, batchData in batch_data_input.items():
                    #     posData = batchData[::group]
                    #     batch_pos_data_input[feedKey] = posData
                    step_pred, step_labels, step_recItemListScore,step_posItem,step_histItems,item_lookup= self.eval(load_sess, batch_data_input)
                    preds.extend(np.reshape(step_pred, -1))
                    labels.extend(np.reshape(step_labels, -1))
                    group_preds.extend(np.reshape(step_pred, (-1, group)))
                    group_labels.extend(np.reshape(step_labels, (-1, group)))

                    recItemListScores.extend(np.squeeze(step_recItemListScore))#[B,vocNum,1]
                    posItems.extend(step_posItem)#[B.]
                    histItems.extend(step_histItems)#[B,T]
        res = cal_metric(labels, preds, self.hparams.metrics)
        if "pairwise_metrics" in self.hparams.values():
            res_pairwise = cal_metric(
                group_labels, group_preds, self.hparams.pairwise_metrics
            )
            res.update(res_pairwise)
        if "other_metrics_basedOn_RecList" in self.hparams.values():
            res_metrics_basedOn_RecList=\
                self.cal_metrics_basedOn_RecList(recItemListScores,posItems,histItems,item_lookup,self.hparams.other_metrics_basedOn_RecList)
            res.update(res_metrics_basedOn_RecList)
        if  self.hparams.expAnalyse:
            if self.hparams.useTimeBochner:
                self.interestSemanticTopkSim(expUserSummary=[expUser,expUserP,expUserIS,step_expUserTime_from_first,group_preds,group_labels],item_lookup=item_lookup)
            else:
                self.interestSemanticTopkSim(
                expUserSummary=[expUser, expUserP, expUserIS, group_preds,
                                group_labels], item_lookup=item_lookup)

        return res

    def cal_metrics_basedOn_RecList(self,recItemListScore,posItem,histItems,item_lookup,metircs):
        res={}
        for metric in metircs:
            if metric.startswith("ILD"):
                ILD_list = [1, 2]
                ks = metric.split("@")
                if len(ks) > 1:
                    ILD_list = [int(token) for token in ks[1].split(";")]
                for k in ILD_list:
                    ILD_temp = np.mean(
                        [
                            self.ILD_score(each_recItemListScore,each_posItem,each_histItems,item_lookup, k)
                            for each_recItemListScore,each_posItem,each_histItems in zip(recItemListScore,posItem,histItems)
                        ]
                    )
                    res["ILD@{0}".format(k)] = round(ILD_temp, 4)
        return res

    def ILD_score(self,recItemListScore,posItem,histItems,item_lookup,k):
        #disType:[disType="Euclidean","cosSim]
        k = min(np.shape(recItemListScore)[-1], k)
        dropHistItem=np.array(list(set(range(self.item_lookup.shape[0]))-set(histItems)))
        dropHistItemScore=recItemListScore[dropHistItem]
        order = np.argsort(dropHistItemScore)[::-1]#降序排序
        recItemListScore_at_k = np.take(dropHistItemScore, order[:k])

        #map2 org id
        itemLookUpIdxs_at_k=dropHistItem[order[:k]]
        itemEmd_at_k=item_lookup[itemLookUpIdxs_at_k]
        #[K,H]
        #cal intra-list dis
        #D_{i,j}=(item_{i}-item_{j})^2=item_{i}^2-2*item_{i}*item_{j}+item_{j}^2
        dis = 1-cosine_similarity(itemEmd_at_k, itemEmd_at_k)#[K,K]
        diag_mask=np.diag(np.diagonal(dis))
        ILD=np.sum(dis-diag_mask)*2/(k*(k-1))

        return ILD
    def interestSemanticTopkSim(self,expUserSummary,item_lookup):
        '''
        输入：
        1. 每个用户不同时刻的K个兴趣 [B,T,K,H]
        2.
        输出：
        1. 计算每个用户的K个兴趣对应的topk sim物品，为兴趣添加语义解释  [U*T,K,H]
        2. 输出每个用户不同时间步的兴趣强度IS值，方便可视化分析用户的兴趣强度随时间的变化 [B,T,K]
        3. 输出用户的交互时间 []
        4. 输出用户的个性化兴趣衰减系数 [U,K]

        Returns:
        '''

        for idx,info in enumerate(expUserSummary):
            dataShape=info[0].shape
            expUserSummary[idx]=np.array(info).reshape((-1,*dataShape))
        if self.hparams.useTimeBochner:
            expUser,expUserP,expUserIS,step_expUserTime_from_first,group_preds,group_labels=expUserSummary
        else:
            expUser,expUserP,expUserIS,group_preds,group_labels=expUserSummary
        userDict,itemDict,cateDict=self.iterator.userdict,self.iterator.itemdict,self.iterator.catedict
        testUser=np.random.choice(list(userDict.values()),5)
        print("+++++++++++++\ntestUser: {}+++++++++++++++++\n".format(testUser))
        #/home/lisong/SNA/Disentangling_Interest_Network_for_Sequential_Recommendation/examples/00_quick_start
        logPath="./expLog"
        for uid in testUser:
            dataIdxOfUid=np.where(expUser==uid)
            if len(dataIdxOfUid[0])>0:
                dataIdxOfUid=dataIdxOfUid[0][0]#==> get scalar idx
            else:
                continue
            userInterest=expUserP[dataIdxOfUid]#[T,S,H]
            userIS=expUserIS[dataIdxOfUid]#[T,S,1]
            if self.hparams.useTimeBochner:
                userTimeToNow=step_expUserTime_from_first[dataIdxOfUid]#[T]
                with open(logPath+"/{}_expData.pkl".format(uid),"wb") as outLog:
                    pickle.dump([dataIdxOfUid,userInterest,userIS,userTimeToNow],outLog)
            else:
                with open(logPath + "/{}_expData.pkl".format(uid), "wb") as outLog:
                    pickle.dump([dataIdxOfUid, userInterest, userIS], outLog)

        with open(logPath + "/dicts.pkl", "wb") as outLog:
            pickle.dump([userDict,itemDict,cateDict], outLog)
        with open(logPath + "/itemLookUp.pkl", "wb") as outLog:
            pickle.dump(item_lookup, outLog)

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
            if not self.hparams.expAnalyse:
                return sess.run([self.pred, self.iterator.labels], feed_dict=feed_dict)
            elif self.hparams.useTimeBochner:
                return sess.run([self.pred, self.iterator.labels,
                                 self.item_lookup,
                                 self.expUser,
                                 self.expUserP,
                                 self.exptUserIS,
                                 self.step_expUserTime_from_first
                                 ], feed_dict=feed_dict)
            else:
                return sess.run([self.pred, self.iterator.labels,
                                 self.item_lookup,
                                 self.expUser,
                                 self.expUserP,
                                 self.exptUserIS], feed_dict=feed_dict)
        else:
            return sess.run([self.pred, self.iterator.labels,self.recItemListScore,self.iterator.items,self.iterator.item_history,self.item_lookup], feed_dict=feed_dict)

