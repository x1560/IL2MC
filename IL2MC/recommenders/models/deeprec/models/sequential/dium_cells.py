# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import tensorflow as tf
from keras.layers.legacy_rnn.rnn_cell_impl import LayerRNNCell
from tensorflow.python.eager import context
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops import init_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.util import nest


_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


class BaseDIUMCell(LayerRNNCell):
    """Cell for Sequential User Matrix"""

    def __init__(
        self,
        num_units,
        slots,
        attention_size,
        input_size,
        activation=None,
        reuse=None,
        kernel_initializer=None,
        bias_initializer=None,
        name=None,
        dtype=None,
        _beta=1.02,
        hparams=None,
        **kwargs
    ):
        super(BaseDIUMCell, self).__init__(_reuse=reuse, name=name, dtype=dtype, **kwargs)
        _check_supported_dtypes(self.dtype)

        if context.executing_eagerly() and context.num_gpus() > 0:
            logging.warn(
                "%s: Note that this cell is not optimized for performance. "
                "Please use keras.layers.cudnn_recurrent.CuDNNGRU for better "
                "performance on GPU.",
                self,
            )

        self._input_size = input_size
        self._slots = slots  # the last channel is reserved for the highway slot
        self._num_units = num_units
        if "2way" in hparams.cell:
            self._real_units = (self._num_units - input_size) // (slots*2)
        else:
            ssl_dim=1 if hparams.use_delta_is_ssl_loss else 0
            self._real_units = (self._num_units-ssl_dim) // slots

        self._beta=_beta
        self.hparams=hparams
        if activation:
            self._activation = activations.get(activation)
        else:
            self._activation = math_ops.tanh
        self._kernel_initializer = initializers.get(kernel_initializer)
        self._bias_initializer = initializers.get(bias_initializer)

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def _basic_build(self, inputs_shape):
        """Common initialization operations for SUM cell and its variants.
        This function creates parameters for the cell.
        """


        h = self._real_units#slot dim
        s = self._slots
        d = inputs_shape[-1]
        if self.hparams.useTimeSliRec:
            d=d-2
        self._erase_W_IP = self.add_variable(
            name="_erase_W_IP", shape=[d + h, h], initializer=self._kernel_initializer
        )
        self._erase_b_IP = self.add_variable(
            name="_erase_b_IP",
            shape=[h],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)
            ),
        )

        self._reset_W_IP = self.add_variable(
            name="_reset_W_IP", shape=[d + h, 1], initializer=self._kernel_initializer
        )
        self._reset_b_IP = self.add_variable(
            name="_reset_b_IP",
            shape=[1],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)
            ),
        )

        self._add_W_IP = self.add_variable(
            name="_add_W_IP", shape=[d + h, h], initializer=self._kernel_initializer
        )
        self._add_b_IP = self.add_variable(
            name="_add_b_IP",
            shape=[h],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)
            ),
        )
        self._erase_W_IS = self.add_variable(
            name="_erase_W_IS", shape=[d+h, h], initializer=self._kernel_initializer
        )
        self._erase_b_IS = self.add_variable(
            name="_erase_b_IS",
            shape=[h],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)
            ),
        )

        self._reset_W_IS = self.add_variable(
            name="_reset_W_IS", shape=[d+h, 1], initializer=self._kernel_initializer
        )
        self._reset_b_IS = self.add_variable(
            name="_reset_b_IS",
            shape=[1],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)
            ),
        )

        self._add_W_IS = self.add_variable(
            name="_add_W_IS", shape=[d+h, h], initializer=self._kernel_initializer
        )
        self._add_b_IS = self.add_variable(
            name="_add_b_IS",
            shape=[h],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)
            ),
        )

        self._beta = self.add_variable(
            name="_beta_no_reg",
            shape=(),
            initializer=tf.compat.v1.constant_initializer(
                np.array([self._beta]), dtype=np.float32
            ),
        )
        self._alpha = self.add_variable(
            name="_alpha_no_reg",
            shape=(),
            initializer=tf.compat.v1.constant_initializer(
                np.array([0.98]), dtype=np.float32
            ),
        )
    @tf_utils.shape_type_conversion
    def build(self, inputs_shape):
        """Initialization operations for SUM cell.
        this function creates all the parameters for the cell.
        """
        if inputs_shape[-1] is None:
            raise ValueError(
                "Expected inputs.shape[-1] to be known, saw shape: %s"
                % str(inputs_shape)
            )
        _check_supported_dtypes(self.dtype)
        d = inputs_shape[-1]  # noqa: F841
        h = self._real_units  # noqa: F841
        s = self._slots  # noqa: F841   num of common channel,the last channel is reserved for the highway slot

        self._basic_build(inputs_shape)

        self.parameter_set = [
            self._erase_W_IP,
            self._erase_b_IP,
            self._reset_W_IP,
            self._reset_b_IP,
            self._add_W_IP,
            self._add_b_IP,

            self._erase_W_IS,
            self._erase_b_IS,
            self._reset_W_IS,
            self._reset_b_IS,
            self._add_W_IS,
            self._add_b_IS,
        ]

        self.built = True

    def call(self, inputs, state):
       return state,state

    def get_config(self):
        return


class DIUMCellIP_MERGE(BaseDIUMCell):
    """A variant of SUM cell, which upgrades the writing attention"""

    @tf_utils.shape_type_conversion
    def build(self, inputs_shape):
        """Initialization operations for SUMV2 cell.
        this function creates all the parameters for the cell.
        """
        if inputs_shape[-1] is None:
            raise ValueError(
                "Expected inputs.shape[-1] to be known, saw shape: %s"
                % str(inputs_shape)
            )
        _check_supported_dtypes(self.dtype)
        d = inputs_shape[-1]
        h = self._real_units#slot dim
        s = self._slots#common channel

        self._basic_build(inputs_shape)

        self._writing_W_IP = self.add_variable(
            name="_writing_W_IP", shape=[d + h, h], initializer=self._kernel_initializer
        )
        self._writing_b_IP = self.add_variable(
            name="_writing_b_IP",
            shape=[h],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)
            ),
        )
        self._writing_W02_IP = self.add_variable(
            name="_writing_W02_IP", shape=[h, s], initializer=self._kernel_initializer
        )

        self.parameter_set = [
            self._erase_W_IP,
            self._erase_b_IP,
            self._reset_W_IP,
            self._reset_b_IP,
            self._add_W_IP,
            self._add_b_IP,

            self._writing_W_IP,
            self._writing_W02_IP,
            self._writing_b_IP,

        ]

        self.built = True

    def squash(self,IP):
        is_mod_2 = tf.reduce_sum(tf.square(IP), axis=-1)  # [B,S]
        is_att_logit_mat = is_mod_2 / (1 + is_mod_2)  # [B,S]
        is_att_weights = tf.nn.softmax(self._beta * is_att_logit_mat, axis=-1)
        is_att_weights = tf.expand_dims(is_att_weights, axis=2)  # [B,S,1]
        return is_att_weights
    def call(self, inputs, state):
        """The real operations for SUMV2 cell to process user behaviors.

        Args:
            inputs: (a batch of) user behaviors at time T
            state:  (a batch of) user states at time T-1

        Returns:
            state: after process the user behavior at time T, returns (a batch of) new user states at time T
            state: after process the user behavior at time T, returns (a batch of) new user states at time T
        """
        _check_rnn_cell_input_dtypes([inputs, state])

        h = self._real_units
        s = self._slots
        ip_state = state
        # n_a, n_b = tf.nn.l2_normalize(last_instance, 1), tf.nn.l2_normalize(inputs, 1)  # [B,D]
        # dist = tf.expand_dims(tf.reduce_sum(input_tensor=n_a * n_b, axis=1), 1)  # [b,1]
        # dist = tf.math.pow(self._alpha, dist)

        ip_state = tf.reshape(ip_state, [-1, s, h])#common+highway channel

        #cos sim
        dotProd=tf.expand_dims(inputs,1)*ip_state#IAI compute   [B,1,H]*[B,S,H]=>  [B,S,H]  A*B
        dotProd=tf.reduce_sum(dotProd, axis=-1)  # [B,S,H]=>[B,S]
        mulMod=tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(inputs),-1)),-1)*\
               tf.sqrt(tf.reduce_sum(tf.square(ip_state),-1))#||A||*||B||  [B,1]*[B,S]==>[B,S]
        ip_att_logit_mat=dotProd/mulMod
        ip_att_weights = tf.nn.softmax(self._beta * ip_att_logit_mat, axis=-1)
        ip_att_weights = tf.expand_dims(ip_att_weights, 2)#[B,S,1]

        #squash IP get ISI
        is_att_weights=self.squash(ip_state)
        # is_att_weights=tf.matmul(is_state[:,:self._slots,:],self._squashW)#[B,S,H] [H,1] => [b,s,1]
        # is_att_weights = tf.nn.softmax(self._beta * is_att_weights, axis=1)

        IPI_ISI_Att = ip_att_weights * is_att_weights
        ip_hat = tf.reduce_sum(
            input_tensor=tf.multiply(ip_state, IPI_ISI_Att), axis=1
        )
        # ip_hat = (ip_hat + ip_state[:, self._slots, :]) / 2#[B,H]

        # is_hat = (is_hat + is_state[:, self._slots, :]) / 2  # [B,H]

        # IP&IS Updating
        ## IP Updating
        ip_writing_input = tf.concat([inputs, ip_hat], axis=1)#[B,d+h]
        ip_att_weights = tf.compat.v1.nn.xw_plus_b(
            ip_writing_input, self._writing_W_IP, self._writing_b_IP
        )#[B,h]
        ip_att_weights = tf.nn.relu(ip_att_weights)
        ip_att_weights = tf.matmul(ip_att_weights, self._writing_W02_IP)
        #B,s
        ip_att_weights = tf.nn.softmax(ip_att_weights, axis=-1)
        ip_att_weights = tf.expand_dims(ip_att_weights, 2)#[B,S,1]

        # ip_att_weights = ip_att_weights * tf.expand_dims(dist, 1)#[b,1,1]
        # Joint IPI ISI Attention

        reset = tf.sigmoid(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, ip_hat], axis=-1), self._reset_W_IP, self._reset_b_IP
            )
        )
        erase = tf.sigmoid(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, ip_hat], axis=-1), self._erase_W_IP, self._erase_b_IP
            )
        )#[B,D+H]
        add = tf.tanh(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, reset * ip_hat], axis=-1), self._add_W_IP, self._add_b_IP
            )
        )

        ip_state01 = ip_state * (
            tf.ones_like(ip_state) - ip_att_weights * tf.expand_dims(erase, 1)
        )
        ip_state01 = ip_state01 + ip_att_weights * tf.expand_dims(erase, 1) * tf.expand_dims(
            add, 1
        )
        ip_state01 = tf.reshape(ip_state01, [-1, self._slots * self._real_units])

        return ip_state01, ip_state01
class DIUMCell(BaseDIUMCell):
    """Time-aware and Interaction-aware Interest Evolution"""

    @tf_utils.shape_type_conversion
    def build(self, inputs_shape):
        """Initialization operations for  cell.
        this function creates all the parameters for the cell.
        """
        if inputs_shape[-1] is None:
            raise ValueError(
                "Expected inputs.shape[-1] to be known, saw shape: %s"
                % str(inputs_shape)
            )
        _check_supported_dtypes(self.dtype)

        h = self._real_units#slot dim
        s = self._slots#common channel
        d = inputs_shape[-1]
        if self.hparams.useTimeSliRec:
            d=d-2
        self.epsilon=1e-7
        self._basic_build(inputs_shape)

        self._writing_W_IP = self.add_variable(
            name="_writing_W_IP", shape=[d + h, h], initializer=self._kernel_initializer
        )
        self._writing_b_IP = self.add_variable(
            name="_writing_b_IP",
            shape=[h],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)
            ),
        )
        self._writing_W02_IP = self.add_variable(
            name="_writing_W02_IP", shape=[h, s], initializer=self._kernel_initializer
        )


        if self.hparams.useTimeSliRec:
            self._time_input_w1 = self.add_variable(
                name="_time_input_w1", shape=[self.hparams.hidden_size], initializer=self._kernel_initializer
            )
            self._time_input_bias1 = self.add_variable(
                name="_time_input_bias1", shape=[self.hparams.hidden_size], initializer=self._kernel_initializer
            )
            self._time_input_w2 = self.add_variable(
                name="_time_input_w2", shape=[self.hparams.hidden_size], initializer=self._kernel_initializer
            )
            self._time_input_bias2 = self.add_variable(
                name="_time_input_bias2", shape=[self.hparams.hidden_size], initializer=self._kernel_initializer
            )
            self._time_kernel_w1 = self.add_variable(
                name="_time_kernel_w1",
                shape=[self._real_units, self._real_units],
                initializer=self._kernel_initializer
            )
            self._time_kernel_t1 = self.add_variable(
                name="_time_kernel_t1",
                shape=[self._real_units, self._real_units],
                initializer=self._kernel_initializer
            )
            self._time_bias1 = self.add_variable(
                name="_time_bias1", shape=[self._real_units], initializer=self._kernel_initializer
            )
            self._time_kernel_w2 = self.add_variable(
                name="_time_kernel_w2",
                shape=[self._real_units, self._real_units],
                initializer=self._kernel_initializer
            )
            self._time_kernel_t2 = self.add_variable(
                name="_time_kernel_t2",
                shape=[self._real_units, self._real_units],
                initializer=self._kernel_initializer
            )
            self._time_bias2 = self.add_variable(
                name="_time_bias2", shape=[self._real_units], initializer=self._kernel_initializer
            )

        if self.hparams.useISIPAttGatingFactor:
            self.ISIPAttGatingFactor= self.add_variable(
                name="_ISIPAttGatingFactor", shape=[],initializer=tf.constant_initializer(0.9)
            )


        self.efficientCapsVersion="enism"
        if self.efficientCapsVersion=="normalVer":
            self.effCapsBias = self.add_variable(shape=[s, 1, s], name='effCapsBias')
        else:
            self.effCapsW = self.add_variable(shape=[s, s, self._real_units, self._real_units],
                                              initializer=self._kernel_initializer,
                                              name='effCapsW')#[capsN, input_N, input_D, capsD]
            self.effCapsBias = self.add_variable(shape=[s, s, 1],  name='effCapsBias')#[capsN, input_N,1]

        self.parameter_set = [
            self._erase_W_IP,
            self._erase_b_IP,
            self._reset_W_IP,
            self._reset_b_IP,
            self._add_W_IP,
            self._add_b_IP,

            self._writing_W_IP,
            self._writing_W02_IP,
            self._writing_b_IP,

            self.effCapsW,
            self.effCapsBias
        ]
        if self.hparams.useTimeSliRec:
            self.parameter_set.extend(
                [
                    self._time_input_w1,
                    self._time_input_bias1,
                    self._time_input_w2,
                    self._time_input_bias2,
                    self._time_kernel_w1,
                    self._time_kernel_t1,
                    self._time_bias1,
                    self._time_kernel_w2,
                    self._time_kernel_t2,
                    self._time_bias2,
                 ]
            )
        self.built = True


    def hintonSquash(self,s):
        n = tf.norm(s, axis=-1, keepdims=True)
        modLen=n ** 2 / (1 + n ** 2)
        return modLen,tf.multiply( modLen/ (n + self.epsilon), s)
    def effCapsSquash(self,s):
        """
        Squash activation used in 'Efficient-CapsNet: Capsule Network with Self-Attention Routing'.
        """
        n = tf.norm(s, axis=-1, keepdims=True)
        mod=1 - 1 / (tf.math.exp(n) + self.epsilon)#[B,S,1]
        return mod , mod* (s/ (n + self.epsilon))

    def efficientCaps(self,primary_caps):
        # version="normalVer" #[normalVer,enism]
        if self.efficientCapsVersion=="normalVer":
            #IP=primary_caps [B,S,H]    B,S,S,H,1
            # U.shape: [None, num_digit_caps, num_primary_caps, dim_primary_caps, 1]
            # primary_caps [None,num_primary_caps,dim_primary_caps]
            # print(primary_caps.shape)#(None,9, 32)
            U = tf.expand_dims(tf.tile(tf.expand_dims(primary_caps, axis=1),[1, self._slots, 1, 1]),
                               axis=-1,
                               name="digit_cap_inputs")
            # print(U.shape)#(None, 9, 9, 32, 1)
            #self.effCapsW  (9, 9, 32, 32)
            # U_hat.shape: [None, num_digit_caps, num_primary_caps, dim_digit_caps]  9,9,32
            # map fn :[num_digit_caps, num_primary_caps, dim_digit_caps]   u_i=(9, 9, 32, 1),effCapsW=(9, 9, 32, 32)  i,j 32
            # U_hat = tf.squeeze(tf.map_fn(lambda u_i: tf.matmul(self.effCapsW, u_i), U),
            #                    axis=-1,
            #                    name="digit_cap_predictions")
            U_hat=tf.squeeze(tf.matmul(tf.expand_dims(self.effCapsW,0), U),axis=-1)#[1,9, 9, 32, 32] (None, 9, 9, 32, 1)

            # A.shape: [None, num_digit_caps, num_primary_caps, num_primary_caps]
            A = tf.matmul(
                U_hat, U_hat, transpose_b=True, name="digit_cap_attentions")/tf.sqrt(tf.cast(self._real_units, tf.float32))
            # print(A.shape)  # (None, 9, 9, 32, 1)
            # C.shape: [None, num_digit_caps, 1, num_primary_caps]
            C = tf.nn.softmax(tf.reduce_sum(A, axis=-2, keepdims=True),
                              axis=-2,
                              name="digit_cap_coupling_coefficients")
            # S.shape: [None, num_digit_caps, dim_digit_caps]
            S = tf.squeeze(tf.matmul(C + self.effCapsBias, U_hat), axis=-2)
            modLen,v = self.effCapsSquash(S)  # v shape=(None,N,D)   [B,S2,H]
            # self.eps = 10e-21
            # n = tf.norm(S, axis=-1, keepdims=True)
            # mod = 1 - 1 / (tf.math.exp(n) + self.eps)  # [B,S,1]

            return modLen, v
        else:
            # [B,S1,H1]  [S2,S1,H1,H]
            u = tf.compat.v1.einsum('...ji,kjiz->...kjz', primary_caps,
                                    self.effCapsW)  # u shape=(None,N,H*W*input_N,D)  [B,S2,S1,H]
            print(u.shape)
            # [Si=0:S1,H1] S2 [H1,H]
            # u = tf.compat.v1.Print(u, [u], message="u", summarize=9)
            c = tf.compat.v1.einsum('...ij,...kj->...i', u,
                                    u)[...,None]  # b shape=(None,N,H*W*input_N,1) -> (None,j,i,1)    [B,i,  S1,S1]
            print(c.shape)

            # c = tf.expand_dims(c, -1)
            # c = tf.compat.v1.Print(c, [c], message="c", summarize=9)
            c = c / tf.sqrt(tf.cast(self._real_units, tf.float32))  # [outs,ins, 1]
            c = tf.nn.softmax(c, axis=1)  # c shape=(None,N,H*W*input_N,1) -> (None,j,i,1)
            c = c + self.effCapsBias  # [outs,ins, 1]
            s = tf.reduce_sum(tf.multiply(u, c), axis=-2)  # s shape=(None,N,D)
            modLen, v = self.effCapsSquash(s)  #modLen,[B,S,1] v shape=(None,N,D)   [B,S2,H]
            print(v.shape)
            return modLen,v

    def squash(self,IP,needSoftmax=True,needExpd=True):
        if self.hparams.squashType=="hintion":
            is_att_weights,_=self.hintonSquash(IP)#return [B,S,1]
        elif self.hparams.squashType=="efficient":
            is_att_weights,_=self.efficientCaps(IP)#return [B,S,1]
        else:
            is_att_weights = self.hintonSquash(IP)  # return [B,S,1]
        print(is_att_weights.shape)
        if needSoftmax:
            is_att_weights = tf.nn.softmax(is_att_weights, axis=-2)# | [b,t,s]
        print(is_att_weights.shape)
        if not needExpd:
            is_att_weights=tf.squeeze(is_att_weights,axis=-1)  #[B,S]
        print(is_att_weights.shape)
        return is_att_weights

    def call(self, inputs, state):
        _check_rnn_cell_input_dtypes([inputs, state])

        h = self._real_units
        s = self._slots

        if self.hparams.use_delta_is_ssl_loss and self.hparams.useSquash:
            ip_state = state[:,:-1]
        else:
            ip_state = state

        ip_state = tf.reshape(ip_state, [-1, s, h])
        if self.hparams.useTimeSliRec:
            time_now_score = tf.expand_dims(inputs[:, -1], -1)
            time_last_score = tf.expand_dims(inputs[:, -2], -1)
            inputs = inputs[:, :-2]
            time_now_input = tf.nn.tanh(
                time_now_score * self._time_input_w1 + self._time_input_bias1
            )  # [B,T,D]
            time_last_input = tf.nn.tanh(
                time_last_score * self._time_input_w2 + self._time_input_bias2
            )  # [B,T,D]

            #time state
            time_now_state = tf.nn.sigmoid(
                   tf.matmul(inputs, self._time_kernel_w1)
                    +tf.matmul(time_now_input, self._time_kernel_t1)
                    + self._time_bias1
            )
            time_last_state =tf.nn.sigmoid (
                    tf.matmul(inputs, self._time_kernel_w2)
                    +tf.matmul(time_last_input, self._time_kernel_t2)
                    + self._time_bias2
            )#[B,D]
        if self.hparams.itemIPSim == "cosSim":
            # cos sim
            norm_inputs = tf.expand_dims(tf.nn.l2_normalize(inputs, axis=-1), 1)  # [B,1,H]
            norm_ip_state = tf.nn.l2_normalize(ip_state, axis=-1)  # [B,S,H]
            ip_att_logit_mat = tf.reduce_sum(norm_inputs * norm_ip_state, axis=-1, keepdims=True)  # [B,S,H]=>[B,S,1]
        elif self.hparams.itemIPSim == "dotProduct":
            ip_att_logit_mat = tf.reduce_sum(tf.expand_dims(inputs, 1) * ip_state, axis=-1, keepdims=True)  # [B,S,1]

        ip_read_att_weights = tf.nn.softmax(self._beta * ip_att_logit_mat, axis=1)  # [B,S,1] along S

        #squash IP get ISI
        if self.hparams.useSquash:
            if self.hparams.use_delta_is_ssl_loss:
                curr_is_att_weights=self.squash(ip_state,False,False)# [B,S]
                is_att_weights=curr_is_att_weights
                # is_att_weights = tf.nn.softmax(self._beta * curr_is_att_weights, axis=-1)
                # print(is_att_weights.shape)
                is_att_weights = tf.expand_dims(is_att_weights, axis=-1)  # [B,S,1]  | [B,T,S,1]
                print(is_att_weights.shape)
            else:
                is_att_weights=self.squash(ip_state)# [B,S,1]


            # ip_read_att_weights = tf.compat.v1.Print(ip_read_att_weights, [ip_read_att_weights[:2,:4,0]],
            #                                          message="ip_read_att_weights",summarize=8)
            # is_att_weights = tf.compat.v1.Print(is_att_weights, [is_att_weights[0,:,0]], message="is_att_weights",summarize=9)

            if self.hparams.useISIPAttGatingFactor:
                # IPI_ISI_Att = ((tf.square(self.ISIPAttGatingFactor)+1)*ip_read_att_weights * is_att_weights)/\
                #               ( tf.square(self.ISIPAttGatingFactor)   *(ip_read_att_weights + is_att_weights))
                # [B,S,1]
                IPI_ISI_Att = self.ISIPAttGatingFactor * ip_read_att_weights +(1-self.ISIPAttGatingFactor)* is_att_weights
            else:
                IPI_ISI_Att = ip_read_att_weights * is_att_weights  # [B,S,1]

            IPI_ISI_Att=tf.nn.softmax(IPI_ISI_Att, axis=1)
            # IPI_ISI_Att = tf.compat.v1.Print(IPI_ISI_Att, [IPI_ISI_Att[:2,:4,0]], message="IPI_ISI_Att",summarize=8)

            ip_hat = tf.reduce_sum(ip_state*IPI_ISI_Att, axis=1)

        else:
            ip_hat = tf.reduce_sum(ip_state * ip_read_att_weights, axis=1)

        ## IP Updating
        ip_writing_input = tf.concat([inputs, ip_hat], axis=1)#[B,d+h]
        ip_writ_att_weights = tf.compat.v1.nn.xw_plus_b(
            ip_writing_input, self._writing_W_IP, self._writing_b_IP
        )#[B,h]
        ip_writ_att_weights = tf.matmul(tf.nn.elu(ip_writ_att_weights), self._writing_W02_IP)
        ip_writ_att_weights = tf.nn.softmax(self._beta *ip_writ_att_weights, axis=-1)


        reset = tf.sigmoid(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, ip_hat], axis=-1), self._reset_W_IP, self._reset_b_IP
            )
        )
        erase = tf.sigmoid(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, ip_hat], axis=-1), self._erase_W_IP, self._erase_b_IP
            )
        )#[B,D+H]
        add = tf.tanh(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, reset * ip_hat], axis=-1), self._add_W_IP, self._add_b_IP
            )
        )

        weight_erase=tf.expand_dims(ip_writ_att_weights,-1) * tf.expand_dims(erase, 1)#[b,s,h]
        # weight_erase=tf.compat.v1.Print(weight_erase,[weight_erase[0:2,0:2,0:5]],message="weight_erase",summarize=20)
        if self.hparams.useTimeSliRec:
            next_ip_state = tf.expand_dims(time_last_state,-2) *ip_state * (
                tf.ones_like(ip_state) - weight_erase
            )#[B,1,D]  [b,s,h]
            next_ip_state = next_ip_state + weight_erase * tf.expand_dims(add, 1)* tf.expand_dims(time_now_state,-2)
        else:
            next_ip_state =  ip_state * (
                    tf.ones_like(ip_state) - weight_erase
            )  # [B,1,D]  [b,s,h]
            next_ip_state = next_ip_state + weight_erase * tf.expand_dims(add, 1)
        #ip_read_att_weights[B,S,1]
        # [0,1]-[0,1]=>[-1,1]
        # SSL on next_is_att
        if self.hparams.use_delta_is_ssl_loss and self.hparams.useSquash:
            version = "BPR"
            if version == "softmaxListRank":
                next_is_att_weights = self.squash(next_ip_state, False, False)  # [B,S]
                delta_is_att = next_is_att_weights - curr_is_att_weights  # [B,S]
                delta_is_ssl_loss = tf.reduce_mean(
                    input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=delta_is_att,
                        labels=ip_writ_att_weights,
                    ), axis=-1, keepdims=True
                )  # [B,1]
            elif version == "BPR":
                next_is_att_weights = self.squash(next_ip_state, False, False)  # [B,S]
                delta_is_att = next_is_att_weights - curr_is_att_weights  # [B,S]
                IAI_ij_diff = tf.expand_dims(ip_writ_att_weights, -1) - tf.expand_dims(ip_writ_att_weights,
                                                                                       -2)  # [B,S,S]
                delta_is_att_ij_diff = tf.expand_dims(delta_is_att, -1) - tf.expand_dims(delta_is_att, -2)  # [B,S,S]
                IAI_ij_diff_greater = tf.greater(IAI_ij_diff, tf.zeros_like(IAI_ij_diff))
                delta_is_att_ij_diff_factor = tf.where(IAI_ij_diff_greater, tf.ones_like(IAI_ij_diff),
                                                       -tf.ones_like(IAI_ij_diff))

                diagEle = tf.linalg.diag(tf.linalg.diag_part(delta_is_att_ij_diff_factor))
                delta_is_att_ij_diff_factor = delta_is_att_ij_diff_factor - diagEle
                delta_is_ssl_loss = -tf.math.log(
                    tf.nn.sigmoid(delta_is_att_ij_diff_factor * delta_is_att_ij_diff))  # [B,S,S]
                delta_is_ssl_loss = tf.reduce_sum(
                    delta_is_ssl_loss, axis=-1, keepdims=False
                )  # [B,S]

                delta_is_ssl_loss = tf.reduce_sum(
                    delta_is_ssl_loss, axis=-1, keepdims=True
                )  # [B,1]

        # ip_read_att_weights[B,S,1]
        ip_state01 = tf.reshape(next_ip_state, [-1, self._slots * self._real_units])
        # ip_state01 = tf.compat.v1.Print(ip_state01, [ip_state01], message="ip_state01", summarize=9)

        if self.hparams.use_delta_is_ssl_loss and self.hparams.useSquash:
            ip_state01 = tf.concat([ip_state01, delta_is_ssl_loss], -1)

        return ip_state01, ip_state01
class DIUMCellSSLIS_timeEmbeddingConcat(BaseDIUMCell):
    """Time-aware and Interaction-aware Interest Evolution"""

    @tf_utils.shape_type_conversion
    def build(self, inputs_shape):
        """Initialization operations for  cell.
        this function creates all the parameters for the cell.
        """
        if inputs_shape[-1] is None:
            raise ValueError(
                "Expected inputs.shape[-1] to be known, saw shape: %s"
                % str(inputs_shape)
            )
        _check_supported_dtypes(self.dtype)

        h = self._real_units#slot dim
        s = self._slots#common channel
        d = inputs_shape[-1]

        self._basic_build(inputs_shape)

        self._writing_W_IP = self.add_variable(
            name="_writing_W_IP", shape=[d + h, h], initializer=self._kernel_initializer
        )
        self._writing_b_IP = self.add_variable(
            name="_writing_b_IP",
            shape=[h],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)
            ),
        )
        self._writing_W02_IP = self.add_variable(
            name="_writing_W02_IP", shape=[h, s], initializer=self._kernel_initializer
        )

        self.parameter_set = [
            self._erase_W_IP,
            self._erase_b_IP,
            self._reset_W_IP,
            self._reset_b_IP,
            self._add_W_IP,
            self._add_b_IP,

            self._writing_W_IP,
            self._writing_W02_IP,
            self._writing_b_IP,

        ]

        self.built = True

    def squash(self,IP,needSoftmax=True,needExpd=True):
        is_mod_2 = tf.reduce_sum(tf.square(IP), axis=-1)  # [B,S] | [B,T,S]
        is_att_weights = is_mod_2 / (1 + is_mod_2)  # [B,S] | [B,T,S]
        if needSoftmax:
            is_att_weights = tf.nn.softmax(self._beta * is_att_weights, axis=-1)# | [b,t,s]
        if needExpd:
            is_att_weights = tf.expand_dims(is_att_weights, axis=-1)  # [B,S,1]  | [B,T,S,1]
        return is_att_weights
    def call(self, inputs, state):
        """
        Args:
            inputs: (a batch of) user behaviors at time T
            state:  (a batch of) user states at time T-1

        Returns:
            state: after process the user behavior at time T, returns (a batch of) new user states at time T
            state: after process the user behavior at time T, returns (a batch of) new user states at time T
        """
        _check_rnn_cell_input_dtypes([inputs, state])

        h = self._real_units
        s = self._slots
        if self.hparams.use_delta_is_ssl_loss:
            ip_state = state[:,:-1]
        else:
            ip_state = state
        # n_a, n_b = tf.nn.l2_normalize(last_instance, 1), tf.nn.l2_normalize(inputs, 1)  # [B,D]
        # dist = tf.expand_dims(tf.reduce_sum(input_tensor=n_a * n_b, axis=1), 1)  # [b,1]
        # dist = tf.math.pow(self._alpha, dist)
        ip_state = tf.reshape(ip_state, [-1, s, h])

        #cos sim
        dotProd=tf.expand_dims(inputs,1)*ip_state#IAI compute   [B,1,H]*[B,S,H]=>  [B,S,H]  A*B
        dotProd=tf.reduce_sum(dotProd, axis=-1)  # [B,S,H]=>[B,S]
        mulMod=tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(inputs),-1)),-1)*\
               tf.sqrt(tf.reduce_sum(tf.square(ip_state),-1))#||A||*||B||  [B,1]*[B,S]==>[B,S]
        ip_att_logit_mat=dotProd/mulMod
        ip_read_att_weights = tf.nn.softmax(self._beta * ip_att_logit_mat, axis=-1)
        ip_read_att_weights = tf.expand_dims(ip_read_att_weights, 2)#[B,S,1]

        #squash IP get ISI
        if self.hparams.use_delta_is_ssl_loss:
            curr_is_att_weights=self.squash(ip_state,False,False)# [B,S,1]
            is_att_weights = tf.nn.softmax(self._beta * curr_is_att_weights, axis=-1)
            is_att_weights = tf.expand_dims(is_att_weights, axis=-1)  # [B,S,1]  | [B,T,S,1]
        else:
            is_att_weights=self.squash(ip_state)# [B,S,1]

        IPI_ISI_Att = ip_read_att_weights * is_att_weights
        ip_hat = tf.reduce_sum(
            input_tensor=tf.multiply(ip_state, IPI_ISI_Att), axis=1
        )
        # ip_hat = (ip_hat + ip_state[:, self._slots, :]) / 2#[B,H]
        # is_hat = (is_hat + is_state[:, self._slots, :]) / 2  # [B,H]
        # IP&IS Updating
        ## IP Updating
        ip_writing_input = tf.concat([inputs, ip_hat], axis=1)#[B,d+h]
        ip_writ_att_weights = tf.compat.v1.nn.xw_plus_b(
            ip_writing_input, self._writing_W_IP, self._writing_b_IP
        )#[B,h]
        ip_writ_att_weights = tf.matmul(tf.nn.elu(ip_writ_att_weights), self._writing_W02_IP)
        #B,s
        ip_writ_att_weights = tf.nn.softmax(ip_writ_att_weights, axis=-1)
        # ip_att_weights = ip_att_weights * tf.expand_dims(dist, 1)#[b,1,1]
        # Joint IPI ISI Attention
        reset = tf.sigmoid(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, ip_hat], axis=-1), self._reset_W_IP, self._reset_b_IP
            )
        )
        erase = tf.sigmoid(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, ip_hat], axis=-1), self._erase_W_IP, self._erase_b_IP
            )
        )#[B,D+H]
        add = tf.tanh(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, reset * ip_hat], axis=-1), self._add_W_IP, self._add_b_IP
            )
        )
        next_ip_state = ip_state * (
            tf.ones_like(ip_state) - tf.expand_dims(ip_writ_att_weights, 2) * tf.expand_dims(erase, 1)
        )
        next_ip_state = next_ip_state + tf.expand_dims(ip_writ_att_weights, 2) * tf.expand_dims(erase, 1) * tf.expand_dims(
            add, 1
        )

        # [0,1]-[0,1]=>[-1,1]
        # SSL on next_is_att
        if self.hparams.use_delta_is_ssl_loss:
            version="BPR"
            if version=="softmaxListRank":
                next_is_att_weights = self.squash(next_ip_state, False, False)  # [B,S]
                delta_is_att = next_is_att_weights - curr_is_att_weights  # [B,S]
                delta_is_ssl_loss = tf.reduce_mean(
                    input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=delta_is_att,
                        labels=ip_writ_att_weights,
                    ),axis=-1,keepdims=True
                )#[B,1]
            elif version=="BPR":
                next_is_att_weights = self.squash(next_ip_state, False, False)  # [B,S]
                delta_is_att = next_is_att_weights - curr_is_att_weights  # [B,S]
                IAI_ij_diff=tf.expand_dims(ip_writ_att_weights,-1)-tf.expand_dims(ip_writ_att_weights,-2)#[B,S,S]
                delta_is_att_ij_diff=tf.expand_dims(delta_is_att,-1)-tf.expand_dims(delta_is_att,-2)#[B,S,S]
                IAI_ij_diff_greater=tf.greater(IAI_ij_diff,tf.zeros_like(IAI_ij_diff))
                delta_is_att_ij_diff_factor=tf.where(IAI_ij_diff_greater,tf.ones_like(IAI_ij_diff),
                                                    -tf.ones_like(IAI_ij_diff))

                diagEle=tf.linalg.diag(tf.linalg.diag_part(delta_is_att_ij_diff_factor))
                delta_is_att_ij_diff_factor=delta_is_att_ij_diff_factor-diagEle
                delta_is_ssl_loss=-tf.math.log(tf.nn.sigmoid(delta_is_att_ij_diff_factor*delta_is_att_ij_diff))#[B,S,S]
                delta_is_ssl_loss = tf.reduce_sum(
                    delta_is_ssl_loss, axis=-1, keepdims=False
                )  # [B,S]

                delta_is_ssl_loss = tf.reduce_sum(
                    delta_is_ssl_loss, axis=-1, keepdims=True
                )  # [B,1]

        #ip_read_att_weights[B,S,1]
        ip_state01 = tf.reshape(next_ip_state, [-1, self._slots * self._real_units])
        if self.hparams.use_delta_is_ssl_loss:
            ip_state01=tf.concat([ip_state01,delta_is_ssl_loss],-1)
        return ip_state01, ip_state01
class DIUM2WayEvoCellV1(BaseDIUMCell):
    """A variant of SUM cell, which upgrades the writing attention"""

    @tf_utils.shape_type_conversion
    def build(self, inputs_shape):
        """Initialization operations for SUMV2 cell.
        this function creates all the parameters for the cell.
        """
        if inputs_shape[-1] is None:
            raise ValueError(
                "Expected inputs.shape[-1] to be known, saw shape: %s"
                % str(inputs_shape)
            )
        _check_supported_dtypes(self.dtype)
        d = inputs_shape[-1]
        h = self._real_units#slot dim
        s = self._slots#common channel

        self._basic_build(inputs_shape)

        self._writing_W_IP = self.add_variable(
            name="_writing_W_IP", shape=[d + h, h], initializer=self._kernel_initializer
        )
        self._writing_b_IP = self.add_variable(
            name="_writing_b_IP",
            shape=[h],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)
            ),
        )
        self._writing_W02_IP = self.add_variable(
            name="_writing_W02_IP", shape=[h, s], initializer=self._kernel_initializer
        )

        self.parameter_set = [
            self._erase_W_IP,
            self._erase_b_IP,
            self._reset_W_IP,
            self._reset_b_IP,
            self._add_W_IP,
            self._add_b_IP,

            self._erase_W_IS,
            self._erase_b_IS,
            self._reset_W_IS,
            self._reset_b_IS,
            self._add_W_IS,
            self._add_b_IS,

            self._writing_W_IP,
            self._writing_W02_IP,
            self._writing_b_IP,

        ]

        self.built = True
    def squashIS(self,IS):
        is_mod_2 = tf.reduce_sum(tf.square(IS), axis=-1)  # [B,S]
        is_att_logit_mat = is_mod_2 / (1 + is_mod_2)  # [B,S]
        is_att_weights = tf.nn.softmax(self._beta * is_att_logit_mat, axis=-1)
        is_att_weights = tf.expand_dims(is_att_weights, axis=2)  # [B,S,1]
        return is_att_weights
    def call(self, inputs, state):
        """The real operations for SUMV2 cell to process user behaviors.

        Args:
            inputs: (a batch of) user behaviors at time T
            state:  (a batch of) user states at time T-1

        Returns:
            state: after process the user behavior at time T, returns (a batch of) new user states at time T
            state: after process the user behavior at time T, returns (a batch of) new user states at time T
        """
        _check_rnn_cell_input_dtypes([inputs, state])

        h = self._real_units
        s = self._slots
        ip_state,is_state, last_instance = state[:, : s * h], state[:, s * h : 2*s*h],state[:,2*s*h:]
        # n_a, n_b = tf.nn.l2_normalize(last_instance, 1), tf.nn.l2_normalize(inputs, 1)  # [B,D]
        # dist = tf.expand_dims(tf.reduce_sum(input_tensor=n_a * n_b, axis=1), 1)  # [b,1]
        # dist = tf.math.pow(self._alpha, dist)

        ip_state = tf.reshape(ip_state, [-1, s, h])#common+highway channel
        is_state = tf.reshape(is_state, [-1, s, h])#common+highway channel

        ip_att_logit_mat=tf.expand_dims(inputs,1)*ip_state[:,:self._slots,:]#IAI compute   [B,1,H]*[B,S,H]=>  [B,S,H]
        ip_att_logit_mat=tf.reduce_sum(ip_att_logit_mat,axis=-1)#[B,S,H]=>[B,S]

        ip_att_weights = tf.nn.softmax(self._beta * ip_att_logit_mat, axis=-1)
        ip_att_weights = tf.expand_dims(ip_att_weights, 2)#[B,S,1]

        #squash IS
        is_att_weights=self.squashIS(is_state[:,:self._slots,:])
        # is_att_weights=tf.matmul(is_state[:,:self._slots,:],self._squashW)#[B,S,H] [H,1] => [b,s,1]
        # is_att_weights = tf.nn.softmax(self._beta * is_att_weights, axis=1)

        #Joint UI Attention
        jointUIAtt=ip_att_weights*is_att_weights
        ip_hat = tf.reduce_sum(
            input_tensor=tf.multiply(ip_state[:, : self._slots, :], jointUIAtt), axis=1
        )
        # ip_hat = (ip_hat + ip_state[:, self._slots, :]) / 2#[B,H]

        is_hat = tf.reduce_sum(
            input_tensor=tf.multiply(is_state[:, : self._slots, :], jointUIAtt), axis=1
        )
        # is_hat = (is_hat + is_state[:, self._slots, :]) / 2  # [B,H]

        # IP&IS Updating
        ## IP Updating
        ip_writing_input = tf.concat([inputs, ip_hat], axis=1)#[B,d+h]
        ip_att_weights = tf.compat.v1.nn.xw_plus_b(
            ip_writing_input, self._writing_W_IP, self._writing_b_IP
        )#[B,h]
        ip_att_weights = tf.nn.relu(ip_att_weights)
        ip_att_weights = tf.matmul(ip_att_weights, self._writing_W02_IP)
        #B,s
        ip_att_weights = tf.nn.softmax(ip_att_weights, axis=-1)
        ip_att_weights = tf.expand_dims(ip_att_weights, 2)#[B,S,1]

        # ip_att_weights = ip_att_weights * tf.expand_dims(dist, 1)#[b,1,1]

        reset = tf.sigmoid(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, ip_hat], axis=-1), self._reset_W_IP, self._reset_b_IP
            )
        )
        erase = tf.sigmoid(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, ip_hat], axis=-1), self._erase_W_IP, self._erase_b_IP
            )
        )#[B,D+H]
        add = tf.tanh(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, reset * ip_hat], axis=-1), self._add_W_IP, self._add_b_IP
            )
        )

        ip_state_part01 = ip_state[:, : self._slots, :]
        ip_state01 = ip_state_part01 * (
            tf.ones_like(ip_state_part01) - ip_att_weights * tf.expand_dims(erase, 1)
        )
        ip_state01 = ip_state01 + ip_att_weights * tf.expand_dims(erase, 1) * tf.expand_dims(
            add, 1
        )

        ip_state01 = tf.reshape(ip_state01, [-1, self._slots * self._real_units])

        ## IS Updating
        reset_IS = tf.sigmoid(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([is_hat], axis=-1), self._reset_W_IS, self._reset_b_IS
            )
        )
        erase_IS = tf.sigmoid(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([is_hat], axis=-1), self._erase_W_IS, self._erase_b_IS
            )
        )
        add_IS = tf.tanh(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([reset_IS * is_hat], axis=-1), self._add_W_IS, self._add_b_IS
            )
        )

        is_state_part01 = is_state[:, : self._slots, :]
        is_state01 = is_state_part01 * (
                tf.ones_like(is_state_part01) - is_att_weights * tf.expand_dims(erase_IS, 1)
        )
        is_state01 = is_state01 + is_att_weights * tf.expand_dims(erase_IS, 1) * tf.expand_dims(
            add_IS, 1
        )
        is_state01 = tf.reshape(is_state01, [-1, self._slots * self._real_units])
        state = tf.concat([ip_state01, is_state01, inputs], axis=-1)

        return state, state

class DIUM2WayEvoCellV2(BaseDIUMCell):
    """A variant of SUM cell, which upgrades the writing attention"""

    @tf_utils.shape_type_conversion
    def build(self, inputs_shape):
        """Initialization operations for SUMV2 cell.
        this function creates all the parameters for the cell.
        """
        if inputs_shape[-1] is None:
            raise ValueError(
                "Expected inputs.shape[-1] to be known, saw shape: %s"
                % str(inputs_shape)
            )
        _check_supported_dtypes(self.dtype)
        d = inputs_shape[-1]
        h = self._real_units#slot dim
        s = self._slots#common channel

        self._basic_build(inputs_shape)

        self._writing_W_IP = self.add_variable(
            name="_writing_W_IP", shape=[d + h, h], initializer=self._kernel_initializer
        )
        self._writing_b_IP = self.add_variable(
            name="_writing_b_IP",
            shape=[h],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)
            ),
        )
        self._writing_W02_IP = self.add_variable(
            name="_writing_W02_IP", shape=[h, s], initializer=self._kernel_initializer
        )

        self.parameter_set = [
            self._erase_W_IP,
            self._erase_b_IP,
            self._reset_W_IP,
            self._reset_b_IP,
            self._add_W_IP,
            self._add_b_IP,

            self._erase_W_IS,
            self._erase_b_IS,
            self._reset_W_IS,
            self._reset_b_IS,
            self._add_W_IS,
            self._add_b_IS,

            self._writing_W_IP,
            self._writing_W02_IP,
            self._writing_b_IP,

        ]

        self.built = True
    def squashIS(self,IS):
        is_mod_2 = tf.reduce_sum(tf.square(IS), axis=-1)  # [B,S]
        is_att_logit_mat = is_mod_2 / (1 + is_mod_2)  # [B,S]
        is_att_weights = tf.nn.softmax(self._beta * is_att_logit_mat, axis=-1)
        is_att_weights = tf.expand_dims(is_att_weights, axis=2)  # [B,S,1]
        return is_att_weights
    def call(self, inputs, state):
        """The real operations for SUMV2 cell to process user behaviors.

        Args:
            inputs: (a batch of) user behaviors at time T
            state:  (a batch of) user states at time T-1

        Returns:
            state: after process the user behavior at time T, returns (a batch of) new user states at time T
            state: after process the user behavior at time T, returns (a batch of) new user states at time T
        """
        _check_rnn_cell_input_dtypes([inputs, state])

        h = self._real_units
        s = self._slots
        ip_state,is_state, last_instance = state[:, : s * h], state[:, s * h : 2*s*h],state[:,2*s*h:]
        # n_a, n_b = tf.nn.l2_normalize(last_instance, 1), tf.nn.l2_normalize(inputs, 1)  # [B,D]
        # dist = tf.expand_dims(tf.reduce_sum(input_tensor=n_a * n_b, axis=1), 1)  # [b,1]
        # dist = tf.math.pow(self._alpha, dist)

        ip_state = tf.reshape(ip_state, [-1, s, h])#common+highway channel
        is_state = tf.reshape(is_state, [-1, s, h])#common+highway channel

        # cos sim
        dotProd = tf.expand_dims(inputs, 1) * ip_state  # IAI compute   [B,1,H]*[B,S,H]=>  [B,S,H]  A*B
        dotProd = tf.reduce_sum(dotProd, axis=-1)  # [B,S,H]=>[B,S]
        mulMod = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(inputs), -1)), -1) * \
                 tf.sqrt(tf.reduce_sum(tf.square(ip_state[:, :self._slots, :]), -1))  # ||A||*||B||  [B,1]*[B,S]==>[B,S]
        ip_att_logit_mat = dotProd / mulMod
        ip_att_weights = tf.nn.softmax(self._beta * ip_att_logit_mat, axis=-1)
        ip_att_weights = tf.expand_dims(ip_att_weights, 2)  # [B,S,1]

        #squash IS
        is_att_weights=self.squashIS(is_state[:,:self._slots,:])
        # is_att_weights=tf.matmul(is_state[:,:self._slots,:],self._squashW)#[B,S,H] [H,1] => [b,s,1]
        # is_att_weights = tf.nn.softmax(self._beta * is_att_weights, axis=1)

        #Joint UI Attention
        jointUIAtt=ip_att_weights*is_att_weights
        ip_hat = tf.reduce_sum(
            input_tensor=tf.multiply(ip_state[:, : self._slots, :], jointUIAtt), axis=1
        )
        # ip_hat = (ip_hat + ip_state[:, self._slots, :]) / 2#[B,H]

        is_hat = tf.reduce_sum(
            input_tensor=tf.multiply(is_state[:, : self._slots, :], jointUIAtt), axis=1
        )
        # is_hat = (is_hat + is_state[:, self._slots, :]) / 2  # [B,H]

        # IP&IS Updating
        ## IP Updating
        # ip_writing_input = tf.concat([inputs, ip_hat], axis=1)#[B,d+h]
        # ip_att_weights = tf.compat.v1.nn.xw_plus_b(
        #     ip_writing_input, self._writing_W_IP, self._writing_b_IP
        # )#[B,h]
        # ip_att_weights = tf.nn.relu(ip_att_weights)
        # ip_att_weights = tf.matmul(ip_att_weights, self._writing_W02_IP)     B,S
        # ip_att_weights = tf.nn.softmax(ip_att_weights, axis=-1)
        # ip_att_weights = tf.expand_dims(ip_att_weights, 2)#[B,S,1]

        # ip_att_weights = ip_att_weights * tf.expand_dims(dist, 1)#[b,1,1]

        reset = tf.sigmoid(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, ip_hat], axis=-1), self._reset_W_IP, self._reset_b_IP
            )
        )
        erase = tf.sigmoid(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, ip_hat], axis=-1), self._erase_W_IP, self._erase_b_IP
            )
        )#[B,D+H]
        add = tf.tanh(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, reset * ip_hat], axis=-1), self._add_W_IP, self._add_b_IP
            )
        )

        ip_state_part01 = ip_state[:, : self._slots, :]
        ip_state01 = ip_state_part01 * (
            tf.ones_like(ip_state_part01) - ip_att_weights * tf.expand_dims(erase, 1)
        )
        ip_state01 = ip_state01 + ip_att_weights * tf.expand_dims(erase, 1) * tf.expand_dims(
            add, 1
        )

        ip_state01 = tf.reshape(ip_state01, [-1, self._slots * self._real_units])

        ## IS Updating
        reset_IS = tf.sigmoid(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs,is_hat], axis=-1), self._reset_W_IS, self._reset_b_IS
            )
        )
        erase_IS = tf.sigmoid(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs,is_hat], axis=-1), self._erase_W_IS, self._erase_b_IS
            )
        )
        add_IS = tf.tanh(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs,reset_IS * is_hat], axis=-1), self._add_W_IS, self._add_b_IS
            )
        )

        is_state_part01 = is_state[:, : self._slots, :]
        is_state01 = is_state_part01 * (
                tf.ones_like(is_state_part01) - is_att_weights * tf.expand_dims(erase_IS, 1)
        )
        is_state01 = is_state01 + is_att_weights * tf.expand_dims(erase_IS, 1) * tf.expand_dims(
            add_IS, 1
        )
        is_state01 = tf.reshape(is_state01, [-1, self._slots * self._real_units])
        state = tf.concat([ip_state01, is_state01, inputs], axis=-1)

        return state, state

class DIUM_ISEvoVecSepGenCell(BaseDIUMCell):
    """A variant of SUM cell, which upgrades the writing attention"""

    @tf_utils.shape_type_conversion
    def build(self, inputs_shape):
        """Initialization operations for SUMV2 cell.
        this function creates all the parameters for the cell.
        """
        if inputs_shape[-1] is None:
            raise ValueError(
                "Expected inputs.shape[-1] to be known, saw shape: %s"
                % str(inputs_shape)
            )
        _check_supported_dtypes(self.dtype)
        d = inputs_shape[-1]
        h = self._real_units#slot dim
        s = self._slots#common channel

        self._basic_build(inputs_shape)

        self._writing_W_IP = self.add_variable(
            name="_writing_W_IP", shape=[d + h, h], initializer=self._kernel_initializer
        )
        self._writing_b_IP = self.add_variable(
            name="_writing_b_IP",
            shape=[h],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)
            ),
        )
        self._writing_W02_IP = self.add_variable(
            name="_writing_W02_IP", shape=[h, s], initializer=self._kernel_initializer
        )

        self.parameter_set = [
            self._erase_W_IP,
            self._erase_b_IP,
            self._reset_W_IP,
            self._reset_b_IP,
            self._add_W_IP,
            self._add_b_IP,

            self._erase_W_IS,
            self._erase_b_IS,
            self._reset_W_IS,
            self._reset_b_IS,
            self._add_W_IS,
            self._add_b_IS,

            self._writing_W_IP,
            self._writing_W02_IP,
            self._writing_b_IP,

        ]

        self.built = True

    def call(self, inputs, state):
        """The real operations for SUMV2 cell to process user behaviors.

        Args:
            inputs: (a batch of) user behaviors at time T
            state:  (a batch of) user states at time T-1

        Returns:
            state: after process the user behavior at time T, returns (a batch of) new user states at time T
            state: after process the user behavior at time T, returns (a batch of) new user states at time T
        """
        _check_rnn_cell_input_dtypes([inputs, state])

        h = self._real_units
        s = self._slots
        ip_state,is_state, last_instance = state[:, : s * h], state[:, s * h : 2*s*h],state[:,2*s*h:]
        n_a, n_b = tf.nn.l2_normalize(last_instance, 1), tf.nn.l2_normalize(inputs, 1)  # [B,D]
        dist = tf.expand_dims(tf.reduce_sum(input_tensor=n_a * n_b, axis=1), 1)  # [b,1]
        dist = tf.math.pow(self._alpha, dist)

        ip_state = tf.reshape(ip_state, [-1, s, h])#common+highway channel
        is_state = tf.reshape(is_state, [-1, s, h])#common+highway channel

        ip_att_logit_mat=tf.expand_dims(inputs,1)*ip_state[:,:self._slots,:]#IAI compute   [B,1,H]*[B,S,H]=>  [B,S,H]
        ip_att_logit_mat=tf.reduce_sum(ip_att_logit_mat,axis=-1)#[B,S,H]=>[B,S]

        ip_att_weights = tf.nn.softmax(self._beta * ip_att_logit_mat, axis=-1)
        ip_att_weights = tf.expand_dims(ip_att_weights, 2)#[B,S,1]

        #squash IS
        is_mod_2=tf.reduce_sum(tf.square(is_state[:,:self._slots,:]),axis=-1)#[B,S]
        is_att_logit_mat=is_mod_2/(1+is_mod_2)#[B,S]
        is_att_weights=tf.nn.softmax(self._beta*is_att_logit_mat,axis=-1)
        is_att_weights=tf.expand_dims(is_att_weights,axis=2)#[B,S,1]
        # is_att_weights=tf.matmul(is_state[:,:self._slots,:],self._squashW)#[B,S,H] [H,1] => [b,s,1]
        # is_att_weights = tf.nn.softmax(self._beta * is_att_weights, axis=1)

        #Joint UI Attention
        jointUIAtt=0.5*ip_att_weights+0.5*is_att_weights
        ip_hat = tf.reduce_sum(
            input_tensor=tf.multiply(ip_state[:, : self._slots, :], jointUIAtt), axis=1
        )
        # ip_hat = (ip_hat + ip_state[:, self._slots, :]) / 2#[B,H]

        is_hat = tf.reduce_sum(
            input_tensor=tf.multiply(is_state[:, : self._slots, :], jointUIAtt), axis=1
        )
        # is_hat = (is_hat + is_state[:, self._slots, :]) / 2  # [B,H]

        # IP&IS Updating
        ## IP Updating
        ip_writing_input = tf.concat([inputs, ip_hat], axis=1)#[B,d+h]
        ip_att_weights = tf.compat.v1.nn.xw_plus_b(
            ip_writing_input, self._writing_W_IP, self._writing_b_IP
        )#[B,h]
        ip_att_weights = tf.nn.relu(ip_att_weights)
        ip_att_weights = tf.matmul(ip_att_weights, self._writing_W02_IP)
        #B,s
        ip_att_weights = tf.nn.softmax(ip_att_weights, axis=-1)
        ip_att_weights = tf.expand_dims(ip_att_weights, 2)#[B,S,1]

        ip_att_weights = ip_att_weights * tf.expand_dims(dist, 1)#[b,1,1]

        reset = tf.sigmoid(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, ip_hat], axis=-1), self._reset_W_IP, self._reset_b_IP
            )
        )
        erase = tf.sigmoid(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, ip_hat], axis=-1), self._erase_W_IP, self._erase_b_IP
            )
        )#[B,D+H]
        add = tf.tanh(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, reset * ip_hat], axis=-1), self._add_W_IP, self._add_b_IP
            )
        )

        ip_state_part01 = ip_state[:, : self._slots, :]
        ip_state01 = ip_state_part01 * (
            tf.ones_like(ip_state_part01) - ip_att_weights * tf.expand_dims(erase, 1)
        )
        ip_state01 = ip_state01 + ip_att_weights * tf.expand_dims(erase, 1) * tf.expand_dims(
            add, 1
        )

        ip_state01 = tf.reshape(ip_state01, [-1, self._slots * self._real_units])

        ## IS Updating

        reset_IS = tf.sigmoid(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([is_state], axis=-1), self._reset_W_IS, self._reset_b_IS
            )
        )# # [B,S,H]  [h, 1] #[B,S,1]
        erase_IS = tf.sigmoid(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([is_state], axis=-1), self._erase_W_IS, self._erase_b_IS
            )
        )# [B,S,H] [h, H] [B,S,H]
        add_IS = tf.tanh(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([reset_IS * is_state], axis=-1), self._add_W_IS, self._add_b_IS
            )
        )# [B,S,H] [h, H] [B,S,H]

        is_state_part01 = is_state[:, : self._slots, :]#[B,S,H]
        is_state01 = is_state_part01 * (
                tf.ones_like(is_state_part01) - jointUIAtt* erase_IS
        )#[B,S,1]
        is_state01 = is_state01 + jointUIAtt * erase_IS * add_IS
        is_state01 = tf.reshape(is_state01, [-1, self._slots * self._real_units])

        state = tf.concat([ip_state01,is_state01, inputs], axis=-1)

        return state, state

class DIUM_W_NoJointAtt_Cell(BaseDIUMCell):
    """A variant of SUM cell, which upgrades the writing attention"""

    @tf_utils.shape_type_conversion
    def build(self, inputs_shape):
        """Initialization operations for SUMV2 cell.
        this function creates all the parameters for the cell.
        """
        if inputs_shape[-1] is None:
            raise ValueError(
                "Expected inputs.shape[-1] to be known, saw shape: %s"
                % str(inputs_shape)
            )
        _check_supported_dtypes(self.dtype)
        d = inputs_shape[-1]
        h = self._real_units#slot dim
        s = self._slots#common channel

        self._basic_build(inputs_shape)

        self._writing_W_IP = self.add_variable(
            name="_writing_W_IP", shape=[d + h, h], initializer=self._kernel_initializer
        )
        self._writing_b_IP = self.add_variable(
            name="_writing_b_IP",
            shape=[h],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)
            ),
        )
        self._writing_W02_IP = self.add_variable(
            name="_writing_W02_IP", shape=[h, s], initializer=self._kernel_initializer
        )

        self.parameter_set = [
            self._erase_W_IP,
            self._erase_b_IP,
            self._reset_W_IP,
            self._reset_b_IP,
            self._add_W_IP,
            self._add_b_IP,

            self._erase_W_IS,
            self._erase_b_IS,
            self._reset_W_IS,
            self._reset_b_IS,
            self._add_W_IS,
            self._add_b_IS,

            self._writing_W_IP,
            self._writing_W02_IP,
            self._writing_b_IP,

        ]

        self.built = True

    def call(self, inputs, state):
        """The real operations for SUMV2 cell to process user behaviors.

        Args:
            inputs: (a batch of) user behaviors at time T
            state:  (a batch of) user states at time T-1

        Returns:
            state: after process the user behavior at time T, returns (a batch of) new user states at time T
            state: after process the user behavior at time T, returns (a batch of) new user states at time T
        """
        _check_rnn_cell_input_dtypes([inputs, state])

        h = self._real_units
        s = self._slots + 1
        ip_state,is_state, last_instance = state[:, : s * h], state[:, s * h : 2*s*h],state[:,2*s*h:]
        n_a, n_b = tf.nn.l2_normalize(last_instance, 1), tf.nn.l2_normalize(inputs, 1)  # [B,D]
        dist = tf.expand_dims(tf.reduce_sum(input_tensor=n_a * n_b, axis=1), 1)  # [b,1]
        dist = tf.math.pow(self._alpha, dist)

        ip_state = tf.reshape(ip_state, [-1, s, h])#common+highway channel
        is_state = tf.reshape(is_state, [-1, s, h])#common+highway channel

        ip_att_logit_mat=tf.expand_dims(inputs,1)*ip_state[:,:self._slots,:]#IAI compute   [B,1,H]*[B,S,H]=>  [B,S,H]
        ip_att_logit_mat=tf.reduce_sum(ip_att_logit_mat,axis=-1)#[B,S,H]=>[B,S]

        ip_att_weights = tf.nn.softmax(self._beta * ip_att_logit_mat, axis=-1)
        ip_att_weights = tf.expand_dims(ip_att_weights, 2)#[B,S,1]

        #squash IS
        is_mod_2=tf.reduce_sum(tf.square(is_state[:,:self._slots,:]),axis=-1)#[B,S]
        is_att_logit_mat=is_mod_2/(1+is_mod_2)#[B,S]
        is_att_weights=tf.nn.softmax(self._beta*is_att_logit_mat,axis=-1)
        is_att_weights=tf.expand_dims(is_att_weights,axis=2)#[B,S,1]
        # is_att_weights=tf.matmul(is_state[:,:self._slots,:],self._squashW)#[B,S,H] [H,1] => [b,s,1]
        # is_att_weights = tf.nn.softmax(self._beta * is_att_weights, axis=1)

        #Joint UI Attention
        ip_hat = tf.reduce_sum(
            input_tensor=tf.multiply(ip_state[:, : self._slots, :], ip_att_weights*is_att_weights), axis=1
        )
        ip_hat = (ip_hat + ip_state[:, self._slots, :]) / 2#[B,H]

        is_hat = tf.reduce_sum(
            input_tensor=tf.multiply(is_state[:, : self._slots, :], ip_att_weights * is_att_weights), axis=1
        )
        is_hat = (is_hat + is_state[:, self._slots, :]) / 2  # [B,H]

        # IP&IS Updating
        ## IP Updating
        ip_writing_input = tf.concat([inputs, ip_hat], axis=1)#[B,d+h]
        ip_att_weights = tf.compat.v1.nn.xw_plus_b(
            ip_writing_input, self._writing_W_IP, self._writing_b_IP
        )#[B,h]
        ip_att_weights = tf.nn.relu(ip_att_weights)
        ip_att_weights = tf.matmul(ip_att_weights, self._writing_W02_IP)
        #B,s
        ip_att_weights = tf.nn.softmax(ip_att_weights, axis=-1)
        ip_att_weights = tf.expand_dims(ip_att_weights, 2)#[B,S,1]

        ip_att_weights = ip_att_weights * tf.expand_dims(dist, 1)#[b,1,1]

        reset = tf.sigmoid(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, ip_hat], axis=-1), self._reset_W_IP, self._reset_b_IP
            )
        )
        erase = tf.sigmoid(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, ip_hat], axis=-1), self._erase_W_IP, self._erase_b_IP
            )
        )#[B,D+H]
        add = tf.tanh(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, reset * ip_hat], axis=-1), self._add_W_IP, self._add_b_IP
            )
        )

        ip_state_part01 = ip_state[:, : self._slots, :]
        ip_state01 = ip_state_part01 * (
            tf.ones_like(ip_state_part01) - ip_att_weights * tf.expand_dims(erase, 1)
        )
        ip_state01 = ip_state01 + ip_att_weights * tf.expand_dims(erase, 1) * tf.expand_dims(
            add, 1
        )

        ip_state01 = tf.reshape(ip_state01, [-1, self._slots * self._real_units])

        ip_start_part02 = ip_state[:, self._slots, :]
        ip_state02 = ip_start_part02 * (tf.ones_like(ip_start_part02) - dist * erase)
        ip_state02 = ip_state02 + dist * erase * add



        ## IS Updating

        reset_IS = tf.sigmoid(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([is_hat], axis=-1), self._reset_W_IS, self._reset_b_IS
            )
        )
        erase_IS = tf.sigmoid(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([is_hat], axis=-1), self._erase_W_IS, self._erase_b_IS
            )
        )
        add_IS = tf.tanh(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([reset_IS * is_hat], axis=-1), self._add_W_IS, self._add_b_IS
            )
        )

        is_state_part01 = is_state[:, : self._slots, :]
        is_state01 = is_state_part01 * (
                tf.ones_like(is_state_part01) - is_att_weights* tf.expand_dims(erase_IS, 1)
        )
        is_state01 = is_state01 + is_att_weights * tf.expand_dims(erase_IS, 1) * tf.expand_dims(
            add_IS, 1
        )
        is_state01 = tf.reshape(is_state01, [-1, self._slots * self._real_units])

        is_start_part02 = is_state[:, self._slots, :]
        is_state02 = is_start_part02 * (tf.ones_like(is_start_part02) - dist * erase_IS)
        is_state02 = is_state02 + dist * erase_IS * add_IS
        state = tf.concat([ip_state01, ip_state02,is_state01,is_state02, inputs], axis=-1)

        return state, state

class DIUM_OnlyIP_Cell(BaseDIUMCell):
    """A variant of SUM cell, which upgrades the writing attention"""

    @tf_utils.shape_type_conversion
    def build(self, inputs_shape):
        """Initialization operations for SUMV2 cell.
        this function creates all the parameters for the cell.
        """
        if inputs_shape[-1] is None:
            raise ValueError(
                "Expected inputs.shape[-1] to be known, saw shape: %s"
                % str(inputs_shape)
            )
        _check_supported_dtypes(self.dtype)
        d = inputs_shape[-1]
        h = self._real_units#slot dim
        s = self._slots#common channel

        self._basic_build(inputs_shape)
        self._writing_W_IP = self.add_variable(
            name="_writing_W_IP", shape=[d + h, h], initializer=self._kernel_initializer
        )
        self._writing_b_IP = self.add_variable(
            name="_writing_b_IP",
            shape=[h],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)
            ),
        )
        self._writing_W02_IP = self.add_variable(
            name="_writing_W02_IP", shape=[h, s], initializer=self._kernel_initializer
        )

        # self._squashW = self.add_variable(
        #     name="_squashW", shape=[h, 1], initializer=self._kernel_initializer
        # )

        self.parameter_set = [
            self.inp2ip,
            self._erase_W_IP,
            self._erase_b_IP,
            self._reset_W_IP,
            self._reset_b_IP,
            self._add_W_IP,
            self._add_b_IP,

            # self._erase_W_IS,
            # self._erase_b_IS,
            # self._reset_W_IS,
            # self._reset_b_IS,
            # self._add_W_IS,
            # self._add_b_IS,

            self._writing_W_IP,
            self._writing_W02_IP,
            self._writing_b_IP,

            # self._squashW
        ]

        self.built = True

    def call(self, inputs, state):
        """The real operations for SUMV2 cell to process user behaviors.

        Args:
            inputs: (a batch of) user behaviors at time T
            state:  (a batch of) user states at time T-1

        Returns:
            state: after process the user behavior at time T, returns (a batch of) new user states at time T
            state: after process the user behavior at time T, returns (a batch of) new user states at time T
        """
        _check_rnn_cell_input_dtypes([inputs, state])

        h = self._real_units
        s = self._slots + 1
        ip_state, last_instance = state[:, : s * h],state[:,s*h:]
        n_a, n_b = tf.nn.l2_normalize(last_instance, 1), tf.nn.l2_normalize(inputs, 1)  # [B,D]
        dist = tf.expand_dims(tf.reduce_sum(input_tensor=n_a * n_b, axis=1), 1)  # [b,1]
        dist = tf.math.pow(self._alpha, dist)

        ip_state = tf.reshape(ip_state, [-1, s, h])#common+highway channel

        ip_att_logit_mat=tf.expand_dims(inputs,1)*ip_state[:,:self._slots,:]#IAI compute   [B,1,H]*[B,S,H]=>  [B,S,H]
        ip_att_logit_mat=tf.reduce_sum(ip_att_logit_mat,axis=-1)#[B,S,H]=>[B,S]

        ip_att_weights = tf.nn.softmax(self._beta * ip_att_logit_mat, axis=-1)
        ip_att_weights = tf.expand_dims(ip_att_weights, 2)#[B,S,1]


        #Joint UI Attention
        ip_hat = tf.reduce_sum(
            input_tensor=tf.multiply(ip_state[:, : self._slots, :], ip_att_weights), axis=1
        )
        ip_hat = (ip_hat + ip_state[:, self._slots, :]) / 2#[B,H]

        # IP&IS Updating
        ## IP Updating
        ip_writing_input = tf.concat([inputs, ip_hat], axis=1)#[B,d+h]
        ip_att_weights = tf.compat.v1.nn.xw_plus_b(
            ip_writing_input, self._writing_W_IP, self._writing_b_IP
        )#[B,h]
        ip_att_weights = tf.nn.relu(ip_att_weights)
        ip_att_weights = tf.matmul(ip_att_weights, self._writing_W02_IP)
        #B,s
        ip_att_weights = tf.nn.softmax(ip_att_weights, axis=-1)
        ip_att_weights = tf.expand_dims(ip_att_weights, 2)#[B,S,1]

        ip_att_weights = ip_att_weights * tf.expand_dims(dist, 1)#[b,1,1]

        reset = tf.sigmoid(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, ip_hat], axis=-1), self._reset_W_IP, self._reset_b_IP
            )
        )
        erase = tf.sigmoid(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, ip_hat], axis=-1), self._erase_W_IP, self._erase_b_IP
            )
        )#[B,D+H]
        add = tf.tanh(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, reset * ip_hat], axis=-1), self._add_W_IP, self._add_b_IP
            )
        )
        ip_state_part01 = ip_state[:, : self._slots, :]
        ip_state01 = ip_state_part01 * (
            tf.ones_like(ip_state_part01) - ip_att_weights * tf.expand_dims(erase, 1)
        )
        ip_state01 = ip_state01 + ip_att_weights * tf.expand_dims(erase, 1) * tf.expand_dims(
            add, 1
        )
        ip_state01 = tf.reshape(ip_state01, [-1, self._slots * self._real_units])

        ip_start_part02 = ip_state[:, self._slots, :]
        ip_state02 = ip_start_part02 * (tf.ones_like(ip_start_part02) - dist * erase)
        ip_state02 = ip_state02 + dist * erase * add

        state = tf.concat([ip_state01, ip_state02, inputs], axis=-1)

        return state,state

def _check_rnn_cell_input_dtypes(inputs):
    for t in nest.flatten(inputs):
        _check_supported_dtypes(t.dtype)


def _check_supported_dtypes(dtype):
    if dtype is None:
        return
    dtype = dtypes.as_dtype(dtype)
    if not (dtype.is_floating or dtype.is_complex):
        raise ValueError(
            "RNN cell only supports floating point inputs, " "but saw dtype: %s" % dtype
        )
