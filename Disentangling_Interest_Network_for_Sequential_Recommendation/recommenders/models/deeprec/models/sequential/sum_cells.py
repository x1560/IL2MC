# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import tensorflow as tf
# from keras.layers.legacy_rnn.rnn_cell_impl import LayerRNNCell
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.eager import context
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.rnn_cell_impl import LayerRNNCell, RNNCell, LSTMStateTuple
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops import init_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.util import nest

from recommenders.models.deeprec.models.sequential.rnn_cell_implement import _Linear

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


class SUMCell(LayerRNNCell):
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
            hparams=None,
            layer_keeps=None,
            ipPool=None,
            readHeads=None,
            **kwargs
    ):
        super(SUMCell, self).__init__(_reuse=reuse, name=name, dtype=dtype, **kwargs)
        _check_supported_dtypes(self.dtype)

        if context.executing_eagerly() and context.num_gpus() > 0:
            logging.warn(
                "%s: Note that this cell is not optimized for performance. "
                "Please use keras.layers.cudnn_recurrent.CuDNNGRU for better "
                "performance on GPU.",
                self,
            )

        self._input_size = input_size
        self._slots = slots - 1  # the last channel is reserved for the highway slot
        self._real_slots = self._slots + 1
        self._num_units = num_units
        self._real_units = (self._num_units - input_size) // slots
        self.hparams = hparams
        self.layer_keeps = layer_keeps
        if activation:
            self._activation = activations.get(activation)
        else:
            self._activation = math_ops.tanh
        self._kernel_initializer = initializers.get(kernel_initializer)
        self._bias_initializer = initializers.get(bias_initializer)
        self.ipPool = ipPool
        self.readHeads=readHeads
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

        d = inputs_shape[-1]

        h = self._real_units
        s = self._slots
        if self.hparams.useTimeInterval:
            d = d - 1
            # s=s-1
        self._erase_W = self.add_variable(
            name="_erase_W", shape=[d + h, h], initializer=self._kernel_initializer
        )
        self._erase_b = self.add_variable(
            name="_erase_b",
            shape=[h],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)
            ),
        )

        self._reset_W = self.add_variable(
            name="_reset_W", shape=[d + h, 1], initializer=self._kernel_initializer
        )
        self._reset_b = self.add_variable(
            name="_reset_b",
            shape=[1],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)
            ),
        )

        self._add_W = self.add_variable(
            name="_add_W", shape=[d + h, h], initializer=self._kernel_initializer
        )
        self._add_b = self.add_variable(
            name="_add_b",
            shape=[h],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)
            ),
        )
        self.heads = self.add_variable(
            name="_heads", shape=[s, d], initializer=self._kernel_initializer
        )

        self._beta = self.add_variable(
            name="_beta_no_reg",
            shape=(),
            initializer=tf.compat.v1.constant_initializer(
                np.array([1.02]), dtype=np.float32
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
        s = self._slots  # noqa: F841

        self._basic_build(inputs_shape)

        self.parameter_set = [
            self._erase_W,
            self._erase_b,
            self._reset_W,
            self._reset_b,
            self._add_W,
            self._add_b,
            self.heads,
        ]

        self.built = True

    def call(self, inputs, state):
        """The real operations for SUM cell to process user behaviors.

        params:
            inputs: (a batch of) user behaviors at time T
            state:  (a batch of) user states at time T-1  shape=[B,_num_units=s*h+d]

        returns:
            state, state:
            - after process the user behavior at time T, returns (a batch of) new user states at time T
            - after process the user behavior at time T, returns (a batch of) new user states at time T
        """
        _check_rnn_cell_input_dtypes([inputs, state])

        h = self._real_units
        s = self._slots + 1  # normal+highway-channel
        state, last = state[:, : s * h], state[:, s * h:]
        state = tf.reshape(state, [-1, s, h])
        # state=tf.compat.v1.Print(state,[state[0,:,:10]],summarize=10*s,message="=====sumv1 state=====\n")
        # last=tf.compat.v1.Print(last,[last[0,:20]],summarize=20,message="\n=====sumv1 high channel state=====\n")
        att_logit_mat = tf.matmul(inputs, self.heads, transpose_b=True)#[B,d]#[d,s]   [B,S]
        # att_logit_mat=tf.compat.v1.Print(att_logit_mat,[att_logit_mat[:10,:5]],summarize=50,message="=====sumv1 att_logit_mat=====\n")
        # factor=tf.maximum(self._beta,100.0)
        att_weights = tf.nn.softmax(self._beta * att_logit_mat, axis=-1)
        att_weights = tf.expand_dims(att_weights, 2)

        h_hat = tf.reduce_sum(
            input_tensor=tf.multiply(state[:, : self._slots, :], att_weights), axis=1
        )
        h_hat = (h_hat + state[:, self._slots, :]) / 2  # fuse highway&normal channel  information

        # local debuff
        n_a, n_b = tf.nn.l2_normalize(last, 1), tf.nn.l2_normalize(inputs, 1)  # l2 norm=x1/|x2|
        dist = tf.expand_dims(tf.reduce_sum(input_tensor=n_a * n_b, axis=1), 1)  # cosin=(x1/|x1|)*(x2/|x2|)
        dist = tf.math.pow(self._alpha, dist)  # alpha^sim  Eq.8
        att_weights = att_weights * tf.expand_dims(dist, 1)  # Eq.8

        # Eq.3-5
        reset = tf.sigmoid(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, h_hat], axis=-1), self._reset_W, self._reset_b
            )
        )
        erase = tf.sigmoid(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, h_hat], axis=-1), self._erase_W, self._erase_b
            )
        )
        add = tf.tanh(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, reset * h_hat], axis=-1), self._add_W, self._add_b
            )
        )

        start_part01 = state[:, : self._slots, :]  # common chanel/slot
        state01 = start_part01 * (
                tf.ones_like(start_part01) - att_weights * tf.expand_dims(erase, 1)
        )  # Eq,6   prev_s*(1-att*ers)
        state01 = state01 + att_weights * tf.expand_dims(erase, 1) * tf.expand_dims(
            add, 1
        )  # Eq.6   +att*ers*add
        state01 = tf.reshape(state01, [-1, self._slots * self._real_units])

        # highway slot,all att score equals 1
        start_part02 = state[:, self._slots, :]
        state02 = start_part02 * (tf.ones_like(start_part02) - dist * erase)
        state02 = state02 + dist * erase * add
        state = tf.concat([state01, state02, inputs], axis=-1)
        return state, state

    def get_config(self):
        config = {
            "num_units": self._num_units,
            "kernel_initializer": initializers.serialize(self._kernel_initializer),
            "bias_initializer": initializers.serialize(self._bias_initializer),
            "activation": activations.serialize(self._activation),
            "reuse": self._reuse,
        }
        base_config = super(SUMCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SUMV2Cell(SUMCell):
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
        h = self._real_units
        s = self._slots

        self._basic_build(inputs_shape)

        self._writing_W = self.add_variable(
            name="_writing_W", shape=[d + h, h], initializer=self._kernel_initializer
        )
        self._writing_b = self.add_variable(
            name="_writing_b",
            shape=[h],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)
            ),
        )
        self._writing_W02 = self.add_variable(
            name="_writing_W02", shape=[h, s], initializer=self._kernel_initializer
        )

        self.parameter_set = [
            self._erase_W,
            self._erase_b,
            self._reset_W,
            self._reset_b,
            self._add_W,
            self._add_b,
            self.heads,
            self._writing_W,
            self._writing_W02,
            self._writing_b,
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
        state, last = state[:, : s * h], state[:, s * h:]
        state = tf.reshape(state, [-1, s, h])

        att_logit_mat = tf.matmul(inputs, self.heads, transpose_b=True)

        att_weights = tf.nn.softmax(self._beta * att_logit_mat, axis=-1)
        att_weights = tf.expand_dims(att_weights, 2)

        h_hat = tf.reduce_sum(
            input_tensor=tf.multiply(state[:, : self._slots, :], att_weights), axis=1
        )
        h_hat = (h_hat + state[:, self._slots, :]) / 2

        # get the true writing attentions
        writing_input = tf.concat([inputs, h_hat], axis=1)
        att_weights = tf.compat.v1.nn.xw_plus_b(
            writing_input, self._writing_W, self._writing_b
        )
        att_weights = tf.nn.relu(att_weights)
        att_weights = tf.matmul(att_weights, self._writing_W02)
        att_weights = tf.nn.softmax(att_weights, axis=-1)
        att_weights = tf.expand_dims(att_weights, 2)

        n_a, n_b = tf.nn.l2_normalize(last, 1), tf.nn.l2_normalize(inputs, 1)
        dist = tf.expand_dims(tf.reduce_sum(input_tensor=n_a * n_b, axis=1), 1)
        dist = tf.math.pow(self._alpha, dist)

        att_weights = att_weights * tf.expand_dims(dist, 1)

        reset = tf.sigmoid(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, h_hat], axis=-1), self._reset_W, self._reset_b
            )
        )
        erase = tf.sigmoid(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, h_hat], axis=-1), self._erase_W, self._erase_b
            )
        )
        add = tf.tanh(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, reset * h_hat], axis=-1), self._add_W, self._add_b
            )
        )

        start_part01 = state[:, : self._slots, :]
        state01 = start_part01 * (
                tf.ones_like(start_part01) - att_weights * tf.expand_dims(erase, 1)
        )
        state01 = state01 + att_weights * tf.expand_dims(erase, 1) * tf.expand_dims(
            add, 1
        )
        state01 = tf.reshape(state01, [-1, self._slots * self._real_units])

        start_part02 = state[:, self._slots, :]
        state02 = start_part02 * (tf.ones_like(start_part02) - dist * erase)
        state02 = state02 + dist * erase * add
        state = tf.concat([state01, state02, inputs], axis=-1)
        return state, state


class SUMV7Cell(SUMCell):
    @property
    def output_size(self):
        if self.hparams.useTimeInterval:
            return self._num_units
        else:
            return self._num_units

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
        h = self._real_units
        s = self._slots
        if self.hparams.useTimeInterval:
            d = d - 1
            # s = s - 1
        self._basic_build(inputs_shape)

        self._writing_W = self.add_variable(
            name="_writing_W", shape=[d + h, 1], initializer=self._kernel_initializer
        )
        self._writing_b = self.add_variable(
            name="_writing_b",
            shape=[1],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)
            ),
        )
        self._writing_W02 = self.add_variable(
            name="_writing_W02", shape=[h, s], initializer=self._kernel_initializer
        )

        self.parameter_set = [
            self._erase_W,
            self._erase_b,
            self._reset_W,
            self._reset_b,
            self._add_W,
            self._add_b,
            self.heads,
            self._writing_W,
            self._writing_W02,
            self._writing_b,
        ]

        self.built = True

    def call(self, inputs, state):
        """The real operations for SUM cell to process user behaviors.

        params:
            inputs: (a batch of) user behaviors at time T
            state:  (a batch of) user states at time T-1  shape=[B,_num_units=s*h+d]

        returns:
            state, state:
            - after process the user behavior at time T, returns (a batch of) new user states at time T
            - after process the user behavior at time T, returns (a batch of) new user states at time T
        """
        _check_rnn_cell_input_dtypes([inputs, state])

        h = self._real_units
        s = self._slots + 1  # normal+highway-channel
        state, last = state[:, : s * h], state[:, s * h:]
        state = tf.reshape(state, [-1, s, h])
        orgInputs=inputs
        inputs=inputs[:,:-1]
        last=last[:,:-1]
        # state=tf.compat.v1.Print(state,[state[0,:,:10]],summarize=10*s,message="=====sumv1 state=====\n")
        # last=tf.compat.v1.Print(last,[last[0,:20]],summarize=20,message="\n=====sumv1 high channel state=====\n")
        # att_logit_mat = tf.matmul(inputs, self.readHeads[:-1,:], transpose_b=True)  # [B,d]#[d,s]   [B,S]
        att_logit_mat = tf.matmul(inputs, self.heads, transpose_b=True)  # [B,d]#[d,s]   [B,S]

        # att_logit_mat=tf.compat.v1.Print(att_logit_mat,[att_logit_mat[:10,:5]],summarize=50,message="=====sumv1 att_logit_mat=====\n")

        # factor=tf.maximum(self._beta,100.0)
        att_weights = tf.nn.softmax(self._beta * att_logit_mat, axis=-1)
        att_weights = tf.expand_dims(att_weights, 2)

        h_hat = tf.reduce_sum(
            input_tensor=tf.multiply(state[:, : self._slots, :], att_weights), axis=1
        )
        h_hat = (h_hat + state[:, self._slots, :]) / 2  # fuse highway&normal channel  information

        # local debuff
        n_a, n_b = tf.nn.l2_normalize(last, 1), tf.nn.l2_normalize(inputs, 1)  # l2 norm=x1/|x2|
        dist = tf.expand_dims(tf.reduce_sum(input_tensor=n_a * n_b, axis=1), 1)  # cosin=(x1/|x1|)*(x2/|x2|)
        dist = tf.math.pow(self._alpha, dist)  # alpha^sim  Eq.8
        att_weights = att_weights * tf.expand_dims(dist, 1)  # Eq.8

        # Eq.3-5
        reset = tf.sigmoid(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, h_hat], axis=-1), self._reset_W, self._reset_b
            )
        )
        erase = tf.sigmoid(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, h_hat], axis=-1), self._erase_W, self._erase_b
            )
        )
        add = tf.tanh(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, reset * h_hat], axis=-1), self._add_W, self._add_b
            )
        )

        start_part01 = state[:, : self._slots, :]  # common chanel/slot
        state01 = start_part01 * (
                tf.ones_like(start_part01) - att_weights * tf.expand_dims(erase, 1)
        )  # Eq,6
        state01 = state01 + att_weights * tf.expand_dims(erase, 1) * tf.expand_dims(
            add, 1
        )  # Eq.6
        state01 = tf.reshape(state01, [-1, self._slots * self._real_units])

        # highway slot,all att score equals 1
        start_part02 = state[:, self._slots, :]
        state02 = start_part02 * (tf.ones_like(start_part02) - dist * erase)
        state02 = state02 + dist * erase * add
        state = tf.concat([state01, state02, orgInputs], axis=-1)
        return state, state


class SUMV6Cell(SUMCell):
    @property
    def output_size(self):
        if self.hparams.useTimeInterval:
            return self._num_units
        else:
            return self._num_units

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
        h = self._real_units
        s = self._slots
        if self.hparams.useTimeInterval:
            d = d - 1

            # s = s - 1
        self._basic_build(inputs_shape)

        self._writing_W = self.add_variable(
            name="_writing_W", shape=[d + h, 1], initializer=self._kernel_initializer
        )
        self._writing_b = self.add_variable(
            name="_writing_b",
            shape=[1],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)
            ),
        )
        self.ers_rst_w = self.add_variable(
            name="ers_rst_w", shape=[d+h, 2*h], initializer=self._kernel_initializer
        )
        self.ers_rst_b = self.add_variable(
            name="ers_rst_b", shape=[2*h], initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)
            ),
        )
        self._candidate_kernel=self.add_variable(
            name="_candidate_kernel", shape=[d+h, h], initializer=self._kernel_initializer
        )
        self._candidate_bias = self.add_variable(
            name="_candidate_bias", shape=[h], initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)
            ),
        )
        self.sim_alpha = self.add_variable(
            name="sim_alpha_no_reg",
            shape=(),
            initializer=tf.compat.v1.constant_initializer(
                np.array([0.98]), dtype=np.float32
            ),
        )
        self.v6heads = self.add_variable(
            name="v6heads", shape=[self._slots + 1, d], initializer=self._kernel_initializer
        )
        self.parameter_set = [
            self._erase_W,
            self._erase_b,
            self._reset_W,
            self._reset_b,
            self._add_W,
            self._add_b,
            self.heads,
            self._writing_W,self._writing_b,
            self.ers_rst_w,self.ers_rst_b,
            self._candidate_kernel,self._candidate_bias,

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
        state, last = state[:, : s * h], state[:, s * h:]
        state = tf.reshape(state, [-1, s, h])
        time_diff = tf.expand_dims(inputs[:, -1], -1)  # [B,1]
        orgInput = inputs
        inputs = inputs[:, :-1]
        # inputs = tf.compat.v1.Print(inputs, [inputs[0, :10]], summarize=10, message="inputs")
        # state = tf.compat.v1.Print(state, [state[0, :, :6]], summarize=36, message="state")
        #
        # get the true writing attentions
        '''
        时间门控：感知当前输入、相关兴趣通道、时间间隔、距离目标交互的时间跨度
        '''
        with tf.compat.v1.variable_scope("writing_input"):
            writing_input = tf.concat([tf.tile(tf.expand_dims(inputs, 1), [1, s, 1]), state], axis=-1)  # [b,s,d+h]
        # att_weights = tf.compat.v1.nn.xw_plus_b(
        #     writing_input, self._writing_W, self._writing_b
        # )  # [b,s,1]
        # att_weights = tf.nn.sigmoid(att_weights) # [b,s,1]
        # # writing_input=tf.nn.l2_normalize(writing_input,-1)
        # # self.v6heads = tf.compat.v1.Print(self.v6heads, [self.v6heads[:,:6]], summarize=36, message="v6heads")
        att_logit_mat = tf.matmul(inputs, self.v6heads, transpose_b=True)#[b,s]
        _beta = tf.maximum(self._beta, 1e8)
        att_weights = tf.nn.softmax(_beta * att_logit_mat, axis=-1)
        att_weights = tf.expand_dims(att_weights, 2)#[b,s,1]
        # # att_weights = tf.compat.v1.Print(att_weights, [att_weights[0, :, 0]], summarize=6, message="att_weights_sfmx")
        # att_weights = tf.compat.v1.Print(att_weights, [att_weights[0, :, 0]], summarize=6, message="att_weights_sfmx")
        #
        # n_a, n_b = tf.nn.l2_normalize(last[:, :-1], 1), tf.nn.l2_normalize(inputs, 1)  # [B,H]
        # dist_sim = tf.reduce_sum(input_tensor=n_a * n_b, axis=1, keepdims=True)  # [B,1]
        # sim_alpha = tf.minimum(self.sim_alpha, 0.98)
        # dist_sim = tf.math.pow(sim_alpha, dist_sim)  # [B,1]
        #
        # factor = tf.maximum(self._alpha, 1)
        # factor=tf.minimum(factor,1.0)
        # time_diff = tf.maximum(time_diff, 0.01)  # [0,99999]==>[0,1]
        # dist_time = time_diff / (time_diff + factor)  # [B,1]
        # dist = (dist_sim + dist_time) / 2
        # dis  时间间隔短或者交互的物品相似，则dis小，更新力度小；时间间隔长或者交互物品差异大，更新力度大；
        # dist=dist_sim
        # att_weights = att_weights * tf.expand_dims(dist, 1)  # [b,s,1]*[b,1,1]
        # att_weights = tf.compat.v1.Print(att_weights, [att_weights[0, :, 0]], summarize=6, message="att_weights_disW")
        with tf.compat.v1.variable_scope("reset_erase"):
            gate_inputs = tf.matmul(writing_input, self.ers_rst_w)#[B,S,D+H]=>[B,S,2*H]
            gate_inputs = tf.nn.bias_add(gate_inputs, self.ers_rst_b)#[B,S,2*H]
            value = tf.sigmoid(gate_inputs)
            reset, erase = tf.split(value=value, num_or_size_splits=2, axis=-1)#[B,S,H],[B,S,H]
            r_state = reset * state#[B,S,H]*[B,S,H]
            candidate = tf.matmul(
                tf.concat([tf.tile(tf.expand_dims(inputs, 1), [1, s, 1]), r_state], -1),
                self._candidate_kernel)  # [b,s,d+h]=>#[b,s,h]

            candidate = tf.nn.bias_add(candidate, self._candidate_bias)#[b,s,h]
            c = tf.tanh(candidate)
            # new_h = att_weights*erase * c + (1 - erase*att_weights) * state#[b,s,h]
            new_h = erase * state + (1 - erase) * c#[b,s,h]
        new_h=att_weights*new_h+(1-att_weights)*state
        state01 = tf.reshape(new_h, [-1, s * self._real_units])
        with tf.compat.v1.variable_scope("stateinput"):
            state = tf.concat([state01, orgInput], axis=-1)
        return state, state

class SUMV6TimeCell(RNNCell):
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
            hparams=None,
            layer_keeps=None,
            ipPool=None,
            readHeads=None,
            **kwargs
    ):
        super(SUMV6TimeCell, self).__init__(_reuse=reuse, name=name, dtype=dtype, **kwargs)
        _check_supported_dtypes(self.dtype)

        if context.executing_eagerly() and context.num_gpus() > 0:
            logging.warn(
                "%s: Note that this cell is not optimized for performance. "
                "Please use keras.layers.cudnn_recurrent.CuDNNGRU for better "
                "performance on GPU.",
                self,
            )

        self._input_size = input_size
        self._slots = slots
        self._real_slots = slots
        self._num_units = num_units
        print("self._num_units ",self._num_units)
        print(" slots ",slots)
        self._real_units = self._num_units // slots
        self.hparams = hparams
        self.layer_keeps = layer_keeps
        self._activation = math_ops.tanh
        self._kernel_initializer = initializers.get(kernel_initializer)
        self._bias_initializer = initializers.get(bias_initializer)
        self.ipPool = ipPool
        self.readHeads = readHeads

        self._state_size = 2 * num_units
        self._output_size = num_units
        self.needHighWay=False

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

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
        d = inputs_shape[-1]-2
        h = self._real_units
        s = self._slots
        if self.needHighWay:
            s=s-1
        self._forget_bias=1.0
        self.ers_rst_w = self.add_variable(
            name="ers_rst_w", shape=[d+h, 4*h], initializer=self._kernel_initializer
        )
        self.ers_rst_b = self.add_variable(
            name="ers_rst_b", shape=[4*h], initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)
            ),
        )


        self._time_input_w1 = vs.get_variable(
            "_time_input_w1", shape=[h], initializer=self._kernel_initializer
        )
        self._time_input_bias1 = vs.get_variable(
            "_time_input_bias1", shape=[h], initializer=self._kernel_initializer
        )
        self._time_input_w2 = vs.get_variable(
            "_time_input_w2", shape=[h], initializer=self._kernel_initializer
        )
        self._time_input_bias2 = vs.get_variable(
            "_time_input_bias2", shape=[h], initializer=self._kernel_initializer
        )
        self._time_kernel_w1 = vs.get_variable(
            "_time_kernel_w1",
            shape=[d, h],
            initializer=self._kernel_initializer,
        )
        self._time_kernel_t1 = vs.get_variable(
            "_time_kernel_t1",
            shape=[h, h],
            initializer=self._kernel_initializer,
        )
        self._time_bias1 = vs.get_variable(
            "_time_bias1", shape=[h], initializer=self._kernel_initializer
        )
        self._time_kernel_w2 = vs.get_variable(
            "_time_kernel_w2",
            shape=[d, h],
            initializer=self._kernel_initializer,
        )
        self._time_kernel_t2 = vs.get_variable(
            "_time_kernel_t2",
            shape=[h, h],
            initializer=self._kernel_initializer,
        )
        self._time_bias2 = vs.get_variable(
            "_time_bias2", shape=[h], initializer=self._kernel_initializer
        )
        self._o_kernel_t1 = vs.get_variable(
            "_o_kernel_t1",
            shape=[h, h],
            initializer=self._kernel_initializer,
        )
        self._o_kernel_t2 = vs.get_variable(
            "_o_kernel_t2",
            shape=[h, h],
            initializer=self._kernel_initializer,
        )
        self.heads = self.add_variable(
            name="_heads", shape=[s, d], initializer=self._kernel_initializer
        )

        self._beta = self.add_variable(
            name="_beta_no_reg",
            shape=(),
            initializer=tf.compat.v1.constant_initializer(
                np.array([1.02]), dtype=np.float32
            ),
        )
        self._writing_W = self.add_variable(
            name="_writing_W", shape=[d + h, h], initializer=self._kernel_initializer
        )
        self._writing_b = self.add_variable(
            name="_writing_b",
            shape=[h],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)
            ),
        )
        self._writing_W02 = self.add_variable(
            name="_writing_W02", shape=[h, 1], initializer=self._kernel_initializer
        )

        self.parameter_set = [
            self.ers_rst_w,self.ers_rst_b,
            self._time_input_w1,  # [1,h]
            self._time_input_bias1 , # [h]
            self._time_kernel_w1,  # [d,h]
            self._time_kernel_t1,  # [h,h]
            self._time_bias1,  # [h]
            self._time_input_w1,  # [1,H]
            self._time_input_bias1,  # [H]
            self._time_input_w2,  # [1,H]
            self._time_input_bias2,  # [H]
            self._time_kernel_w1,  # [D,H]
            self._time_kernel_t1,  # [H,H]
            self._time_bias1,  # [H]
            self.heads,
            self._beta,
            self._writing_W,
            self._writing_W02,
            self._writing_b,
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
        time_now_score = tf.expand_dims(inputs[:, -1], -1)#[B,1]
        time_last_score = tf.expand_dims(inputs[:, -2], -1)
        inputs = inputs[:, :-2]
        num_proj = self._num_units

        c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
        m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_proj])  # [b,S*h]

        resp_m_prev = tf.reshape(m_prev, [-1,self._real_slots,self._real_units])#[b,s,h]
        resp_c_prev = tf.reshape(c_prev, [-1, self._real_slots, self._real_units])  # [b,s,h]
        needAtt=True
        evoWay = "single"
        if self.needHighWay:
            resp_m_prev=resp_m_prev[:,:-1,:]
            high_m_prev=resp_m_prev[:,-1,:]
            resp_c_prev = resp_c_prev[:, :-1, :]
            high_c_prev = resp_c_prev[:, -1, :]
        if needAtt:
            factor=tf.maximum(self._beta,2.0)
            # inputs=tf.compat.v1.Print(inputs,[inputs[:10,:7]],summarize=70,message="=====sumv1 inputs inputs=====\n")

            inp_tile=tf.tile(tf.expand_dims(inputs,axis=1),[1,self._real_slots,1])#[B,1,H]=>[B S H]
            # self.readHeads=tf.compat.v1.Print(self.readHeads,[self.readHeads[:10,:,:5]],summarize=200,message="=====sumv1 att_logit readHeads=====\n")
            writing_input = tf.concat([inp_tile, self.readHeads], axis=-1)#[B S H]#[B,S,D]
            print("writing_input shape, ", writing_input.shape)#[B,S,D+H]
            att_weights = tf.compat.v1.nn.xw_plus_b(
                writing_input, self._writing_W, self._writing_b
            )#[B,S,h]
            print("att_weights shape, ", att_weights.shape)
            att_weights = tf.nn.relu(att_weights)
            att_weights = tf.matmul(att_weights, self._writing_W02)
            att_weights = tf.nn.softmax(factor*att_weights, axis=1)#[B,S,1]
            # att_weights=tf.compat.v1.Print(att_weights,[att_weights[:10,:5,0]],summarize=50,message="=====sumv1 att_logit_mat=====\n")

            print("att_weights shape, ",att_weights.shape)
        '''
        时间门控：感知当前输入、相关兴趣通道、时间间隔、距离目标交互的时间跨度
        '''

        time_now_input = tf.nn.tanh(
            time_now_score * self._time_input_w1 + self._time_input_bias1
        )#[B,H]
        time_last_input = tf.nn.tanh(
            time_last_score * self._time_input_w2 + self._time_input_bias2
        )#[B,H]
        time_now_state = (
                math_ops.matmul(inputs, self._time_kernel_w1)#[D,H]
                + math_ops.matmul(time_now_input, self._time_kernel_t1)#[H,H]
                + self._time_bias1
        )#[B,H]
        time_last_state = (
                math_ops.matmul(inputs, self._time_kernel_w2)
                + math_ops.matmul(time_last_input, self._time_kernel_t2)
                + self._time_bias2
        )#[B,H]
        #single evo

        if evoWay=="single":
            time_now_input = tf.expand_dims(time_now_input, 1)
            time_last_input = tf.expand_dims(time_last_input, 1)
            time_now_state = tf.expand_dims(time_now_state, 1)  # [B,1,H]
            time_last_state = tf.expand_dims(time_last_state, 1)  # [B,1,H]
            writing_input = tf.concat([tf.tile(tf.expand_dims(inputs, 1), [1,self._real_slots, 1]),
                                       resp_m_prev], axis=-1)  # [b,s,d+h]
            gate_inputs = tf.matmul(writing_input, self.ers_rst_w)  # [B,S,D+H]=>[B,S,2*H]
            value = tf.nn.bias_add(gate_inputs, self.ers_rst_b)  # [B,S,2*H]
            i, j, f, o = tf.split(value=value, num_or_size_splits=4, axis=-1)  # [B,S,H],[B,S,H]
            with tf.compat.v1.variable_scope("outgate"):
                o = (
                    o
                    + math_ops.matmul(time_now_input, self._o_kernel_t1)#[B,1,H] #[H,H]
                    + math_ops.matmul(time_last_input, self._o_kernel_t2)#[B,1,H] #[H,H]
                )
                print("ogate shaep: ",o.shape)
                print("time_now_input shaep: ",time_now_input.shape)
                print("time_last_input shaep: ",time_last_input.shape)
            c_prev=tf.reshape(c_prev,[-1,self._real_slots,self._real_units])
            with tf.compat.v1.variable_scope("cgate"):
                c = tf.nn.sigmoid(f + self._forget_bias) * tf.nn.sigmoid(
                    time_last_state
                ) * c_prev + tf.nn.sigmoid(i) * tf.nn.sigmoid(time_now_state) * self._activation(j)
            with tf.compat.v1.variable_scope("mgate"):
                m = tf.nn.sigmoid(o) * self._activation(c)
                print("mgate shaep: ", m.shape)

            if needAtt:
                with tf.compat.v1.variable_scope("att_merge"):
                    c=att_weights*c+(1-att_weights)*resp_c_prev#[B,S,H]
                    m=att_weights*m+(1-att_weights)*resp_m_prev

            with tf.compat.v1.variable_scope("new_state"):
                new_state = array_ops.concat([tf.reshape(c,[-1,self._real_slots*self._real_units]),
                                              tf.reshape(m,[-1,self._real_slots*self._real_units])], 1)
                print("new_state shaep: ", new_state.shape)
            # m=tf.compat.v1.Print(m,[m[:10,:,:5]],summarize=200,message="=====sumv1 state mat =====\n")

            return tf.reshape(m,[-1,self._real_slots*self._real_units]), new_state


        if evoWay == "merge":
            h_hat=None
            writing_input = tf.concat([inputs,h_hat], axis=-1)  # [b,d+h]
            gate_inputs = tf.matmul(writing_input, self.ers_rst_w)  # [B,D+H]=>[B,2*H]
            value = tf.nn.bias_add(gate_inputs, self.ers_rst_b)  # [B,2*H]
            i, j, f, o = tf.split(value=value, num_or_size_splits=4, axis=-1)  # [B,H]
            with tf.compat.v1.variable_scope("outgate"):
                o = (
                        o
                        + math_ops.matmul(time_now_input, self._o_kernel_t1)  # [B,H] #[H,H]
                        + math_ops.matmul(time_last_input, self._o_kernel_t2)  # [B,H] #[H,H]
                )
                print("ogate shaep: ", o.shape)
                print("time_now_input shaep: ", time_now_input.shape)
                print("time_last_input shaep: ", time_last_input.shape)

            with tf.compat.v1.variable_scope("cgate"):
                c_high = tf.nn.sigmoid(f + self._forget_bias) * tf.nn.sigmoid(time_last_state) * high_c_prev + \
                         tf.nn.sigmoid(i) * tf.nn.sigmoid(time_now_state) * self._activation(j)
                m_high=tf.nn.sigmoid(o) * self._activation(c_high)

                time_now_input = tf.expand_dims(time_now_input, 1)
                time_last_input = tf.expand_dims(time_last_input, 1)
                time_now_state = tf.expand_dims(time_now_state, 1)  # [B,1,H]
                time_last_state = tf.expand_dims(time_last_state, 1)  # [B,1,H]
                i=tf.expand_dims(i,1)
                j=tf.expand_dims(j,1)
                f=tf.expand_dims(f,1)
                o=tf.expand_dims(o,1)
                c = tf.nn.sigmoid(f + self._forget_bias) * tf.nn.sigmoid(time_last_state) \
                    * resp_c_prev + att_weights * tf.nn.sigmoid(i) * tf.nn.sigmoid(time_now_state) * self._activation(j)
                print("c,chigh shape is {},{}".format(c.shape,c_high.shape))

            with tf.compat.v1.variable_scope("mgate"):
                m = tf.nn.sigmoid(o) * self._activation(c)
                print("mgate shaep: ", m.shape)
            if needAtt:
                with tf.compat.v1.variable_scope("att_merge"):
                    c = att_weights * c + (1 - att_weights) * resp_c_prev  # [B,S,H]
                    m = att_weights * m + (1 - att_weights) * resp_m_prev

            with tf.compat.v1.variable_scope("merge_state"):
                c_merge = tf.concat([c, tf.expand_dims(c_high, 1)], 1)  # [B,S+1,H]
                m_merge = tf.concat([m, tf.expand_dims(m_high, 1)], 1)  # [B,S+1,H]

            with tf.compat.v1.variable_scope("new_state"):
                new_state = array_ops.concat([tf.reshape(c_merge, [-1, self._real_slots * self._real_units]),
                                              tf.reshape(m_merge, [-1, self._real_slots * self._real_units])], 1)
                print("new_state shaep: ", new_state.shape)

            return tf.reshape(m_merge, [-1, self._real_slots * self._real_units]), new_state
class SUMV5Cell(SUMCell):
    @property
    def output_size(self):
        if self.hparams.useTimeInterval:
            return self._num_units
        else:
            return self._num_units

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
        h = self._real_units
        s = self._slots
        if self.hparams.useTimeInterval:
            d = d - 1
            # s = s - 1
        self._basic_build(inputs_shape)

        self._writing_W = self.add_variable(
            name="_writing_W", shape=[d + h, h], initializer=self._kernel_initializer
        )
        self._writing_b = self.add_variable(
            name="_writing_b",
            shape=[h],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)
            ),
        )
        self._writing_W02 = self.add_variable(
            name="_writing_W02", shape=[h, s], initializer=self._kernel_initializer
        )

        self.parameter_set = [
            self._erase_W,
            self._erase_b,
            self._reset_W,
            self._reset_b,
            self._add_W,
            self._add_b,
            self.heads,
            self._writing_W,
            self._writing_W02,
            self._writing_b,
        ]

        self.built = True

    def ipPoolQuery(self, query, ipPool):
        with tf.compat.v1.variable_scope("ipPoolQuery"):
            query_size = query.shape[-1]
            attention_w = tf.compat.v1.get_variable(name="attention_w", shape=[query_size, self._real_units],
                                                    initializer=self._kernel_initializer, )
            attention_b = tf.compat.v1.get_variable(name="attention_b", shape=[self._real_units],
                                                    initializer=self._kernel_initializer, )
            realQuery = tf.compat.v1.nn.xw_plus_b(query, attention_w, attention_b)
        # [B,D]  [poolNum,D]
        realQuery = tf.expand_dims(realQuery, axis=1)  # [B,1,D]
        pool = tf.expand_dims(ipPool, axis=0)  # [1,poolNum,D]
        poolSoftBeta = 1000
        att_logit = tf.nn.softmax(
            poolSoftBeta * tf.reduce_sum(realQuery * pool, axis=-1, keepdims=True)
            , axis=-2)  # [B,poolNum,1]
        collb_ip = tf.reduce_sum(pool * att_logit, axis=1)  # [B,poolNum,D]=>[B,D]
        return collb_ip

    def KL(self, p, q):
        return tf.reduce_sum(p * tf.math.log(p / (q + 1e-8)), axis=-1, keepdims=True)  # [B,D]=>[B,1]

    def JS(self, p, q):
        M = 0.5 * (p + q)  # [B,D]
        return 0.5 * (self.KL(p, M) + self.KL(q, M))

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
        state, last = state[:, : s * h], state[:, s * h:]
        state = tf.reshape(state, [-1, s, h])
        time_diff = tf.expand_dims(inputs[:, -1], -1)  # [B,1]
        orgInput = inputs
        inputs = inputs[:, :-1]

        att_logit_mat = tf.matmul(inputs, self.heads, transpose_b=True)
        att_weights = tf.nn.softmax(self._beta * att_logit_mat, axis=-1)
        att_weights = tf.expand_dims(att_weights, 2)

        h_hat = tf.reduce_sum(
            input_tensor=tf.multiply(state[:, : self._slots, :], att_weights), axis=1
        )
        h_hat = (h_hat + state[:, self._slots, :]) / 2

        # get the true writing attentions
        writing_input = tf.concat([inputs, h_hat], axis=1)
        att_weights = tf.compat.v1.nn.xw_plus_b(
            writing_input, self._writing_W, self._writing_b
        )
        att_weights = tf.nn.relu(att_weights)
        att_weights = tf.matmul(att_weights, self._writing_W02)
        att_weights = tf.nn.softmax(att_weights, axis=-1)
        att_weights = tf.expand_dims(att_weights, 2)

        n_a, n_b = tf.nn.l2_normalize(last[:, :-1], 1), tf.nn.l2_normalize(inputs, 1)
        dist = tf.expand_dims(tf.reduce_sum(input_tensor=n_a * n_b, axis=1), 1)
        dist = tf.math.pow(self._alpha, dist)

        att_weights = att_weights * tf.expand_dims(dist, 1)

        reset = tf.sigmoid(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, h_hat], axis=-1), self._reset_W, self._reset_b
            )
        )
        erase = tf.sigmoid(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, h_hat], axis=-1), self._erase_W, self._erase_b
            )
        )
        add = tf.tanh(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, reset * h_hat], axis=-1), self._add_W, self._add_b
            )
        )

        start_part01 = state[:, : self._slots, :]
        state01 = start_part01 * (
                tf.ones_like(start_part01) - att_weights * tf.expand_dims(erase, 1)
        )
        state01 = state01 + att_weights * tf.expand_dims(erase, 1) * tf.expand_dims(
            add, 1
        )
        state01 = tf.reshape(state01, [-1, self._slots * self._real_units])
        start_part02 = state[:, self._slots, :]
        state02 = start_part02 * (tf.ones_like(start_part02) - dist * erase)
        state02 = state02 + dist * erase * add
        state = tf.concat([state01, state02, orgInput], axis=-1)
        return state, state


class SUMV4Cell(SUMCell):
    @property
    def output_size(self):
        if self.hparams.useTimeInterval:
            return 1
        else:
            return self._num_units

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
        h = self._real_units
        s = self._slots
        if self.hparams.useTimeInterval:
            d = d - 1
            # s = s - 1
        self._basic_build(inputs_shape)

        self._writing_W = self.add_variable(
            name="_writing_W", shape=[d + h, h], initializer=self._kernel_initializer
        )
        self._writing_b = self.add_variable(
            name="_writing_b",
            shape=[h],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)
            ),
        )
        self._writing_W02 = self.add_variable(
            name="_writing_W02", shape=[h, s], initializer=self._kernel_initializer
        )

        self.parameter_set = [
            self._erase_W,
            self._erase_b,
            self._reset_W,
            self._reset_b,
            self._add_W,
            self._add_b,
            self.heads,
            self._writing_W,
            self._writing_W02,
            self._writing_b,
        ]

        self.built = True

    def ipPoolQuery(self, query, ipPool):
        with tf.compat.v1.variable_scope("ipPoolQuery"):
            query_size = query.shape[-1]
            attention_w = tf.compat.v1.get_variable(name="attention_w", shape=[query_size, self._real_units],
                                                    initializer=self._kernel_initializer, )
            attention_b = tf.compat.v1.get_variable(name="attention_b", shape=[self._real_units],
                                                    initializer=self._kernel_initializer, )
            realQuery = tf.compat.v1.nn.xw_plus_b(query, attention_w, attention_b)
        # [B,D]  [poolNum,D]
        realQuery = tf.expand_dims(realQuery, axis=1)  # [B,1,D]
        pool = tf.expand_dims(ipPool, axis=0)  # [1,poolNum,D]
        poolSoftBeta = 1000
        att_logit = tf.nn.softmax(
            poolSoftBeta * tf.reduce_sum(realQuery * pool, axis=-1, keepdims=True)
            , axis=-2)  # [B,poolNum,1]
        collb_ip = tf.reduce_sum(pool * att_logit, axis=1)  # [B,poolNum,D]=>[B,D]
        return collb_ip

    def KL(self, p, q):
        return tf.reduce_sum(p * tf.math.log(p / (q + 1e-8)), axis=-1, keepdims=True)  # [B,D]=>[B,1]

    def JS(self, p, q):
        M = 0.5 * (p + q)  # [B,D]
        return 0.5 * (self.KL(p, M) + self.KL(q, M))

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
        '''
        0:s-1 : normal
        s-1: time interval 
        s: highway
        '''
        if not self.hparams.useTimeInterval:
            return

        state, last = state[:, : s * h], state[:, s * h:]
        state = tf.reshape(state, [-1, s, h])

        time_diff = tf.expand_dims(inputs[:, -1], -1)  # [B,1]
        orgInput = inputs
        inputs = inputs[:, :-1]

        # with tf.compat.v1.variable_scope("timeDiff", reuse=tf.compat.v1.AUTO_REUSE):
        # time diff embedding
        # self._time_diff_input_w1 = tf.compat.v1.get_variable(
        #     name="_time_diff_input_w1", shape=[h], initializer=self._kernel_initializer
        # )
        # self._time_diff_input_bias1 = tf.compat.v1.get_variable(
        #     name="_time_diff_input_bias1", shape=[h], initializer=self._kernel_initializer
        # )
        # time_diff_in = tf.nn.sigmoid(
        #     time_diff * self._time_diff_input_w1 + self._time_diff_input_bias1
        # )  # [B,D]

        # time_diff_last_state=state[:,self._slots-1,:]
        # time diff evo
        # self.rz_time_diff_w = tf.compat.v1.get_variable(
        #     name="rz_time_diff_w", shape=[2*h,2*h], initializer=self._kernel_initializer
        # )
        #
        # self.rz_time_diff_b = tf.compat.v1.get_variable(
        #     name="rz_time_diff_b", shape=[2*h], initializer=self._kernel_initializer
        # )
        # self.h_time_diff_w = tf.compat.v1.get_variable(
        #     name="h_time_diff_w", shape=[2 * h,  h], initializer=self._kernel_initializer
        # )

        # self.h_time_diff_b = tf.compat.v1.get_variable(
        #     name="h_time_diff_b", shape=[h], initializer=self._kernel_initializer
        # )
        # rz_time_diff=tf.nn.sigmoid(tf.compat.v1.nn.xw_plus_b(tf.concat([time_diff_last_state,time_diff_in],axis=-1),
        #                                        self.rz_time_diff_w ,self.rz_time_diff_b ))
        #
        # reset_time_diff=rz_time_diff[:,:h]
        # z_time_diff=rz_time_diff[:,h:]
        # h_time_diff_hat=tf.nn.tanh(tf.compat.v1.nn.xw_plus_b(
        #     tf.concat([reset_time_diff*time_diff_last_state,time_diff_in],-1),
        #     self.h_time_diff_w,self.h_time_diff_b
        # ))
        # self.time_diff_pred_w = tf.compat.v1.get_variable(
        #     name="time_diff_pred_w", shape=[2*h, 1], initializer=self._kernel_initializer
        # )
        #
        # self.time_diff_pred_b = tf.compat.v1.get_variable(
        #     name="time_diff_pred_b", shape=[1], initializer=self._kernel_initializer
        # )
        # time_diff_next_state=(1-z_time_diff)*time_diff_last_state+z_time_diff*h_time_diff_hat#[B,D]
        # time_diff_last_pred=tf.nn.relu(tf.compat.v1.nn.xw_plus_b(
        #     time_diff_last_state,self.time_diff_pred_w ,self.time_diff_pred_b
        # ))# timediff >=0

        # time_diff_alpha=tf.nn.sigmoid(time_diff-time_diff_last_pred)#[B,1]
        # time_diff_alpha=tf.nn.sigmoid(tf.compat.v1.nn.xw_plus_b(
        #     tf.concat([time_diff_last_state,time_diff_in],axis=-1), self.time_diff_pred_w,self.time_diff_pred_b
        # ))#[B,1]  [0,1]
        # time_diff_loss=tf.reduce_mean(tf.math.square(time_diff_last_state-time_diff_in),axis=-1,keepdims=True)#[B,D]=>[B,1]

        if self.hparams.ipW_att_way == "sumAtt":
            att_logit_mat = tf.matmul(inputs, self.heads, transpose_b=True)
            att_weights = tf.nn.softmax(self._beta * att_logit_mat, axis=-1)
            att_weights = tf.expand_dims(att_weights, 2)
            h_hat = tf.reduce_sum(
                input_tensor=tf.multiply(state[:, : self._slots, :], att_weights), axis=1
            )
            # h_hat = (h_hat + state[:, self._slots, :]) / 2

        if self.hparams.useShareIpPool:
            h_pool = self.ipPoolQuery(inputs, self.ipPool)  # [B,D]
            h_high = state[:, self._slots, :]
            D_h_hat = h_hat.shape[-1]
            self.gatingFuseColl_w = tf.compat.v1.get_variable(name="gatingFuseColl_w", shape=[D_h_hat * 3, D_h_hat],
                                                              dtype=tf.float32, )
            self.gatingFuseColl_b = tf.compat.v1.get_variable(name="gatingFuseColl_b", shape=[D_h_hat],
                                                              dtype=tf.float32,
                                                              initializer=tf.compat.v1.zeros_initializer(), )
            # comp JS divergence
            if self.hparams.useperson_coll_diff_beta:
                person_coll_diff_beta = self.JS(tf.nn.softmax(h_hat, axis=-1),
                                                tf.nn.softmax(h_pool, axis=-1))  # =>[B,1]
                h_hat = tf.compat.v1.nn.xw_plus_b(
                    tf.concat([h_hat, h_high, person_coll_diff_beta * h_pool], -1),
                    self.gatingFuseColl_w, self.gatingFuseColl_b
                )

            # else:
            #     h_hat = tf.compat.v1.nn.xw_plus_b(
            #         tf.concat([h_hat, h_high, h_pool, time_diff_in], -1),
            #         self.gatingFuseColl_w, self.gatingFuseColl_b
            #     )
        # get the true writing attentions
        writing_input = tf.concat([inputs, h_hat], axis=1)
        att_weights = tf.compat.v1.nn.xw_plus_b(
            writing_input, self._writing_W, self._writing_b
        )
        att_weights = tf.nn.relu(att_weights)
        att_weights = tf.matmul(att_weights, self._writing_W02)
        att_weights = tf.nn.softmax(att_weights, axis=-1)
        att_weights = tf.expand_dims(att_weights, 2)

        n_a, n_b = tf.nn.l2_normalize(last[:, :-1], 1), tf.nn.l2_normalize(inputs, 1)
        dist = tf.expand_dims(tf.reduce_sum(input_tensor=n_a * n_b, axis=1), 1)
        dist = tf.math.pow(self._alpha, dist)

        att_weights = att_weights * tf.expand_dims(dist, 1)

        reset = tf.sigmoid(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, h_hat], axis=-1), self._reset_W, self._reset_b
            )
        )
        erase = tf.sigmoid(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, h_hat], axis=-1), self._erase_W, self._erase_b
            )
        )
        add = tf.tanh(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, reset * h_hat], axis=-1), self._add_W, self._add_b
            )
        )

        state_part01 = state[:, : self._slots, :]
        state01 = state_part01 * (
                tf.ones_like(state_part01) - att_weights * tf.expand_dims(erase, 1)
        )
        state01 = state01 + att_weights * tf.expand_dims(erase, 1) * tf.expand_dims(
            add, 1
        )
        state01 = tf.reshape(state01, [-1, (self._slots) * self._real_units])  # [b,]
        state_part03 = state[:, self._slots, :]  # b,D
        state03 = state_part03 * (tf.ones_like(state_part03) - dist * erase)
        state03 = state03 + dist * erase * add
        state = tf.concat([state01, state03, orgInput], axis=-1)

        # print(h_hat)
        # print(tf.expand_dims(time_diff_loss,-1))
        # print(tf.expand_dims(person_coll_diff_beta,-1))
        # loss=tf.concat([h_hat,
        #                 time_diff_loss,
        #                 time_diff_loss],axis=-1)#[B,D+2]

        loss = person_coll_diff_beta  # [B,1]
        return loss, state  # (outputs,state)


class SUMV3Cell(SUMCell):
    """A variant of SUM cell, which upgrades the writing attention"""

    @property
    def output_size(self):
        if self.hparams.use_delta_is_ssl_loss:
            return 1
        else:
            return self._num_units

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
        h = self._real_units
        s = self._slots

        self._basic_build(inputs_shape)

        self._writing_W = self.add_variable(
            name="_writing_W", shape=[d + h, h], initializer=self._kernel_initializer
        )
        self._writing_b = self.add_variable(
            name="_writing_b",
            shape=[h],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)
            ),
        )
        self._writing_W02 = self.add_variable(
            name="_writing_W02", shape=[h, s], initializer=self._kernel_initializer
        )

        self.parameter_set = [
            self._erase_W,
            self._erase_b,
            self._reset_W,
            self._reset_b,
            self._add_W,
            self._add_b,
            self.heads,
            self._writing_W,
            self._writing_W02,
            self._writing_b,
        ]

        self.built = True

    def _activate(self, logit, activation):
        if activation == "sigmoid":
            return tf.nn.sigmoid(logit)
        elif activation == "softmax":
            return tf.nn.softmax(logit)
        elif activation == "relu":
            return tf.nn.relu(logit)
        elif activation == "tanh":
            return tf.nn.tanh(logit)
        elif activation == "elu":
            return tf.nn.elu(logit)
        elif activation == "identity":
            return tf.identity(logit)
        else:
            raise ValueError("this activations not defined {0}".format(activation))

    def _dropout(self, logit, keep_prob):
        """Apply drops upon the input value.

        Args:
            logit (object): The input value.
            keep_prob (float): The probability of keeping each element.

        Returns:
            object: A tensor of the same shape of logit.
        """
        return tf.nn.dropout(x=logit, rate=1 - (keep_prob))

    def _active_layer(self, logit, activation, layer_idx=-1):
        """Transform the input value with an activation. May use dropout.

        Args:
            logit (object): Input value.
            activation (str): A string indicating the type of activation function.
            layer_idx (int): Index of current layer. Used to retrieve corresponding parameters

        Returns:
            object: A tensor after applying activation function on logit.
        """
        if layer_idx >= 0 and self.hparams.user_dropout:
            logit = self._dropout(logit, self.layer_keeps[layer_idx])
        return self._activate(logit, activation)

    def _fcn_net(self, model_output, layer_sizes, scope, first_layer_size=None, reuse=None):
        """Construct the MLP part for the model.

        Args:
            model_output (object): The output of upper layers, input of MLP part
            layer_sizes (list): The shape of each layer of MLP part
            scope (object): The scope of MLP part

        Returns:
            object: Prediction logit after fully connected layer.
        """
        hparams = self.hparams
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            if first_layer_size is None:
                last_layer_size = model_output.shape[-1]
            else:
                last_layer_size = first_layer_size
            layer_idx = 0
            hidden_nn_layers = []
            hidden_nn_layers.append(model_output)
            with tf.compat.v1.variable_scope(
                    "sum_nn_part", initializer=self._kernel_initializer, reuse=reuse
            ) as scope:
                for idx, layer_size in enumerate(layer_sizes):
                    curr_w_nn_layer = tf.compat.v1.get_variable(
                        name="sum_w_nn_layer" + str(layer_idx),
                        shape=[last_layer_size, layer_size],
                        dtype=tf.float32,
                    )
                    curr_b_nn_layer = tf.compat.v1.get_variable(
                        name="sum_b_nn_layer" + str(layer_idx),
                        shape=[layer_size],
                        dtype=tf.float32,
                        initializer=tf.compat.v1.zeros_initializer(),
                    )

                    curr_hidden_nn_layer = (
                            tf.tensordot(
                                hidden_nn_layers[layer_idx], curr_w_nn_layer, axes=1
                            )
                            + curr_b_nn_layer
                    )

                    scope = "nn_part" + str(idx)
                    activation = hparams.activation[idx]

                    curr_hidden_nn_layer = self._active_layer(
                        logit=curr_hidden_nn_layer, activation=activation, layer_idx=idx
                    )
                    hidden_nn_layers.append(curr_hidden_nn_layer)
                    layer_idx += 1
                    last_layer_size = layer_size

                w_nn_output = tf.compat.v1.get_variable(
                    name="sum_w_nn_output", shape=[last_layer_size, 1], dtype=tf.float32
                )
                b_nn_output = tf.compat.v1.get_variable(
                    name="sum_b_nn_output",
                    shape=[1],
                    dtype=tf.float32,
                    initializer=tf.compat.v1.zeros_initializer(),
                )

                nn_output = (
                        tf.tensordot(hidden_nn_layers[-1], w_nn_output, axes=1)
                        + b_nn_output
                )
        return nn_output

    def _attention_fcn(self, query, interestStateMat):
        """Apply attention by fully connected layers.

        Args:
            query (object): The embedding of target item which is regarded as a query in attention operations.
            user_embedding (object): The output of RNN layers which is regarded as user modeling.

        Returns:
            object: Weighted sum of user modeling.
        """
        hparams = self.hparams
        with tf.compat.v1.variable_scope("attention_fcn"):
            query_size = query.shape[-1]
            attention_mat = tf.compat.v1.get_variable(
                name="attention_mat",
                shape=[interestStateMat.shape.as_list()[-1], query_size],
                initializer=self._kernel_initializer,
            )
            att_inputs = tf.tensordot(interestStateMat, attention_mat, [[2], [0]])

            queries = tf.reshape(
                tf.tile(query, [1, att_inputs.shape[1]]), tf.shape(input=att_inputs)
            )
            last_hidden_nn_layer = tf.concat(
                [att_inputs, queries, att_inputs - queries, att_inputs * queries], -1
            )
            att_fnc_output = self._fcn_net(
                last_hidden_nn_layer, hparams.att_fcn_layer_sizes, scope="sum_att_fcn"
            )  # B S D
            att_fnc_output = tf.squeeze(att_fnc_output, -1)
            att_weights = tf.nn.softmax(
                att_fnc_output,
                name="sum_att_weights",
            )
            output = interestStateMat * tf.expand_dims(att_weights, -1)
            return output

    def multiHeadAtt(self, query, key, value):
        with tf.compat.v1.variable_scope("multiHeadAtt", reuse=tf.compat.v1.AUTO_REUSE):
            heads = 2
            Q_size = query.shape.as_list()[-1]
            K_size = key.shape.as_list()[-1]
            V_size = K_size
            head_dim = K_size // heads

            w_mhAtt_Q = tf.compat.v1.get_variable(name="w_mhAtt_Q", shape=[Q_size, head_dim * heads], dtype=tf.float32)
            b_mhAtt_Q = tf.compat.v1.get_variable(name="b_mhAtt_Q", shape=[head_dim * heads], dtype=tf.float32,
                                                  initializer=tf.compat.v1.zeros_initializer(), )
            w_mhAtt_K = tf.compat.v1.get_variable(name="w_mhAtt_K", shape=[K_size, head_dim * heads],
                                                  dtype=tf.float32, )
            b_mhAtt_K = tf.compat.v1.get_variable(name="b_mhAtt_K", shape=[head_dim * heads], dtype=tf.float32,
                                                  initializer=tf.compat.v1.zeros_initializer(), )
            w_mhAtt_V = tf.compat.v1.get_variable(name="w_mhAtt_V", shape=[V_size, head_dim * heads],
                                                  dtype=tf.float32, )
            b_mhAtt_V = tf.compat.v1.get_variable(name="b_mhAtt_V", shape=[head_dim * heads], dtype=tf.float32,
                                                  initializer=tf.compat.v1.zeros_initializer(), )
            mult_prj_Q = tf.compat.v1.nn.xw_plus_b(query, w_mhAtt_Q, b_mhAtt_Q)  # [B,D=d*headNum]
            mult_prj_Q = tf.reshape(mult_prj_Q, shape=[-1, heads, head_dim])
            mult_prj_Q = tf.expand_dims(mult_prj_Q, 1)  # [B,1,heads,d]
            mult_prj_K = tf.compat.v1.nn.xw_plus_b(key, w_mhAtt_K, b_mhAtt_K)  # [B,S,D=d*headNum]
            mult_prj_K = tf.reshape(mult_prj_K, shape=[-1, self._slots, heads, head_dim])  # [B,S,heads,d]

            mult_prj_V = tf.compat.v1.nn.xw_plus_b(value, w_mhAtt_V, b_mhAtt_V)  # [B,S,D=d*headNum]
            mult_prj_V = tf.reshape(mult_prj_V, shape=[-1, self._slots, heads, head_dim])  # [B,S,heads,d]

            # multiHeadAtt
            att_logit = tf.reduce_sum(mult_prj_Q * mult_prj_K, -1, keepdims=True)  # [B,S,heads,d]=>#[B,S,heads,1]
            att_logit = tf.nn.softmax(self._beta * att_logit)  # [B,S,heads,1]
            h_hat = tf.reduce_sum(att_logit * mult_prj_V, axis=1)  # [B,S,heads,d]==>B head d
            h_hat = tf.reshape(h_hat, [-1, heads * head_dim])
        return h_hat

    def ipPoolQuery(self, query, ipPool):
        with tf.compat.v1.variable_scope("ipPoolQuery"):
            query_size = query.shape[-1]
            attention_w = tf.compat.v1.get_variable(name="attention_w", shape=[query_size, self._real_units],
                                                    initializer=self._kernel_initializer, )
            attention_b = tf.compat.v1.get_variable(name="attention_b", shape=[self._real_units],
                                                    initializer=self._kernel_initializer, )
            realQuery = tf.compat.v1.nn.xw_plus_b(query, attention_w, attention_b)
        # [B,D]  [poolNum,D]
        realQuery = tf.expand_dims(realQuery, axis=1)  # [B,1,D]
        pool = tf.expand_dims(ipPool, axis=0)  # [1,poolNum,D]
        poolSoftBeta = 1000
        att_logit = tf.nn.softmax(
            poolSoftBeta * tf.reduce_sum(realQuery * pool, axis=-1, keepdims=True)
            , axis=-2)  # [B,poolNum,1]
        collb_ip = tf.reduce_sum(pool * att_logit, axis=1)  # [B,poolNum,D]=>[B,D]
        return collb_ip

    def ipPoolUpdate(self, query, h_hat, ipPool):
        '''

        Args:
            query: [B,D1]
            h_hat: [B,D2]
            pool:  [num,D2]
        Returns:

        '''
        with tf.compat.v1.variable_scope("ipPoolUpdate"):
            query_size = query.shape[-1]
            attention_w = tf.compat.v1.get_variable(name="ipPoolUpdate_att_w", shape=[query_size, self._real_units],
                                                    initializer=self._kernel_initializer, )
            attention_b = tf.compat.v1.get_variable(name="ipPoolUpdate_att_b", shape=[self._real_units],
                                                    initializer=self._kernel_initializer, )
            realQuery = tf.compat.v1.nn.xw_plus_b(query, attention_w, attention_b)
            realQuery = tf.expand_dims(realQuery, axis=1)  # [B,1,D]
            pool = tf.expand_dims(ipPool, axis=0)  # [1,poolNum,D]
            h_hat = tf.expand_dims(h_hat, axis=1)  # [B,1,D]
            poolSoftBeta = 1000
            att_logit = tf.nn.softmax(
                poolSoftBeta * tf.reduce_sum(realQuery * pool, axis=-1, keepdims=True)
                , axis=-2)  # [B,poolNum,1]
            h_hat_update_pool = tf.reduce_sum(h_hat * att_logit, 0)  # [B,poolNum,D]=>[poolNum,D]

            # write
            fea = tf.concat([ipPool, h_hat_update_pool], -1)  # [poolNum,2*D]
            ipPool_write_w = tf.compat.v1.get_variable(name="ipPoolUpdate_write_w", shape=[fea.shape[-1], 1],
                                                       initializer=self._kernel_initializer, )
            ipPool_write_b = tf.compat.v1.get_variable(name="ipPoolUpdate_write_b", shape=[1],
                                                       initializer=self._kernel_initializer, )
            w_alpha = tf.compat.v1.nn.xw_plus_b(fea, ipPool_write_w, ipPool_write_b)  # [poolNum,1]
            w_alpha = tf.nn.sigmoid(w_alpha)
            return w_alpha * ipPool + (1 - w_alpha) * h_hat_update_pool  # [poolNum,D]

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
        state, last = state[:, : s * h], state[:, s * h:]
        state = tf.reshape(state, [-1, s, h])

        if self.hparams.ipW_att_way == "sumAtt":
            att_logit_mat = tf.matmul(inputs, self.heads, transpose_b=True)
            att_weights = tf.nn.softmax(self._beta * att_logit_mat, axis=-1)
            att_weights = tf.expand_dims(att_weights, 2)
            print(state[:, : self._slots, :])
            print(att_weights)
            h_hat = tf.reduce_sum(
                input_tensor=tf.multiply(state[:, : self._slots, :], att_weights), axis=1
            )

        if self.hparams.ipW_att_way == "dimnAtt":
            ip_state = state[:, : self._slots, :]
            with tf.compat.v1.variable_scope(
                    "itemIPSim", initializer=self._kernel_initializer, reuse=tf.compat.v1.AUTO_REUSE
            ) as scope:
                itemIPSim_w = tf.compat.v1.get_variable(
                    name="itemIPSim_w",
                    shape=[inputs.shape[-1], h],
                    dtype=tf.float32,
                )
                itemIPSim_b = tf.compat.v1.get_variable(
                    name="itemIPSim_b",
                    shape=[h],
                    dtype=tf.float32,
                    initializer=tf.compat.v1.zeros_initializer(),
                )
                ip_inputs = tf.compat.v1.nn.xw_plus_b(inputs, itemIPSim_w, itemIPSim_b)
            if self.hparams.itemIPSim == "cosSim":
                # cos sim
                norm_inputs = tf.expand_dims(tf.nn.l2_normalize(ip_inputs, axis=-1), 1)  # [B,1,H]
                norm_ip_state = tf.nn.l2_normalize(ip_state, axis=-1)  # [B,S,H]
                ip_att_logit_mat = tf.reduce_sum(norm_inputs * norm_ip_state, axis=-1,
                                                 keepdims=True)  # [B,S,H]=>[B,S,1]
            elif self.hparams.itemIPSim == "dotProduct":
                ip_att_logit_mat = tf.reduce_sum(tf.expand_dims(ip_inputs, 1) * ip_state, axis=-1,
                                                 keepdims=True)  # [B,S,1]

            att_weights = tf.nn.softmax(self._beta * ip_att_logit_mat, axis=1)  # [B,S,1] along S
            h_hat = tf.reduce_sum(
                input_tensor=tf.multiply(ip_state, att_weights), axis=1
            )  # [B,D]
        if self.hparams.ipW_att_way == "attFcn":
            h_hat = self._attention_fcn(inputs, state[:, : self._slots, :])
            h_hat = tf.reduce_sum(input_tensor=h_hat, axis=1)
        if self.hparams.ipW_att_way == "multiHeadAtt":
            ip = state[:, : self._slots, :]  # [B,S,D]
            h_hat = self.multiHeadAtt(inputs, ip, ip)

        if self.hparams.useShareIpPool:
            # self.ipPool = tf.compat.v1.Print(self.ipPool, [self.ipPool[:2, :10]], message="********self.cell.ipPool:\n",
            #                                  summarize=20)
            h_pool = self.ipPoolQuery(inputs, self.ipPool)
            h_highway = state[:, self._slots, :]  # [B,D]
            D_h_hat = h_hat.shape[-1]
            h_cat = tf.concat([h_hat, h_highway, h_pool], axis=-1)
            if self.hparams.gatingFuseColl:
                with tf.compat.v1.variable_scope(
                        "gatingFuseColl", initializer=self._kernel_initializer, reuse=tf.compat.v1.AUTO_REUSE
                ) as scope:
                    if self.hparams.gatingWay == "mapping":
                        gatingFuseColl_w = tf.compat.v1.get_variable(name="gatingFuseColl_w",
                                                                     shape=[D_h_hat * 3, D_h_hat], dtype=tf.float32, )
                        gatingFuseColl_b = tf.compat.v1.get_variable(name="gatingFuseColl_b", shape=[D_h_hat],
                                                                     dtype=tf.float32,
                                                                     initializer=tf.compat.v1.zeros_initializer(), )

                        h_hat = tf.compat.v1.nn.xw_plus_b(
                            h_cat,
                            gatingFuseColl_w, gatingFuseColl_b
                        )
                    if self.hparams.gatingWay == "weiSum":
                        gatingFuseColl_w = tf.compat.v1.get_variable(name="gatingFuseColl_w",
                                                                     shape=[D_h_hat * 3, 3], dtype=tf.float32, )
                        gatingFuseColl_b = tf.compat.v1.get_variable(name="gatingFuseColl_b", shape=[3],
                                                                     dtype=tf.float32,
                                                                     initializer=tf.compat.v1.zeros_initializer(), )
                        fusingScore = tf.nn.sigmoid(tf.compat.v1.nn.xw_plus_b(
                            tf.concat([h_hat, h_highway, h_pool], axis=-1),
                            gatingFuseColl_w, gatingFuseColl_b
                        ))  # [B,3]
                        # fusingScore=tf.nn.softmax(fusingScore,axis=-1)#[B,3]
                        h_hat = tf.reshape(h_cat, shape=[-1, 3, D_h_hat]) * tf.expand_dims(fusingScore,
                                                                                           -1)  # [B,3,D]*[B,3,1]
                        h_hat = tf.reduce_mean(h_hat, axis=-2)  # [B,3,D]=>[B,D]

            else:
                h_hat = (h_hat + state[:, self._slots, :] + h_pool) / 3
            if self.hparams.useShareIpPoolWrite:
                self.ipPool = self.ipPoolUpdate(inputs, h_hat, self.ipPool)
        else:
            h_hat = (h_hat + state[:, self._slots, :]) / 2

        # get the true writing attentions
        writing_input = tf.concat([inputs, h_hat], axis=1)
        att_weights = tf.compat.v1.nn.xw_plus_b(
            writing_input, self._writing_W, self._writing_b
        )
        att_weights = tf.nn.relu(att_weights)
        att_weights = tf.matmul(att_weights, self._writing_W02)
        att_weights = tf.nn.softmax(att_weights, axis=-1)
        att_weights = tf.expand_dims(att_weights, 2)

        n_a, n_b = tf.nn.l2_normalize(last, 1), tf.nn.l2_normalize(inputs, 1)
        dist = tf.expand_dims(tf.reduce_sum(input_tensor=n_a * n_b, axis=1), 1)
        dist = tf.math.pow(self._alpha, dist)

        att_weights = att_weights * tf.expand_dims(dist, 1)

        reset = tf.sigmoid(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, h_hat], axis=-1), self._reset_W, self._reset_b
            )
        )
        erase = tf.sigmoid(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, h_hat], axis=-1), self._erase_W, self._erase_b
            )
        )
        add = tf.tanh(
            tf.compat.v1.nn.xw_plus_b(
                tf.concat([inputs, reset * h_hat], axis=-1), self._add_W, self._add_b
            )
        )

        start_part01 = state[:, : self._slots, :]
        state01 = start_part01 * (
                tf.ones_like(start_part01) - att_weights * tf.expand_dims(erase, 1)
        )
        state01 = state01 + att_weights * tf.expand_dims(erase, 1) * tf.expand_dims(
            add, 1
        )
        state01 = tf.reshape(state01, [-1, self._slots * self._real_units])

        start_part02 = state[:, self._slots, :]
        state02 = start_part02 * (tf.ones_like(start_part02) - dist * erase)
        state02 = state02 + dist * erase * add
        state = tf.concat([state01, state02, inputs], axis=-1)

        return state, state  # (outputs,state)


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
