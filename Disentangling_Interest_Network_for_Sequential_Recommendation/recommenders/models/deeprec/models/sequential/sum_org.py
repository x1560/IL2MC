import numpy as np
import tensorflow as tf
from tensorflow.compat.v1.nn import dynamic_rnn
from recommenders.models.deeprec.models.sequential.sequential_base_model import (
    SequentialBaseModel,
)
from recommenders.models.deeprec.models.sequential.sum_cells_org import (
    SUMCell,
    SUMV2Cell
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
            self.history_embedding = tf.concat(
                [self.item_history_embedding, self.cate_history_embedding], 2
            )
            cell = self._create_sumcell()
            self.cell = cell
            cell.model = self
            final_state = self._build_sum(cell)

            final_state, att_weights = self._attention_query_by_state(
                final_state, self.target_item_embedding
            )
            model_output = tf.concat([final_state, self.target_item_embedding], 1)
            if "other_metrics_basedOn_RecList" in self.hparams.values():
                item_lookup=tf.concat([self.item_lookup,self.cateIdOfAllItems_embedding],-1)#[vocNum,H1+H2]
                allItemsEmd = tf.expand_dims(item_lookup, 0)  # [vocNum,H]==>[1,vocNum,H]
                final_state = tf.expand_dims(final_state, 1)  # [B,H]=>[B,1,H]
                final_state = final_state[::self.hparams.test_num_ngs + 1, :, :]
                vocNum = allItemsEmd.shape[1]
                final_state = tf.tile(final_state, [1, vocNum, 1])
                allItemsEmd = tf.tile(allItemsEmd, [tf.shape(final_state)[0], 1, 1])
                self.recItemListScoreModelOut = tf.concat([final_state, allItemsEmd], -1)
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
                )

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
                )

            else:
                att_res = seq_output

        return att_res, att_weights

    def _create_sumcell(self):
        """Create a SUM cell

        Returns:
            object: An initialized SUM cell
        """
        hparams = self.hparams
        input_embedding_dim = self.history_embedding.shape[-1]
        input_params = [
            hparams.hidden_size * hparams.slots + input_embedding_dim,
            hparams.slots,
            hparams.attention_size,
            input_embedding_dim,
        ]
        sumcells = {"SUM": SUMCell, "SUMV2": SUMV2Cell}
        sumCell = sumcells[hparams.cell]
        res = None
        if hparams.cell in ["SUM", "SUMV2"]:
            res = sumCell(*input_params)
        else:
            raise ValueError("ERROR! Cell type not support: {0}".format(hparams.cell))
        return res

    def _build_sum(self, cell):
        """Generate  user memory states from behavior sequence

        Args:
            object: An initialied SUM cell.

        Returns:
            object: A flatten representation of user memory states, in the shape of (BatchSize, SlotsNum x HiddenSize)
        """
        hparams = self.hparams
        with tf.compat.v1.variable_scope("sum"):
            self.mask = self.iterator.mask
            self.sequence_length = tf.reduce_sum(input_tensor=self.mask, axis=1)

            rum_outputs, final_state = dynamic_rnn(
                cell,
                inputs=self.history_embedding,
                dtype=tf.float32,
                sequence_length=self.sequence_length,
                scope="sum",
                initial_state=cell.zero_state(
                    tf.shape(input=self.history_embedding)[0], tf.float32
                ),
            )

            final_state = final_state[:, : hparams.slots * hparams.hidden_size]

            self.heads = cell.heads
            self.alpha = cell._alpha
            self.beta = cell._beta
            # tf.compat.v1.summary.histogram("SUM_outputs", rum_outputs)

        return final_state
