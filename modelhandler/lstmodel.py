import tensorflow as tf
import functools
from modelhandler.bn_lstm import BNLSTMCell


# Hafner, Danijar. Structuring Your TensorFlow Models, 2016.
def define_scope(func):
    attribute = '_cache_' + func.__name__

    @property
    @functools.wraps(func)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(func.__name__, reuse=tf.AUTO_REUSE):
                setattr(self, attribute, func(self))
        return getattr(self, attribute)

    return decorator


def extract_axis_1(data, ind):
    """
    Get specified elements along the first axis of tensor.
    :param data: Tensorflow tensor that will be subsetted.
    :param ind: Indices to take (one for each element along axis 0 of data).
    :return: Subsetted tensor.
    """

    batch_range = tf.range(tf.shape(data)[0])
    indices = tf.stack([batch_range, ind], axis=1)
    res = tf.gather_nd(data, indices)

    return res


class LSTModel(object):
    def __init__(self, placeholders,
                 hyperparams, batch_n, features_len, labels_len, m1_labels, seed_value, is_training):

        # placeholders
        self.features_placeholder = placeholders['features']
        self.label_placeholder = placeholders['labels']
        self.netw_seq_placeholder = placeholders['netw_sequence']
        try:
            self.state_placeholder = placeholders['states']
        except KeyError:
            self.state_placeholder = None

        self.features_len = features_len
        self.labels_len = labels_len
        self.m1_labels = m1_labels
        self.seed_value = seed_value
        self.is_training = is_training

        # hyperparameters
        self.batch_n = batch_n
        self.netw_seq = hyperparams['netw_sequence']
        self.units_n = hyperparams['units_n']
        self.layers_n = hyperparams['layers_n']
        self.dropout_r = hyperparams['dropout_r'] if is_training else 0.
        self.learning_r = hyperparams['learning_r']
        # self.decay_r = hyperparams['decay_r']

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.prediction
        self.optimize
        self.error
        # self.add_global_step

    @define_scope
    def prediction(self):

        # tf.nn.rnn_cell.GRUCell(
        #     self.units_n, kernel_initializer=tf.contrib.layers.xavier_initializer(seed=self.seed_value)
        # ) for _ in range(self.layers_n)

        # LSTM cells
        if self.is_training:
            netw_cell = tf.nn.rnn_cell.MultiRNNCell([
                tf.nn.rnn_cell.DropoutWrapper(
                    tf.nn.rnn_cell.LSTMCell(
                        num_units=self.units_n, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(
                            uniform=True, seed=self.seed_value, dtype=tf.float32
                        )
                    ), output_keep_prob=1.-self.dropout_r, seed=self.seed_value
                ) for _ in range(self.layers_n)
            ])

        else:
            netw_cell = tf.nn.rnn_cell.MultiRNNCell([
                tf.nn.rnn_cell.LSTMCell(
                    num_units=self.units_n, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(
                        uniform=True, seed=self.seed_value, dtype=tf.float32
                    )
                ) for _ in range(self.layers_n)
            ])

        # LSTM States
        if self.state_placeholder is not None:
            state_per_layer_list = tf.unstack(self.state_placeholder, axis=0)
            rnn_tuple_state = tuple(
                [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[i][0], state_per_layer_list[i][1])
                 for i in range(self.layers_n)]
            )
        else:
            rnn_tuple_state = None

        '''
        RETURN:
            i. output = outputs/activations for all time sequences (output[lastseq] identical to states[lastlayer].h)
            shape: [batch_n, sequence_max_n, units_n]
            ii. states = tuples of hidden state and final output
                states[layer1].c = hidden state for layer1
                states[layer1].h = final output/activation for layer1 (even with dynamic lengths)
            shape for last layer [-1]: (c=[batch_n, units_n], h=[batch_n, units_n])
            
            NOTE: output vectors contain "dropout'ed" vectors, states will retain the "original" vectors
        '''

        with tf.variable_scope('network') as network_scope:
            output, state = tf.nn.dynamic_rnn(
                netw_cell,
                self.features_placeholder,
                sequence_length=self.netw_seq_placeholder,
                initial_state=rnn_tuple_state,  # zero state
                dtype=tf.float32,
                time_major=False,
                scope=network_scope
            )
            # output = tf.reshape(output, [-1, self.units_n])  # resize to 2D (optional)

        # with tf.variable_scope('dropout'):
        #     output_dropped = tf.layers.dropout(
        #         state[-1][1] if self.m1_labels else output,  # lstm
        #         # output[:, -1, :],  # gru
        #         rate=self.dropout_r,
        #         seed=self.seed_value,
        #         training=self.is_training
        #     )

        # Dense layer,  shape: [batch_n, labels_len]
        with tf.variable_scope('classifier'):
            logits = tf.layers.dense(
                # dropout will only be applied to output (not state)
                extract_axis_1(output, self.netw_seq_placeholder - 1) if self.m1_labels else output,
                self.labels_len,
                activation=None,
                use_bias=True,
                kernel_initializer=tf.contrib.layers.xavier_initializer(seed=self.seed_value),
                bias_initializer=tf.zeros_initializer(),
                trainable=True,
                name="layer_output"
                # reuse=True
            )
            # logits = tf.reshape(logits, [self.batch_n, self.sequence_max_n, self.labels_len])  # resize back to 3D

        # compute Loss,  shape: [batch_n]
        # m:n - seq2seq.sequence_loss, with sequence mask and averaging over batches
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.label_placeholder,
            logits=logits,
            name="softmax_crossentropy"
        ) if self.m1_labels else tf.contrib.seq2seq.sequence_loss(
            logits=logits,
            targets=self.label_placeholder,
            weights=tf.sequence_mask(
                self.netw_seq_placeholder, maxlen=self.netw_seq, dtype=tf.float32
            ),  # for masking variable sequences
            average_across_timesteps=False,
            average_across_batch=True
        )

        # KDD99_10
        # class_weights = tf.constant([[0.000587, 0.955254, 0.009385, 0.000084, 0.034691]])  # norm u2r probe dos r2l
        # one_hot_labels = tf.one_hot(self.label_placeholder, self.labels_len)
        # weights = tf.reduce_sum(class_weights * one_hot_labels, axis=1)
        # cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
        #     labels=one_hot_labels,
        #     logits=logits
        # )
        # cross_entropy = cross_entropy * weights

        loss = tf.reduce_mean(cross_entropy)

        return (output, state), logits, loss

    @define_scope
    def optimize(self):

        _, _, loss = self.prediction
        tf.summary.scalar('loss', loss)

        tvars = tf.trainable_variables()

        # Gradient Clipping
        clipped_gradients, _ = tf.clip_by_global_norm(
            tf.gradients(loss, tvars, name="gradients"),
            clip_norm=5,
            name="gradient_clipping"
        )

        # Adam Optimizer
        opt = tf.train.AdamOptimizer(
            self.learning_r,
            name="adam_optimizer"
        )

        # Update gradients
        update_step = opt.apply_gradients(
            zip(clipped_gradients, tvars),
            name="apply_gradients",
            global_step=self.global_step
        )

        return update_step, tf.summary.merge_all(), self.global_step  # (op, global_step)

    @define_scope
    def error(self):

        outputs, logits, loss = self.prediction

        # double check softmax when using M:N labelling scheme
        pred = tf.argmax(
            tf.nn.softmax(
                logits if self.m1_labels else tf.reshape(
                    logits, [-1, self.labels_len])[:tf.reduce_sum(self.netw_seq_placeholder)]
            ),
            axis=1,
            output_type=tf.int32
        )  # shape: [batch_n]

        truth = self.label_placeholder if self.m1_labels else tf.reshape(
            self.label_placeholder, [-1])[:tf.reduce_sum(self.netw_seq_placeholder)]

        return outputs, truth, pred, loss, tf.reduce_mean(
            tf.cast(tf.equal(pred, truth), tf.float32)  # shape: [batch_n]
        )  # ((output, state), ground truth, predictions, loss, accuracy)
