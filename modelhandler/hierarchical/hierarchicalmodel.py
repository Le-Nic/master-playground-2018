import tensorflow as tf
import functools
from modelhandler.hierarchical.model_components import task_specific_attention


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


class LSTModel(object):
    def __init__(self, placeholders,
                 hyperparams, batch_n, features_len, labels_len, m1_labels, seed_value, is_training):

        # placeholders
        self.features_placeholder = placeholders['features']
        self.label_placeholder = placeholders['labels']
        self.netw_seq_placeholder = placeholders['netw_sequence']
        self.host_seq_placeholder = placeholders['host_sequence']
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
        self.batch_n = batch_n  # different batch for train/dev/test-sets
        self.netw_seq = hyperparams['netw_sequence']  # previously: sequence_max_n
        self.host_seq = hyperparams['host_sequence']  # previously: sequence_max_n
        self.netw_output_n = hyperparams['netw_output_n']
        self.host_output_n = hyperparams['host_output_n']
        self.units_n = hyperparams['units_n']
        self.layers_n = hyperparams['layers_n']
        self.dropout_r = hyperparams['dropout_r'] if is_training else 0.
        self.learning_r = hyperparams['learning_r']

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.prediction
        self.optimize
        self.error
        # self.add_global_step

    @define_scope
    def prediction(self):

        if self.is_training:
            host_cell = tf.nn.rnn_cell.MultiRNNCell([
                tf.nn.rnn_cell.DropoutWrapper(
                    tf.nn.rnn_cell.GRUCell(
                        num_units=self.units_n, dtype=tf.float32, kernel_initializer=tf.contrib.layers.xavier_initializer(
                            uniform=True, seed=self.seed_value, dtype=tf.float32
                        )
                    ), output_keep_prob=1. - self.dropout_r, seed=self.seed_value
                ) for _ in range(self.layers_n)
            ])
            netw_cell = tf.nn.rnn_cell.MultiRNNCell([
                tf.nn.rnn_cell.DropoutWrapper(
                    tf.nn.rnn_cell.GRUCell(
                        num_units=self.units_n, dtype=tf.float32, kernel_initializer=tf.contrib.layers.xavier_initializer(
                            uniform=True, seed=self.seed_value, dtype=tf.float32
                        )
                    ), output_keep_prob=1.-self.dropout_r, seed=self.seed_value
                ) for _ in range(self.layers_n)
            ])

        else:
            host_cell = tf.nn.rnn_cell.MultiRNNCell([
                tf.nn.rnn_cell.GRUCell(
                    num_units=self.units_n, dtype=tf.float32, kernel_initializer=tf.contrib.layers.xavier_initializer(
                        uniform=True, seed=self.seed_value, dtype=tf.float32
                    )
                ) for _ in range(self.layers_n)
            ])
            netw_cell = tf.nn.rnn_cell.MultiRNNCell([
                tf.nn.rnn_cell.GRUCell(
                    num_units=self.units_n, dtype=tf.float32, kernel_initializer=tf.contrib.layers.xavier_initializer(
                        uniform=True, seed=self.seed_value, dtype=tf.float32
                    )
                ) for _ in range(self.layers_n)
            ])

        # Host-level e.g. (64 ,8, 8, 238) -> (512, 8, 238)
        host_level_inputs = tf.reshape(self.features_placeholder, [
            self.batch_n * self.netw_seq,
            self.host_seq,
            self.features_len
        ])

        # (512)
        host_level_lengths = tf.reshape(self.host_seq_placeholder, [self.batch_n * self.netw_seq])

        with tf.variable_scope('host') as host_scope:
            # (512, 8, 32), e.g. units_n = 32
            # word_encoder_output, _ = bidirectional_rnn(
            #     cell_fw=host_cell, cell_bw=host_cell,
            #     # cell=host_cell,
            #     inputs_embedded=host_level_inputs,
            #     input_lengths=host_level_lengths,
            #     scope=host_scope
            # )
            host_encoder_output, _ = tf.nn.dynamic_rnn(
                host_cell,
                host_level_inputs,
                sequence_length=host_level_lengths,
                # initial_state=rnn_tuple_state,  # zero state
                dtype=tf.float32,
                time_major=False,
                scope=host_scope
            )

            with tf.variable_scope('attention') as attention_scope:
                # (512, 238), e.g. host_output_n = 238
                host_level_output = task_specific_attention(
                    host_encoder_output,
                    self.host_output_n,
                    initializer=tf.contrib.layers.xavier_initializer(seed=self.seed_value),
                    scope=attention_scope
                )

        # Network-level (64, 8, 238)
        network_level_inputs = tf.reshape(host_level_output, [self.batch_n, self.netw_seq, self.host_output_n])

        with tf.variable_scope('network') as network_scope:
            # (64, 8, 32)
            # network_encoder_output, _ = bidirectional_rnn(
            #     cell_fw=netw_cell, cell_bw=netw_cell,
            #     # cell=host_cell,
            #     inputs_embedded=network_level_inputs,
            #     input_lengths=self.netw_seq_placeholder,
            #     scope=network_scope
            # )
            network_encoder_output, _ = tf.nn.dynamic_rnn(
                netw_cell,
                network_level_inputs,
                sequence_length=self.netw_seq_placeholder,
                dtype=tf.float32,
                time_major=False,
                scope=network_scope
            )

            with tf.variable_scope('attention') as attention_scope:
                # (64, 238), e.g. netw_output_n = 238
                network_level_output = task_specific_attention(
                    network_encoder_output,
                    self.netw_output_n,
                    initializer=tf.contrib.layers.xavier_initializer(seed=self.seed_value),
                    scope=attention_scope
                )

        with tf.variable_scope('classifier'):
            # (64, 2),  e.g. labels_len = 2
            logits = tf.layers.dense(
                network_level_output,
                self.labels_len,
                activation=None,
                use_bias=True,
                kernel_initializer=tf.contrib.layers.xavier_initializer(seed=self.seed_value),
                bias_initializer=tf.zeros_initializer(),
                trainable=True,
                name="layer_output"
                # reuse=True
            )

            # class_weights = tf.constant([0.053, 0.947])  # 1/19 vs 18/19
            # if self.labels_len != 2:
            #     print("Please specify weights for multiclass")
            #     exit()
            #
            # # specify the weights for each sample in the batch (without having to compute the onehot label matrix)
            # weights = tf.gather(class_weights, self.label_placeholder)
            # cross_entropy = tf.losses.sparse_softmax_cross_entropy(self.label_placeholder, logits, weights)

            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.label_placeholder, logits=logits, name="softmax_crossentropy")

            loss = tf.reduce_mean(cross_entropy)

            return network_level_output, logits, loss

            # prediction = tf.argmax(logits, axis=-1)
            # return prediction

    @define_scope
    def optimize(self):
        # normal to attack ratio (~18.2968 : 1)

        _, _, loss = self.prediction
        tf.summary.scalar('loss', loss)

        tvars = tf.trainable_variables()

        # Gradient Clipping
        grads, global_norm = tf.clip_by_global_norm(
            tf.gradients(loss, tvars),
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
            zip(grads, tvars),
            name="apply_gradients",
            global_step=self.global_step
        )

        return update_step, tf.summary.merge_all(), self.global_step  # (op, global_step)

    @define_scope
    def error(self):

        outputs, logits, loss = self.prediction

        pred = tf.argmax(
            tf.nn.softmax(logits),
            axis=1,
            output_type=tf.int32
        )  # shape: [batch_n]

        truth = self.label_placeholder

        return outputs, truth, pred, loss, tf.reduce_mean(
            tf.cast(tf.equal(pred, truth), tf.float32)  # shape: [batch_n]
        )  # (ground truth, predictions, loss, accuracy)
