import tensorflow as tf
import functools


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
                 hyperparams, features_len, labels_len, seed_value, is_training):

        # placeholders
        self.features_placeholder = placeholders['features']
        self.label_placeholder = placeholders['labels']
        try:
            self.seq_placeholder = placeholders['sequences']
        except KeyError:
            self.seq_placeholder = None
        # self.state_placeholder = state_placeholder

        self.features_len = features_len
        self.labels_len = labels_len
        self.seed_value = seed_value
        self.is_training = is_training

        # hyperparameters
        self.batch_n = hyperparams['batch_n'] if is_training else 1
        self.sequence_max_n = hyperparams['sequence_max_n']

        self.units_n = hyperparams['units_n']
        self.layers_n = hyperparams['layers_n']
        self.dropout_r = hyperparams['dropout_r'] if is_training else 0.

        self.learning_r = hyperparams['learning_r']
        self.decay_r = hyperparams['decay_r']

        self.global_step = tf.Variable(0, trainable=False)

        self.prediction
        self.optimize
        self.error
        # self.add_global_step

    @define_scope
    def prediction(self):

        # LSTM Cells
        cells = [
            tf.nn.rnn_cell.LSTMCell(
                num_units=self.units_n, dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer(
                    uniform=True, seed=self.seed_value, dtype=tf.float32
                ), name="lstm_cell"  # , reuse=True
            ) for _ in range(self.layers_n)
        ]
        cells_stacked = tf.nn.rnn_cell.MultiRNNCell(cells)

        # Dropout wrapper
        if self.is_training and self.dropout_r > 0:
            cells_stacked = tf.contrib.rnn.DropoutWrapper(
                cells_stacked,
                output_keep_prob=(1. - self.dropout_r),
                variational_recurrent=False,  # https://arxiv.org/abs/1512.05287
                seed=self.seed_value
            )

        '''
        RETURN:
            i. output = outputs/activations for all time sequences
            ii. states = tuples of hidden state and final output
                states[layer1].c = hidden state for layer1
                states[layer1].h = final output/activation (even with dynamic lengths)
        '''
        # LSTM layer,  shape: [batch_n, sequence_max_n, units_n], (c=[batch_n, units_n], h=[batch_n, units_n])
        _, states = tf.nn.dynamic_rnn(
            cells_stacked,
            self.features_placeholder,
            sequence_length=self.seq_placeholder if self.sequence_max_n else None,
            initial_state=None,  # zero state
            dtype=tf.float32,
            time_major=False
        )
        # output = tf.reshape(output, [-1, self.units_n])  # resize to 2D (optional)
        # tf.summary.histogram("states", states)

        # TODO: REDUCE WEIGHTS OF "normal" INSTANCES

        # Dense layer,  shape: [batch_n, labels_len]
        logits = tf.layers.dense(
            states[-1].h,
            # tf.layers.batch_normalization(output[:, -1, :]),  # batch normalization?
            self.labels_len,
            activation=None,
            use_bias=True,
            kernel_initializer=None,
            bias_initializer=tf.zeros_initializer(),
            trainable=True,
            name="layer_output"
            # reuse=True
        )
        # logits = tf.reshape(logits, [self.batch_n, self.sequence_max_n, self.labels_len])  # resize back to 3D

        return logits

    @define_scope
    def optimize(self):

        class_weights = tf.constant([[1.0, 2.0, 2.0, 2.0, 2.0]])
        one_hot_labels = tf.one_hot(self.label_placeholder, self.labels_len)
        # deduce weights for batch samples based on their true label
        weights = tf.reduce_sum(class_weights * one_hot_labels, axis=1)
        # compute your (unweighted) softmax cross entropy loss
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(
            labels=one_hot_labels,
            logits=self.prediction
        )
        # apply the weights, relying on broadcasting of the multiplication
        weighted_losses = unweighted_losses * weights
        # reduce the result to get your final loss
        loss = tf.reduce_mean(weighted_losses)

        # compute Loss,  shape: [batch_n]
        # m:n - seq2seq.sequence_loss, with sequence mask and averaging over batches
        # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     labels=self.label_placeholder,
        #     logits=self.prediction,
        #     name="softmax_crossentropy"
        # )

        # loss = tf.reduce_mean(cross_entropy)  # shape: 1

        # Learning Rate decay (uses Adam)
        # learning_rate = tf.train.exponential_decay(
        #     self.learning_r, self.global_step, decay_steps=1,
        #     decay_rate=self.decay_r, name="lr_decay"
        # )

        # calculate & clip Gradients
        trainables = tf.trainable_variables()  # returns all variables with trainable=True
        gradients = tf.gradients(loss, trainables, name="gradients")
        clipped_gradients, _ = tf.clip_by_global_norm(
            gradients, clip_norm=5,  # clipping ratio: 5
            name="gradient_clipping"
        )

        # Optimization (minimize)
        update_step = tf.train.AdamOptimizer(
            # learning_rate=learning_rate,
            name="adam_optimizer"
        ).apply_gradients(
            zip(clipped_gradients, trainables),
            name="apply_gradients", global_step=tf.train.get_or_create_global_step())

        return update_step, loss

    @define_scope
    def error(self):

        pred = tf.argmax(tf.nn.softmax(self.prediction), axis=1, output_type=tf.int32)  # shape: [batch_n]
        pred_positive = tf.equal(pred, self.label_placeholder)

        return pred, tf.reduce_mean(
            tf.cast(pred_positive, tf.float32)  # shape: [batch_n]
        )  # shape: 1
    #
    # @define_scope
    # def add_global_step(self):
    #     return self.global_step.assign_add(1)
