import tensorflow as tf
import functools


# Hafner, Danijar. Structuring Your TensorFlow Models, 2016.
def define_scope(func):
    attribute = '_cache_' + func.__name__

    @property
    @functools.wraps(func)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(func.__name__):
                setattr(self, attribute, func(self))
        return getattr(self, attribute)

    return decorator


class LSTModel(object):
    def __init__(self, features_placeholder, label_placeholder, seq_placeholder, state_placeholder,
                 features_len, batch_n, sequence_max_n, labels_len, units_n, layers_n,
                 learning_r, decay_r, seed_value):

        # placeholders
        self.features_placeholder = features_placeholder
        self.label_placeholder = label_placeholder
        self.seq_placeholder = seq_placeholder
        self.state_placeholder = state_placeholder

        # datasets' hyperparameters
        self.features_len = features_len
        self.batch_n = batch_n
        self.sequence_max_n = sequence_max_n
        self.labels_len = labels_len

        # models' hyperparameters
        self.units_n = units_n
        self.layers_n = layers_n

        self.learning_r = learning_r
        self.decay_r = decay_r
        self.seed_value = seed_value

        self.global_step = tf.Variable(0, trainable=False)

        self.prediction
        self.optimize
        self.error
        self.add_global_step

    @define_scope
    def prediction(self):

        # set up LSTM cells & states
        state_per_layer = tf.unstack(self.state_placeholder, axis=0)
        state_current = tuple(
            [tf.nn.rnn_cell.LSTMStateTuple(c=state_per_layer[i][0], h=state_per_layer[i][1])
             for i in range(self.layers_n)])

        cells = [tf.nn.rnn_cell.LSTMCell(num_units=self.units_n, dtype=tf.float32)
                 for _ in range(self.layers_n)]
        cells_stacked = tf.nn.rnn_cell.MultiRNNCell(cells)

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
            sequence_length=self.seq_placeholder,
            initial_state=state_current,
            dtype=tf.float32,
            time_major=False)
        # output = tf.reshape(output, [-1, self.units_n])  # resize to 2D (optional)

        # TODO: REDUCE WEIGHTS OF "normal" INSTANCES
        # TODO: check lstm and output layer initializer

        # Dense layer,  shape: [batch_n, labels_len]
        logits = tf.layers.dense(
            states[-1].h,
            # tf.layers.batch_normalization(output[:, -1, :]),  # batch normalization?
            self.labels_len,
            activation=None,
            use_bias=True,
            kernel_initializer=tf.contrib.layers.xavier_initializer(seed=self.seed_value, dtype=tf.float32),   # tanh
            bias_initializer=tf.zeros_initializer(),
            trainable=True
            # reuse=True
        )
        # logits = tf.reshape(logits, [self.batch_n, self.sequence_max_n, self.labels_len])  # resize back to 3D

        return logits

    @define_scope
    def optimize(self):
        logits = self.prediction

        # compute Loss,  shape: [batch_n]
        # m:n - seq2seq.sequence_loss, with sequence mask and averaging over batches
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.label_placeholder,
            logits=logits
        )
        loss = tf.reduce_mean(cross_entropy)  # shape: 1

        # Learning Rate decay
        learning_rate = tf.train.exponential_decay(self.learning_r, self.global_step,
                                                   decay_steps=1, decay_rate=self.decay_r)

        # calculate & clip Gradients
        trainables = tf.trainable_variables()  # returns all variables with trainable=True
        gradients = tf.gradients(loss, trainables)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=5)  # clipping ratio: 5

        # Optimization (minimize)
        update_step = tf.train.AdamOptimizer(learning_rate=learning_rate) \
            .apply_gradients(zip(clipped_gradients, trainables))

        return update_step, loss

    @define_scope
    def error(self):
        predictions = tf.argmax(tf.nn.softmax(self.prediction), axis=1, output_type=tf.int32)  # shape: [batch_n]

        return tf.reduce_mean(
            tf.cast(tf.not_equal(predictions, self.label_placeholder), tf.int32)  # shape: [batch_n]
        )  # shape: 1

    @define_scope
    def add_global_step(self):
        return self.global_step.assign_add(1)
