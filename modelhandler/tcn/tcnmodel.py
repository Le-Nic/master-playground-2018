import tensorflow as tf
import functools
from modelhandler.tcn.tcn import TemporalConvNet


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


class TCNModel(object):
    def __init__(self, placeholders,
                 hyperparams, batch_n, features_len, labels_len, m1_labels, seed_value, is_training):

        # placeholders
        self.features_placeholder = placeholders['features']
        self.label_placeholder = placeholders['labels']
        self.netw_seq_placeholder = placeholders['netw_sequence']

        self.batch_n = batch_n  # different batch for train/dev/test-sets
        self.features_len = features_len
        self.labels_len = labels_len
        self.m1_labels = m1_labels
        self.seed_value = seed_value
        self.is_training = is_training

        # hyperparameters
        self.netw_seq = hyperparams['netw_sequence']
        # self.netw_output_n = hyperparams['netw_output_n']
        self.channels_n = hyperparams['channels_n']
        self.kernels_n = hyperparams['kernels_n']

        # self.units_n = hyperparams['units_n']
        # self.layers_n = hyperparams['layers_n']
        self.dropout_r = hyperparams['dropout_r'] if is_training else 0.
        self.learning_r = hyperparams['learning_r']

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.prediction
        self.optimize
        self.error
        # self.add_global_step

    @define_scope
    def prediction(self):

        output = TemporalConvNet(
            input_layer=self.features_placeholder,
            num_channels=self.channels_n,
            sequence_length=self.netw_seq,
            kernel_size=self.kernels_n,
            dropout=self.dropout_r if self.is_training else .0,
            atten=False,
            seed=self.seed_value
        )

        with tf.variable_scope('classifier'):
            # (64, 2),  e.g. labels_len = 2
            logits = tf.layers.dense(
                output[:, -1, :],
                self.labels_len,
                activation=None,
                use_bias=True,
                kernel_initializer=tf.contrib.layers.xavier_initializer(seed=self.seed_value),
                bias_initializer=tf.zeros_initializer(),
                trainable=True,
                name="layer_output"
                # reuse=True
            )

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.label_placeholder, logits=logits, name="softmax_crossentropy")

        loss = tf.reduce_mean(cross_entropy)

        return output[:, -1, :], logits, loss

    @define_scope
    def optimize(self):
        # normal to attack ratio (~18.2968 : 1)

        _, _, loss = self.prediction
        tf.summary.scalar('loss', loss)

        # tvars = tf.trainable_variables()
        #
        # # Gradient Clipping
        # grads, _ = tf.clip_by_global_norm(
        #     tf.gradients(loss, tvars),
        #     clip_norm=5,
        #     name="gradient_clipping"
        # )
        #
        # # Adam Optimizer
        # opt = tf.train.AdamOptimizer(
        #     self.learning_r,
        #     name="adam_optimizer"
        # )
        #
        # # Update gradients
        # update_step = opt.apply_gradients(
        #     zip(grads, tvars),
        #     name="apply_gradients",
        #     global_step=self.global_step
        # )

        opt = tf.train.AdamOptimizer(
            self.learning_r,
            name="adam_optimizer"
        )
        grads, tvars = zip(*opt.compute_gradients(loss))

        grads, _ = tf.clip_by_global_norm(
            grads,
            clip_norm=5,
            name="gradient_clipping"
        )

        update_step = opt.apply_gradients(
            zip(grads, tvars),
            name="apply_gradients",
            global_step=self.global_step
        )

        return update_step, tf.summary.merge_all(), self.global_step  # (op, global_step)

    @define_scope
    def error(self):

        output, logits, loss = self.prediction

        pred = tf.argmax(
            tf.nn.softmax(logits),
            axis=1,
            output_type=tf.int32
        )  # shape: [batch_n]

        truth = self.label_placeholder

        return output, truth, pred, loss, tf.reduce_mean(
            tf.cast(tf.equal(pred, truth), tf.float32)  # shape: [batch_n]
        )  # (ground truth, predictions, loss, accuracy)
