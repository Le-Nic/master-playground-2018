import tensorflow as tf

try:
    from tensorflow.contrib.rnn import LSTMStateTuple
except ImportError:
    LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple


def bidirectional_rnn(cell_fw, cell_bw, inputs_embedded, input_lengths, scope=None):
    """Bidirecional RNN with concatenated outputs and states"""
    with tf.variable_scope(scope or "birnn") as scope:
        ((fw_outputs, bw_outputs), (fw_state, bw_state)) = (
            tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=inputs_embedded,
                sequence_length=input_lengths,
                dtype=tf.float32,
                # swap_memory=True,
                time_major=False,
                scope=scope)
        )
        outputs = tf.concat((fw_outputs, bw_outputs), 2)

        def concatenate_state(_fw_state, _bw_state):
            if isinstance(_fw_state, LSTMStateTuple):
                state_c = tf.concat((_fw_state.c, _bw_state.c), 1, name='bidirectional_concat_c')
                state_h = tf.concat((_fw_state.h, _bw_state.h), 1, name='bidirectional_concat_h')
                _state = LSTMStateTuple(c=state_c, h=state_h)
                return _state

            elif isinstance(_fw_state, tf.Tensor):
                _state = tf.concat((_fw_state, _bw_state), 1, name='bidirectional_concat')
                return _state

            # multilayer
            elif isinstance(_fw_state, tuple) and isinstance(_bw_state, tuple) and len(_fw_state) == len(_bw_state):
                _state = tuple(concatenate_state(fw, bw) for fw, bw in zip(_fw_state, _bw_state))
                return _state

            else:
                raise ValueError(
                    'unknown state type: {}'.format((_fw_state, _bw_state)))

        state = concatenate_state(fw_state, bw_state)
        return outputs, state


def rnn(cell, inputs_embedded, input_lengths, scope=None):
    with tf.variable_scope(scope or "rnn") as scope:
        output, state = tf.nn.dynamic_rnn(
            cell=cell,
            inputs=inputs_embedded,
            sequence_length=input_lengths,
            # initial_state=rnn_tuple_state,  # zero state
            dtype=tf.float32,
            time_major=False,
            scope=scope
        )

        return output, state


def task_specific_attention(
        inputs, output_size, initializer, activation_fn=tf.tanh, scope=None):
    """
    Performs task-specific attention reduction, using learned
    attention context vector (constant within task of interest).
    Args:
        inputs: Tensor of shape [batch_size, units, input_size]
            `input_size` must be static (known)
            `units` axis will be attended over (reduced from output)
            `batch_size` will be preserved
        output_size: Size of output's inner (feature) dimension
    Returns:
        outputs: Tensor of shape [batch_size, output_dim].
    """
    assert len(inputs.get_shape()) == 3 and inputs.get_shape()[-1].value is not None

    with tf.variable_scope(scope or 'attention') as scope:
        attention_context_vector = tf.get_variable(
            name='attention_context_vector',
            shape=[output_size],
            initializer=initializer,
            dtype=tf.float32
        )
        input_projection = tf.contrib.layers.fully_connected(
            inputs,
            output_size,
            activation_fn=activation_fn,
            scope=scope
        )

        vector_attn = tf.reduce_sum(tf.multiply(input_projection, attention_context_vector), axis=2, keepdims=True)
        attention_weights = tf.nn.softmax(vector_attn, axis=1)
        weighted_projection = tf.multiply(input_projection, attention_weights)

        outputs = tf.reduce_sum(weighted_projection, axis=1)

        return outputs
