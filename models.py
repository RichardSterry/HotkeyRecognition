from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def load_variables_from_checkpoint(sess, start_checkpoint):
    """Utility function to centralize checkpoint restoration.
    Args:
    sess: TensorFlow session.
    start_checkpoint: Path to saved checkpoint on disk.
    """

    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, start_checkpoint)


def create_model(fingerprint_input,
                 model_architecture='conv',
                 input_feature_dimensions=(105, 40),
                 target_classes=12,
                 is_training=True):

    if model_architecture == 'conv':
        return create_conv_model(fingerprint_input=fingerprint_input,
                                 input_feature_dimensions=input_feature_dimensions,
                                 target_classes=target_classes,
                                 is_training=is_training)
    elif model_architecture == 'fc':
        return create_fc_model(fingerprint_input=fingerprint_input,
                               input_feature_dimensions=input_feature_dimensions,
                               target_classes=target_classes,
                               is_training=is_training)
    else:
        raise Exception('model_architecture argument "' + model_architecture + '" not recognized')


def create_fc_model(fingerprint_input,
                    input_feature_dimensions=(98, 40),
                    hidden_layer_nodes=[128],
                    target_classes=12,
                    is_training=True):

    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

    input_time_size = input_feature_dimensions[0]
    input_frequency_size = input_feature_dimensions[1]

    logits = []

    for i in range(len(hidden_layer_nodes)):

        if i == 0:
            # First hidden layer
            weights = tf.Variable(
                tf.truncated_normal([input_time_size * input_frequency_size, hidden_layer_nodes[i]], stddev=0.001))
        else:
            weights = tf.Variable(
                tf.truncated_normal([hidden_layer_nodes[i-1], hidden_layer_nodes[i]], stddev=0.001))

        bias = tf.Variable(tf.zeros([hidden_layer_nodes[i]]))

        if i == 0:
            logits.append(tf.matmul(fingerprint_input, weights) + bias)
        else:
            logits.append(tf.matmul(logits[-1], weights) + bias)

        activation = tf.nn.sigmoid(logits)

        if is_training:
            dropout = tf.nn.dropout(activation, dropout_prob)
        else:
            dropout = activation

    # Last layer
    weights = tf.Variable(
        tf.truncated_normal([hidden_layer_nodes[-1], target_classes], stddev=0.001))
    bias = tf.Variable(tf.zeros([target_classes]))
    logits.append(tf.matmul(logits[-1], weights) + bias)

    if is_training:
        return logits[-1], dropout_prob
    else:
        return logits[-1]


def create_conv_model(fingerprint_input,
                      input_feature_dimensions=(105, 40),
                      target_classes=12,
                      is_training=True):

    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

    input_time_size = input_feature_dimensions[0]
    input_frequency_size = input_feature_dimensions[1]


    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])

    # ========= Convolution Layer 1 =========
    first_filter_width = 8
    first_filter_height = 20
    first_filter_count = 64

    first_weights = tf.Variable(
        tf.truncated_normal(
            [first_filter_height, first_filter_width, 1, first_filter_count],
            stddev=0.01))

    first_bias = tf.Variable(tf.zeros([first_filter_count]))
    first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, 1, 1, 1], 'SAME') + first_bias
    first_relu = tf.nn.relu(first_conv)

    if is_training:
        first_dropout = tf.nn.dropout(first_relu, dropout_prob)
    else:
        first_dropout = first_relu

    max_pool = tf.nn.max_pool(first_dropout, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    # ========= Convolution Layer 2 =========
    second_filter_width = 4
    second_filter_height = 10
    second_filter_count = 64

    second_weights = tf.Variable(
        tf.truncated_normal(
            [
                second_filter_height, second_filter_width, first_filter_count,
                second_filter_count
            ],
            stddev=0.01))

    second_bias = tf.Variable(tf.zeros([second_filter_count]))
    second_conv = tf.nn.conv2d(max_pool, second_weights, [1, 1, 1, 1],
                               'SAME') + second_bias
    second_relu = tf.nn.relu(second_conv)

    if is_training:
        second_dropout = tf.nn.dropout(second_relu, dropout_prob)
    else:
        second_dropout = second_relu

    second_conv_shape = second_dropout.get_shape()
    second_conv_output_width = second_conv_shape[2]
    second_conv_output_height = second_conv_shape[1]
    second_conv_element_count = int(
        second_conv_output_width * second_conv_output_height *
        second_filter_count)

    flattened_second_conv = tf.reshape(second_dropout,
                                       [-1, second_conv_element_count])

    # ========= Fully Connected Layer 1 =========
    final_fc_weights = tf.Variable(
        tf.truncated_normal(
            [second_conv_element_count, target_classes], stddev=0.01))
    final_fc_bias = tf.Variable(tf.zeros([target_classes]))
    final_fc = tf.matmul(flattened_second_conv, final_fc_weights) + final_fc_bias

    if is_training:
        return final_fc, dropout_prob
    else:
        return final_fc