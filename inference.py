from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import argparse
import tensorflow as tf
from tensorflow.python.ops.gen_audio_ops import *


def load_audio_bytes(filename):

    if not filename or not tf.gfile.Exists(filename):
        tf.logging.fatal('Audio file does not exist %s', filename)

    with open(filename, 'rb') as wav_file:
        wav_data = wav_file.read()

    return wav_data


def load_frozen_graph(filename):
    """
    Loads a frozen graph
    """

    if not filename or not tf.gfile.Exists(filename):
        tf.logging.fatal('Graph file does not exist %s', filename)

    with tf.gfile.FastGFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

        start_time = time.time()
        tf.import_graph_def(graph_def, name='')
        print('Model load time: %.5f sec' % (time.time() - start_time))


def load_word_labels(filename):
    """
    Read in labels, one label per line.
    """
    return [line.rstrip() for line in tf.gfile.GFile(filename)]


def execute_graph(wav_data, labels, input_layer_name='wav_data:0', output_layer_name='labels_softmax:0', num_top_predictions=3, repeat=100):
    """
    Runs the audio data through the graph and prints predictions.
    """

    with tf.Session() as sess:
        # Feed the audio data as input to the graph.
        #   predictions  will contain a two-dimensional array, where one
        #   dimension represents the input image count, and the other has
        #   predictions per class
        softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)

        start_time = time.time()

        for i in range(repeat):
            predictions, = sess.run(softmax_tensor, {input_layer_name: wav_data})

        tot_time_taken = time.time() - start_time

        # Sort to show labels in order of confidence
        top_k = predictions.argsort()[-num_top_predictions:][::-1]

        print('\n[*] Predictions =>')
        for i, node_id in enumerate(top_k):
            human_string = labels[node_id]
            score = predictions[node_id]
            print('\t[Top-%d] %s \t(score = %.5f)' % (i+1, human_string, score))

        print('\n[*] Total inference time = %.5f sec' % tot_time_taken)
        if repeat > 1:
            print('[*] Average inference time = %.5f sec (repeated = %d) ' % (tot_time_taken / repeat, repeat))

    return 0


def parse_arg():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--wav', type=str, default='', help='Input audio file.')

    parser.add_argument(
        '--model', type=str, default='../train/conv_g.pb', help='Pre-trained frozen model.')

    parser.add_argument(
        '--labels', type=str, default='../train/conv_labels_g.txt', help='Path to the file containing word labels.')

    parser.add_argument(
        '--input_name', type=str, default='wav_data:0', help='Name of input node in model.')

    parser.add_argument(
        '--output_name', type=str, default='labels_softmax:0', help='Name of output node in the model.')

    parser.add_argument(
        '--top_k', type=int, default=3, help='Number of results to show.')

    parser.add_argument(
        '--repeat', type=int, default=100, help='Number of times to repeat inference.')

    FLAGS, unparsed = parser.parse_known_args()

    return FLAGS


def main():
    # Getting input parameters
    FLAGS = parse_arg()

    load_frozen_graph(FLAGS.model)

    word_labels = load_word_labels(FLAGS.labels)

    wav_data = load_audio_bytes(FLAGS.wav)

    execute_graph(wav_data, word_labels, num_top_predictions=FLAGS.top_k, repeat=FLAGS.repeat)


if __name__ == '__main__':
    main()
