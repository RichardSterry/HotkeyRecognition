from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path

import tensorflow as tf
from tensorflow.python.framework import graph_util
import models
from tensorflow.python.ops.gen_audio_ops import decode_wav, audio_spectrogram, mfcc

FLAGS = None


def prepare_model_parameters():
    """
    Prepares all the parameters for training
    """
    FLAGS_dict = vars(FLAGS)

    FLAGS_dict['audio_samples'] = int(FLAGS.sampling_rate * FLAGS.clip_duration_ms / 1000)
    FLAGS_dict['window_size_samples'] = int(FLAGS.sampling_rate * FLAGS.window_size_ms / 1000)
    FLAGS_dict['window_stride_samples'] = int(FLAGS.sampling_rate * FLAGS.window_stride_ms / 1000)
    length_minus_window = FLAGS_dict['audio_samples'] - FLAGS_dict['window_size_samples']
    FLAGS_dict['context_window'] = 1 + int(length_minus_window / FLAGS_dict['window_stride_samples'])
    FLAGS_dict['target_class_no'] = len(FLAGS_dict['wanted_words'].split(',')) + 2


def create_inference_graph():

    wav_data_placeholder = tf.placeholder(tf.string, [], name='wav_data')
    decoded_sample_data = decode_wav(
        wav_data_placeholder,
        desired_channels=1,
        desired_samples=FLAGS.audio_samples,
        name='decoded_sample_data')

    spectrogram = audio_spectrogram(
        decoded_sample_data.audio,
        window_size=FLAGS.window_size_samples,
        stride=FLAGS.window_stride_samples,
        magnitude_squared=True)

    fingerprint_input = mfcc(
        spectrogram,
        decoded_sample_data.sample_rate,
        dct_coefficient_count=FLAGS.dct_coefficient_count)

    fingerprint_frequency_size = FLAGS.dct_coefficient_count
    fingerprint_time_size = FLAGS.context_window
    reshaped_input = tf.reshape(fingerprint_input, [
        -1, fingerprint_time_size * fingerprint_frequency_size
    ])

    logits = models.create_model(
        reshaped_input,
        model_architecture=FLAGS.model_architecture,
        input_feature_dimensions=(FLAGS.context_window, FLAGS.dct_coefficient_count),
        target_classes=FLAGS.target_class_no,
        is_training=False)

    # Create an output to use for inference.
    tf.nn.softmax(logits, name='labels_softmax')


def main():
    # Create the model and load its weights.
    sess = tf.InteractiveSession()

    create_inference_graph()

    models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)

    # Turn all the variables into inline constants inside the graph and save it.
    frozen_graph_def = graph_util.convert_variables_to_constants(
        sess, sess.graph_def, ['labels_softmax'])

    tf.train.write_graph(
        frozen_graph_def,
        os.path.dirname(FLAGS.output_file),
        os.path.basename(FLAGS.output_file),
        as_text=False)

    tf.logging.info('Saved frozen graph to %s', FLAGS.output_file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sampling_rate',
        type=int,
        default=16000,
        help='Expected sample rate of the wavs', )
    parser.add_argument(
        '--clip_duration_ms',
        type=int,
        default=1000,
        help='Expected duration in milliseconds of the wavs', )
    parser.add_argument(
        '--clip_stride_ms',
        type=int,
        default=30,
        help='How often to run recognition. Useful for models with cache.', )
    parser.add_argument(
        '--window_size_ms',
        type=float,
        default=30.0,
        help='How long each spectrogram timeslice is', )
    parser.add_argument(
        '--window_stride_ms',
        type=float,
        default=10.0,
        help='How long each spectrogram timeslice is', )
    parser.add_argument(
        '--dct_coefficient_count',
        type=int,
        default=40,
        help='How many bins to use for the MFCC fingerprint', )
    parser.add_argument(
        '--start_checkpoint',
        type=str,
        default='',
        help='If specified, restore this pre-trained model before any training.')
    parser.add_argument(
        '--model_architecture',
        type=str,
        default='conv',
        help='What model architecture to use')
    parser.add_argument(
        '--wanted_words',
        type=str,
        default='yes,no,up,down,left,right,on,off,stop,go',
        help='Words to use (others will be added to an unknown label)', )
    parser.add_argument(
        '--output_file', type=str, default='mytest', help='Where to save the frozen graph.')

    FLAGS, _ = parser.parse_known_args()
    prepare_model_parameters()
    main()

