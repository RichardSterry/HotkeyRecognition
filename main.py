#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from utility import Utility
import tensorflow as tf
import models
from audio import AudioProcessor
from tensorflow.python.platform import gfile

# Variable that will hold all the parameters of the experiment
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
    FLAGS_dict['time_shift_samples'] = int((FLAGS.time_shift_ms * FLAGS.sampling_rate) / 1000)
    show_parameters()


def show_parameters():
    FLAGS_dict = vars(FLAGS)

    print('\n[*] Starting Key-word identification training ...\n')
    print('[*] Parameters of the experiment =>')
    for k, v in FLAGS_dict.items():
        print('\t{} : {}'.format(k, v))


def main():

    # We want to see all the logging messages for this tutorial.
    tf.logging.set_verbosity(tf.logging.INFO)

    # Starting a new TensorFlow session
    sess = tf.InteractiveSession()

    # Step 1: INPUT placeholder for feeding in audio features
    fingerprint_input = tf.placeholder(
        tf.float32, [None, FLAGS.context_window * FLAGS.dct_coefficient_count], name='fingerprint_input')

    # Step 2: MODEL Construction
    logits, dropout_prob = models.create_model(
        fingerprint_input,
        model_architecture=FLAGS.model_architecture,
        input_feature_dimensions=(FLAGS.context_window, FLAGS.dct_coefficient_count),
        target_classes=FLAGS.target_class_no,
        is_training=True)

    # Step 3: GROUND-TRUTH placeholder for measuring error
    ground_truth_input = tf.placeholder(
        tf.float32, [None, FLAGS.target_class_no], name='groundtruth_input')

    # Step 4: LOSS function
    with tf.name_scope('cross_entropy'):
        cross_entropy_mean = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                            labels=ground_truth_input, logits=logits))
    tf.summary.scalar('cross_entropy', cross_entropy_mean)

    # Step 5: MINIMIZING LOSS   # Todo: Use a different optimizer
    with tf.name_scope('train'):
        learning_rate_input = tf.placeholder(
            tf.float32, [], name='learning_rate_input')
        train_step = tf.train.GradientDescentOptimizer(
            learning_rate_input).minimize(cross_entropy_mean)

    # Step 6: ASSESSING Performance
    predicted_indices = tf.argmax(logits, 1)
    expected_indices = tf.argmax(ground_truth_input, 1)
    correct_prediction = tf.equal(predicted_indices, expected_indices)
    confusion_matrix = tf.confusion_matrix(expected_indices, predicted_indices)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # ============= INITIALIZATION OF VARIABLES / LOGGER =============

    training_steps_list = list(map(int, FLAGS.how_many_training_steps.split(',')))
    learning_rates_list = list(map(float, FLAGS.learning_rate.split(',')))

    if len(training_steps_list) != len(learning_rates_list):
        raise Exception(
            '--how_many_training_steps and --learning_rate must be of equal length '
            'lists, but are %d and %d long instead' % (len(training_steps_list),
                                                       len(learning_rates_list)))

    global_step = tf.train.get_or_create_global_step()
    increment_global_step = tf.assign(global_step, global_step + 1)

    saver = tf.train.Saver(tf.global_variables())

    # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
    merged_summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                         sess.graph)
    validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')

    tf.global_variables_initializer().run()

    start_step = 1

    if FLAGS.start_checkpoint:
        models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)
        start_step = global_step.eval(session=sess)

    tf.logging.info('Training from step: %d ', start_step)

    # Save graph in .pbtxt.
    tf.train.write_graph(sess.graph_def, FLAGS.train_dir,
                         FLAGS.model_architecture + '.pbtxt')

    audio_processor = AudioProcessor()
    audio_processor.prepare_audio_indices(dataset_folder=FLAGS.data_dir,
                                          wanted_words=FLAGS.wanted_words.split(','),
                                          validation_percentage=FLAGS.validation_percentage,
                                          testing_percentage=FLAGS.testing_percentage,
                                          silence_percentage=FLAGS.silence_percentage,
                                          unknown_percentage=FLAGS.unknown_percentage)

    audio_processor.prepare_background_data(dataset_folder=FLAGS.data_dir)
    audio_processor.audio_feature_extraction_graph(no_samples=FLAGS.audio_samples,
                                                   window_size_samples=FLAGS.window_size_samples,
                                                   window_stride_samples=FLAGS.window_stride_samples,
                                                   dct_coefficient_count=FLAGS.dct_coefficient_count)

    # Save list of words.
    with gfile.GFile(os.path.join(FLAGS.train_dir, FLAGS.model_architecture + '_labels.txt'), 'w') as f:
        f.write('\n'.join(audio_processor.prepare_word_list(FLAGS.wanted_words.split(','))))

    # ============= TRAINING LOOP =============
    #training_steps_max = 100
    training_steps_max = np.sum(training_steps_list)

    for training_step in range(start_step, training_steps_max + 1):

        # Identifying learning rate
        training_steps_sum = 0
        for i in range(len(training_steps_list)):
            training_steps_sum += training_steps_list[i]
            if training_step <= training_steps_sum:
                learning_rate_value = learning_rates_list[i]
                break

        #print('learning_rate_value: ', learning_rate_value)

        # Pull a batch of audio samples that we'll use for training
        batch_data, batch_ground_truth = \
            audio_processor.get_batch_data(sess=sess,
                                           batch_size=FLAGS.batch_size,
                                           feature_dimension=FLAGS.context_window * FLAGS.dct_coefficient_count,
                                           offset=0,                    # Todo: add parameter
                                           time_shift=FLAGS.time_shift_samples,
                                           audio_samples=FLAGS.audio_samples,
                                           background_frequency=FLAGS.background_frequency,
                                           background_volume=FLAGS.background_volume,
                                           mode='training')

        # Run the graph with this batch of training data.
        train_summary, train_accuracy, cross_entropy_value, _, _ = sess.run(
            [
                merged_summaries, accuracy, cross_entropy_mean, train_step, increment_global_step
            ],
            feed_dict={
                fingerprint_input: batch_data,
                ground_truth_input: batch_ground_truth,
                learning_rate_input: learning_rate_value,
                dropout_prob: 0.5                           # Todo: add parameter
            })

        train_writer.add_summary(train_summary, training_step)
        tf.logging.info('Step #%d: rate %f, accuracy %.1f%%, cross entropy %f' %
                        (training_step, learning_rate_value, train_accuracy * 100,
                         cross_entropy_value))

        # SAVING MODEL CHECKPOINT
        if training_step % FLAGS.save_step_interval == 0 or training_step == training_steps_max:
            checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.model_architecture + '.ckpt')
            tf.logging.info('Saving to "%s-%d"', checkpoint_path, training_step)
            saver.save(sess, checkpoint_path, global_step=training_step)

        # CHECKING VALIDATION PERFORMANCE
        if training_step % FLAGS.eval_step_interval == 0 or training_step == training_steps_max:
            tf.logging.info('Estimating performance on Validation set ...')

            val_set_size = len(audio_processor.data_index['validation'])

            validation_fingerprints, validation_ground_truth = \
                audio_processor.get_batch_data(sess=sess,
                                               batch_size=-1,
                                               feature_dimension=FLAGS.context_window * FLAGS.dct_coefficient_count,
                                               offset=0,
                                               time_shift=0,
                                               audio_samples=FLAGS.audio_samples,
                                               background_frequency=FLAGS.background_frequency,
                                               background_volume=FLAGS.background_volume,
                                               mode='validation')

            # Run a validation step and capture training summaries for TensorBoard
            # with the `merged` op.
            validation_summary, validation_accuracy, conf_matrix = sess.run(
                [merged_summaries, accuracy, confusion_matrix],
                feed_dict={
                    fingerprint_input: validation_fingerprints,
                    ground_truth_input: validation_ground_truth,
                    dropout_prob: 1.0
                })

            validation_writer.add_summary(validation_summary, training_step)

            tf.logging.info('Confusion Matrix:\n %s' % (conf_matrix))
            tf.logging.info('Step %d: Validation accuracy = %.1f%% (N=%d)' %
                            (training_step, validation_accuracy * 100, val_set_size))

    # CHECKING FINAL TEST PERFORMANCE
    tf.logging.info('Estimating performance on Testing set ...')
    test_set_size = len(audio_processor.data_index['testing'])

    test_fingerprints, test_ground_truth = \
        audio_processor.get_batch_data(sess=sess,
                                       batch_size=-1,
                                       feature_dimension=FLAGS.context_window * FLAGS.dct_coefficient_count,
                                       offset=0,  # Todo: add parameter
                                       time_shift=0,  # Todo: add parameter
                                       audio_samples=FLAGS.audio_samples,  # Todo: add parameter
                                       background_frequency=FLAGS.background_frequency,
                                       background_volume=FLAGS.background_volume,
                                       mode='testing')

    # Run a test and capture training summaries for TensorBoard
    # with the `merged` op.
    test_summary, test_accuracy, conf_matrix = sess.run(
        [merged_summaries, accuracy, confusion_matrix],
        feed_dict={
            fingerprint_input: test_fingerprints,
            ground_truth_input: test_ground_truth,
            dropout_prob: 1.0
        })

    validation_writer.add_summary(test_summary)

    tf.logging.info('Confusion Matrix:\n %s' % (conf_matrix))
    tf.logging.info('Final Test accuracy = %.1f%% (N=%d)' %
                    (test_accuracy * 100, test_set_size))

    # Stopping TensorFlow session
    sess.close()


if __name__ == '__main__':
    # Parsing experimental set up
    FLAGS, _ = Utility.parse_arg()
    prepare_model_parameters()

    # Calling the main function
    main()
