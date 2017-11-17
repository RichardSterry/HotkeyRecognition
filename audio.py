from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import glob
import hashlib
import random
import numpy as np
from utility import Utility
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python.ops import io_ops
from tensorflow.python.ops.gen_audio_ops import decode_wav, audio_spectrogram, mfcc

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # 134,217,727
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
SILENCE_LABEL = '_silence_'
UNKNOWN_LABEL = '_unknown_'
RANDOM_SEED = 59185


class AudioProcessor(object):

    def __init__(self, debug=True):
        self.debug = debug
        self.data_index = {'training': [], 'testing': [], 'validation': []}
        self.all_words = dict()
        self.word_index = dict()
        self.background_data = []

        # Placeholders for Feature extraction
        self.wav_filename_placeholder_ = None
        self.foreground_volume_placeholder_ = None
        self.time_shift_padding_placeholder_ = None
        self.time_shift_offset_placeholder_ = None
        self.background_data_placeholder_ = None
        self.background_volume_placeholder_ = None
        self.feature_extractor = None

    @staticmethod
    def get_word_from_filepath(file_path):
        word = re.search('.*/([^/]+)/.*.wav', file_path).group(1).lower()

        return word

    @staticmethod
    def summary_of_indices(data_dict):

        print('Training files: {:,}'.format(len(data_dict['training'])))
        print('Testing files: {:,}'.format(len(data_dict['testing'])))
        print('Validation files: {:,}'.format(len(data_dict['validation'])))

        print('Total files: {:,}'.format(len(data_dict['training']) + len(data_dict['testing']) + len(data_dict['validation'])))

    @staticmethod
    def determine_set(file_name, validation_percentage=10, testing_percentage=10, verbose=False):
        base_name = os.path.basename(file_name)

        hash_name = re.sub(r'_nohash_.*$', '', base_name)

        hash_name_hashed = hashlib.sha1(str(hash_name).encode('utf-8')).hexdigest()

        percentage_hash = ((int(hash_name_hashed, 16) %
                            (MAX_NUM_WAVS_PER_CLASS + 1)) *
                           (100.0 / MAX_NUM_WAVS_PER_CLASS))

        if percentage_hash < validation_percentage:
            result = 'validation'
        elif percentage_hash < validation_percentage + testing_percentage:
            result = 'testing'
        else:
            result = 'training'

        if verbose:
            print('base_name: ', base_name)
            print('hash_name: ', hash_name)
            print('hash_name_hashed: ', hash_name_hashed)
            print('percentage_hash: ', percentage_hash)
            print('result: ', result)

        return result

    @staticmethod
    def get_all_file_paths(dataset_folder, expr='/*/', file_type='wav'):

        audio_files = []
        subdir = glob.glob(dataset_folder + expr)
        for d in subdir:
            files = glob.glob(d + '*.{}'.format(file_type))
            audio_files.extend(files)

        return audio_files

    @staticmethod
    def prepare_word_list(wanted_words):
        return [SILENCE_LABEL, UNKNOWN_LABEL] + wanted_words

    def audio_feature_extraction_graph(self,
                                       no_samples=16000,
                                       window_size_samples=300,
                                       window_stride_samples=150,
                                       dct_coefficient_count=40):
        """
        Resposible for extracting features from audio.

        :return:
        """

        # Step 1: Reading audio samples from wav file
        self.wav_filename_placeholder_ = tf.placeholder(tf.string, [])
        wav_loader = io_ops.read_file(self.wav_filename_placeholder_)
        wav_decoder = decode_wav(wav_loader, desired_channels=1, desired_samples=no_samples)

        # Step 2: Adjusting the volume of the audio
        self.foreground_volume_placeholder_ = tf.placeholder(tf.float32, [])
        scaled_foreground = tf.multiply(wav_decoder.audio, self.foreground_volume_placeholder_)

        # Step 3: Shifting audio and padding with 0
        self.time_shift_padding_placeholder_ = tf.placeholder(tf.int32, [2, 2])
        padded_foreground = tf.pad(scaled_foreground, self.time_shift_padding_placeholder_, mode='CONSTANT')
        self.time_shift_offset_placeholder_ = tf.placeholder(tf.int32, [2])
        sliced_foreground = tf.slice(padded_foreground, self.time_shift_offset_placeholder_, [no_samples, -1])

        # Step 4: Mixing background noise
        self.background_data_placeholder_ = tf.placeholder(tf.float32, [no_samples, 1])
        self.background_volume_placeholder_ = tf.placeholder(tf.float32, [])
        scaled_background = tf.multiply(self.background_data_placeholder_, self.background_volume_placeholder_)
        background_add = tf.add(scaled_background, sliced_foreground)
        background_clamp = tf.clip_by_value(background_add, -1.0, 1.0)

        # Step 5: Extracting spectrogram
        spectrogram = audio_spectrogram(
            background_clamp,
            window_size=window_size_samples,
            stride=window_stride_samples,
            magnitude_squared=True)

        # Step 6: Extracting MFCC
        self.feature_extractor = mfcc(
            spectrogram,
            wav_decoder.sample_rate,
            dct_coefficient_count=dct_coefficient_count)

        return self.feature_extractor

    def prepare_word_index(self, wanted_words):

        # Generating word_index
        self.word_index[SILENCE_LABEL] = 0
        self.word_index[UNKNOWN_LABEL] = 1
        for i, word in enumerate(wanted_words):
            self.word_index[word] = i + 2

    def prepare_audio_indices(self,
                              dataset_folder,
                              wanted_words,
                              validation_percentage=10,
                              testing_percentage=10,
                              silence_percentage=10,
                              unknown_percentage=10):

        random.seed(RANDOM_SEED)

        # Generating word_index
        self.prepare_word_index(wanted_words)

        unknown_word_index = {'training': [], 'testing': [], 'validation': []}

        # Fetching all file names from the dataset folder
        all_audio_files = self.get_all_file_paths(dataset_folder)
        #print('[*] Total number of audio files found: {:d}'.format(len(all_audio_files)))
        print('[*] Generating training, validation and testing indices ... ', end='')

        for wav_path in all_audio_files:

            #print('wav_path: ', wav_path)
            word = self.get_word_from_filepath(wav_path)
            #print('word: ', word)

            if word == BACKGROUND_NOISE_DIR_NAME:
                continue

            # Keeping track of all key-words
            self.all_words[word] = True

            # Determining the set where the file would go
            set_index = self.determine_set(wav_path,
                                           validation_percentage=validation_percentage,
                                           testing_percentage=testing_percentage)

            if word in wanted_words:
                self.data_index[set_index].append({'label': word, 'file': wav_path})
            else:
                unknown_word_index[set_index].append({'label': word, 'file': wav_path})


        #print(self.all_words.keys())

        # Adding Silence: Data will be multiplied by zero later, so the content doesn't matter.
        silence_wav_path = self.data_index['training'][0]['file']
        for set_index in ['validation', 'testing', 'training']:
            set_size = len(self.data_index[set_index])
            silence_size = int(np.ceil(set_size * silence_percentage / 100))

            for _ in range(silence_size):
                self.data_index[set_index].append({
                    'label': SILENCE_LABEL,
                    'file': silence_wav_path
                })

            # Pick some unknowns to add to each partition of the data set.
            random.shuffle(unknown_word_index[set_index])
            unknown_size = int(np.ceil(set_size * unknown_percentage / 100))
            self.data_index[set_index].extend(unknown_word_index[set_index][:unknown_size])


        # Making sure that the ordering is random.
        for set_index in ['validation', 'testing', 'training']:
            random.shuffle(self.data_index[set_index])

        #print('Data index after adding silence and unknown:')
        #self.summary_of_indices(self.data_index)
        print('!Done.')

    def prepare_background_data(self, dataset_folder):
        background_dir = os.path.join(dataset_folder, BACKGROUND_NOISE_DIR_NAME)
        bg_files = self.get_all_file_paths(dataset_folder=background_dir, expr='/', file_type='wav')

        #print(bg_files)

        with tf.Session(graph=tf.Graph()) as sess:
            wav_filename_placeholder = tf.placeholder(tf.string, [])
            wav_loader = io_ops.read_file(wav_filename_placeholder)
            wav_decoder = decode_wav(wav_loader, desired_channels=1)

            for wav_path in bg_files:
                audio_samples = sess.run(wav_decoder,
                                         feed_dict={wav_filename_placeholder: wav_path}).audio.flatten()
                self.background_data.append(audio_samples)

        #print(self.background_data)

    def get_batch_data(self,
                       sess,
                       batch_size=100,
                       feature_dimension=105 * 40,
                       offset=0,
                       time_shift=0,
                       audio_samples=16000,
                       background_frequency=0.8,
                       background_volume=0.1,
                       mode='training',
                       verbose=False):
        """
        Generates (batch) data for training/validation/testing

        Args:
            sess:                   TensorFlow session that was active when audio processor was created
            batch_size:             Target number of audio files
            feature_dimension:      Number of features extracted from audio frame
            offset:                 Where to start within the list while fetching audio data
            time_shift:             How much to randomly shift the clips by in time
            audio_samples:          Number of audio samples in an window
            background_frequency:   Probability of adding background noise to audio
            background_volume:      How loud the background noise will be
            mode:                   Which audio file partition to use. Must be in
                                    ['training', 'testing', 'validation']
            verbose:                Verbosity of the batch processing, True to print intermediate results

        Returns:
            batch_data:             Batch data preprocessed containing audio measurements
            batch_labels:           Labels for the batch data in one-hot form

        """

        # Fetching all candidate audio files
        wav_files = self.data_index[mode]

        if verbose:
            print('Mode = {:s}, wav_files found = {:,}'.format(mode, len(wav_files)))

        if batch_size == -1:
            # Consider all files
            wav_file_count = len(wav_files)
        else:
            wav_file_count = max(0, min(batch_size, len(wav_files) - offset))

        # Data and labels will be populated and returned.
        output_classes = len(self.word_index)

        # Allocating space for the out put
        batch_data = np.zeros((wav_file_count, feature_dimension))
        batch_labels = np.zeros((wav_file_count, output_classes))

        # Deciding if to use background data
        use_background_data = self.background_data and (mode == 'training')

        for i in range(offset, offset + wav_file_count):

            # Selecting a audio file
            if batch_size == -1:
                index = i
            else:
                index = np.random.randint(len(wav_files))

            audio_file_info = wav_files[index]

            if verbose:
                print(audio_file_info)

            # If we're time shifting, set up the offset for this sample.
            if time_shift > 0:
                time_shift_amount = np.random.randint(-time_shift, time_shift)
            else:
                time_shift_amount = 0

            if time_shift_amount > 0:
                time_shift_padding = [[time_shift_amount, 0], [0, 0]]
                time_shift_offset = [0, 0]
            else:
                time_shift_padding = [[0, -time_shift_amount], [0, 0]]
                time_shift_offset = [-time_shift_amount, 0]

            input_dict = {
                self.wav_filename_placeholder_: audio_file_info['file'],
                self.time_shift_padding_placeholder_: time_shift_padding,
                self.time_shift_offset_placeholder_: time_shift_offset,
            }

            #if verbose:
            #    print('input_dict: ', input_dict)

            # Choose a section of background noise to mix in.
            if use_background_data:
                background_index = np.random.randint(len(self.background_data))
                background_samples = self.background_data[background_index]

                background_offset = np.random.randint(
                    0, len(background_samples) - audio_samples)
                background_clipped = background_samples[background_offset:(
                    background_offset + audio_samples)]
                background_reshaped = background_clipped.reshape([audio_samples, 1])

                if np.random.uniform(0, 1) < background_frequency:
                    background_volume = np.random.uniform(0, background_volume)
                else:
                    background_volume = 0
            else:
                background_reshaped = np.zeros([audio_samples, 1])
                background_volume = 0

            input_dict[self.background_data_placeholder_] = background_reshaped
            input_dict[self.background_volume_placeholder_] = background_volume

            # If we want silence, mute out the main sample but leave the background.
            if audio_file_info['label'] == SILENCE_LABEL:
                input_dict[self.foreground_volume_placeholder_] = 0
            else:
                input_dict[self.foreground_volume_placeholder_] = 1

            # Run the feature extraction graph to produce training data
            #if verbose:
            #    print('input_dict: ', input_dict)

            batch_data[i - offset, :] = sess.run(self.feature_extractor, feed_dict=input_dict).flatten()

            # Check for word, if not present then return index for Unknown (i.e., 1)
            label_index = self.word_index.get(audio_file_info['label'], 1)
            batch_labels[i - offset, label_index] = 1

        return batch_data, batch_labels


def test_feature_extractor():
    A = AudioProcessor()
    sess = tf.InteractiveSession()
    samples = 16000
    A.audio_feature_extraction_graph(no_samples=samples)

    input_dict = dict()
    input_dict[A.wav_filename_placeholder_] = '../wav/sb_right.wav'
    input_dict[A.foreground_volume_placeholder_] = 1.0
    input_dict[A.time_shift_padding_placeholder_] = [[0, 0], [0, 0]]
    input_dict[A.time_shift_offset_placeholder_] = [0, 0]
    input_dict[A.background_data_placeholder_] = np.random.random((samples, 1))
    input_dict[A.background_volume_placeholder_] = 0.0

    features = sess.run(A.feature_extractor, feed_dict=input_dict)

    print(features[0].shape)
    plt.matshow(features[0])

    sess.close()
    plt.show()


def test_get_batch_data():
    A = AudioProcessor()
    A.prepare_audio_indices(dataset_folder=FLAGS.data_dir, wanted_words=FLAGS.wanted_words.split(','),
                            validation_percentage=FLAGS.validation_percentage,
                            testing_percentage=FLAGS.testing_percentage,
                            silence_percentage=FLAGS.silence_percentage,
                            unknown_percentage=FLAGS.unknown_percentage)

    A.prepare_background_data(dataset_folder=FLAGS.data_dir)

    sess = tf.InteractiveSession()
    samples = 16000

    A.audio_feature_extraction_graph(no_samples=samples)

    batch_data, batch_labels = A.get_batch_data(sess, batch_size=36, verbose=True)
    print('batch_data.shape: ', batch_data.shape)
    plt.matshow(batch_labels)
    plt.show()

    sess.close()


if __name__ == '__main__':
    FLAGS, _ = Utility.parse_arg()

    test_feature_extractor()

    #test_get_batch_data()




