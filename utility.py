import os
import argparse


class Utility(object):
    """
    A class for storing a number of useful functions
    """

    def __init__(self, debug=True):
        self.debug = debug

    @staticmethod
    def parse_arg():
        """
        Parsing input arguments.

        Returns:
            Parsed data
        """

        parser = argparse.ArgumentParser(description='[*] Keyword recognition.')

        # Optional arguments
        parser.add_argument(
            '--data_dir',
            type=str,
            default='../dataset/',
            help="Where to put the audio dataset")

        parser.add_argument(
            '--model_path',
            type=str,
            default='../models',
            help='Path to the folder containing pre-trained model(s) [../models]')

        # ======= TARGET VOCABULARY =======
        parser.add_argument(
            '--wanted_words',
            type=str,
            default='yes,no,up,down,left,right,on,off,stop,go',
            help='Words to use (others will be added to an unknown label)')

        parser.add_argument(
            '--silence_percentage',
            type=float,
            default=10.0,
            help='How much of the training data should be silence.')

        parser.add_argument(
            '--unknown_percentage',
            type=float,
            default=10.0,
            help='How much of the training data should be unknown words.')

        parser.add_argument(
            '--testing_percentage',
            type=int,
            default=10,
            help='What percentage of the audio files to use as a test set.')

        parser.add_argument(
            '--validation_percentage',
            type=int,
            default=10,
            help='What percentage of the audio files to use as a validation set.')

        parser.add_argument(
            '--sampling_rate',
            type=int,
            default=16000,
            help='Expected sample rate of the audio')

        parser.add_argument(
            '--background_volume',
            type=float,
            default=0.1,
            help='How loud the background noise should be, between 0 and 1.')

        parser.add_argument(
            '--background_frequency',
            type=float,
            default=0.8,
            help='How many of the training samples have background noise mixed in.')

        parser.add_argument(
            '--clip_duration_ms',
            type=int,
            default=1000,
            help='Expected duration in milliseconds of the audio')

        parser.add_argument(
            '--window_size_ms',
            type=float,
            default=30.0,
            help='How long each spectrogram timeslice is')

        parser.add_argument(
            '--window_stride_ms',
            type=float,
            default=10.0,
            help='How long is the stride')

        parser.add_argument(
            '--dct_coefficient_count',
            type=int,
            default=40,
            help='How many bins to use for the MFCC fingerprint')

        parser.add_argument(
            '--how_many_training_steps',
            type=str,
            default='15000,3000',
            help='How many training loops to run')

        parser.add_argument(
            '--time_shift_ms',
            type=float,
            default=100.0,
            help='Range to randomly shift the training audio by in time.')

        parser.add_argument(
            '--learning_rate',
            type=str,
            default='0.001,0.0001',
            help='How large a learning rate to use when training.')

        parser.add_argument(
            '--batch_size',
            type=int,
            default=100,
            help='How many items to train with at once')

        parser.add_argument(
            '--summaries_dir',
            type=str,
            default='../logs',
            help='Where to save summary logs for TensorBoard.')

        parser.add_argument(
            '--eval_step_interval',
            type=int,
            default=400,
            help='How often to evaluate the training results.')

        parser.add_argument(
            '--train_dir',
            type=str,
            default='../train',
            help='Directory to write event logs and checkpoint.')

        parser.add_argument(
            '--save_step_interval',
            type=int,
            default=100,
            help='Save model checkpoint every save_steps.')

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

        FLAGS, unparsed = parser.parse_known_args()

        return FLAGS, unparsed

    @staticmethod
    def prepare_directory(path_to_folder):
        """
        Checks for an existing directory. If the directory is not present then the function creates the directory.

        Parameters:
            path_to_folder: Path to the directory

        """

        if not os.path.exists(path_to_folder):
            os.mkdir(path_to_folder)
            print('=> Directory {} created.'.format(path_to_folder))
        else:
            print('[*] Directory {} exists.'.format(path_to_folder))



if __name__ == '__main__':
    FLAGS, unparsed = Utility.parse_arg()

    FLAGS_dict = vars(FLAGS)

    for k, v in FLAGS_dict.items():
        print('\t{} : {}'.format(k, v))
