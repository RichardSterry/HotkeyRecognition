import os
import warnings
import numpy as np
from scipy.io import wavfile


class WavAudioReader(object):

    def __init__(self):
        pass

    @staticmethod
    def load_audio_file(filename, normalize=True, verbose=False):
        """Loads an audio file and returns a float PCM-encoded array of samples.

        Args:
        filename: Path to the .wav file to load.

        Returns:
        Numpy array holding the sample data as floats between -1.0 and 1.0.
        """

        sampling_freq, samples = WavAudioReader.read_wav_file(filename)

        # Identifying bits
        if samples.dtype == 'int16':
            nb_bits = 16  # 16-bit wav files
        elif samples.dtype == 'int32':
            nb_bits = 32  # 32-bit wav files

        # enforcing the range to [-1,+1]
        if normalize:
            max_nb_bit = float(2 ** (nb_bits - 1))
            samples = samples / max_nb_bit

        if verbose:
            print('Sampling Freq.: {:,} Hz'.format(sampling_freq))
            print('Audio samples: {:,}'.format(samples.size))
            print('Audio duration: {:.3f} seconds'.format(samples.size / sampling_freq))
            print('Bit depth: {:d} bits'.format(nb_bits))
            print('Sample range: [{:.1f}, {:.1f}]'.format(np.min(samples), np.max(samples)))

        return sampling_freq, np.array(samples, dtype=np.float32)

    @classmethod
    def read_wav_file(cls, filename):

        assert os.path.exists(filename), 'Invalid file_path!!!'

        # Suppressing one warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sampling_freq, samples = wavfile.read(filename)

        return sampling_freq, samples


if __name__ == '__main__':
    WR = WavAudioReader()
    sampling_freq, audio_samples = WR.load_audio_file('../wav/sb_no.wav', verbose=True)