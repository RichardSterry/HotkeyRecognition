import os
import tensorflow as tf
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio



BACKGROUND_NOISE_DIR_NAME = '_background_noise_'


def test_audio_google():
    #from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

    background_data = []

    with tf.Session(graph=tf.Graph()) as sess:
        wav_filename_placeholder = tf.placeholder(tf.string, [])
        wav_loader = io_ops.read_file(wav_filename_placeholder)
        wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
        search_path = os.path.join('/tmp/speech_dataset/', BACKGROUND_NOISE_DIR_NAME, '*.wav')
        for wav_path in gfile.Glob(search_path):
            print('Reading: ', wav_path)
            wav_data = sess.run(
                wav_decoder,
                feed_dict={wav_filename_placeholder: wav_path}).audio.flatten()
            background_data.append(wav_data)

        print(background_data[0].shape)
        print(background_data[0][1900:2000])



def test_audio():
    from audio import WavAudioReader

    WReader = WavAudioReader()
    sampling_freq, audio_samples = WReader.load_audio_file(os.path.join('/tmp/speech_dataset/',
                                                         BACKGROUND_NOISE_DIR_NAME,
                                                         'exercise_bike.wav'))

    print(audio_samples.shape)
    print(audio_samples[1900:2000])


def get_audio(file_name):
    from audio import WavAudioReader

    WReader = WavAudioReader()
    sampling_freq, audio_samples = WReader.load_audio_file(file_name)

    print(audio_samples)


def prepare_processing_graph(desired_samples=16000, wav_file='sb_down1s.wav'):
    sess = tf.InteractiveSession()

    wav_filename_placeholder_ = tf.placeholder(tf.string, [])
    wav_loader = io_ops.read_file(wav_filename_placeholder_)
    wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1, desired_samples=desired_samples)
    foreground_volume_placeholder_ = tf.placeholder(tf.float32, [])
    scaled_foreground = tf.multiply(wav_decoder.audio, foreground_volume_placeholder_)

    # Shift the sample's start position, and pad any gaps with zeros.
    time_shift_padding_placeholder_ = tf.placeholder(tf.int32, [2, 2])
    time_shift_offset_placeholder_ = tf.placeholder(tf.int32, [2])

    padded_foreground = tf.pad(scaled_foreground, time_shift_padding_placeholder_, mode='CONSTANT')
    sliced_foreground = tf.slice(padded_foreground, time_shift_offset_placeholder_, [desired_samples, -1])

    audio = sess.run(wav_decoder, feed_dict={wav_filename_placeholder_: wav_file})
    print(audio.audio)

    sess.close()

if __name__ == '__main__':
    #test_audio_google()
    #test_audio()

    prepare_processing_graph()

    #get_audio('../wav/sb_down1s.wav')