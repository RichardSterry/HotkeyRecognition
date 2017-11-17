from audio import WavAudioReader


SILENCE_LABEL = '_silence_'
UNKNOWN_WORD_LABEL = '_unknown_'

class KeyWordAudioReader(WavAudioReader):
    
    def __init__(self, debug=True):
        super(KeyWordAudioReader, self).__init__()


    @staticmethod
    def prepare_words_list(wanted_words):
        """Prepends common tokens to the custom word list.
        Args:
          wanted_words: List of strings containing the custom words.
        Returns:
          List with the standard silence and unknown tokens added.
        """
        return [SILENCE_LABEL, UNKNOWN_WORD_LABEL] + wanted_words



if __name__ == '__main__':
    wl = KeyWordAudioReader.prepare_words_list(['a', 'b'])
    print(wl)
