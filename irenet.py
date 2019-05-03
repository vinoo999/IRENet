import text_encoder as WordEncoder
class IreNet(object):
    def __init__(self, *args):
        pass
    
    def _build_arch(self):
        word_encoder = WordEncoder
        self.word_input = word_encoder.input