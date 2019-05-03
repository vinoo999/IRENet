from text_encoder import WordEncoder
class IreNet(object):
    def __init__(self, *args):
        self.word_encoder = WordEncoder()
    
    def _build_arch(self, caption):
        self.caption_embedding = self.word_encoder.transform(caption)


if __name__ == "__main__":
    irenet = IreNet()
