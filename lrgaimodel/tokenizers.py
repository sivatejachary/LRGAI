from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

class Tokenizer:
    def __init__(self):
        self.tokenizer = HFTokenizer(BPE())
        self.tokenizer.pre_tokenizer = Whitespace()
    def train(self, files):
        trainer = BpeTrainer(special_tokens=["<unk>", "<s>", "</s>"])
        self.tokenizer.train(files, trainer)
    def encode(self, text):
        return self.tokenizer.encode(text).ids
    def decode(self, ids):
        return self.tokenizer.decode(ids)
