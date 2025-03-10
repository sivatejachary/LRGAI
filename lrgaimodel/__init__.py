from .model import LRGAIModel
from .tokenizers import Tokenizer
from .retrieval import Retriever
from .generation import Generator

# Lazy loading variables
_model = None
_tokenizer = None
_retriever = None
_generator = None

def load_model():
    global _model, _tokenizer, _retriever, _generator
    if _model is None:
        _tokenizer = Tokenizer()
        _retriever = Retriever()
        _model = LRGAIModel()
        _generator = Generator(_model, _retriever)
    return _model, _tokenizer, _retriever, _generator

__all__ = ["LRGAIModel", "Tokenizer", "Retriever", "Generator", "load_model"]
