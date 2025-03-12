import json
import os
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer

Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."


class LAM:
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self.model_initialized = False

    @property
    def model(self):
        if not self.model_initialized:
            raise RuntimeError("Model is not initialized yet.")
        return self._model

    @property
    def tokenizer(self):
        if not self.model_initialized:
            raise RuntimeError("Tokenizer is not initialized yet.")
        return self._tokenizer

    def build(self, ckpt_dir: str, tokenizer_path: str, max_seq_len: int, max_batch_size: int, model_parallel_size: Optional[int] = None, seed: int = 1):
        """Build the LAM instance and initialize the model and tokenizer."""
        if not self.model_initialized:
            start_time = time.time()

            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group("nccl")
            if not model_parallel_is_initialized():
                if model_parallel_size is None:
                    model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
                initialize_model_parallel(model_parallel_size)

            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)

            # Load model checkpoint and tokenizer lazily
            checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
            assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
            ckpt_path = checkpoints[get_model_parallel_rank()]
            checkpoint = torch.load(ckpt_path, map_location="cpu")

            with open(Path(ckpt_dir) / "params.json", "r") as f:
                params = json.loads(f.read())

            model_args: ModelArgs = ModelArgs(max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params)
            self._tokenizer = Tokenizer(model_path=tokenizer_path)
            model_args.vocab_size = self.tokenizer.n_words

            self._model = Transformer(model_args)
            self._model.load_state_dict(checkpoint, strict=False)

            self.model_initialized = True
            print(f"Model loaded in {time.time() - start_time:.2f} seconds")

        return self

    def generate(self, prompt_tokens: List[List[int]], max_gen_len: int, temperature: float = 0.6, top_p: float = 0.9, logprobs: bool = False, echo: bool = False) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """Generate text sequences based on provided prompts using the lazy-loaded model."""
        return self.model.generate(prompt_tokens, max_gen_len, temperature, top_p, logprobs, echo)

    def text_completion(self, prompts: List[str], temperature: float = 0.6, top_p: float = 0.9, max_gen_len: Optional[int] = None, logprobs: bool = False, echo: bool = False) -> List[CompletionPrediction]:
        """Generate text completions for a list of prompts."""
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        generation_tokens, generation_logprobs = self.generate(prompt_tokens=prompt_tokens, max_gen_len=max_gen_len, temperature=temperature, top_p=top_p, logprobs=logprobs, echo=echo)

        if logprobs:
            return [{"generation": self.tokenizer.decode(t), "logprobs": logprobs_i} for t, logprobs_i in zip(generation_tokens, generation_logprobs)]
        return [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]
