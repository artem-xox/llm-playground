from dataclasses import dataclass

import torch


@dataclass
class NanoConfig:
    # training
    batch_size: int = 64
    max_iters: int = 500
    eval_interval: int = 50
    eval_iters: int = 20
    learning_rate: float = 3e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # model
    block_size: int = 256
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    dropout: float = 0.2

    @property
    def head_size(self) -> int:
        return self.n_embd // self.n_head
