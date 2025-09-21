import torch

from src.models.nano.config import NanoConfig
from src.models.nano.model import GPTLanguageModel


def main():
    ckpt = torch.load("models/nano/model.pt", map_location="cpu")
    config = NanoConfig(**ckpt["config"])
    model = GPTLanguageModel(ckpt["vocab_size"], config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    itos = ckpt["vocab"]["itos"]

    def decode(indices):
        return "".join([itos[i] for i in indices])

    context = torch.zeros((1, 1), dtype=torch.long)
    with torch.no_grad():
        out = model.generate(context, max_new_tokens=500)[0].tolist()

    print(decode(out))


if __name__ == "__main__":
    main()
