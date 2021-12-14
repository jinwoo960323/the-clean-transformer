import torch.nn
from pytorch_lightning import LightningModule


class Transformer(LightningModule):
    def __init__(self, hidden_size: int, ffn_size: int,
                 vocab_size: int, max_length: int,
                 pad_token_id: int, heads: int, depth: int,
                 dropout: float, lr: float):  # noqa
        super().__init__()
        self.save_hyperparameters()
        # TODO: implement transformer
        self.token_embeddings = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        self.encoder = Encoder()
        self.decoder = Decoder()

    def training_step_end(self, *args, **kwargs) -> STEP_OUTPUT:
        pass

class Encoder(torch.nn.Module):
    raise NotImplementedError

class Decoder(torch.nn.Module):
    raise NotImplementedError