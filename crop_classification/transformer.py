import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base model for this and many
    other models.
    """

    def __init__(self, encoder, src_embed, batch_size, d_model, n_class):
        super(TransformerEncoder, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        # self.linear = nn.Linear(d_model, n_class)
        self.linear = nn.Linear(25, n_class)
        self.m = nn.AdaptiveAvgPool2d(5)

    def forward(self, src, src_mask):
        "Take in and process masked src and target sequences."
        # print('src:',src.shape)
        enc = self.encoder(src, src_mask)
        # flat = enc.reshape(-1).unsqueeze(0)
        # print('enc:',enc.shape)
        changed = self.m(enc)
        # print('changed',changed.shape)
        changed2 = changed.reshape(changed.shape[0],-1)
        # print('changed2',changed2.shape)

        logits = self.linear(changed2)
        # logits = self.linear(enc)
        # result = F.softmax(lin, dim=1)
        # print('logits:',logits.shape)
        return logits

    def encode(self, src, src_mask):
        embedded = self.src_embed(src)

        return self.encoder(embedded, src_mask)
        # return self.encoder(self.src_embed(src), src_mask)