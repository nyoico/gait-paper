import math
import torch
import torch.nn as nn


class ConvEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, embed_dim, kernel_size=1)
        self.bn = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        return y


class PosEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=512, dropout=0.1):
        super().__init__()

        pe = torch.zeros(max_len, embed_dim) 
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1) 
        div = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) * -(math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe.unsqueeze(0))  
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=False)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Conv1d(embed_dim, ff_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(ff_dim, embed_dim, kernel_size=1)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        attn_out, _ = self.self_attn(
            src, src, src,
            need_weights=True,
            average_attn_weights=False
        )
        src = self.norm1(src + self.dropout(attn_out))

        y = self.ff(src.permute(1, 2, 0)).permute(2, 0, 1)
        src = self.norm2(src + self.dropout(y))

        return src


class Decoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.masked_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=False)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=False)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Conv1d(embed_dim, ff_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(ff_dim, embed_dim, kernel_size=1)
        )
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_pad_mask=None, memory_key_padding_mask=None):

        # masked self-attention
        m1, _ = self.masked_attn( 
            tgt, tgt, tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_pad_mask,
            need_weights=False
        )
        tgt = self.norm1(tgt + self.dropout(m1))

        # cross attention
        m2, _ = self.cross_attn(
            tgt, memory, memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False
        )
        tgt = self.norm2(tgt + self.dropout(m2))

        # feed-forward network
        y = self.ff(tgt.permute(1, 2, 0)).permute(2, 0, 1)
        tgt = self.norm3(tgt + self.dropout(y))

        return tgt


def make_casual_mask(seq_len, device):
    return torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
        diagonal=1
    )


def make_causal_mask(seq_len, device):
    return make_casual_mask(seq_len, device)


class Model(nn.Module):
    def __init__(self, num_classes, embed_dim, sensor_dim, num_heads, num_layers, ff_dim, dropout, max_len):
        super().__init__()

        # Conv Embedding
        self.sensorL_embed = ConvEmbedding(sensor_dim, embed_dim)
        self.sensorR_embed = ConvEmbedding(sensor_dim, embed_dim)

        # pos encoding
        self.pos_enc = PosEncoding(embed_dim, max_len=max_len, dropout=dropout)

        # Encoder
        self.encoder = nn.ModuleList(
            [Encoder(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )

        # Decoder Input Embedding
        self.tgt_embed = nn.Embedding(num_classes, embed_dim)

        # Decoder
        self.decoder = nn.ModuleList(
            [Decoder(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )

        # Linear
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, sensorL, sensorR, tgt_seq, tgt_mask=None, return_attn=False, return_prob=False):
        x_left = self.sensorL_embed(sensorL).permute(0, 2, 1)   
        x_right = self.sensorR_embed(sensorR).permute(0, 2, 1) 
        x = x_left + x_right
        x = self.pos_enc(x)
        x = x.permute(1, 0, 2) 

        for layer in self.encoder:
            x = layer(x)
        memory = x

        y = self.tgt_embed(tgt_seq)  
        y = self.pos_enc(y).permute(1, 0, 2)  

        if tgt_mask is None:
            tgt_mask = make_casual_mask(y.size(0), y.device)

        for layer in self.decoder:
            y = layer(y, memory, tgt_mask)

        logits = self.classifier(self.dropout(y))  

        if return_attn:
            pass

        if return_prob:
            return torch.softmax(logits, dim=-1)

        return logits
