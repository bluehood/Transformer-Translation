import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model).to(self.device) # Query transformation
        self.W_k = nn.Linear(d_model, d_model).to(self.device) # Key transformation
        self.W_v = nn.Linear(d_model, d_model).to(self.device) # Value transformation
        self.W_o = nn.Linear(d_model, d_model).to(self.device) # Output transformation
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output.to(self.device)
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2).to(self.device)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model).to(self.device)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q).to(self.device))
        K = self.split_heads(self.W_k(K).to(self.device))
        V = self.split_heads(self.W_v(V).to(self.device))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        output = self.W_o(self.combine_heads(attn_output).to(self.device))
        return output.to(self.device)

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fc1 = nn.Linear(d_model, d_ff).to(self.device)
        self.fc2 = nn.Linear(d_ff, d_model).to(self.device)
        self.relu = nn.ReLU().to(self.device)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x.to(self.device)))).to(self.device)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        pe = torch.zeros(max_seq_length, d_model).to(self.device)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1).to(self.device)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).to(self.device)
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        x = x.to(self.device)
        return x + self.pe[:, :x.size(1)].to(self.device)
    

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.self_attn = MultiHeadAttention(d_model, num_heads).to(self.device)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff).to(self.device)
        self.norm1 = nn.LayerNorm(d_model).to(self.device)
        self.norm2 = nn.LayerNorm(d_model).to(self.device)
        self.dropout = nn.Dropout(dropout).to(self.device)
        
    def forward(self, x, mask):
        x = x.to(self.device)
        mask = mask.to(self.device)
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.self_attn = MultiHeadAttention(d_model, num_heads).to(self.device)
        self.cross_attn = MultiHeadAttention(d_model, num_heads).to(self.device)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff).to(self.device)
        self.norm1 = nn.LayerNorm(d_model).to(self.device)
        self.norm2 = nn.LayerNorm(d_model).to(self.device)
        self.norm3 = nn.LayerNorm(d_model).to(self.device)
        self.dropout = nn.Dropout(dropout).to(self.device)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        x = x.to(self.device)
        enc_output = enc_output.to(self.device)
        src_mask = src_mask.to(self.device)
        tgt_mask = tgt_mask.to(self.device)

        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model).to(self.device)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model).to(self.device)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length).to(self.device)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout).to(self.device) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout).to(self.device) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size).to(self.device)
        self.dropout = nn.Dropout(dropout).to(self.device)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask.to(self.device)
        return src_mask.to(self.device), tgt_mask.to(self.device)

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output
