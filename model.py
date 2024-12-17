import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.num_heads == 0
        
        self.num_heads = config.num_heads
        self.head_size = config.n_embd // config.num_heads
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, q, k=None, v=None, mask=None, is_causal=False):
        batch_size = q.size(0)
        
        if k is None:
            k = q
        if v is None:
            v = q
            
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v) 
        
        q = q.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)
        
        if is_causal:
            seq_len = q.size(-2)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device), diagonal=1)
            scores.masked_fill_(causal_mask, float('-inf'))
        
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores.masked_fill_(~mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = attn @ v 
        
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.n_embd)
        out = self.out_proj(out)
        
        return out

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Regular transformer block with self-attention"""
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffwd = FeedForward(config)

    def forward(self, x, mask=None, is_causal=False):
        # Self attention
        attn_output = self.attn(
            self.ln1(x),
            mask=mask,
            is_causal=is_causal
        )
        x = x + attn_output
        
        # Feed forward
        x = x + self.ffwd(self.ln2(x))
        return x

class CrossAttentionBlock(nn.Module):
    """Transformer block with cross-attention to encoder output"""
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.self_attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.cross_attn = MultiHeadAttention(config)
        self.ln3 = nn.LayerNorm(config.n_embd)
        self.ffwd = FeedForward(config)

    def forward(self, x, enc_out, self_mask=None, cross_mask=None):
        # Self attention with causal masking
        x = x + self.self_attn(
            self.ln1(x),
            mask=self_mask,
            is_causal=True
        )
        
        # Cross attention to encoder output
        x = x + self.cross_attn(
            q=self.ln2(x),
            k=enc_out,
            v=enc_out,
            mask=cross_mask
        )
        
        # Feed forward
        x = x + self.ffwd(self.ln3(x))
        return x

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        
    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask)
        return self.ln_f(x)

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Pre-cross attention transformer blocks
        self.pre_blocks = nn.ModuleList([
            Block(config) for _ in range(config.n_pre_cross_layer)
        ])
        # Cross attention blocks
        self.cross_blocks = nn.ModuleList([
            CrossAttentionBlock(config) for _ in range(config.n_cross_layer)
        ])
        self.ln_f = nn.LayerNorm(config.n_embd)
        
    def forward(self, x, enc_out, padding_mask=None, cross_mask=None):
        # First run through pre-cross attention blocks with causal masking
        for block in self.pre_blocks:
            x = block(x, padding_mask, is_causal=True)
            
        # Then through cross attention blocks
        # Cross attention blocks still use causal masking for self-attention
        for block in self.cross_blocks:
            x = block(x, enc_out, padding_mask, cross_mask)
            
        return self.ln_f(x)

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Source and target embeddings
        self.src_tok_emb = nn.Embedding(config.src_vocab_size, config.n_embd)
        self.tgt_tok_emb = nn.Embedding(config.tgt_vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.dropout)
        
        # Encoder and Decoder
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        
        # Output projection
        self.head = nn.Linear(config.n_embd, config.tgt_vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, src_ids, tgt_ids, src_mask=None, tgt_mask=None):
        B, T = src_ids.size()
    
        # Source embedding
        src_emb = self.src_tok_emb(src_ids)
        src_pos = self.pos_emb[:, :T, :]
        x = self.drop(src_emb + src_pos)
        
        # Encode
        encoder_out = self.encoder(x, src_mask)
        
        # Target embedding
        tgt_emb = self.tgt_tok_emb(tgt_ids)
        tgt_pos = self.pos_emb[:, :tgt_ids.size(1), :]
        y = self.drop(tgt_emb + tgt_pos)
        
        # Decode
        y = self.decoder(y, encoder_out, tgt_mask, src_mask)
        
        # Project to vocabulary
        logits = self.head(y)
        
        return logits

    def generate(self, src_ids, max_new_tokens, temperature=1.0, top_k=None):
        """Generate translation tokens autoregressively"""
        self.eval()
        B, T = src_ids.size()
        
        # Create source padding mask (1 for tokens, 0 for padding)
        src_mask = (src_ids != 0).unsqueeze(1).unsqueeze(2).to(dtype=torch.bool)
        
        # First encode the source sequence
        src_emb = self.src_tok_emb(src_ids)
        pos_emb = self.pos_emb[:, :T, :]
        x = self.drop(src_emb + pos_emb)
        encoder_out = self.encoder(x, src_mask)
        
        # Initialize target sequence with START token
        tgt_ids = torch.full((B, 1), fill_value=2, dtype=torch.long, device=src_ids.device)  # Assume 2 is START token
        
        for _ in range(max_new_tokens):
            # Cut off if sequence is too long
            if tgt_ids.size(1) > self.config.block_size:
                break
                
            # Create target padding mask (1 for tokens, 0 for padding)
            tgt_mask = (tgt_ids != 0).unsqueeze(1).unsqueeze(2).to(dtype=torch.bool)
            
            # Get embeddings for target sequence
            tgt_emb = self.tgt_tok_emb(tgt_ids)
            pos_emb = self.pos_emb[:, :tgt_ids.size(1), :]
            y = self.drop(tgt_emb + pos_emb)
            
            # Decode
            y = self.decoder(y, encoder_out, tgt_mask, src_mask)
            
            # Project to vocabulary
            logits = self.head(y)
            
            # Only take the last token's logits
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append next token to sequence
            tgt_ids = torch.cat((tgt_ids, next_token), dim=1)
            
            # Stop if we hit the EOS token (assume 3 is EOS token)
            if (next_token == 3).any():
                break
        
        return tgt_ids

class TransformerConfig:
    """Configuration class to store the configuration of a `Transformer`."""
    def __init__(
        self,
        src_vocab_size=50257,
        tgt_vocab_size=50257,
        block_size=1024,
        n_layer=12,  # Number of encoder layers
        n_pre_cross_layer=6,  # Number of decoder layers before cross attention
        n_cross_layer=6,  # Number of decoder layers with cross attention
        n_embd=768,
        num_heads=12,
        dropout=0.1
    ):
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_pre_cross_layer = n_pre_cross_layer
        self.n_cross_layer = n_cross_layer
        self.n_embd = n_embd
        self.num_heads = num_heads
        self.dropout = dropout