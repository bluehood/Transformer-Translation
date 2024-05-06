import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        Initialize the MultiHeadAttention layer with the specified model dimension and number of heads.
        
        Parameters:
            d_model (int): The model's dimension.
            num_heads (int): The number of attention heads.
        
        Returns:
            None
        """
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize dimensions
        self.d_model = d_model # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_model // num_heads # Dimension of each head's key, query, and value
        
        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model) # Query transformation
        self.W_k = nn.Linear(d_model, d_model) # Key transformation
        self.W_v = nn.Linear(d_model, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Apply scaled dot-product attention to the given query, key, and value tensors.

        Args:
            Q (torch.Tensor): Query tensor of shape (batch_size, query_length, d_model).
            K (torch.Tensor): Key tensor of shape (batch_size, key_length, d_model).
            V (torch.Tensor): Value tensor of shape (batch_size, key_length, d_model).
            mask (torch.Tensor, optional): Mask tensor of shape (batch_size, query_length, key_length). Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, query_length, d_model).
        """
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        """
        Split the input tensor into multiple heads for multi-head attention.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_length, d_model).

        Returns:
            torch.Tensor: The reshaped tensor of shape (batch_size, num_heads, seq_length, d_k).
        """
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        """
        Combine the multiple heads back to the original shape.

        Parameters:
            x (torch.Tensor): The input tensor of shape (batch_size, num_heads, seq_length, d_k).

        Returns:
            torch.Tensor: The reshaped tensor of shape (batch_size, seq_length, d_model).
        """
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        """
        Apply linear transformations and split heads.
        Perform scaled dot-product attention.
        Combine heads and apply output transformation.
        
        Args:
            Q (torch.Tensor): Query tensor of shape (batch_size, query_length, d_model).
            K (torch.Tensor): Key tensor of shape (batch_size, key_length, d_model).
            V (torch.Tensor): Value tensor of shape (batch_size, key_length, d_model).
            mask (torch.Tensor, optional): Mask tensor of shape (batch_size, query_length, key_length). Defaults to None.
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, query_length, d_model).
        """
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output
    

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        """
        Initialize the PositionWiseFeedForward module with the specified model dimension and feed-forward dimension.

        Parameters:
            d_model (int): The model's dimension.
            d_ff (int): The feed-forward dimension.

        Returns:
            None
        """
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Apply forward pass through the PositionWiseFeedForward module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, d_model).
        """
        return self.fc2(self.relu(self.fc1(x)))
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        """
        Initialize the PositionalEncoding module with the specified model dimension and maximum sequence length.
        
        Parameters:
            d_model (int): The model's dimension.
            max_seq_length (int): The maximum sequence length.
        
        Returns:
            None
        """
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        """
        Apply forward pass through the PositionalEncoding module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, d_model).
        """
        return x + self.pe[:, :x.size(1)]
    

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        """
        Initialize the EncoderLayer module with the specified model dimension, number of heads, feed-forward dimension, and dropout rate.
        
        Parameters:
            d_model (int): The model's dimension.
            num_heads (int): The number of attention heads.
            d_ff (int): The feed-forward dimension.
            dropout (float): The dropout rate.
            
        Returns:
            None
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        """
        Apply the forward pass through the EncoderLayer.

        Parameters:
            x (torch.Tensor): The input tensor.
            mask (torch.Tensor): The mask tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        """
        Initializes a DecoderLayer object.

        Args:
            d_model (int): The dimensionality of the model.
            num_heads (int): The number of attention heads.
            d_ff (int): The dimensionality of the feed-forward layer.
            dropout (float): The dropout rate.

        Returns:
            None
        """
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        """
        Apply the forward pass through the DecoderLayer.

        Parameters:
            x (torch.Tensor): The input tensor.
            enc_output (torch.Tensor): The encoder output tensor.
            src_mask (torch.Tensor): The source mask tensor.
            tgt_mask (torch.Tensor): The target mask tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
    

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        """
        Initializes a Transformer model.

        Args:
            src_vocab_size (int): The size of the source vocabulary.
            tgt_vocab_size (int): The size of the target vocabulary.
            d_model (int): The dimensionality of the model.
            num_heads (int): The number of attention heads.
            num_layers (int): The number of layers in the model.
            d_ff (int): The dimensionality of the feed-forward layer.
            max_seq_length (int): The maximum sequence length.
            dropout (float): The dropout rate.

        Returns:
            None
        """
        super(Transformer, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        """
        Generate masks for the source and target sequences.

        Parameters:
            src (torch.Tensor): The source sequence tensor.
            tgt (torch.Tensor): The target sequence tensor.

        Returns:
            torch.Tensor: The source mask tensor.
            torch.Tensor: The target mask tensor.
        """
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        nopeak_mask = nopeak_mask.to(self.device)
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        """
        Apply the forward pass through the Transformer model.

        Args:
            src (torch.Tensor): The source sequence tensor.
            tgt (torch.Tensor): The target sequence tensor.

        Returns:
            torch.Tensor: The output tensor.

        This method takes in source and target sequences as input tensors and applies the forward pass through the Transformer model. It generates masks for the source and target sequences using the `generate_mask` method. It then embeds the source and target sequences using the `encoder_embedding` and `decoder_embedding` layers, respectively. The embedded sequences are passed through a series of encoder and decoder layers, which consist of self-attention and feed-forward neural networks. The output of the decoder layers is passed through a fully connected layer (`fc`) to produce the final output tensor.
        """
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