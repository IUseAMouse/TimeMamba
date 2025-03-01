import torch
import torch.nn as nn
import math

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=4096, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Initialize the rotation matrices
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Create cache for fast inference
        self._rotary_cache = None
        
    def forward(self, seq_len):
        if self._rotary_cache is not None and seq_len <= self._rotary_cache.shape[0]:
            return self._rotary_cache[:seq_len]
            
        # Position indices
        pos = torch.arange(seq_len, device=self.inv_freq.device)
        # Outer product of positions and frequencies
        freqs = torch.einsum('i,j->ij', pos, self.inv_freq)
        # Complex exponential: cos(theta) + i*sin(theta)
        emb = torch.cat((freqs, freqs), dim=-1)
        # Cache for reuse
        self._rotary_cache = emb
        
        return emb

class ScaleAwareRoPE(RotaryPositionalEmbedding):
    def __init__(self, dim, max_seq_len=4096, base=10000, scale_factor=1.0):
        super().__init__(dim, max_seq_len, base)
        self.scale_factor = scale_factor
        
    def forward(self, seq_len):
        # Scale the positions based on the pyramid level
        pos = torch.arange(seq_len, device=self.inv_freq.device) * self.scale_factor
        freqs = torch.einsum('i,j->ij', pos, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb

class LevelPositionalEncoding(nn.Module):
    def __init__(self, d_model, level=0, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Scale factor based on pyramid level (coarser levels have larger scale)
        scale_factor = 1.0 / (2 ** level)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Scale the positions based on level
        position = position * scale_factor
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

def apply_rotary_pos_emb(q, k, pos_emb):
    # Extract the first half and second half of q and k
    q_cos = q[..., :q.shape[-1]//2]
    q_sin = q[..., q.shape[-1]//2:]
    k_cos = k[..., :k.shape[-1]//2]
    k_sin = k[..., k.shape[-1]//2:]
    
    # Get cos and sin components from positional embedding
    cos = pos_emb[..., :pos_emb.shape[-1]//2]
    sin = pos_emb[..., pos_emb.shape[-1]//2:]
    
    # Apply rotary embeddings
    q_embed = torch.cat([
        q_cos * cos - q_sin * sin,
        q_sin * cos + q_cos * sin
    ], dim=-1)
    
    k_embed = torch.cat([
        k_cos * cos - k_sin * sin,
        k_sin * cos + k_cos * sin
    ], dim=-1)
    
    return q_embed, k_embed