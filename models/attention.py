import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.positional_encoding import RotaryPositionalEmbedding, apply_rotary_pos_emb

class RoPEMultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len=4096):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # RoPE embedding
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)
        
    def forward(self, query, key, value, attn_mask=None):
        batch_size, q_len, _ = query.shape
        _, kv_len, _ = key.shape
        
        # Linear projections and reshape for multi-head
        q = self.q_proj(query).reshape(batch_size, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(key).reshape(batch_size, kv_len, self.num_heads, self.head_dim)
        v = self.v_proj(value).reshape(batch_size, kv_len, self.num_heads, self.head_dim)
        
        # Get positional embeddings
        pos_emb = self.rope(max(q_len, kv_len))
        
        # Apply rotary embeddings to q and k
        q_pos = pos_emb[:q_len].unsqueeze(0).unsqueeze(2)  # [1, q_len, 1, head_dim]
        k_pos = pos_emb[:kv_len].unsqueeze(0).unsqueeze(2)  # [1, kv_len, 1, head_dim]
        
        q, k = apply_rotary_pos_emb(q, k, q_pos if q_len <= kv_len else k_pos)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch, num_heads, q_len, head_dim]
        k = k.transpose(1, 2)  # [batch, num_heads, kv_len, head_dim]
        v = v.transpose(1, 2)  # [batch, num_heads, kv_len, head_dim]
        
        # Compute attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [batch, num_heads, q_len, kv_len]
        
        # Apply attention mask if provided
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, -1e9)
            
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        # Compute contextualized representations
        context = torch.matmul(attn_probs, v)  # [batch, num_heads, q_len, head_dim]
        context = context.transpose(1, 2).reshape(batch_size, q_len, self.d_model)
        
        # Project to output
        output = self.out_proj(context)
        
        return output