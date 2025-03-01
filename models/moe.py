import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import RoPEMultiheadAttention

class PatternMoE(nn.Module):
    def __init__(self, hidden_dim, num_experts=8, k=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.k = k  # Keeping original parameter name
        
        # Router network for each level's features
        self.router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts)
        )
        
        # Experts for different pattern types with proper padding
        # Each expert processes the input differently
        self.local_experts = nn.ModuleList([
            nn.Sequential(
                # Ensure same length with proper padding
                nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, 
                          padding='same',  # Changed to 'same' padding to preserve sequence length
                          dilation=2**(i % 3 + 1)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1)
            ) for i in range(num_experts // 2)
        ])
        
        self.global_experts = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                RoPEMultiheadAttention(d_model=hidden_dim, num_heads=4),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim)
            ) for _ in range(num_experts - num_experts // 2)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)
                
    def forward(self, level_features):
        """
        Process features from each pyramid level with MoE
        
        Args:
            level_features: List of tensors [B, C, L_i] for each level i
            
        Returns:
            processed_features: List of processed feature tensors
            routing_info: Dictionary with routing information
        """
        if not isinstance(level_features, list):
            raise ValueError("Expected level_features to be a list of tensors, one for each pyramid level")
        
        # Process each level separately
        processed_features = []
        all_routing_probs = []
        expert_usage_counts = torch.zeros(self.num_experts, device=level_features[0].device)
        
        for level_idx, x in enumerate(level_features):
            # Transpose to [B, L, C] for processing
            x = x.transpose(1, 2)  # [B, L, C]
            batch_size, seq_len, channels = x.shape
            
            # Get routing logits from average of sequence
            router_input = self.norm(x.mean(dim=1))  # [batch, channels]
            routing_logits = self.router(router_input)  # [batch, num_experts]
            
            # Select top-k experts
            routing_probs = F.softmax(routing_logits, dim=-1)
            topk_probs, topk_indices = torch.topk(routing_probs, self.k, dim=-1)
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)  # Normalize
            
            # Create expert outputs tensor that matches input shape exactly
            expert_outputs = torch.zeros_like(x)
            
            for batch_idx in range(batch_size):
                # Process input through selected experts
                for k_idx in range(self.k):
                    expert_idx = topk_indices[batch_idx, k_idx].item()
                    expert_prob = topk_probs[batch_idx, k_idx].item()
                    
                    # Count expert usage
                    expert_usage_counts[expert_idx] += 1
                    
                    if expert_idx < len(self.local_experts):
                        # Local expert (convolutional)
                        x_batch = x[batch_idx:batch_idx+1].transpose(1, 2)  # [1, C, L]
                        
                        # Apply the expert
                        expert_out = self.local_experts[expert_idx](x_batch)
                        
                        # Ensure output has correct sequence length
                        if expert_out.shape[2] != seq_len:
                            # Resize if needed (shouldn't happen with 'same' padding)
                            expert_out = F.interpolate(expert_out, size=seq_len, mode='linear')
                            
                        expert_out = expert_out.transpose(1, 2)  # [1, L, C]
                        
                    else:
                        # Global expert (attention-based)
                        global_idx = expert_idx - len(self.local_experts)
                        x_trans = x[batch_idx:batch_idx+1]  # [1, L, C]
                        
                        # Apply sequential components
                        expert_out = self.global_experts[global_idx][0](x_trans)  # LayerNorm
                        expert_out = expert_out + self.global_experts[global_idx][1](
                            expert_out, expert_out, expert_out)  # Attention
                        expert_out = expert_out + self.global_experts[global_idx][2:](expert_out)  # FFN
                    
                    # Ensure shape is exactly [L, C] before adding
                    if expert_out.shape[1:] != (seq_len, channels):
                        print(f"Warning: Expert {expert_idx} output shape {expert_out.shape} " +
                              f"doesn't match expected ({seq_len}, {channels})")
                        # Resize to match
                        expert_out = F.interpolate(
                            expert_out.transpose(1, 2), 
                            size=seq_len, 
                            mode='linear'
                        ).transpose(1, 2)
                    
                    # Add weighted expert output
                    expert_outputs[batch_idx] += expert_prob * expert_out.squeeze(0)
            
            # Add residual connection
            expert_outputs = expert_outputs + x
            
            # Transpose back to [B, C, L] format
            expert_outputs = expert_outputs.transpose(1, 2)
            
            # Add to processed features
            processed_features.append(expert_outputs)
            all_routing_probs.append(routing_probs)
        
        # Normalize usage counts by batch size * levels
        expert_usage = expert_usage_counts / (len(level_features) * level_features[0].shape[0])
        
        # Return expert outputs and routing info for monitoring
        return processed_features, {
            "routing_probs": all_routing_probs,
            "expert_usage": expert_usage
        }