import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model core - the heart of Mamba.
    """
    def __init__(self, d_model, d_state, dropout=0.0):
        super().__init__()
        # B and C matrices
        self.B = nn.Parameter(torch.randn(d_model, d_state) / d_state**0.5)
        self.C = nn.Parameter(torch.randn(d_model, d_state) / d_state**0.5)
        
        # Parameter for discretization step (delta)
        self.log_delta = nn.Parameter(torch.zeros(d_model))
        
        # Dropout for state
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, state=None):
        """
        x: (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape
        d_state = self.B.shape[1]
        
        # Initialize state if not provided - shape should be (batch, d_model, d_state)
        if state is None:
            state = torch.zeros(batch, d_model, d_state, device=x.device)
        
        # Get discrete-time parameters
        delta = F.softplus(self.log_delta)  # (d_model,)
        
        # Create output sequence
        output = torch.zeros_like(x)
        
        # Discretize continuous parameters - shape: (d_model, d_state)
        deltaB = torch.exp(-delta.unsqueeze(-1) * self.B)  # (d_model, d_state)
        
        # Selective scan - process sequence step by step
        for t in range(seq_len):
            # Get current input - shape: (batch, d_model)
            xt = x[:, t]  # (batch, d_model)
            
            # Compute state update (discretized SSM)
            # For each feature dimension:
            # state' = A * state + B * input
            
            # A * state: element-wise multiplication with deltaB
            # Shape: (batch, d_model, d_state)
            state_A = state * deltaB.unsqueeze(0)
            
            # B * input: outer product of input and B
            # Shape: (batch, d_model, d_state)
            input_B = xt.unsqueeze(-1) * self.B.unsqueeze(0)
            
            # Combined update with delta-scaled input term
            state = state_A + (1 - deltaB.unsqueeze(0)) * input_B
            
            # Apply dropout to the state
            state = self.dropout(state)
            
            # Compute output: C * state (for each feature dimension)
            # Shape multiplication: (batch, d_model, d_state) × (d_model, d_state) -> (batch, d_model)
            yt = torch.sum(state * self.C.unsqueeze(0), dim=-1)
            
            # Store in output sequence
            output[:, t] = yt
            
        return output, state


class MambaBlock(nn.Module):
    """
    Complete Mamba block with input projection, selective SSM, and output projection
    """
    def __init__(
        self, 
        d_model, 
        d_state=16, 
        d_conv=4, 
        expand_factor=2, 
        dropout=0.1
    ):
        super().__init__()
        
        # Expansion factor for SiLU activation
        self.expand = expand_factor
        expanded_dim = int(expand_factor * d_model)
        
        # Input projection and expansion
        self.in_proj = nn.Linear(d_model, expanded_dim * 2)  # Split into SSM input and gate
        
        # Local convolution for short-range mixing
        self.conv = nn.Conv1d(
            expanded_dim, 
            expanded_dim, 
            kernel_size=d_conv,
            padding=d_conv-1,
            groups=expanded_dim
        )
        self.conv_activation = nn.SiLU()
        
        # Selective SSM - the heart of Mamba
        self.ssm = SelectiveSSM(expanded_dim, d_state, dropout=dropout)
        
        # Layer normalization and projections
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(expanded_dim, d_model)
        
        # Initialize parameters
        self._init_weights()
        
    def _init_weights(self):
        # Initialize conv as almost identity - important for stability
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        
        # Initialize output projection with small weights
        nn.init.normal_(self.out_proj.weight, std=0.02)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        # Apply layer norm
        x_ln = self.norm(x)
        
        # Project input and split to get SSM input and gate
        x_proj = self.in_proj(x_ln)
        x_proj_expanded, x_gate = torch.chunk(x_proj, 2, dim=-1)
        
        # Apply local convolution (for short-range mixing)
        # Transpose for conv1d which expects [B, C, L]
        x_conv = x_proj_expanded.transpose(1, 2)
        x_conv = self.conv(x_conv)[:, :, :x.size(1)]  # Causal: trim to input length
        x_conv = x_conv.transpose(1, 2)  # Back to [B, L, C]
        x_conv = self.conv_activation(x_conv)
        
        # Apply selective SSM to the convolution output
        x_ssm, _ = self.ssm(x_conv)
        
        # Apply gating mechanism with sigmoid
        x_gated = x_ssm * torch.sigmoid(x_gate)
        
        # Project back to d_model dimension
        output = self.out_proj(x_gated)
        
        # Residual connection
        return output + x


class MambaSequenceModel(nn.Module):
    """
    A complete Mamba-based sequence model with multiple stacked blocks
    """
    def __init__(
        self, 
        d_model=128, 
        n_layer=4,
        d_state=16, 
        d_conv=4, 
        expand_factor=2, 
        dropout=0.1
    ):
        super().__init__()
        
        # Input projection
        self.in_proj = nn.Linear(1, d_model)
        
        # Stack of Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand_factor=expand_factor,
                dropout=dropout
            )
            for _ in range(n_layer)
        ])
        
        # Output projection
        self.out_proj = nn.Linear(d_model, 1)
        
        # Initialize
        self._init_weights()
        
    def _init_weights(self):
        # Initialize input projection
        nn.init.normal_(self.in_proj.weight, std=0.02)
        if self.in_proj.bias is not None:
            nn.init.zeros_(self.in_proj.bias)
            
        # Initialize output projection
        nn.init.normal_(self.out_proj.weight, std=0.02)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)
        
    def forward(self, x):
        """
        x: (batch, channels, seq_len) - typical for time series
        """
        # Reshape input to (batch, seq_len, channels)
        x = x.transpose(1, 2)
        
        # Project to d_model dimension
        x = self.in_proj(x)
        
        # Pass through Mamba layers
        for layer in self.layers:
            x = layer(x)
            
        # Project back to output dimension
        x = self.out_proj(x)
        
        # Reshape back to (batch, channels, seq_len)
        return x.transpose(1, 2)


class MambaFeatureExtractor(nn.Module):
    """
    Mamba feature extractor that matches the interface expected by the TP-MoE architecture
    """
    def __init__(self, d_model=128, d_state=16, expand=2, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Mamba blocks
        self.mamba = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=4,  # Small kernel for local mixing
            expand_factor=expand,
            dropout=dropout
        )
        
        # Additional feed-forward with residual for feature enrichment
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * expand),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expand, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        x: (batch, channels, seq_len) - time series format
        """
        # Transpose to (batch, seq_len, channels)
        x = x.transpose(1, 2)
        
        # Process with Mamba
        x = self.mamba(x)
        
        # Apply feed-forward network with residual
        residual = x
        x = self.ff(x) + residual
        
        # Transpose back to (batch, channels, seq_len)
        return x.transpose(1, 2)