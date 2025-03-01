import torch
import torch.nn as nn
import torch.nn.functional as F

from .decomposition import TemporalPyramidDecomposition
from .moe import PatternMoE
from .integration import PatternIntegration
from .mamba import MambaFeatureExtractor

class TPMoEBlock(nn.Module):
    def __init__(
        self,
        input_dim=1,
        output_dim=1,
        hidden_dim=128,
        levels=4,
        base_window=16,
        num_experts=8,
        k=4,  
        dropout=0.1,
        use_mamba=True,
        d_state=16
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.use_mamba = use_mamba
        
        # Input projection if input_dim != 1
        self.input_proj = None
        if input_dim != 1:
            self.input_proj = nn.Conv1d(input_dim, 1, kernel_size=1)
        
        # Global processing with Mamba first (if enabled)
        if use_mamba:
            # Project to hidden_dim before Mamba
            self.pre_mamba_proj = nn.Conv1d(1, hidden_dim, kernel_size=1)
            
            # Global Mamba for long-range dependency modeling
            self.global_mamba = MambaFeatureExtractor(
                d_model=hidden_dim,
                d_state=d_state,
                expand=2,
                dropout=dropout
            )
            
            # Project back to single channel for pyramid decomposition
            self.post_mamba_proj = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        
        # Temporal pyramid decomposition (applied after global processing)
        self.pyramid_decomposition = TemporalPyramidDecomposition(
            levels=levels, 
            base_window=base_window
        )
        
        # Feature extraction from pyramid levels
        self.level_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(2, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Dropout(dropout)
            ) 
            for _ in range(levels)
        ])
        
        # Pattern-specific MoE processing - FIXED PARAMETERS
        self.pattern_moe = PatternMoE(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            k=k,
            dropout=dropout
        )
        
        # Pattern integration
        self.pattern_integration = PatternIntegration(
            hidden_dim=hidden_dim,
            levels=levels
        )
        
        # Output projection
        self.output_proj = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)
        
    def forward(self, x):
        # Handle input projection if needed
        if self.input_proj is not None:
            x = self.input_proj(x)
        
        # Global Mamba processing first (if enabled)
        if self.use_mamba:
            # Project to hidden dimension
            x_mamba = self.pre_mamba_proj(x)
            
            # Apply Mamba to capture global dependencies
            x_mamba = self.global_mamba(x_mamba)
            
            # Project back to single channel for pyramid decomposition
            x = self.post_mamba_proj(x_mamba)
            
            # Store global features for later skip connection
            global_features = x_mamba
        
        # Now decompose into temporal pyramid
        pyramid = self.pyramid_decomposition(x)
        
        # Extract features from each level
        level_features = []
        for i, (approx, detail) in enumerate(pyramid):
            # Combine approximation and detail coefficients
            combined = torch.cat([approx, detail], dim=1)
            features = self.level_encoders[i](combined)
            
            # Add global context if Mamba was used
            if self.use_mamba:
                # Adapt global features to current level's resolution
                target_length = features.shape[2]
                adapted_global = F.adaptive_avg_pool1d(global_features, target_length)
                
                # Add global context as residual connection
                features = features + adapted_global
                
            level_features.append(features)
            
        # Process with pattern-specific MoE
        processed_features, routing_info = self.pattern_moe(level_features)
        
        # Integrate patterns
        integrated = self.pattern_integration(processed_features)
        
        # Project to output dimension
        output = self.output_proj(integrated)
        
        return output, routing_info