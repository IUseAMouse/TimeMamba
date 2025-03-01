import torch
import torch.nn as nn
import torch.nn.functional as F

from .tp_moe_block import TPMoEBlock

class ResidualTPMoEBlock(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, **kwargs):
        super().__init__()
        self.tp_moe = TPMoEBlock(input_dim=input_dim, output_dim=output_dim, **kwargs)
        
        # Residual connection projection if dimensions don't match
        self.residual_proj = None
        if input_dim != output_dim:
            self.residual_proj = nn.Conv1d(input_dim, output_dim, kernel_size=1)
            
    def forward(self, x):
        identity = x
        if self.residual_proj is not None:
            identity = self.residual_proj(x)
            
        out, routing_info = self.tp_moe(x)
        
        # Fix sequence length mismatch - resize identity to match out's sequence length
        if identity.shape[2] != out.shape[2]:
            identity = F.adaptive_avg_pool1d(identity, out.shape[2])
            
        return out + identity, routing_info


class StackedTPMoE(nn.Module):
    def __init__(
        self, 
        num_blocks=3, 
        hidden_dim=128, 
        forecast_len=96, 
        sequence_len=384, 
        dropout=0.1,
        use_mamba=True,
        d_state=16,
        **block_args
    ):
        super().__init__()
        self.num_blocks = num_blocks
        
        # First block: input_dim=1, output_dim=hidden_dim
        self.blocks = nn.ModuleList([
            ResidualTPMoEBlock(
                input_dim=1, 
                output_dim=hidden_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                use_mamba=use_mamba,
                d_state=d_state,
                **block_args
            )
        ])
        
        # Middle blocks: input_dim=hidden_dim, output_dim=hidden_dim
        for _ in range(num_blocks - 2):
            self.blocks.append(
                ResidualTPMoEBlock(
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    use_mamba=use_mamba,
                    d_state=d_state,
                    **block_args
                )
            )
        
        # Last block: input_dim=hidden_dim, output_dim=1
        if num_blocks > 1:
            self.blocks.append(
                ResidualTPMoEBlock(
                    input_dim=hidden_dim,
                    output_dim=1,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    use_mamba=use_mamba,
                    d_state=d_state,
                    **block_args
                )
            )
            
        # Forecast projection: takes the last 'sequence_len' points and projects to 'forecast_len'
        self.forecast_proj = nn.Linear(sequence_len, forecast_len)
        
    def forward(self, x):
        # Input shape: [batch, 1, sequence_len]
        features = x
        all_routing_info = []
        
        # Process through stacked blocks
        for block in self.blocks:
            features, routing_info = block(features)
            all_routing_info.append(routing_info)
        
        # Project to forecast length
        # First transpose to get sequence dimension as last dim for linear projection
        features = features.transpose(1, 2)  # [batch, sequence_len, 1]
        output = self.forecast_proj(features)  # [batch, forecast_len, 1]
        output = output.transpose(1, 2)  # [batch, 1, forecast_len]
        
        return output, all_routing_info