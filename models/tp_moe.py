import torch
import torch.nn as nn
import torch.nn.functional as F
from .tp_moe_block import TPMoEBlock

class TemporalPyramidMoE(nn.Module):
    def __init__(
        self,
        sequence_len=384,
        forecast_len=96,
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
        self.sequence_len = sequence_len
        self.forecast_len = forecast_len
        
        # Use a single TPMoEBlock for backward compatibility
        self.block = TPMoEBlock(
            input_dim=1,
            output_dim=1,  
            hidden_dim=hidden_dim,
            levels=levels,
            base_window=base_window,
            num_experts=num_experts,
            k=k,
            use_mamba=use_mamba,
            d_state=d_state
        )
        
        # Instead of a Linear layer, use a Conv1d layer for sequence-to-sequence projection
        # This avoids the shape mismatch issues with the Linear layer
        self.forecast_conv = nn.Conv1d(
            in_channels=1,                # Input channels from TPMoEBlock
            out_channels=1,               # Keep same channel dimension
            kernel_size=sequence_len,     # Look at the entire sequence
            stride=1,
            padding=0
        )
        
        # Add a final adaptive pooling layer to ensure forecast length
        self.final_pool = nn.AdaptiveAvgPool1d(forecast_len)
        
    def forward(self, x):
        # Process through a single block
        features, routing_info = self.block(x)
        
        # Print input shape (for debugging)
        batch_size, channels, seq_len = features.shape
        print(f"Features from TPMoEBlock: batch={batch_size}, channels={channels}, seq_len={seq_len}")
        
        # Ensure we always have the right number of channels (1)
        if channels != 1:
            # Convert multi-channel to single channel if needed
            features = features.mean(dim=1, keepdim=True)
            print(f"After channel reduction: {features.shape}")
        
        # Apply our convolutional projection
        # This works with any input sequence length
        if seq_len >= self.sequence_len:
            # If sequence is long enough, use the conv directly
            output = self.forecast_conv(features)
            print(f"After forecast_conv: {output.shape}")
        else:
            # If sequence is too short, pad it first
            padding_size = self.sequence_len - seq_len
            features_padded = F.pad(features, (0, padding_size))
            output = self.forecast_conv(features_padded)
            print(f"After padding and forecast_conv: {output.shape}")
        
        # Ensure output has exactly the forecast length
        output = self.final_pool(output)
        print(f"Final output shape: {output.shape}")
        
        return output, routing_info