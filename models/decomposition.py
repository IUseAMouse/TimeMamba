import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalPyramidDecomposition(nn.Module):
    def __init__(self, levels=4, base_window=16):
        super().__init__()
        self.levels = levels
        self.base_window = base_window
        
        # Learnable wavelet-inspired filters for decomposition
        self.decomposers = nn.ModuleList([
            nn.Conv1d(1, 2, 
                    kernel_size=base_window * 2**i, 
                    stride=base_window // 2 if i == 0 else base_window * 2**(i-1), 
                    padding=base_window // 2 if i == 0 else base_window * 2**(i-1))
            for i in range(levels)
        ])
        
    def forward(self, x):
        # x shape: [batch, 1, sequence_length]
        pyramid = []
        x_current = x
        
        # Build multi-resolution representation
        for i, decomposer in enumerate(self.decomposers):
            # Check if current sequence length is valid for decomposition
            if x_current.shape[2] <= 1:
                # Sequence is too small, use placeholder values for this and remaining levels
                dummy_approx = x_current.clone()  # Use current tensor as approximation
                dummy_detail = torch.zeros_like(x_current)  # Zero tensor for detail
                
                # Add current level
                pyramid.append((dummy_approx, dummy_detail))
                
                # Add remaining levels with the same dummy values
                for _ in range(i+1, self.levels):
                    pyramid.append((dummy_approx, dummy_detail))
                
                # Exit early
                break
            
            # Split into approximate and detail coefficients
            coeffs = decomposer(x_current)
            approx, detail = coeffs[:, 0:1], coeffs[:, 1:2]
            pyramid.append((approx, detail))
            
            if i < self.levels - 1:
                # Only downsample if sequence length is sufficient
                if x_current.shape[2] <= 2:
                    # Can't downsample further, use current tensor for next level
                    x_current = approx
                else:
                    x_current = F.avg_pool1d(x_current, kernel_size=2, stride=2)
            
        return pyramid