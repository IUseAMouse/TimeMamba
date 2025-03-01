import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import RoPEMultiheadAttention

class PatternIntegration(nn.Module):
    def __init__(self, hidden_dim, levels):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.levels = levels
        
        # Cross-attention between levels
        self.level_attentions = nn.ModuleList([
            RoPEMultiheadAttention(hidden_dim, num_heads=4)
            for _ in range(levels)
        ])
        
        # Frequency domain analysis
        self.freq_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            for _ in range(levels)
        ])
        
        # Layer normalizations for each level
        self.level_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(levels)
        ])
        
        # Final integration
        self.final_attention = RoPEMultiheadAttention(hidden_dim, num_heads=4)
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        # Fusion layer that combines all level features
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * levels, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
    def forward(self, level_features):
        """
        Process features from multiple pyramid levels
        
        Args:
            level_features: List of tensors [B, C, L_i] for each level i
        
        Returns:
            Integrated feature tensor [B, C, L_0]
        """
        # Defensive check to avoid None return
        if not isinstance(level_features, list):
            print(f"Warning: level_features is not a list but {type(level_features)}")
            if level_features is None:
                # Return a dummy tensor if input is None (should never happen in normal operation)
                return torch.zeros(1, self.hidden_dim, 10)
            # Try to use the input directly if it's already a tensor
            if isinstance(level_features, torch.Tensor):
                return level_features
        
        if len(level_features) != self.levels:
            print(f"Warning: Expected {self.levels} levels but got {len(level_features)}")
            # Pad with empty levels if needed
            while len(level_features) < self.levels:
                # Create a dummy tensor with same batch size as first level
                dummy = torch.zeros_like(level_features[0])
                level_features.append(dummy)
            # Or truncate if we have too many
            level_features = level_features[:self.levels]
        
        enhanced_features = []
        
        # First pass: Apply cross-attention between each level and all other levels
        for i in range(self.levels):
            # Get current level features and transpose for attention [B, L, C]
            current_level = level_features[i].transpose(1, 2)
            
            # Apply normalization - on the last dimension (channels)
            level_norm = self.level_norms[i](current_level)
            
            # Initialize with original features
            enhanced = current_level
            
            # Apply cross-attention with every other level
            for j in range(self.levels):
                if i != j:
                    # Get other level features and transpose
                    other_level = level_features[j].transpose(1, 2)
                    
                    # Adaptive resize the other level to match current level length
                    target_len = current_level.shape[1]
                    
                    # Use interpolate for sequence dimension
                    if other_level.shape[1] != target_len:
                        other_level = F.interpolate(
                            other_level.transpose(1, 2), 
                            size=target_len, 
                            mode='linear'
                        ).transpose(1, 2)
                    
                    # Apply normalization to other level
                    other_norm = self.level_norms[j](other_level)
                    
                    # Apply cross-attention
                    attn_out = self.level_attentions[i](level_norm, other_norm, other_norm)
                    enhanced = enhanced + attn_out
            
            # Skip frequency analysis if sequence is too short
            if enhanced.shape[1] > 1:
                try:
                    # Apply frequency domain analysis
                    # First transpose back to [B, C, L] for FFT
                    enhanced_t = enhanced.transpose(1, 2)
                    freq = torch.fft.rfft(enhanced_t)
                    
                    # Handle complex output
                    freq_real = freq.real
                    freq_imag = freq.imag
                    
                    # Convert back to [B, L, C] for further processing
                    freq_real = freq_real.transpose(1, 2)
                    freq_imag = freq_imag.transpose(1, 2)
                    
                    # Handle shapes - concatenate real and imaginary parts
                    min_len = min(freq_real.shape[1], freq_imag.shape[1])
                    freq_real = freq_real[:, :min_len]
                    freq_imag = freq_imag[:, :min_len]
                    
                    # Resize back to original length if needed
                    if min_len != enhanced.shape[1]:
                        # Pad with zeros
                        pad_len = enhanced.shape[1] - min_len
                        freq_real = F.pad(freq_real, (0, 0, 0, pad_len))
                        freq_imag = F.pad(freq_imag, (0, 0, 0, pad_len))
                    
                    # Combine real and imaginary parts
                    freq_combined = torch.cat([freq_real, freq_imag], dim=-1)
                    
                    # Transform frequency features
                    freq_features = self.freq_transforms[i](freq_combined)
                    
                    # Combine with enhanced features
                    enhanced = enhanced + freq_features
                except Exception as e:
                    print(f"Warning: FFT analysis failed for level {i}, skipping. Error: {str(e)}")
            
            # Store as [B, C, L] for consistency
            enhanced_features.append(enhanced.transpose(1, 2))
        
        # Make sure we have at least one valid feature tensor
        if not enhanced_features:
            print("Warning: No valid enhanced features produced")
            # Return the first input level as fallback
            return level_features[0] 
        
        # Resize all features to match the first level's length
        base_length = enhanced_features[0].shape[2]
        aligned_features = []
        
        for i, feat in enumerate(enhanced_features):
            if feat is None:
                print(f"Warning: enhanced feature {i} is None, skipping")
                continue
                
            if i == 0:
                aligned_features.append(feat)
            else:
                # Resize to match base length
                try:
                    resized = F.interpolate(feat, size=base_length, mode='linear')
                    aligned_features.append(resized)
                except Exception as e:
                    print(f"Warning: Failed to resize feature {i}. Error: {str(e)}")
                    # Skip this feature
        
        # If we have no aligned features, return the first enhanced feature
        if not aligned_features:
            print("Warning: No aligned features available")
            return enhanced_features[0]
        
        try:
            # Convert to [B, L, C] for concatenation
            aligned_features_t = [f.transpose(1, 2) for f in aligned_features]
            
            # Concatenate all aligned features for final fusion
            concat_features = torch.cat(aligned_features_t, dim=-1)
            
            # Apply fusion
            integrated = self.fusion(concat_features)
            
            # Convert back to [B, C, L] for output
            return integrated.transpose(1, 2)
        except Exception as e:
            print(f"Warning: Failed during final fusion. Error: {str(e)}")
            # Return the first aligned feature as fallback
            return aligned_features[0]