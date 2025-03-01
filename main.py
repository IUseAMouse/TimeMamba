import torch
from models import StackedTPMoE, TemporalPyramidMoE

def test_mamba_first_architecture():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create dummy data
    batch_size = 4
    sequence_len = 384
    x = torch.randn(batch_size, 1, sequence_len)
    
    print(f"Testing Mamba-first architecture with input shape: {x.shape}")
    
    # Test single block model
    model_single = TemporalPyramidMoE(
        sequence_len=sequence_len,
        forecast_len=96,
        hidden_dim=64,
        levels=4,
        base_window=16,
        num_experts=8,
        k=4,
        use_mamba=True,
        d_state=16
    )
    
    # Count parameters
    single_params = sum(p.numel() for p in model_single.parameters())
    print(f"Single block model parameters: {single_params:,}")
    
    # Forward pass
    output_single, routing_info_single = model_single(x)
    print(f"Single block output shape: {output_single.shape}")
    
    return model_single

if __name__ == "__main__":
    test_mamba_first_architecture()