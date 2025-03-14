# TimeMamba

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**TimeMamba** is a PyTorch implementation of the Mamba state space model for time series forecasting. It leverages the powerful Mamba architecture to provide efficient and accurate predictions for time series data and O(n) memory complexity.

## Features

- 🚀 **Efficient Implementation**: Optimized for both memory usage and speed using selective state space models
- 📈 **Time Series Forecasting**: Direct multi-step forecasting for various horizons
- 🔌 **PyTorch Lightning Integration**: Easy training, validation, and testing with built-in best practices
- 🧩 **Extensible Architecture**: Modular design that can be easily adapted for different time series tasks

## Installation

TimeMamba uses [uv](https://github.com/astral-sh/uv) for dependency management for faster and more reliable Python package handling.

### Prerequisites

- Python 3.8+
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/IUseAMouse/TimeMamba.git
cd TimeMamba

# Create a virtual environment and install dependencies
make setup
make installdeps

# Alternatively, if you don't have make installed:
uv venv
uv pip install -e .
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt
```

## Quick Start

Here's a simple example of how to use TimeMamba for forecasting:

```bash
import torch
from src.models.mamba import MambaForecastingModel

# Create sample time series data
batch_size, seq_len, input_features = 32, 100, 1
x = torch.randn(batch_size, seq_len, input_features)

# Initialize model
model = MambaForecastingModel(
    input_size=input_features,  
    output_size=1,              # Dimension of prediction targets
    d_model=128,                # Hidden dimension
    n_layers=4,                 # Number of Mamba blocks
    d_state=16,                 # State space model dimension
    forecast_horizon=24         # Predict 24 steps ahead
)

# Generate forecasts
with torch.no_grad():
    predictions = model(x)  # Shape: [batch_size, forecast_horizon, output_size]
    
print(f"Input shape: {x.shape}")
print(f"Prediction shape: {predictions.shape}")
```

This is how you can train a model :

```bash
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

# Prepare data (x: inputs, y: targets)
train_data = TensorDataset(x_train, y_train)
val_data = TensorDataset(x_val, y_val)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_data, batch_size=32, num_workers=4)

# Create model
model = MambaForecastingModel(
    input_size=1,
    output_size=1,
    d_model=128,
    n_layers=4,
    forecast_horizon=24,
    learning_rate=1e-3
)

# Train with PyTorch Lightning
trainer = pl.Trainer(
    max_epochs=100,
    accelerator="auto",
    devices=1,
    gradient_clip_val=1.0
)
trainer.fit(model, train_loader, val_loader)

# Save the trained model
trainer.save_checkpoint("mamba_forecast_model.ckpt")
```
