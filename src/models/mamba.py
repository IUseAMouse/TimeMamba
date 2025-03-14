import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Optional, Tuple, Union, Dict, Any, List
from einops import rearrange, repeat


class ParallelSelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state, dropout=0.0):
        super().__init__()
        
        # SSM parameters
        self.A = nn.Parameter(torch.randn(d_model, d_state))
        self.B = nn.Parameter(torch.randn(d_model, d_state))
        
        # Replace parameter with actual projection layer
        self.C_proj = nn.Linear(d_model, d_model)
        
        # Optional selective projection
        self.D_proj = nn.Linear(d_model, d_model)
        
        # Time-step parameter
        self.log_delta = nn.Parameter(torch.zeros(d_model))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        batch, seq_len, d_model = x.shape
        d_state = self.B.shape[1]
        
        # Get discrete-time parameters
        delta = F.softplus(self.log_delta)  # (d_model,)
        
        # Discretize continuous parameters
        A_diag = torch.exp(-delta)  # (d_model,)
        
        # Compute input projection
        x_proj = self.C_proj(x)  # (batch, seq_len, d_model)
        
        # Apply selective projection
        x_proj = x_proj * torch.sigmoid(self.D_proj(x))
        
        # Apply mask to input if provided
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)  # [batch, seq_len, 1]
            x_proj = x_proj * mask_expanded
        
        # Sequential implementation for debugging
        h = torch.zeros(batch, d_model, d_state, device=x.device)
        output = torch.zeros_like(x)
        
        for t in range(seq_len):
            # If using a mask, reset the state when masked
            if mask is not None and t > 0:
                # Create mask tensor for this timestep
                mask_t = mask[:, t].unsqueeze(-1).unsqueeze(-1)  # [batch, 1, 1]
                # Reset state where mask is False
                h = h * mask_t
            
            # Update hidden state: h_t = A * h_{t-1} + B * x_t
            h = h * A_diag.unsqueeze(-1).unsqueeze(0) + \
                x_proj[:, t, :].unsqueeze(-1) * self.B.unsqueeze(0)
            
            # Compute output: y_t = C * h_t
            output[:, t, :] = self.dropout((h * self.A.unsqueeze(0)).sum(dim=-1))
            
            # Ensure masked positions have exactly zero output
            if mask is not None and not mask[:, t].all():
                output[:, t, :] = output[:, t, :] * mask[:, t].unsqueeze(-1)
        
        return output, h


class MambaBlock(nn.Module):
    """
    Improved Mamba block with efficient computation and support for masking
    """
    def __init__(
        self, 
        d_model, 
        d_state=16, 
        d_conv=4, 
        expand_factor=2, 
        dropout=0.1,
        use_gradient_checkpointing=False
    ):
        super().__init__()
        
        # Expansion factor for SiLU activation
        self.expand = expand_factor
        expanded_dim = int(expand_factor * d_model)
        
        # Input projection and expansion
        self.in_proj = nn.Linear(d_model, expanded_dim * 2)  # Split into SSM input and gate
        
        # Local convolution for short-range mixing (causal)
        self.conv = nn.Conv1d(
            expanded_dim, 
            expanded_dim, 
            kernel_size=d_conv,
            padding=d_conv-1,
            groups=expanded_dim
        )
        self.conv_activation = nn.SiLU()
        
        # Selective SSM - efficient parallelized version
        self.ssm = ParallelSelectiveSSM(expanded_dim, d_state, dropout=dropout)
        
        # Layer normalization and projections
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(expanded_dim, d_model)
        
        # Gradient checkpointing for memory efficiency
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
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

    def _forward_impl(self, x, mask=None):
        """
        Implementation of the forward pass
        
        Args:
            x: (batch, seq_len, d_model)
            mask: Optional[Tensor] - (batch, seq_len) boolean mask (True for valid positions)
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
        
        # Apply mask if provided
        if mask is not None:
            x_conv = x_conv * mask.unsqueeze(-1)
        
        # Apply selective SSM to the convolution output
        x_ssm, _ = self.ssm(x_conv, mask)
        
        # Apply gating mechanism with sigmoid
        x_gated = x_ssm * torch.sigmoid(x_gate)
        
        # Project back to d_model dimension
        output = self.out_proj(x_gated)
        
        # Apply mask if provided
        if mask is not None:
            output = output * mask.unsqueeze(-1)
        
        # Residual connection
        return output + x

    def forward(self, x, mask=None):
        """
        Forward pass with optional gradient checkpointing for memory efficiency
        """
        if self.use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, mask
            )
        else:
            return self._forward_impl(x, mask)


class MambaForecastingModel(pl.LightningModule):
    """
    Enhanced Mamba model for time series forecasting with support for:
    - Variable input lengths
    - Configurable forecast horizons
    - Direct multi-step forecasting
    - Efficient training on large datasets
    """
    def __init__(
        self,
        input_size: int = 1,  # Number of input features
        output_size: int = 1,  # Number of output features
        d_model: int = 128,   # Hidden dimension
        n_layers: int = 4,    # Number of Mamba blocks
        d_state: int = 16,    # SSM state dimension
        d_conv: int = 4,      # Convolution kernel size
        expand_factor: int = 2,  # Expansion factor
        dropout: float = 0.1,
        forecast_horizon: int = 24,  # Default forecast length
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        gradient_clip_val: float = 1.0,
        use_gradient_checkpointing: bool = True  # Memory optimization
    ):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Input embedding
        self.input_embedding = nn.Linear(input_size, d_model)
        
        # Stack of Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand_factor=expand_factor,
                dropout=dropout,
                use_gradient_checkpointing=use_gradient_checkpointing
            )
            for _ in range(n_layers)
        ])
        
        # Layer norm before output
        self.norm = nn.LayerNorm(d_model)
        
        # CRITICAL FIX: Always set forecast_horizon regardless of mode
        self.forecast_horizon = forecast_horizon       
        
        # Multi-step direct forecasting
        self.output_projection = nn.Linear(d_model, output_size * forecast_horizon)
        
        # Configure training settings
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip_val = gradient_clip_val
        
        # Metrics
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
    
    def forward(self, x, mask=None):
        """
        Forward pass through the model
        
        Args:
            x: Tensor of shape [batch, seq_len, input_size] - Input time series
            mask: Optional[Tensor] - Mask for variable-length sequences
            
        Returns:
            Tensor - Forecasted values with dimensions:
                    - Single-step autoregressive: [batch, output_size] (to be implemented)
                    - Multi-step: [batch, horizon, output_size]
        """
        batch_size, seq_len, _ = x.shape
        
        # Embed inputs
        x = self.input_embedding(x)
        
        # Process through Mamba layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Apply final normalization
        x = self.norm(x)
        
        # Direct multi-horizon forecasting
        x = x[:, -1]  # Use last time step features
        x = self.output_projection(x)  # Project to forecast horizon
        return x.reshape(batch_size, self.forecast_horizon, -1)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        y_hat = self(x)
        
        # Calculate loss (MSE for forecasting)
        loss = F.mse_loss(y_hat, y)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        
        # For additional metrics calculation
        self.training_step_outputs.append({'loss': loss, 'y': y, 'y_hat': y_hat})
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        # Forward pass - explicitly use the target's horizon
        y_hat = self(x)
        
        # Calculate loss
        loss = F.mse_loss(y_hat, y)
        
        # Calculate MAE for interpretability
        mae = F.l1_loss(y_hat, y)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_mae', mae, prog_bar=True, sync_dist=True)
        
        # Store for epoch-end computations
        self.validation_step_outputs.append({'loss': loss, 'mae': mae, 'y': y, 'y_hat': y_hat})
        
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        
        # Forward pass - explicitly use the target's horizon
        y_hat = self(x)
        
        # Calculate loss
        loss = F.mse_loss(y_hat, y)
        
        # Calculate MAE for interpretability
        mae = F.l1_loss(y_hat, y)
        
        # Log metrics
        self.log('t_loss', loss, prog_bar=True, sync_dist=True)
        self.log('tmae', mae, prog_bar=True, sync_dist=True)
        
        # Store for epoch-end computations
        self.test_step_outputs.append({'loss': loss, 'mae': mae, 'y': y, 'y_hat': y_hat})
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Cosine annealing with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10,
            T_mult=2
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch'
            }
        }

    def on_train_epoch_end(self):
        # Clear saved outputs
        self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self):
        # Clear saved outputs
        self.validation_step_outputs.clear()
    
    def on_test_epoch_end(self):
        # Clear saved outputs
        self.test_step_outputs.clear()
    
    def count_parameters(self):
        """
        Calculate the number of trainable parameters in the model
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def on_fit_start(self):
        """
        Log the number of model parameters when training starts
        """
        num_params = self.count_parameters()
        # Check if logger exists before attempting to log
        if self.logger is not None:
            self.logger.log_hyperparams({"model_parameters": num_params})
        else:
            # Else print instead of log when no logger is available
            print(f"Model parameters: {num_params}")
        
        # Print model architecture summary
        print("\nModel Architecture:")
        print(f"- Input Size: {self.hparams.input_size}")
        print(f"- Output Size: {self.hparams.output_size}")
        print(f"- Hidden Dimension: {self.hparams.d_model}")
        print(f"- Number of Layers: {self.hparams.n_layers}")
        print(f"- Forecast Horizon: {self.hparams.forecast_horizon}")


class MambaMixModule(pl.LightningModule):
    """
    Advanced Mamba architecture that supports both classification and forecasting
    with mixed-task capabilities. Suitable for large-scale training.
    """
    def __init__(
        self,
        input_size: int = 1,
        output_size: int = 1,
        num_classes: Optional[int] = None,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        dropout: float = 0.1,
        forecast_horizon: int = 24,
        task_type: str = 'forecasting',  # 'forecasting', 'classification', or 'multi_task'
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        gradient_clip_val: float = 1.0,
        use_gradient_checkpointing: bool = True
    ):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        self.task_type = task_type
        
        # Input embedding
        self.input_embedding = nn.Linear(input_size, d_model)
        
        # Stack of Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand_factor=expand_factor,
                dropout=dropout,
                use_gradient_checkpointing=use_gradient_checkpointing
            )
            for _ in range(n_layers)
        ])
        
        # Layer norm before output
        self.norm = nn.LayerNorm(d_model)
        
        # Task-specific heads
        if task_type in ['forecasting', 'multi_task']:
            self.forecasting_head = nn.Linear(d_model, output_size * forecast_horizon)
            self.forecast_horizon = forecast_horizon
            
        if task_type in ['classification', 'multi_task']:
            assert num_classes is not None, "num_classes must be provided for classification tasks"
            self.classification_head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(d_model, num_classes)
            )
        
        # Configure training
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip_val = gradient_clip_val
    
    def _process_backbone(self, x, mask=None):
        """Process input through the backbone network"""
        # Embed inputs
        x = self.input_embedding(x)
        
        # Process through Mamba layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Apply final normalization
        return self.norm(x)
    
    def forward(self, x, mask=None, task=None):
        """
        Forward pass with task specification
        
        Args:
            x: Tensor [batch, seq_len, input_size] - Input time series
            mask: Optional[Tensor] - Mask for variable-length sequences
            task: Optional[str] - Specify which task to perform ('forecasting' or 'classification')
                  If None, uses the default task_type
            
        Returns:
            Tensor - Either classification logits or forecasted values
        """
        batch_size, seq_len, _ = x.shape
        
        # Determine task
        if task is None:
            task = self.task_type
            
        # If multi_task, default to forecasting unless specified
        if task == 'multi_task':
            task = 'forecasting'
        
        # Process through backbone
        features = self._process_backbone(x, mask)
        
        # Task-specific outputs
        if task == 'classification':
            # For classification, use pooled representation
            # Reshape to (batch, d_model, seq_len) for pooling
            features_t = features.transpose(1, 2)
            return self.classification_head(features_t)
            
        elif task == 'forecasting':
            # For forecasting, use the last time step
            last_hidden = features[:, -1]
            
            # Direct multi-step forecasting
            output = self.forecasting_head(last_hidden)
            output_size = self.hparams.output_size
            output = output.view(batch_size, self.forecast_horizon, output_size)
            
            return output
    
    def training_step(self, batch, batch_idx):
        # Unpack the batch according to the task
        if self.task_type == 'forecasting':
            x, y = batch
            y_hat = self.forward(x, task='forecasting')
            loss = F.mse_loss(y_hat, y)
            self.log('train_loss', loss, prog_bar=True, sync_dist=True)
            
        elif self.task_type == 'classification':
            x, y = batch
            y_hat = self.forward(x, task='classification')
            loss = F.cross_entropy(y_hat, y)
            preds = torch.argmax(y_hat, dim=1)
            acc = (preds == y).float().mean()
            self.log('train_loss', loss, prog_bar=True, sync_dist=True)
            self.log('train_acc', acc, prog_bar=True, sync_dist=True)
            
        elif self.task_type == 'multi_task':
            # Expect a dictionary with task-specific data
            x = batch['input']
            forecast_target = batch.get('forecast_target')
            class_target = batch.get('class_target')
            
            # Compute losses for available tasks
            loss = 0
            if forecast_target is not None:
                forecast_pred = self.forward(x, task='forecasting')
                forecast_loss = F.mse_loss(forecast_pred, forecast_target)
                loss += forecast_loss
                self.log('train_forecast_loss', forecast_loss, prog_bar=True, sync_dist=True)
                
            if class_target is not None:
                class_pred = self.forward(x, task='classification')
                class_loss = F.cross_entropy(class_pred, class_target)
                loss += class_loss
                
                preds = torch.argmax(class_pred, dim=1)
                acc = (preds == class_target).float().mean()
                self.log('train_class_loss', class_loss, prog_bar=True, sync_dist=True)
                self.log('train_acc', acc, prog_bar=True, sync_dist=True)
            
            self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        # Similar structure to training_step with appropriate metrics
        if self.task_type == 'forecasting':
            x, y = batch
            y_hat = self.forward(x, task='forecasting')
            loss = F.mse_loss(y_hat, y)
            mae = F.l1_loss(y_hat, y)
            
            self.log('val_loss', loss, prog_bar=True, sync_dist=True)
            self.log('val_mae', mae, prog_bar=True, sync_dist=True)
        
        elif self.task_type == 'classification':
            x, y = batch
            y_hat = self.forward(x, task='classification')
            loss = F.cross_entropy(y_hat, y)
            preds = torch.argmax(y_hat, dim=1)
            acc = (preds == y).float().mean()
            
            self.log('val_loss', loss, prog_bar=True, sync_dist=True)
            self.log('val_acc', acc, prog_bar=True, sync_dist=True)
        
        elif self.task_type == 'multi_task':
            x = batch['input']
            forecast_target = batch.get('forecast_target')
            class_target = batch.get('class_target')
            
            loss = 0
            if forecast_target is not None:
                forecast_pred = self.forward(x, task='forecasting')
                forecast_loss = F.mse_loss(forecast_pred, forecast_target)
                forecast_mae = F.l1_loss(forecast_pred, forecast_target)
                loss += forecast_loss
                
                self.log('val_forecast_loss', forecast_loss, prog_bar=True, sync_dist=True)
                self.log('val_forecast_mae', forecast_mae, prog_bar=True, sync_dist=True)
                
            if class_target is not None:
                class_pred = self.forward(x, task='classification')
                class_loss = F.cross_entropy(class_pred, class_target)
                loss += class_loss
                
                preds = torch.argmax(class_pred, dim=1)
                acc = (preds == class_target).float().mean()
                
                self.log('val_class_loss', class_loss, prog_bar=True, sync_dist=True)
                self.log('val_acc', acc, prog_bar=True, sync_dist=True)
            
            self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        # Similar to validation_step but with more detailed metrics
        if self.task_type == 'forecasting':
            x, y = batch
            y_hat = self.forward(x, task='forecasting')
            loss = F.mse_loss(y_hat, y)
            mae = F.l1_loss(y_hat, y)
            
            # Step-wise metrics for multi-step forecasting
            if y.dim() > 2:
                for i in range(y.size(1)):
                    step_loss = F.mse_loss(y_hat[:, i], y[:, i])
                    step_mae = F.l1_loss(y_hat[:, i], y[:, i])
                    self.log(f'test_loss_step_{i+1}', step_loss, sync_dist=True)
                    self.log(f'test_mae_step_{i+1}', step_mae, sync_dist=True)
            
            self.log('test_loss', loss, prog_bar=True, sync_dist=True)
            self.log('test_mae', mae, prog_bar=True, sync_dist=True)
        
        # Similar implementations for classification and multi-task
        # [omitted for brevity but would follow validation_step pattern]
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers with cosine schedule and warm restarts"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10,
            T_mult=2
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch'
            }
        }
    
    def count_parameters(self):
        """Calculate the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def on_fit_start(self):
        """Log model statistics when training starts"""
        num_params = self.count_parameters()
        self.logger.log_hyperparams({"model_parameters": num_params})
        print(f"Model initialized with {num_params:,} trainable parameters")
        
        # Print detailed model info
        print("\nModel Architecture:")
        print(f"- Task Type: {self.hparams.task_type}")
        print(f"- Input Size: {self.hparams.input_size}")
        print(f"- Output Size: {self.hparams.output_size}")
        if hasattr(self.hparams, 'num_classes'):
            print(f"- Number of Classes: {self.hparams.num_classes}")
        print(f"- Hidden Dimension: {self.hparams.d_model}")
        print(f"- Number of Layers: {self.hparams.n_layers}")
        print(f"- State Dimension: {self.hparams.d_state}")
        if self.task_type in ['forecasting', 'multi_task']:
            print(f"- Forecast Horizon: {self.hparams.forecast_horizon}")
        print(f"- Gradient Checkpointing: {self.hparams.use_gradient_checkpointing}\n")