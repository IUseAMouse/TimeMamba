import torch
import torch.nn as nn
import pytorch_lightning as pl

class TemporalPyramidMoELightning(pl.LightningModule):
    def __init__(
        self,
        model_type='stacked',  # 'single' or 'stacked'
        num_blocks=3,
        sequence_len=384,
        forecast_len=96,
        hidden_dim=128,
        levels=4,
        base_window=16,
        num_experts=8,
        top_k=4,
        learning_rate=1e-4,
        dropout=0.1
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model parameters
        self.sequence_len = sequence_len
        self.forecast_len = forecast_len
        self.learning_rate = learning_rate
        
        # Initialize the correct model type
        if model_type == 'single':
            from models import TemporalPyramidMoE
            self.model = TemporalPyramidMoE(
                sequence_len=sequence_len,
                forecast_len=forecast_len,
                hidden_dim=hidden_dim,
                levels=levels,
                base_window=base_window,
                num_experts=num_experts,
                top_k=top_k,
                dropout=dropout
            )
        else:  # 'stacked'
            from models import StackedTPMoE
            self.model = StackedTPMoE(
                num_blocks=num_blocks,
                hidden_dim=hidden_dim,
                forecast_len=forecast_len,
                sequence_len=sequence_len,
                levels=levels,
                base_window=base_window,
                num_experts=num_experts,
                top_k=top_k,
                dropout=dropout
            )
        
    def forward(self, x):
        return self.model(x)
    
    def _compute_loss(self, batch, batch_idx):
        x, y = batch
        y_hat, routing_info = self(x)
        
        # Standard MSE loss
        mse_loss = F.mse_loss(y_hat, y)
        
        # Note: In a full implementation, we would add SoftDTW loss here
        # For simplicity, we're only using MSE in this example
        # soft_dtw_loss = compute_soft_dtw(y_hat, y)
        # total_loss = (1 - self.sdtw_weight) * mse_loss + self.sdtw_weight * soft_dtw_loss
        
        # For now, return only MSE
        total_loss = mse_loss
        
        # Add regularization for expert utilization if needed
        # expert_loss = compute_expert_balance_loss(routing_info)
        # total_loss = total_loss + expert_loss
        
        return total_loss, mse_loss, routing_info
    
    def training_step(self, batch, batch_idx):
        loss, mse, routing_info = self._compute_loss(batch, batch_idx)
        
        # Log metrics
        self.log('train_loss', loss)
        self.log('train_mse', mse)
        
        # Log expert utilization
        if 'local_routing' in routing_info:
            local_expert_usage = routing_info['local_routing']['expert_usage']
            global_expert_usage = routing_info['global_routing']['expert_usage']
            
            self.log('local_expert_usage', local_expert_usage.mean())
            self.log('global_expert_usage', global_expert_usage.mean())
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, mse, _ = self._compute_loss(batch, batch_idx)
        
        # Log metrics
        self.log('val_loss', loss)
        self.log('val_mse', mse)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, mse, _ = self._compute_loss(batch, batch_idx)
        
        # Log metrics
        self.log('test_loss', loss)
        self.log('test_mse', mse)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }