# DEPRECATED (2026-09) - see src/timessm/. This file is the March 2025 Mamba
# attempt, kept as is (no-delete policy): its "selective" SSM is a static LTI
# scan whose divide-by-A^i recurrence is unstable at long lengths. It runs on
# the `legacy` extra (uv sync --extra legacy).
import lightning.pytorch as pl
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from src.models.mamba import MambaForecastingModel
from src.models.data_module import TSFDataModule  

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import resource
soft, hard = resource.getrlimit(resource.RLIMIT_AS)
resource.setrlimit(resource.RLIMIT_AS, (6 * 1024 * 1024 * 1024, hard))

mlf_logger = MLFlowLogger(
    experiment_name="mamba_forecasting",
    tracking_uri="mlruns",
    run_name="mamba_forecast_run"
)

data_module = TSFDataModule(
    data_dir="src/dataset/",
    seq_length=60,
    forecast_horizon=24,
    batch_size=3,  
    stride=60,     
    normalize=True,
    train_ratio=0.7,
    val_ratio=0.15,
    file_pattern="*.tsf",
    verbose=True
)

# Initialize model
model = MambaForecastingModel(
    input_size=1,
    output_size=1,
    d_model=128,
    n_layers=4,
    d_state=16,
    forecast_horizon=24,
    learning_rate=1e-4
)

trainer = pl.Trainer(
    max_epochs=100,
    accelerator="auto",
    devices=1,
    accumulate_grad_batches=4,
    logger=mlf_logger,
    callbacks=[
        ModelCheckpoint(dirpath="checkpoints/", filename="mamba-{epoch:02d}-{val_loss:.4f}", 
                       monitor="val_loss", save_top_k=3),
        EarlyStopping(monitor="val_loss", patience=10),
        LearningRateMonitor()
    ],
    log_every_n_steps=5
)


print("Starting training...")
trainer.fit(model, data_module)

results = trainer.test(model, data_module)
print(f"Test results: {results}")