"""
Train an SSMForecaster from scratch on the pinball with the TimeJEPA stack.

    python scripts/train_ssm.py --config-name ssm_mini_v3 wandb.run_name=ssm-mini-v3

The datamodule block and the FinetuneModule kwargs are those of
TimeJEPA/scripts/train.py (finetune branch), key for key, so the corpus
(lotsa_v3, rationing, balanced sampling), the geometry randomization, the
schedule and the checkpointing are the champion's recipe. train.py is not an
importable module, so `apply_schedule_fraction` (8 lines) is recopied.
Additions: `training.delta_scales` / `training.p_delta_scale` (multi-rate
training on the Delta knob, SSMFinetuneModule) and `model.ssm.*`.
"""

import logging
import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timejepa.data.datamodule import MultiDatasetMonashDataModule  # noqa: E402
from timessm.model import build_from_config  # noqa: E402
from timessm.training import SSMFinetuneModule  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def apply_schedule_fraction(trainer_kwargs: dict, cfg) -> dict:
    """Copy of TimeJEPA/scripts/train.py: training.schedule_fraction < 1 -
    the run ends where the cosine ends (limit_train_batches), unless the
    config already limits it. Inert at 1.0."""
    frac = float(cfg.training.get("schedule_fraction", 1.0))
    if not (0.0 < frac <= 1.0):
        raise ValueError("training.schedule_fraction must be in (0, 1]")
    if frac < 1.0 and trainer_kwargs.get("limit_train_batches") is None:
        trainer_kwargs["limit_train_batches"] = frac
        logger.info(f"schedule_fraction={frac}: cosine annealed and run bounded at "
                    f"{frac:.0%} of the epoch")
    return trainer_kwargs


def build_datamodule(cfg: DictConfig) -> MultiDatasetMonashDataModule:
    """TimeJEPA/scripts/train.py datamodule block, finetune branch."""
    aug_root = cfg.get("augmentations") or {}
    aug_cfg = aug_root.get("finetune") if aug_root else None
    if aug_cfg is not None:
        aug_cfg = OmegaConf.to_container(aug_cfg, resolve=True)
        if not aug_cfg.get("enabled", True):
            aug_cfg = None
    logger.info("Augmentations (finetune): " + ("enabled" if aug_cfg else "disabled"))
    return MultiDatasetMonashDataModule(
        data_dir=cfg.data.data_dir,
        context_length=cfg.model.seq_length,
        prediction_length=cfg.model.prediction_length,
        datasets=cfg.data.get("datasets_finetune"),
        dataset_pattern=cfg.data.get("dataset_pattern", "*.npy"),
        combine_mode=cfg.data.get("combine_mode", "concatenate"),
        balanced_sampling=cfg.data.balanced_sampling,
        sampling_temperature=cfg.data.sampling_temperature,
        max_oversample_ratio=cfg.data.max_oversample_ratio,
        ration_oversample=bool(cfg.data.get("ration_oversample", False)),
        batch_size=cfg.data.batch_size,
        stride=cfg.data.stride,
        normalize_mode=cfg.data.normalize_mode,
        normalizer_type=cfg.data.normalizer_type,
        clip_outliers=cfg.data.clip_outliers,
        clip_sigma=cfg.data.clip_sigma,
        train_val_test_split=cfg.data.train_val_test_split,
        augmentation_config=aug_cfg,
        multi_resolution_factors=list(cfg.data.get("multi_resolution_factors_finetune") or [1]),
        p_multi_resolution=float(cfg.data.get("p_multi_resolution_finetune", 0.0)),
        cross_resolution=bool(cfg.model.get("cross_resolution", False)),
        short_series_windows=bool(cfg.data.get("short_series_windows", False)),
        short_min_context=int(cfg.data.get("short_min_context", 16)),
        short_min_target=int(cfg.data.get("short_min_target", 4)),
        seed=cfg.data.seed,
        num_workers=int(cfg.data.get("num_workers", 4)),
        persistent_workers=bool(cfg.data.get("persistent_workers", False)),
        use_mmap=bool(cfg.data.get("use_mmap", False)),
    )


def build_module(cfg: DictConfig, model) -> SSMFinetuneModule:
    """TimeJEPA/scripts/train.py FinetuneModule kwargs, key for key, plus the
    multi-rate keys. The JEPA-side terms (anchor, joint, critic, score) are
    read at their defaults: the SSM has no latent target to anchor on."""
    loss = cfg.training.loss
    if float(loss.get("lambda_anchor", 0.0)) or float(loss.get("lambda_joint", 0.0)) \
            or float(loss.get("lambda_score", 0.0)) or list(loss.get("critic_steps") or []):
        raise ValueError("anchor / joint / critic / score terms need a JEPA model; "
                         "SSMForecaster trains on the pinball alone")
    return SSMFinetuneModule(
        model=model,
        delta_scales=list(cfg.training.get("delta_scales") or []),
        p_delta_scale=float(cfg.training.get("p_delta_scale", 0.0)),
        extend_horizon_queries=cfg.training.get("extend_horizon_queries", False),
        pretrained_encoder_path=cfg.training.get("pretrained_encoder_path"),
        finetune_mode=cfg.training.get("finetune_mode", "full_finetune"),
        unfreeze_after_epoch=cfg.training.unfreeze_after_epoch,
        loss_type=loss.finetune_type,
        learning_rate=cfg.training.optimizer.learning_rate,
        weight_decay=cfg.training.optimizer.weight_decay,
        encoder_lr_multiplier=cfg.training.optimizer.get("encoder_lr_multiplier", 1.0),
        betas=tuple(cfg.training.optimizer.betas),
        warmup_epochs=cfg.training.lr_scheduler.warmup_epochs,
        max_epochs=cfg.training.max_epochs,
        schedule_fraction=float(cfg.training.get("schedule_fraction", 1.0)),
        lr_scheduler=cfg.training.lr_scheduler.type,
        min_lr=cfg.training.lr_scheduler.min_lr,
        dropout=cfg.training.dropout,
        context_lengths=list(cfg.training.get("context_lengths") or []),
        p_random_context_finetune=float(cfg.training.get("p_random_context_finetune", 0.0)),
        log_every_n_steps=cfg.training.log_every_n_steps,
    )


@hydra.main(version_base=None, config_path="../configs", config_name="ssm_mini_v3")
def main(cfg: DictConfig):
    logger.info("=" * 80)
    logger.info("CONFIGURATION")
    logger.info("=" * 80)
    logger.info("\n" + OmegaConf.to_yaml(cfg))
    pl.seed_everything(cfg.data.seed, workers=True)

    datamodule = build_datamodule(cfg)
    datamodule.prepare_data()

    model = build_from_config(cfg)
    parts = model.get_num_params()
    logger.info(f"SSMForecaster {cfg.model.name}: {parts['total']:,} parameters "
                f"(blocks {parts['blocks']:,}, decoder {parts['decoder']:,})")
    pl_module = build_module(cfg, model)
    logger.info(f"Multi-rate training: delta_scales={pl_module.delta_scales} "
                f"p={pl_module.p_delta_scale}")

    checkpoint_dir = Path(cfg.data.checkpoint_dir) / cfg.model.name / "pretrain_False"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor=cfg.checkpoint.monitor,
        mode=cfg.checkpoint.mode,
        save_top_k=cfg.checkpoint.save_top_k,
        save_last=cfg.checkpoint.save_last,
        filename=cfg.checkpoint.filename,
        auto_insert_metric_name=bool(cfg.checkpoint.get("auto_insert_metric_name", False)),
        verbose=True,
    )]
    if cfg.early_stopping.enabled:
        callbacks.append(EarlyStopping(
            monitor=cfg.early_stopping.monitor, patience=cfg.early_stopping.patience,
            mode=cfg.early_stopping.mode, min_delta=cfg.early_stopping.min_delta,
            verbose=True))
    callbacks.append(LearningRateMonitor(logging_interval="step"))

    wandb_logger = WandbLogger(
        project=cfg.wandb.project, entity=cfg.wandb.entity,
        name=cfg.wandb.run_name or cfg.model.name, tags=cfg.wandb.tags,
        config=OmegaConf.to_container(cfg, resolve=True), log_model=cfg.wandb.log_model,
    )
    trainer_kwargs = dict(
        accelerator=cfg.trainer.accelerator, logger=wandb_logger,
        devices=cfg.trainer.devices, precision=cfg.trainer.precision,
        max_epochs=cfg.trainer.max_epochs,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        val_check_interval=cfg.trainer.val_check_interval,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        callbacks=callbacks, default_root_dir=cfg.data.output_dir,
        deterministic=cfg.trainer.deterministic, strategy=cfg.trainer.strategy,
        use_distributed_sampler=cfg.trainer.use_distributed_sampler,
    )
    for key in ("limit_train_batches", "limit_val_batches"):
        if cfg.trainer.get(key) is not None:
            trainer_kwargs[key] = cfg.trainer.get(key)
    apply_schedule_fraction(trainer_kwargs, cfg)
    trainer = pl.Trainer(**trainer_kwargs)

    logger.info("STARTING TRAINING")
    trainer.fit(pl_module, datamodule=datamodule,
                ckpt_path=cfg.training.get("resume_ckpt", None))
    logger.info("TRAINING COMPLETE")
    logger.info(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
