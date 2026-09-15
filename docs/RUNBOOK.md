# Runbook TimeSSM

## Mise en place

```bash
cd TimeMamba && git checkout timessm
uv sync                       # ../TimeJEPA doit exister (dépendance éditable)
uv run pytest -q              # 29 tests ; les anciens tests Mamba : uv sync --extra legacy
```

Sur le pod : même pile que TimeJEPA (torch cu128, voir le piège CUDA dans
`../TimeJEPA/PLAN.md`). Les données sont lues dans `../TimeJEPA/data/`
(corpus `processed/lotsa_v3`, GIFT `gift_eval`).

## Entraînement (spike, 1 jour GPU)

```bash
python scripts/train_ssm.py --config-name ssm_mini_v3 wandb.run_name=ssm-mini-v3
```

Recette du champion copiée (voir l'en-tête de la config) ; `schedule_fraction: 0.3`
borne le run à 30 % de l'époque ; checkpoints tous les 5 % dans
`checkpoints/timessm_mini_v3_zs/pretrain_False/`. Témoins W&B à vérifier dans les
premières minutes : `aug/delta_scale` (les cinq valeurs), `aug/delta_neq1_frac`
(≈ 0.5), `geometry/context_len`.

Mémoire : batch 128 × accumulation 3 (1152 effectif). Si OOM :
`data.batch_size=64 trainer.accumulate_grad_batches=6`.

## Évaluation

```bash
CK=checkpoints/timessm_mini_v3_zs/pretrain_False/<ckpt>
scripts/eval_ssm.sh $CK +tta_flip=true +ratein=mix +ratein_pool=true   # stack officiel (P-SSM.1)
scripts/eval_ssm.sh $CK +ratein=backtest +ratein_pool=true             # décimation, sélection dure
scripts/eval_ssm.sh $CK +ratein=delta +ratein_pool=true                # le bouton (P-SSM.2)
scripts/eval_ssm.sh $CK +ratein=oracle                                  # table oracle-k (diagnostic)
```

Résultats dans `evaluation/timessm_mini_v3_zs/<ckpt>/gift<tag>/` (cache par
config, comme TimeJEPA). Tous les checkpoints d'un run, en série, digest et
table dans `logs/` :

```bash
scripts/eval_checkpoints_ssm.sh checkpoints/timessm_mini_v3_zs/pretrain_False            # stack officiel
STACK="+ratein=backtest +ratein_pool=true" scripts/eval_checkpoints_ssm.sh <dir>       # décimation dure
STACK="+ratein=delta +ratein_pool=true" scripts/eval_checkpoints_ssm.sh <dir>          # le bouton
scripts/eval_checkpoints_ssm.sh <dir> +gift_batch_size=8                                # pendant un run
```

## Bras à plage large (reprise du meilleur checkpoint, ~12 h)

```bash
python scripts/train_ssm.py --config-name ssm_mini_v3_wide \
    '+training.pretrained_encoder_path="checkpoints/timessm_mini_v3_zs/pretrain_False/<best>.ckpt"' \
    wandb.run_name=ssm-mini-v3-wide 2>&1 | tee logs/train_ssm_wide.log
STACK="+ratein=delta +ratein_pool=true" scripts/eval_checkpoints_ssm.sh checkpoints/timessm_mini_v3_wide_zs/pretrain_False
STACK="+ratein=backtest +ratein_pool=true" scripts/eval_checkpoints_ssm.sh checkpoints/timessm_mini_v3_wide_zs/pretrain_False
```

Refus attendus : `+ratein_w` (pas de FiLM), `+refine`, `+ttt` (pas d'encodeur
JEPA), `+ratein=delta` sur un modèle sans `rate_knob`.

## Scaling : FSDP et checkpointing d'activations (2026-09-13)

```bash
# DDP + recompute des blocs en backward (la mémoire est dans les activations, pas les poids)
python scripts/train_ssm.py --config-name ssm_mini_v3 model.ssm.activation_checkpointing=true ...
# FSDP (poids, gradients, états Adam shardés ; un unit par bloc ; checkpoints en un fichier)
python scripts/train_ssm.py --config-name ssm_mini_v3 trainer.strategy=fsdp model.ssm.activation_checkpointing=true ...
```

À 2.5-20M, DDP suffit (Adam fp32 = 16 octets/paramètre, 320 Mo à 20M) ; FSDP est là pour
au-delà, et n'est revendiqué qu'après un vrai run multi-GPU. Le checkpointing coûte ~30 % de
calcul et divise la mémoire d'activations par le nombre de blocs. Pod 8× 5090 : 15 vCPU pour
8 processus, mettre `data.num_workers=1`.

## Machine louée (8 GPU) : préflight puis run 10M en DDP (2026-09-14)

```bash
# 0. corpus : network volume attaché, ou copie déréférencée depuis l'ancien pod
tar -C /workspace/TimeJEPA/data/processed -chf - lotsa_v3 | ssh <pod> 'tar -C /workspace/TimeJEPA/data/processed -xf -'
# 1. tout ce qui doit être vert avant de payer (10-15 min) : tests, audit corpus, profil 1 pas, smoke DDP 30 pas + rechargement
scripts/preflight.sh                      # CONFIG=ssm_mid_v3 par défaut (DEVICES=3 sur le pod 3090)
# 2. le run (P-SSM.4 gravée dans la config)
python scripts/train_ssm.py --config-name ssm_mid_v3 wandb.run_name=ssm-mid-v3 2>&1 | tee logs/train_ssm_mid.log
# défaut = pod 3x3090 (batch 64 x acc 6) ; pod 8 GPU : data.batch_size=48 trainer.accumulate_grad_batches=3 data.num_workers=2
# budget en FENÊTRES : l'époque du sampler dépend du batch (voir l'en-tête de la config) ; schedule_fraction 0.05 à batch 64 = 298M fenêtres
# l'allocateur tourne en expandable_segments (train_ssm.py) : un OOM à 18 Gio alloués + 4.6 Gio réservés était de la fragmentation
# 3. évals sur un GPU pendant le run
STACK="+tta_flip=true +ratein=mix +ratein_pool=true" EVAL_CONFIG=ssm_mid_v3_eval scripts/eval_checkpoints_ssm.sh checkpoints/timessm_mid_v3_zs/pretrain_False +gift_batch_size=32
```

DDP est le chemin validé ; FSDP (`trainer.strategy=fsdp`) n'a jamais tourné en multi-GPU : un smoke
de 15 min sur le pod loué avant toute mention, jamais comme chemin principal.

## Doctrine

Une variable par bras, prédictions gravées dans `docs/EXPERIMENTAL_LOG.md`
avant le lancement, rien de réglé sur le test GIFT (oracle = diagnostic),
jamais de suppression de fichier, code et commentaires en anglais, docs en
français.
