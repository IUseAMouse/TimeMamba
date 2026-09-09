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
config, comme TimeJEPA). Tous les checkpoints d'un run :
`../TimeJEPA/scripts/eval_checkpoints.sh` attend une config TimeJEPA ; utiliser
une boucle sur `scripts/eval_ssm.sh` (ordre `ls -tr`).

Refus attendus : `+ratein_w` (pas de FiLM), `+refine`, `+ttt` (pas d'encodeur
JEPA), `+ratein=delta` sur un modèle sans `rate_knob`.

## Doctrine

Une variable par bras, prédictions gravées dans `docs/EXPERIMENTAL_LOG.md`
avant le lancement, rien de réglé sur le test GIFT (oracle = diagnostic),
jamais de suppression de fichier, code et commentaires en anglais, docs en
français.
