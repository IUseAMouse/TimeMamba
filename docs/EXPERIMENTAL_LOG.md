# Registre expérimental TimeSSM

Newest-first. Même doctrine que `../TimeJEPA/docs/EXPERIMENTAL_LOG.md` : prédictions
gravées avant chaque run, une variable par bras, oracle = diagnostic jamais officiel.

## Journal des mises à jour

- **2026-09-09 (TimeSSM SPIKE — CODE LIVRÉ, NON COURU ; P-SSM.0 TENUE : l'équivariance
  au rythme passe à 1e-8)** — Branche `timessm` de TimeMamba, dépendance éditable sur
  TimeJEPA (aucune copie de code hors les 8 lignes d'`apply_schedule_fraction` et le bloc
  datamodule de `train.py`, non importables). Livré : `S4DLayer` (diagonal, ZOH exact,
  noyau Vandermonde + convolution FFT, récurrence `step`, `delta_scale` par batch ou par
  item, `Re(a) = −exp(·) < 0` par construction), `GatedSSMBlock` (sans conv depthwise par
  défaut : une conv est un filtre en pas, pas en temps physique — le test montre qu'elle
  casse l'équivalence décimation ≡ Δ/k ; `selective_readout` en ablation, porte sur la
  LECTURE seulement), `SSMForecaster` duck-typé JEPATST (RobustScale + RevIN + un token par
  pas + 6 blocs + tête quantile 1536 cross-attentive ; rollout autonome par un token futur
  appris, horizon libre 8-900 sans boucle ; `forecast(x, n, w)` avec `w` = facteur de Δ ;
  `rate_knob = 'delta'`, `core_prefixes` pour le chargeur), `SSMFinetuneModule` (module
  finetune de TimeJEPA + tirage d'un facteur Δ ∈ {0.25, 0.5, 1, 2, 4} avec p 0.5 par batch
  d'entraînement, cible inchangée, témoins `aug/delta_scale`), `train_ssm.py`, config
  `ssm_mini_v3` (recette du champion résolue et copiée ; déviations déclarées : batch 128 ×
  acc. 3, LR unique, multi-rythme, pas de conv), `eval_ssm.sh`. Côté TimeJEPA :
  `+ratein=delta` (même sélecteur backtest, contexte natif, `w = 1/k`, fan à l'horizon
  natif, diag `knob`), gardes `check_model_flags`, `model.builder` dans
  `create_model_from_config`, `core_prefixes` dans `load_checkpoint`. **Tests** (29 verts
  ici, 4 nouveaux dans TimeJEPA, anciens tests Mamba 20 verts sous l'extra `legacy`) :
  équivariance sur entrée constante par blocs (état et sortie, couche et bloc, 1e-8 en
  float64), écart croissant en k sur une sinusoïde, FFT == récurrence à L = 1024, |Ā| < 1
  et gradient fini à 1024 (l'échec du scan de 2025), `w` par item == boucle, contrat
  `forecast` à n ∈ {8, 128, 256, 900}, FinetuneModule (loss finie, gradients jusqu'aux
  blocs, groupes encoder/decoder), taille 2-4M (mesuré : 3.0M dont 0.77M de tête), harnais
  GIFT en off / flip / mix-pool / backtest / delta, aller-retour checkpoint par le chargeur
  TimeJEPA. **Smoke** : train 20 pas CPU sur corpus factice (loss 0.83 → val 0.734) ; éval
  du checkpoint smoke (46k paramètres, non entraîné) sur m_dense/D/short : off = delta à
  K = 1 bit-identiques (1.1638 / 0.1417), backtest choisit k = 6 par décimation, delta
  refuse tous les k (ratios 1.01-1.19 : le bouton d'un modèle non entraîné au multi-rythme
  ne transfère pas — c'est précisément ce que le run doit trancher). Coût du backtest à
  11 candidats : ~2× celui de TimeJEPA (contexte natif à chaque k au lieu de L/k).

  **Prédictions gravées** (checkpoint 5 % = 1/6 du run borné à 30 %) :
  - **P-SSM.0** (tenue le jour 1) : le test d'équivariance passe ; sinon rien ne se lance.
  - **P-SSM.1** (stack flip + mix + pool, décimation, modèle-agnostique) : à 5 %, CRPS
    entre 0.545 et 0.575 (head8 5 % standard 0.5585, S4-c 0.5506) ; > 0.60 ⇒ architecture
    ou recette à revoir avant toute question de rythme.
  - **P-SSM.2** (le verdict) : sur le MÊME checkpoint, `+ratein=delta` ≥ `+ratein=backtest`
    d'au moins 0.3 pt de CRPS, et delta sur les 32 configs > 5 % de l'oracle-k capture
    ≥ 50 % du gain oracle (RateIN classique : 37 %). ÉCHEC-DIAGNOSTIC si delta ≤
    décimation : le bouton est exact mais la lecture/tête n'est pas invariante ⇒ vérifier
    `aug/delta_scale`, second bras `training.p_delta_scale=1.0`.
  - **P-SSM.3** (ablation, seulement si P-SSM.1 tient) : `model.ssm.selective_readout=true`
    ± 0.3 pt à 5 %.

  Commandes : runbook. À lancer par l'utilisateur après le verdict anneal-30 et xres.
