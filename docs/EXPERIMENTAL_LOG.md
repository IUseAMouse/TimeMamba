# Registre expérimental TimeSSM

Newest-first. Même doctrine que `../TimeJEPA/docs/EXPERIMENTAL_LOG.md` : prédictions
gravées avant chaque run, une variable par bras, oracle = diagnostic jamais officiel.

## Journal des mises à jour

- **2026-09-13 (P-SSM.2b NE SE RÉPLIQUE PAS sur 1.2850 : hybride k ≤ 4 0.7920 / 0.5471 / couv.
  0.725 contre backtest dur 0.7901 / 0.5475 / 0.720 — 0.04 pt, bruit)** — Bilan sur deux
  checkpoints : +0.47 pt (1.2942), +0.04 pt (1.2850). Le bouton sur sa plage entraînée n'est
  pas un gain établi : match nul avec la décimation. Même dessin par config (gains à k ≤ 4
  par le bouton : electricity/15T/short 0.103 vs 0.112, ett1/15T/long, m_dense/H/medium ;
  pertes quand l'hybride glisse vers une grosse décimation : bizitobs_service/short k32,
  saugeen/D k24, solar/10T/medium k8 0.369 vs 0.326, ett1/H/long k6). Le checkpoint plus
  entraîné change de cadence moins souvent (35/97 contre 41/97) : le gain du rythme
  décroît avec l'entraînement, bouton comme décimation. À conditions identiques, 1.2942
  reste devant 1.2850 (0.5445 contre 0.5475 en backtest dur) : les 0.7689 de MASE du stack
  sur 1.2850 viennent des couches, pas du checkpoint ; champion inchangé. Wide est le
  dernier test du bouton (P-SSM.2c gravée) ; en cas d'échec, la ligne du papier est un
  résultat négatif propre : le corps LTI tient, le bouton ne bat pas la décimation.

- **2026-09-13 (P-SSM.2b TIENT sur le champion 1.2942 : hybride bouton k ≤ 4 + décimation
  au-delà 0.7906 / 0.5398 / couv. 0.765 contre backtest dur 0.7892 / 0.5445 / 0.768, dur, sans
  flip, 97 configs : CRPS −0.47 pt (seuil 0.3), MASE +0.14 dans le bruit)** — Sur sa plage
  entraînée [1/4, 4], le bouton bat la décimation. Par config : gains là où le sélecteur
  prend le bouton à k ≤ 4 (m_dense/H/medium k4 vs 2 : 0.129 vs 0.147 ; us_births/D k3 vs 1 ;
  jena/H/long k3 vs 2 ; electricity/15T/short k4 vs 4 : 0.099 vs 0.109 ; solar/H/short k3
  vs 2), pertes là où l'hybride glisse vers une grosse décimation que le backtest pur ne
  prenait pas (solar/10T/medium k12 vs 3 : 0.376 vs 0.329 ; kdd/H/long k8 vs 3 ; bizitobs_l2c
  k6-8 vs 2-4) — la table mixte bouton / décimation déplace le winner's curse. Le stack
  flip + mix vaut 1.6 pt sur ce checkpoint (0.5445 → 0.5282), même ordre que sur TimeJEPA.
  Décision : pas de mode supplémentaire dans le harnais ; wide (bouton sur toute la plage,
  plus de candidats décimés) est le test propre. Second checkpoint (1.2850) en cours.

- **2026-09-13 (TABLE COMPLÈTE À 97 SUR 11 CHECKPOINTS, 5 % à 55 % du run : PLATEAU à
  0.528-0.533 depuis 25 % ; champion inchangé `1.2942` ; décision de couper le run)** —
  Stack flip + mix + pool : 5 % 0.5761 · 10 % 0.5419 · 15 % 0.5305 · 20 % 0.5334 · 25 %
  **0.5282** (0.7717, couv. 0.805) · 30 % 0.5301 · 35 % 0.5311 · 40 % 0.5292 · 45 % 0.5296 ·
  50 % 0.5278 (0.7762, couv. 0.719) · 55 % 0.5289 (0.7689, couv. 0.729). Huit checkpoints
  consécutifs sous ou au niveau de head8 (0.5340). Le 50 % gagne 0.04 pt de CRPS sur le
  25 % et lui perd 0.45 pt de MASE et 8.6 pt de couverture : même bande, le 25 % reste le
  champion (seul à gagner sur les trois axes). Couverture en dérive descendante depuis 25 %
  (0.805 → 0.66-0.73, la pinball resserre le fan), à traiter au papier comme limite
  partagée avec le transformer. Les 45 % restants (2.3 j) sont la fin du cosinus, sans gain
  attendu (anneal-30) : run coupé, GPU au bras wide-Δ depuis le dernier checkpoint (1.2850,
  le plus entraîné), P-SSM.2c gravée plus haut.

- **2026-09-13 (PRÉPARATION DU SCALING : checkpointing d'activations et option FSDP, code
  livré, non couru sur GPU)** — Décision utilisateur : après le verdict wide et un second
  seed, scaler TimeSSM sur un pod loué (8× RTX 5090, ~1 $/h/GPU). À un token par pas, la
  mémoire est dans les activations (batch × 1280 × 2·d_model × blocs), pas dans les poids :
  `model.ssm.activation_checkpointing` recompute chaque bloc en backward (train seulement,
  inerte en eval ; test : mêmes sorties et mêmes gradients en float64). `trainer.strategy:
  fsdp` construit une `FSDPStrategy` (FULL_SHARD, un unit par `GatedSSMBlock`, checkpoints
  en un fichier pour le harnais, policy de checkpointing si demandée) ; test de
  construction seulement, la validation multi-GPU se fait sur le pod avant toute mention
  au CV. Budget chiffré (registre TimeJEPA du jour) : un 10M à exposition du champion SSM
  ≈ 15 h ≈ 120 $ sur 8× 5090, un 20M le double ; un 100M à un token par pas n'entre pas
  dans 1 000 $ (≈ 5 j de 8× H100) sans patcher, ce qui casserait le bouton Δ. Corpus v3
  reconstructible par `../TimeJEPA/scripts/build_corpus_v3.sh` (révisions HF épinglées,
  audit 106 fichiers / 15.85 Md).

- **2026-09-13 (CORRECTION D'ÉCHELLE : les « % » des checkpoints SSM sont des % du RUN borné à
  30 %, pas de l'époque ; le champion « 25 % » est à 7.5 % de l'époque, 223M fenêtres, contre
  746M pour head8 à 25 % de l'époque)** — L'époque du sampler vaut 2.98 Md de fenêtres quel
  que soit le batch (7.76M × 128 ou 2.59M × 384 par GPU) ; le run SSM est borné à 30 % de
  l'époque (894M fenêtres, = anneal-30) et ses checkpoints tombent tous les 5 % du run (1.5 %
  de l'époque). Table compute : head8 25 % époque 746M fenêtres / 1.4 j GPU (0.5340) ; head8
  5 % époque 149M / 0.3 j (0.5585) ; anneal-30 30 % époque 894M / 1.7 j (0.5375) ; SSM 6 %
  époque 179M / 1.0 j (0.5334) ; SSM 7.5 % époque 223M / 1.3 j (0.5282, champion) ; SSM 9 %
  époque 268M / 1.6 j (0.5301, 2e sous head8) ; SSM fin de run 30 % époque 894M / 5.2 j. Le
  SSM coûte 3× par fenêtre (2000 contre 6100 fenêtres/s sur 3 GPU) : à temps GPU égal il bat
  le champion, et avec 3.3× moins de données. Conséquence pour la coupe : le « pic à 25 % puis
  dérive » de TimeJEPA (G7.3c) est à 25 % de l'ÉPOQUE, où le SSM n'est pas encore (LR à 80 %
  du pic) — laisser courir au moins jusqu'à 15-20 % de l'époque avant de décider. Le stack
  30 %-du-run (1.2936-v1) : 0.7760 / 0.5301 / couv. 0.693 sur 97, deuxième checkpoint sous
  head8 ; couverture en baisse (0.805 → 0.693), à suivre.

- **2026-09-13 (NOUVEAU CHAMPION : `epoch00_valloss1.2942` (25 % du run), stack flip + mix +
  pool, 0.7717 / 0.5282 / couverture 0.805 sur 97 configs, contre head8 0.7842 / 0.5340 /
  0.756)** — Table à 97 : 5 % 0.5761, 10 % 0.5419, 15 % 0.5305, 20 % 0.5334, 25 % 0.5282.
  Trois checkpoints consécutifs à ou sous le champion ; le 25 % gagne sur les trois axes,
  MASE −1.25 pt, CRPS −0.58 pt (au bord de la bande de bruit, 0.6 pt), couverture +4.9 pt —
  c'est la MASE et la couverture qui sortent le verdict du doute, pas le CRPS seul. Un
  scratch LTI de 2.5M, sans pretrain, sans attention dans le corps, sans conv, à un quart de
  son budget, à inférence strictement identique (RateIN par décimation, sans le bouton).
  Lecture : sur ce corpus le prior « état continu + horloge » vaut plus que l'attention ; le
  pretrain JEPA (≈ 1 pt) n'était pas ce qui manquait. Réserves : 30-45 % encore sur des
  sous-ensembles (71-74 configs, OOM d'une éval parallèle), à compléter ; couverture à
  suivre (0.679-0.720 sur les sous-ensembles suivants). Suite : passe complète 30-45 %,
  série delta-k4 par checkpoint, bras wide depuis le meilleur.

- **2026-09-12 (TABLE À 97 SUR LES PREMIERS CHECKPOINTS : P-SSM.1 TIENT, égalité avec le
  champion à 15-20 % du budget ; bras à plage large prêt, non lancé)** — Stack officiel
  (flip + mix + pool, décimation), GPU libres, batch 64, 97 configs : 5 % 0.8560 / 0.5761 /
  couv. 0.763 ; 10 % 0.7948 / 0.5419 / 0.760 ; 15 % 0.7757 / 0.5305 / 0.730 ; 20 % 0.7839 /
  0.5334 / 0.732 ; champion head8 0.7842 / 0.5340 / 0.756. P-SSM.1 (bande 0.545-0.575 à 5 %) :
  0.5761 à 5 % est au bord, avec le 5 % au milieu du warmup (LR 1.5e-4) ; à 10 % le modèle est
  sous la bande. Le 15 % est le meilleur chiffre jamais mesuré sur ce projet, MASE comprise,
  mais à 0.35 pt du champion il est dans la bande de bruit checkpoint à checkpoint mesurée
  sur anneal-30 (0.5352-0.5412) : pas un nouveau champion par notre propre règle ; couverture
  plus basse (0.730). Pente aplatie entre 15 et 20 % ; val_loss en plateau 1.289-1.295 depuis
  20 %. LEÇON D'INSTRUMENT : les tables de la veille mélangeaient des sous-ensembles de 55 à
  70 configs (OOM pendant le run, les lourdes manquantes) et donnaient 0.508-0.524 — jamais
  comparables ; `eval_checkpoints_ssm.sh` affiche désormais n_cfg et cached/computed/failed.
  Budget réel : le run borné à 30 % de l'époque = 2.33M batchs par GPU à 5.2 it/s = 5.2 jours
  (pas 2.7, erreur de ma part au lancement), checkpoint tous les 6 h ; à 42 % du run le 12/09.
  **Décision à prendre** : couper au meilleur checkpoint et lancer le bras à plage large, ou
  laisser finir (3 jours). **Bras à plage large** (`configs/ssm_mini_v3_wide.yaml`) : reprise
  des POIDS du meilleur checkpoint (`pretrained_encoder_path`, optimiseur et cosinus neufs),
  `delta_scales` = les 13 valeurs de K_CANDIDATES de 1/48 à 4, p 0.7, LR 1e-4, budget 3 % de
  l'époque (10 % du spike, ~12 h), 5 checkpoints ; smoke CPU passé (52 clés chargées).
  **P-SSM.2c** (gravée) : sur le checkpoint wide, `+ratein=delta` sans max_k ≥ `+ratein=backtest`
  de 0.3 pt et disparition des pertes à k ≥ 16 (bizitobs_service, bizitobs_l2c) ; ÉCHEC si
  delta ≤ backtest : le bouton ne transfère pas aux grands k même entraîné, l'hybride reste
  la couche officielle.

- **2026-09-11 (P-SSM.2 ÉCHOUE sur le checkpoint 1.3022 : delta 0.8276 / 0.5558 contre
  backtest 0.7953 / 0.5409, dur, sans flip, 97 configs ; DIAGNOSTIC MESURÉ : le bouton gagne
  sur sa plage entraînée (k ≤ 8) et perd au-delà (k ≥ 16))** — Run corrigé (schedule),
  checkpoint ~10 % du budget (val 1.3022). Sélecteur identique des deux côtés, pooling CRPS,
  marge 5 %. Delta active le bouton sur 28 configs, la décimation sur 32. Par config
  (`compare_subset.py`) : à k égal 16 (bizitobs_service ×3, bizitobs_application/long) delta
  perd 11-23 % — à w = 1/16 le bouton est moins bon que la décimation par 16 ; là où delta
  choisit un k énorme (sz_taxi 48, kdd/D 24, bizitobs_l2c/medium 32) le backtest accepte un w
  que le test punit (winner's curse hors distribution) ; à k ≤ 8 des deux côtés delta gagne
  (bitbrains_fast_storage/5T/long 0.606 contre 0.898, /medium 0.642 contre 0.691, bitbrains_rnd
  /5T/long, solar/H/short, ett2/H où delta refuse à raison un k = 6 que la décimation accepte à
  tort). Cause : `delta_scales` [0.25..4] n'entraîne le bouton que jusqu'à k = 4, le harnais
  demande w jusqu'à 1/48 ; à Δ/48 les constantes de temps sont 48× hors de tout ce que le
  modèle a vu, alors qu'un contexte décimé par 48 reste un contexte court à Δ, en
  distribution. Deux suites : (1) garde `+ratein_delta_max_k=N` dans le harnais (bouton
  jusqu'à N, décimation au-delà, même sélecteur ; livrée, test stub), à mesurer sur 1.3022 à
  N = 4 et 8 ; (2) second bras d'entraînement, `delta_scales` sur toute la plage demandée
  (1/48 à 4). Aussi : P-SSM.1 encore à lire sur 97 (stack complet sur 55 : 0.5045 contre
  0.5041 pour le champion sur les mêmes configs — égalité, pas domination ; les 42 manquantes
  sont les longues, tombées en OOM pendant le run). Vitesse : m4_monthly et
  temperature_rain dominent l'éval (48k et 96k instances), batch 64 puis 8.

  **P-SSM.2b** (hybride, même checkpoint) : `+ratein=delta +ratein_delta_max_k=4` bat
  `+ratein=backtest` d'au moins 0.3 pt de CRPS ; N = 8 dans le bruit de N = 4. ÉCHEC si
  l'hybride ≤ backtest : le bouton n'apporte rien même sur sa plage, et la décimation
  reste la couche officielle.

- **2026-09-10 (PREMIER RUN INVALIDE : schedule étiré ×3 par l'accumulation ; correctifs
  bf16 et vitesse ; run à relancer)** — Trois problèmes rencontrés au lancement sur le pod,
  tous corrigés et poussés : (1) `bf16-mixed` plante dans la FFT (cuFFT sans noyau
  bfloat16, autocast ne convertit pas les FFT) → convolution en float32 sous autocast
  (`4ed7a73`) ; (2) 2.7 it/s : un `w` uniforme par batch était traité comme « un Δ par item »,
  noyau dupliqué 128× et 128 FFT du noyau par bloc → chemin à noyau unique (`e568a5d`),
  5.3 it/s ensuite ; (3) le scheduler de TimeJEPA comptait des batchs alors qu'il avance par
  pas d'optimiseur : avec `accumulate_grad_batches: 3`, warmup et cosinus ×3, TOUT le run
  borné à 30 % était du warmup (LR 1.5e-5 au checkpoint 5 %, val 1.33, courbe `lr-AdamW` à
  pente 1/3 du scratch head8) → corrigé côté TimeJEPA (`// accumulate`). Le checkpoint 5 %
  de ce run ne mesure pas l'architecture ; son éval (stack, 24 configs vues : bitbrains
  0.47-0.88, electricity/15T 0.086) n'est pas retenue. Vitesse : le SSM à un token par pas
  coûte ~6× TimeJEPA par échantillon (1280 tokens contre 127 patchs, FFT bornée par la
  mémoire) ; le verdict P-SSM.2 se lit au checkpoint 5 % (~6 h à 5.3 it/s), le run à 30 %
  (~5 j) n'est engagé que si P-SSM.1 tient. Prédictions P-SSM.0-3 inchangées.

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
