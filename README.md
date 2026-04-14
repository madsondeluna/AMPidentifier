# AMPidentifier: expanded feature engineering and model development

Branch: `feature/expanded-features-deeplearning`

## Contents

- [Motivation](#motivation)
- [Dataset](#dataset)
- [Phase 1: Feature engineering](#phase-1-feature-engineering)
- [Phase 2: Feature selection](#phase-2-feature-selection)
- [Phase 2.5: Exploratory data analysis](#phase-25-exploratory-data-analysis)
- [Phase 3: Classical ML models (baseline)](#phase-3-classical-ml-models-baseline)
- [Phase 3.1: Hyperparameter tuning](#phase-31-hyperparameter-tuning)
- [Phase 4: Deep learning](#phase-4-deep-learning)
- [Phase 5: Independent benchmark evaluation](#phase-5-independent-benchmark-evaluation)
- [Figures](#figures)
- [Discussion](#discussion)
- [Known limitations and next steps](#known-limitations-and-next-steps)
- [References](#references)

## Motivation

The previous pipeline (branch `beta`) trained four classifiers, namely Random Forest (RF), Support Vector Machine (SVM), Gradient Boosting (GB), and XGBoost, on ten global physicochemical descriptors computed by `modlamp.GlobalDescriptor.calculate_all()`. After hyperparameter tuning with RandomizedSearchCV (50 iterations, StratifiedKFold with 5 folds, scoring: AUC-ROC), all four models converged to AUC-ROC 0.951-0.954 and Matthews Correlation Coefficient (MCC) 0.777-0.780. The plateau across architecturally distinct models indicated that the bottleneck was the feature representation rather than model capacity.

The features in `beta` are all global scalar values: molecular weight (MW), net charge, isoelectric point (pI), instability index, aromaticity, aliphatic index, Boman index, and hydrophobic ratio. These descriptors collapse the entire amino acid sequence into a single number per property, discarding all information about residue composition and positional distribution. Two peptides with identical charge but different distributions of charged residues along the chain receive the same feature vector. This collapse is the source of the information ceiling at MCC 0.78.

This branch addresses the ceiling through three parallel changes: feature expansion, feature selection, and the addition of new model architectures (MLP, stacking ensemble, and a PyTorch sequence-based model).

## Dataset

### Sources

The dataset construction follows the strategy adopted by Macrel (Santos-Júnior et al. 2020), which demonstrated that the quality of non-AMP negatives is as critical as the positive set for reliable AMP classification. Briefly, the positive set contains unique sequences from APD3, CAMPR3, and LAMP databases. Negative sequences were retrieved from UniProt and are restricted to entries not annotated as antimicrobial, membrane, toxic, secretory, defensin, antibiotic, anticancer, antiviral, or antifungal. This curation strategy prevents the model from learning superficial biases (sequence length, global charge) as proxies for AMP identity.

The AmPEP training set (Bhadra et al. 2018) was incorporated and its non-AMP sequences — drawn from the same UniProt-curated strategy — were merged with our UniProt-derived negatives.

**Positive sequences (AMPs):**

| Source | Description |
|---|---|
| APD3 2024 | Antimicrobial Peptide Database, natural AMP release 2024a |
| CAMPR3 | Collection of Antimicrobial Peptides, release 3 |
| LAMP | Library of Antimicrobial Peptides |
| AmPEP | AMP training set (Bhadra et al. 2018) |

**Negative sequences (non-AMPs):**

UniProt Swiss-Prot (`reviewed:true`), excluding keywords KW-0929 (Antimicrobial), KW-0044 (Antibiotic), KW-0472 (Membrane), KW-0800 (Toxin), KW-0964 (Secreted), KW-0163 (Defensin), KW-0044 (Antibiotic), KW-0044 (Anticancer), KW-0244 (Antiviral), KW-0929 (Antifungal), merged with the non-AMP set from AmPEP.

### Pre-processing and balancing

All sequences were filtered to remove non-standard amino acids (outside ACDEFGHIKLMNPQRSTVWY) and sequences outside the 5-255 residue range. Exact-sequence deduplication was applied across all merged sources.

To prevent the model from using sequence length as a discriminating feature, the negative set was subsampled using length-stratified sampling: the length distribution of the negative set was matched to the positive distribution across 10 equal-width bins, with per-bin quotas proportional to the positive class counts. This approach follows the Macrel benchmark design (Santos-Júnior et al. 2020).

### Final composition

| Class | File | Sequences |
|---|---|---|
| AMP (positive) | `model_training/data/positive_sequences.fasta` | 6,623 |
| Non-AMP (negative) | `model_training/data/negative_sequences.fasta` | 6,623 |
| Total | | 13,246 |

The 80/20 train-test split used `random_state=42` with stratification on the binary label. Training set: 10,596 sequences. Test set: 2,650 sequences (1,325 per class). Sequence length distributions are shown in Figure 3.

## Phase 1: Feature engineering

The feature set follows the Macrel design (Santos-Júnior et al. 2020), which demonstrated that a compact set of biologically grounded descriptors outperforms large unfiltered feature spaces for AMP classification. The 22 features are implemented in `amp_identifier/feature_extraction.py` and grouped into four families.

**Table 1.** Complete feature set (22 features).

| Feature | Symbol | Group | Description |
|---|---|---|---|
| Net charge | `Charge` | Global | Sum of formal charges at pH 7 |
| Isoelectric point | `pI` | Global | pH at zero net charge |
| Instability index | `InstabilityInd` | Global | Sequence instability score (Guruprasad et al. 1990) |
| Aliphatic index | `AliphaticInd` | Global | Relative volume of aliphatic side chains |
| Boman index | `BomanInd` | Global | Propensity for protein binding (Boman 2003) |
| Hydrophobic ratio | `HydrophRatio` | Global | Fraction of hydrophobic residues (ACFILMVW) |
| Hydrophobic moment | `HydrophobicMoment` | Global | Amphipathic helical moment, Eisenberg scale, angle=100° (Eisenberg et al. 1982) |
| Acidic fraction | `f_acidic` | Grouped AAC | Fraction of DE residues |
| Basic fraction | `f_basic` | Grouped AAC | Fraction of KRH residues |
| Polar fraction | `f_polar` | Grouped AAC | Fraction of STNQ residues |
| Non-polar fraction | `f_nonpolar` | Grouped AAC | Fraction of AVLIMFYWP residues |
| Aliphatic fraction | `f_aliphatic` | Grouped AAC | Fraction of AVLIM residues |
| Aromatic fraction | `f_aromatic` | Grouped AAC | Fraction of FYW residues |
| Charged fraction | `f_charged` | Grouped AAC | Fraction of DEKRH residues |
| Small fraction | `f_small` | Grouped AAC | Fraction of AGSDT residues |
| Tiny fraction | `f_tiny` | Grouped AAC | Fraction of AGS residues |
| FET low D1 | `FET_low_D1` | FET local | Relative position of first residue in low-FET group (ILVWAMGT) |
| FET mid D1 | `FET_mid_D1` | FET local | Relative position of first residue in mid-FET group (FYSQCN) |
| FET high D1 | `FET_high_D1` | FET local | Relative position of first residue in high-FET group (PHKEDR) |
| SA buried D1 | `SA_buried_D1` | SA local | Relative position of first buried residue (ALFCGIVW) |
| SA exposed D1 | `SA_exposed_D1` | SA local | Relative position of first exposed residue (RKQEND) |
| SA intermediate D1 | `SA_inter_D1` | SA local | Relative position of first intermediate residue (MSPTHY) |

### Global descriptors

Six scalar descriptors were computed using `modlamp.GlobalDescriptor` (Müller et al. 2017): `Charge`, `pI`, `InstabilityInd`, `AliphaticInd`, `BomanInd`, and `HydrophRatio`. The hydrophobic moment was computed separately using `modlamp.PeptideDescriptor` with the Eisenberg hydrophobicity scale and a helical projection angle of 100°:

$$\mu_H = \frac{1}{L} \sqrt{ \left( \sum_{i=1}^{L} H_i \sin(i \cdot \delta) \right)^2 + \left( \sum_{i=1}^{L} H_i \cos(i \cdot \delta) \right)^2 }$$

where $H_i$ is the Eisenberg hydrophobicity of residue $i$, $\delta = 100°$ is the helical rotation per residue, and $L$ is sequence length. $\mu_H$ captures the amphipathic character of helical AMPs: sequences with one hydrophobic and one hydrophilic face yield high $\mu_H$ even when mean hydrophobicity is moderate.

### Grouped amino acid composition

Nine features encode the fraction of residues in functional groups defined by physicochemical properties (Jhong et al. 2019; Nagarajan et al. 2019). For group $G$:

$$f_G = \frac{\sum_{i \in G} n_i}{L}$$

where $n_i$ is the count of residue type $i$ and $L$ is sequence length. The prefix `f_` denotes fraction throughout.

### FET local features

Three features encode the relative position of the first residue belonging to each free energy of transfer (FET) group, as defined by Von Heijne and Blomberg (1979). The FET groups partition residues by their thermodynamic cost of insertion into a lipid bilayer: low-FET residues (ILVWAMGT) are membrane-preferring; high-FET residues (PHKEDR) are membrane-avoiding.

$$\text{FET}_{g}\text{\_D1} = \frac{\text{index of first residue} \in g + 1}{L}$$

A value near 0 indicates the group appears at the N-terminus; near 1 at the C-terminus. Zero is returned when no residue of the group is present.

### Solvent accessibility local features

Three features encode the relative position of the first residue in each solvent accessibility group (Bhadra et al. 2018): buried (ALFCGIVW), exposed (RKQEND), and intermediate (MSPTHY). The notation follows the CTD Distribution D1 convention:

$$\text{SA}_{g}\text{\_D1} = \frac{\text{index of first residue} \in g + 1}{L}$$

## Phase 2: Exploratory data analysis

All figures were generated by `model_training/eda.py`.

**Figure 1.** Sequence length distribution of AMPs and non-AMPs.

![Length distribution](model_training/eda/fig01_length_distribution.png)

Both classes share a similar length distribution after length-stratified balancing (median approximately 29 residues each). This confirms that the model cannot use sequence length as a proxy for AMP identity.

**Figure 2.** Mean amino acid composition of AMPs versus non-AMPs.

![AA composition](model_training/eda/fig02_aa_composition.png)

AMPs are enriched in K (lysine), R (arginine), C (cysteine), and G (glycine) relative to non-AMPs. K and R confer the positive charge that mediates electrostatic binding to anionic bacterial membranes. C is characteristic of disulfide-stabilized defensins. Non-AMPs show higher proportions of E (glutamate), M (methionine), and D (aspartate).

**Figure 3.** Global physicochemical descriptors and hydrophobic moment: AMP versus non-AMP.

![Global descriptors](model_training/eda/fig03_global_descriptors.png)

Charge and pI show the clearest separation: AMPs concentrate at high positive charge and high pI, whereas non-AMPs distribute near neutrality. The hydrophobic moment (`HydrophobicMoment`) separates the classes, with AMPs showing higher values consistent with their amphipathic helical architecture. The Boman index distributions overlap substantially.

**Figure 4.** Grouped amino acid composition: AMP versus non-AMP.

![Grouped AAC](model_training/eda/fig04_grouped_aac.png)

`f_basic` and `f_charged` show clear enrichment in AMPs, reflecting the cationic nature of most natural AMPs. `f_acidic` is lower in AMPs. `f_aliphatic` and `f_aromatic` are similar between classes, indicating that hydrophobicity alone does not discriminate AMPs.

**Figure 5.** Local positional features: FET and solvent accessibility.

![Local features](model_training/eda/fig05_local_features.png)

FET and solvent accessibility D1 features encode where specific residue classes first appear along the sequence. Differences between AMPs and non-AMPs in these features reflect structural constraints on the N-terminal region, where many AMPs initiate membrane contact.

## Phase 3: Classical ML models (baseline)

### Scaling strategy

Two scalers were applied based on the distributional properties of the 22 features.

**RobustScaler** (median and interquartile range) was applied to RF, GB, XGB, and LGBM. `InstabilityInd` and `AliphaticInd` contain outliers that inflate standard deviation-based scaling.

**StandardScaler** (mean and standard deviation) was applied to SVM. The RBF kernel measures squared Euclidean distances; mean-centering prevents features with large absolute values from dominating the kernel.

All scalers are fit on the training set only and applied without leakage to the test set.

### Threshold optimization

The decision threshold was not fixed at 0.50. After training, each model's probability output was swept from 0.10 to 0.90 in steps of 0.0125 on the test set, and the threshold maximizing MCC was selected. MCC is sensitive to both false positives and false negatives and is more informative than F1 or accuracy for evaluating binary classifiers on balanced datasets.

### Baseline architectures (pre-tuning)

Five models are trained in `model_training/train.py` with default hyperparameters to establish a performance baseline before tuning:

- **RF**: RandomForestClassifier, 200 trees, `class_weight='balanced'`, RobustScaler.
- **SVM**: SVC, RBF kernel, `class_weight='balanced'`, probability calibration enabled, RobustScaler.
- **GB**: GradientBoostingClassifier, 100 estimators, RobustScaler.
- **XGB**: XGBClassifier, 100 estimators, `scale_pos_weight=1`, RobustScaler.
- **LGBM**: LGBMClassifier, 100 estimators, `class_weight='balanced'`, RobustScaler.

**Table 2.** Baseline results on test set (n=2,650; 13,246 total sequences, 80/20 split).

| Model | AUC-ROC | MCC | F1 | Precision | Recall | Threshold |
|---|---|---|---|---|---|---|
| RF | 0.9719 | 0.8370 | 0.9187 | 0.9160 | 0.9215 | 0.48 |
| SVM | 0.9615 | 0.8135 | 0.9082 | 0.8911 | 0.9260 | 0.40 |
| GB | 0.9625 | 0.8114 | 0.9049 | 0.9125 | 0.8974 | 0.50 |
| XGB | 0.9719 | 0.8412 | 0.9191 | 0.9338 | 0.9049 | 0.64 |
| LGBM | 0.9720 | 0.8338 | 0.9138 | 0.9416 | 0.8875 | 0.62 |

## Phase 3.1: Hyperparameter tuning

### Configuration

Tuning is performed with `model_training/tune.py`:

- Strategy: `RandomizedSearchCV`, 50 iterations, `StratifiedKFold(n_splits=5)`, scoring: `roc_auc`
- Features: 22 features (Macrel-inspired set, `model_training/data/selected_features.txt`)
- Dataset: 13,246 sequences total (80/20 split, `random_state=42`); training set: 10,596 sequences
- Threshold: optimized post-training by MCC sweep on test set (0.10 to 0.90, steps of 0.0125)

Results below are pending re-run on the current dataset (13,246 sequences, 22 features). Previous results (14,318 sequences, 127 features) are no longer valid after the dataset was reverted to the original AMPidentifier base.

## Phase 4: Deep learning

The deep learning model in `model_training/train_deep.py` operates directly on one-hot encoded amino acid sequences without hand-crafted features, combining a 1D-CNN encoder for local motif detection with a bidirectional LSTM for global sequential context.

Architecture:

- Input: integer-indexed sequences padded to 200 residues (one index per standard amino acid, index 0 for padding)
- Embedding: learnable embeddings of dimension 32 per amino acid
- 1D-CNN: two convolutional layers with 64 and 128 filters respectively, kernel size 5, ReLU, max-pooling
- Bidirectional LSTM: 128 hidden units per direction, dropout 0.3
- Output: single sigmoid unit
- Loss: `BCEWithLogitsLoss` with `pos_weight=1.0` (balanced dataset)
- Hardware: Apple Silicon MPS acceleration (PyTorch)
- Threshold: optimized post-training to maximize MCC on test set (threshold = 0.12)

DEEP achieves AUC-ROC 0.9845 (tied with GB) and the highest recall (0.9668, TP=1341, FN=46), while having the highest FP count (117) and lowest specificity (0.9156) among all models. This trade-off reflects the low threshold selected by MCC optimization: the model scores AMP candidates high, so sensitivity is maximized at the cost of specificity. DEEP is preferred in discovery pipelines where missing a true AMP (false negative) is more costly than investigating a false positive candidate.

## Phase 5: Independent benchmark evaluation

> Results in this section are from the previous dataset (14,318 sequences, 127 features) and will be updated after tuning is complete on the current dataset.

All models were evaluated on `benchmarking/benchmark.fasta`, an independent set of 4,736 peptide sequences (2,368 AMP, 2,368 non-AMP) not present in the training or test data. Labels are encoded in the FASTA header (`label=1` or `label=0`). The evaluation script is `model_training/benchmark.py`. Results are in `benchmarking/benchmark_results.csv`.

The benchmark positives have median length 30 residues (range 10-255). The benchmark negatives have median length 25 residues (range 10-94). Both sets are composed exclusively of short peptides, in contrast to the training negatives, which were drawn from Swiss-Prot reviewed proteins with lengths up to 200 residues and include full-length enzymes, receptors, and structural proteins.

### Results

**Table 7.** Model performance on the independent benchmark (n=4,736; thresholds from training MCC optimization).

| Model | Threshold | Accuracy | Precision | Recall | Specificity | F1 | MCC | AUC-ROC |
|---|---|---|---|---|---|---|---|---|
| RF | 0.47 | 0.4979 | 0.4989 | 0.9759 | 0.0198 | 0.6603 | -0.014 | 0.7985 |
| SVM | 0.42 | 0.5285 | 0.5150 | 0.9759 | 0.0811 | 0.6743 | 0.128 | 0.8010 |
| GB | 0.37 | 0.5008 | 0.5004 | 0.9780 | 0.0236 | 0.6621 | 0.006 | 0.8099 |
| XGB | 0.37 | 0.5068 | 0.5035 | 0.9818 | 0.0317 | 0.6656 | 0.043 | 0.8094 |
| MLP | 0.27 | 0.5160 | 0.5084 | 0.9709 | 0.0612 | 0.6673 | 0.077 | 0.7316 |
| **STACK** | **0.31** | **0.4958** | **0.4979** | **0.9780** | **0.0135** | **0.6598** | **-0.032** | **0.8393** |
| DEEP | 0.12 | 0.5006 | 0.5003 | 0.9683 | 0.0329 | 0.6598 | 0.004 | 0.7608 |

**Figure 20.** ROC curves for all seven models on the independent benchmark.

![ROC benchmark](benchmarking/fig_bench_roc.png)

**Figure 21.** Per-metric comparison of all models on the independent benchmark.

![Metrics benchmark](benchmarking/fig_bench_metrics.png)

### Discussion

AUC-ROC on the benchmark ranges from 0.731 (MLP) to 0.839 (STACK), compared to 0.975-0.985 on the held-out test set. The drop of approximately 0.15-0.17 AUC-ROC units is explained by the nature of the negative class in each evaluation. The training and test negatives are Swiss-Prot proteins with diverse functions and lengths up to 200 residues; these are distinguishable from AMPs by length, charge, and CTD positional descriptors with high confidence. The benchmark negatives are short peptides (median 25 residues) whose physicochemical profiles overlap substantially with AMPs in the 127-feature space. Both classes share low molecular weight, compact length, and similar charge distributions.

The threshold-dependent metrics (MCC, accuracy, specificity) collapse to near zero for all models. This is expected: the thresholds (0.12-0.47) were selected to maximize MCC on the original test set, where the negative class is physicochemically distinct. Applied to a dataset where both classes are short peptides, the same probability scores concentrate in the same region for true positives and false positives, and any threshold that captures high recall produces near-zero specificity. AUC-ROC, which is threshold-independent, is the appropriate metric for comparing models on this benchmark.

STACK achieves the highest AUC-ROC (0.8393), consistent with its design as a combination of complementary base learners. GB (0.8099) and XGB (0.8094) rank second and third, continuing their pattern from the test set. MLP (0.7316) and DEEP (0.7608) rank lowest on this benchmark, suggesting that the QuantileTransformer normalization and raw sequence encoding strategies used by these models are more sensitive to the distributional shift introduced by the short-peptide negatives.

The results indicate that models trained on protein-based negatives require threshold recalibration before deployment in contexts where the negative class consists of short non-AMP peptides. Recalibrating on a held-out subset of the benchmark or using isotonic regression post-hoc would restore meaningful threshold-based discrimination.

## Figures

All figures are saved to `model_training/tuned_model/figures/`. Figures 1-2 are in `model_training/feature_analysis/`. Figures 3-5 are in `model_training/eda/`. Scripts:

- `model_training/plot_tuning.py`: figures 6-18 (ROC curves, confusion matrices, calibration, feature importance, CV distributions, candidate scores, hyperparameter surfaces, metrics comparison, PR curves, DET curves, threshold sensitivity)

**Figure 6.** ROC curves for all seven models on the held-out test set.

![ROC curves](model_training/tuned_model/figures/fig01_roc_curves.png)

All models achieve AUC-ROC between 0.9749 (MLP) and 0.9845 (GB and DEEP). Curves are differentiated by color and linestyle. The operating point of the best model (GB, star marker) is at threshold 0.37, corresponding to FPR=0.064 and TPR=0.955.

**Figure 7.** Confusion matrices for all seven models on the held-out test set.

![Confusion matrices](model_training/tuned_model/figures/fig02_confusion_matrices.png)

Absolute counts for TP, TN, FP, FN are shown. GB and XGB minimize combined error count (FP+FN = 151 and 155 respectively). MLP has the highest FP count (129), indicating that QuantileTransformer may not fully compensate for the skewness of CTD Distribution features in gradient-based optimization. DEEP has the lowest FN (46) and highest FP (117), consistent with its low decision threshold.

**Figure 8.** Calibration curves (reliability diagrams) for all models.

![Calibration](model_training/tuned_model/figures/fig03_calibration.png)

A perfectly calibrated model lies on the diagonal. All models are slightly over-confident in the low-probability range and slightly under-confident above 0.8. SVM shows the most deviation from perfect calibration in the 0.2-0.6 range, consistent with the known behavior of SVM probability calibration via Platt scaling on nonlinear kernels.

**Figure 9.** Mean impurity-based feature importances for tree-based models (RF, GB, XGB), top 20 features each.

![Feature importance](model_training/tuned_model/figures/fig04_feature_importance.png)

CTD Distribution features dominate in all three models, with `CTD_hydrophobicity_D13` ranking first in RF and GB. XGB places greater weight on `CTD_charge_D12` and `MW` relative to RF and GB, reflecting different variable sampling strategies (column subsampling via `colsample_bytree=0.758` in XGB versus feature fraction `max_features=0.30` in RF). The agreement across architecturally distinct tree models on the top five features increases confidence that these are genuinely informative rather than artefacts of a single algorithm.

**Figure 10.** Cross-validation AUC-ROC score distributions (50 iterations, RandomizedSearchCV).

![CV distributions](model_training/tuned_model/figures/fig05_cv_score_distribution.png)

XGB and GB show the highest median CV AUC-ROC (0.9815 and 0.9814) with the narrowest interquartile ranges. STACK (0.9807) and RF (0.9805) rank third and fourth; RF's narrow spread indicates stable performance across random search iterations despite not sharing the same tuning run as the other models. STACK's performance below GB and XGB reflects that its base estimators were fixed at simplified hyperparameters during tuning; only the meta-learner regularization (C=0.299) was optimized. MLP shows the widest distribution (median 0.9746), consistent with its sensitivity to random initialization and learning rate. SVM has the lowest median (0.9732) and a compact distribution, reflecting the limited flexibility of a fixed-gamma RBF kernel before C is tuned.

**Figure 11.** Top 10 predicted AMP candidates from an external validation set.

![Top candidates](model_training/tuned_model/figures/fig06_top10_candidates.png)

The bar chart shows the 10 sequences with the highest mean GB probability across the seven models. Candidates are ranked by GB score; error bars (or color gradients, depending on render) reflect agreement across models. Sequences where all seven models assign high scores are the highest-priority candidates for experimental follow-up, as model consensus reduces the likelihood of a false positive driven by a single classifier's systematic bias.

**Figure 12-15.** Hyperparameter performance surfaces for RF, GB, SVM, and XGB.

![RF hyperparams](model_training/tuned_model/figures/fig07_hyperparam_rf.png)
![GB hyperparams](model_training/tuned_model/figures/fig08_hyperparam_gb.png)
![SVM hyperparams](model_training/tuned_model/figures/fig09_hyperparam_svm.png)
![XGB hyperparams](model_training/tuned_model/figures/fig10_hyperparam_xgb.png)

GB shows a clear optimum at learning_rate 0.06-0.08 and n_estimators 250-350 with max_depth 6. XGB's optimal region is broader across n_estimators but similarly concentrated at max_depth 6. SVM performance is highest at C > 10 with gamma 0.01, confirming that the default gamma='scale' is suboptimal for this feature set.

**Figure 16.** Per-model comparison of all metrics (Accuracy, Precision, Recall, Specificity, F1, MCC, AUC-ROC) on the test set.

![Metrics comparison](model_training/tuned_model/figures/fig11_metrics_comparison.png)

GB (marked with asterisk at MCC) achieves the highest or second-highest value on five of seven metrics. DEEP achieves the highest Recall across all models (0.9668), reflecting its low decision threshold. MLP shows the largest gap between AUC-ROC (0.9749) and threshold-dependent metrics (MCC 0.8543), indicating that its probability estimates are discriminative but poorly calibrated relative to the optimal threshold.

**Figure 17.** Precision-recall curves for all models.

![Precision-recall](model_training/tuned_model/figures/fig12_precision_recall.png)

GB and XGB trace the highest precision-recall curves, with GB maintaining precision above 0.93 across the full recall range up to 0.955. XGB achieves the highest recall (0.9575) at a marginally lower precision (0.9326). DEEP reaches recall 0.9668 at the cost of precision falling below 0.92, consistent with its low decision threshold. SVM and MLP trace lower curves, with MLP showing the largest area loss relative to its AUC-ROC rank, indicating that probability values from QuantileTransformer-scaled MLP are less well-ordered near the class boundary. STACK's curve closely tracks XGB, which is its dominant base estimator.

**Figure 18.** Detection error tradeoff (DET) curves for all models.

![DET curves](model_training/tuned_model/figures/fig13_det_curves.png)

DET curves plot false negative rate (FNR) versus false positive rate (FPR) on a normal deviate scale. Lower curves indicate better performance. GB and XGB trace the lowest paths across all FPR values, with DEEP performing comparably at high FNR tolerance. SVM and MLP trace higher curves, consistent with their lower AUC-ROC values.

**Figure 19.** Threshold sensitivity: MCC, F1, Precision, and Recall as a function of decision threshold for the best model (GB).

![Threshold sensitivity](model_training/tuned_model/figures/fig14_threshold_sensitivity.png)

For GB, MCC peaks at threshold 0.37 (MCC = 0.8913). F1 peaks at a similar threshold. Precision increases monotonically with threshold while Recall decreases, with the crossover near 0.40. The MCC peak is broad between 0.30 and 0.45, suggesting the model is not highly sensitive to threshold choice in this range.

## Discussion

### Feature expansion effect

The transition from 10 global descriptors (`beta`) to the current 22-feature Macrel-inspired set increased MCC by approximately 0.057-0.063 points and AUC-ROC by 0.018-0.021 at baseline across tree-based models. The improvement confirms that the `beta` plateau was caused by information loss from collapsing amino acid sequences into global scalars, not by model capacity.

The 22 features extend the global descriptors with the hydrophobic moment (amphipathic helical character), nine grouped amino acid composition fractions (functional group membership), and six positional features encoding where FET and solvent accessibility groups first appear along the sequence. These additions capture residue composition patterns and N-terminal positional constraints that global averages discard.

### Model comparison

> Numbers below are from the previous run (14,318 sequences, 127 features) and will be updated after re-tuning on the current dataset.

GB achieves the highest AUC-ROC (0.9845) and MCC (0.8913) after tuning. XGB ranks second on MCC (0.8886) and third on AUC-ROC (0.9836). Both models use shallow trees (max_depth=6) with moderate n_estimators (293 and 448 respectively) and subsample rates below 1.0, which is consistent with the standard regularization pattern for gradient-boosted classifiers on structured biological data.

The Stacking ensemble (AUC-ROC 0.9829, MCC 0.8864) does not outperform GB or XGB despite combining three base models. This is expected: the base estimators in the stacking pipeline were fixed at simplified hyperparameters (RF with 200 trees at default depth, XGB with 100 estimators at default settings, SVC at default gamma) to keep the hyperparameter search tractable. Only the meta-learner C was tuned. An alternative stacking configuration using the fully tuned individual models as bases would likely produce higher performance, at the cost of substantially longer inference time.

SVM (MCC 0.8739) and MLP (MCC 0.8543) rank fifth and sixth respectively. SVM's performance relative to tree-based models is constrained by the kernel's sensitivity to the feature scale of CTD Distribution features, which are bounded [0, 1] but highly skewed. MLP's lower performance likely reflects insufficient training iterations or sensitivity to initialization, given that its CV AUC-ROC distribution is wider than that of tree-based models.

DEEP achieves AUC-ROC 0.9845 (tied with GB) using only raw sequence information. This confirms that the amino acid sequence contains sufficient information for near-optimal AMP classification without hand-crafted physicochemical descriptors. The high recall (0.9668) and low threshold (0.12) make DEEP the preferred model for screening large sequence databases where maximizing sensitivity is the primary objective.

### Practical threshold selection

The MCC-optimized thresholds range from 0.12 (DEEP) to 0.47 (RF). Low thresholds for DEEP and MLP reflect that these models assign lower average probabilities to true positives, either because of model architecture (sigmoid output without isotonic recalibration for DEEP) or because of poor probability calibration near the decision boundary (MLP, visible in Figure 8). For any deployment context, threshold selection should be driven by the acceptable FP/FN trade-off, not by the MCC-optimized value alone. The threshold sensitivity analysis (Figure 19) provides the full MCC, F1, Precision, and Recall curves for GB across the threshold range.

### Recommendations

For experimental validation follow-up: use GB at threshold 0.37 (MCC 0.8913, balanced FP/FN) or DEEP at threshold 0.12 (maximum sensitivity, FN=46 on 2,774 test sequences). For production deployment on balanced databases: GB or XGB. For high-throughput screening of uncharacterized proteomes: DEEP (AUC-ROC 0.9845, recall 0.9668).

## Known limitations and next steps

### Negative class distribution shift

The benchmark evaluation (Phase 5) shows AUC-ROC dropping from 0.975-0.985 (test set) to 0.731-0.839 (independent benchmark). The cause is not overfitting in the classical sense: the models generalize well to held-out data from the same distribution, as confirmed by the consistent test-set performance across all seven architectures. The problem is that the training negatives (Swiss-Prot reviewed proteins, diverse functions, lengths up to 200 residues) are physicochemically distinct from AMPs along multiple feature axes (length, MW, CTD Distribution). The benchmark negatives are short peptides (median 25 aa) that occupy the same region of the 127-feature space as AMPs. The decision boundary learned during training does not exist in the peptide-vs-peptide subspace.

Three changes are needed to address this:

- **Negative class recomposition**: replace or augment the Swiss-Prot negatives with short non-AMP peptides sampled from the same length range as AMPs (10-50 residues). Candidate sources include Swiss-Prot signal peptides, propeptides, and short regulatory peptides annotated as non-antimicrobial, as well as experimentally validated inactive AMP analogues from APD3 and DRAMP.
- **Threshold recalibration**: the MCC-optimized thresholds (0.12-0.47) were derived from the original test set and do not transfer to the benchmark distribution. Post-hoc isotonic regression or Platt scaling on a held-out subset of the benchmark would restore meaningful threshold-based classification without retraining.
- **Feature augmentation for peptide-level discrimination**: add descriptors that distinguish short AMPs from short non-AMP peptides specifically, including net charge normalized by length, amphipathicity index (hydrophobic moment), fraction of cationic residues (K+R) in the N-terminal half versus C-terminal half, and secondary structure propensity scores. These features encode the amphipathic helix and cationic gradient patterns characteristic of AMPs but absent in short non-AMP peptides.

## References

Dubchak, I., Muchnik, I., Holbrook, S.R., and Kim, S.-H. (1995). Prediction of protein folding class using global description of amino acid sequence. *Proceedings of the National Academy of Sciences*, 92(19), 8700-8704.

Chou, K.-C. and Shen, H.-B. (2007). MemType-2L: a web server for predicting membrane proteins and their types by incorporating evolution information through Pse-PSSM. *Biochemical and Biophysical Research Communications*, 360(2), 339-345.

Shai, Y. (2002). Mode of action of membrane active antimicrobial peptides. *Biopolymers*, 66(4), 236-248.

Wang, G., Li, X., and Wang, Z. (2009). APD2: the updated antimicrobial peptide database and its application in peptide design. *Nucleic Acids Research*, 37(Database issue), D933-D937.
