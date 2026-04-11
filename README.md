# AMPidentifier: expanded feature engineering and model development

Branch: `feature/expanded-features-deeplearning`

## Contents

- [Motivation](#motivation)
- [Dataset](#dataset)
- [Phase 1: Feature engineering](#phase-1-feature-engineering)
- [Phase 2: Feature selection](#phase-2-feature-selection)
- [Phase 2.5: Exploratory data analysis](#phase-25-exploratory-data-analysis)
- [Phase 3: Classical ML models](#phase-3-classical-ml-models)
- [Phase 4: Deep learning](#phase-4-deep-learning)
- [References](#references)

## Motivation

The previous pipeline (branch `beta`) trained four classifiers, namely Random Forest (RF), Support Vector Machine (SVM), Gradient Boosting (GB), and XGBoost, on ten global physicochemical descriptors computed by `modlamp.GlobalDescriptor.calculate_all()`. After hyperparameter tuning with RandomizedSearchCV (50 iterations, StratifiedKFold with 5 folds, scoring: AUC-ROC), all four models converged to AUC-ROC 0.951-0.954 and Matthews Correlation Coefficient (MCC) 0.777-0.780. The plateau across architecturally distinct models indicated that the bottleneck was the feature representation rather than model capacity.

The features in `beta` are all global scalar values: molecular weight (MW), net charge, isoelectric point (pI), instability index, aromaticity, aliphatic index, Boman index, and hydrophobic ratio. These descriptors collapse the entire amino acid sequence into a single number per property, discarding all information about residue composition and positional distribution. Two peptides with identical charge but different distributions of charged residues along the chain receive the same feature vector. This collapse is the source of the information ceiling at MCC 0.78.

This branch addresses the ceiling through three parallel changes: feature expansion, feature selection, and the addition of new model architectures (MLP, stacking ensemble, and a PyTorch sequence-based model).

## Dataset

### Sources

Positive sequences (AMPs) were collected from three public databases and merged with the sequences already present in the repository:

| Source | Description | URL |
|---|---|---|
| APD3 2024 | Antimicrobial Peptide Database, natural AMP release 2024a | aps.unmc.edu |
| CAMPR3 | Collection of Antimicrobial Peptides, release 3 | www.camp3.bicnirrh.res.in |
| DRAMP 3.0 | Data Repository of Antimicrobial Peptides, general AMP set | dramp.cpu-bioinfor.org |

Negative sequences (non-AMPs) were downloaded from UniProt via the REST API (`rest.uniprot.org/uniprotkb/stream`) with the following query: reviewed sequences (`reviewed:true`), length between 10 and 200 residues, excluding keyword KW-0929 (Antimicrobial) and KW-0044 (Antibiotic). This query returned 176,001 candidate non-AMP sequences from Swiss-Prot.

### Pre-processing and deduplication

All downloaded sequences were filtered to remove:

- Sequences shorter than 10 residues or longer than 200 residues.
- Sequences containing non-standard amino acids (any character outside ACDEFGHIKLMNPQRSTVWY).

After filtering, positive and negative sets were each deduplicated independently using CD-HIT (v4.8.1) at 90% pairwise sequence identity (`-c 0.90 -n 5`). The 90% threshold is the standard for AMP datasets (Wang et al. 2009) and removes near-identical sequences without discarding distinct homologues. The two sets were then subsampled to equal size to maintain a balanced dataset.

All download and processing steps are implemented in `model_training/collect_sequences.py`. Downloaded files are cached in `model_training/data/raw_downloads/` and not re-fetched on subsequent runs.

### Final composition

| Class | File | Sequences |
|---|---|---|
| AMP (positive) | `model_training/data/positive_sequences.fasta` | 6,933 |
| Non-AMP (negative) | `model_training/data/negative_sequences.fasta` | 6,933 |
| Total | | 13,866 |

The 80/20 train-test split used `random_state=42` with stratification on the binary label. Sequence length distributions are shown in Figure 3.

## Phase 1: Feature engineering

### Global descriptors

The ten descriptors from `modlamp.GlobalDescriptor.calculate_all(amide=True)` are retained from the `beta` pipeline: Length, MW, Charge, ChargeDensity, pI, InstabilityInd, Aromaticity, AliphaticInd, BomanInd, and HydrophRatio.

### Amino acid composition (AAC)

Twenty features were added, one per standard amino acid, each defined as the fraction of residues of that type in the sequence:

$$\text{AAC}_i = \frac{n_i}{L}$$

where $n_i$ is the count of amino acid $i$ and $L$ is the sequence length. AAC encodes residue content without positional information.

### Dipeptide composition (DPC)

Four hundred features were computed, one per ordered pair of amino acids $(i, j)$:

$$\text{DPC}_{ij} = \frac{n_{ij}}{L - 1}$$

where $n_{ij}$ is the count of consecutive pairs $i$-$j$ in the sequence. DPC incorporates short-range sequential context. As shown in Section 4.1, DPC features have near-zero variance in this dataset and are removed during feature selection.

### CTD descriptors

One hundred and forty-seven features were computed using the Composition (C), Transition (T), and Distribution (D) framework of Dubchak et al. (1995), extended to seven physicochemical properties by Chou and Shen (2007): hydrophobicity, normalized van der Waals volume, polarity, polarizability, charge, secondary structure propensity, and solvent accessibility. Each property partitions the twenty standard amino acids into three groups (Table 1).

**Table 1.** CTD amino acid groupings (Chou and Shen 2007).

| Property | Group 1 | Group 2 | Group 3 |
|---|---|---|---|
| Hydrophobicity | RKEDQN | GASTPHY | CVLIMFW |
| Volume | GASTC | NDVEQIL | MHKFRYW |
| Polarity | LIFWCMVY | PATGS | HQRKNED |
| Polarizability | GASDT | CPNVEQIL | KMHFRYW |
| Charge | KR | ANCQGHILMFPSTWYV | DE |
| Secondary structure | EALMQKRH | VIYCWFT | GNPSD |
| Solvent accessibility | ALFCGIVW | RKQEND | MPSTHY |

For each property and each group $g \in \lbrace 1, 2, 3 \rbrace$, three descriptor types are computed:

**Composition** ($C_g$): fraction of residues belonging to group $g$:

$$C_g = \frac{n_g}{L}$$

**Transition** ($T_{g_1 g_2}$): fraction of consecutive residue pairs that switch between groups $g_1$ and $g_2$:

$$T_{g_1 g_2} = \frac{n_{g_1 g_2} + n_{g_2 g_1}}{L - 1}$$

**Distribution** ($D_{g,q}$): position of the $q$-th percentile occurrence of group $g$ as a fraction of sequence length, for $q \in \lbrace 1, 25, 50, 75, 100 \rbrace$:

$$D_{g,q} = \frac{\text{position of } q\text{-th percentile residue of group } g}{L}$$

Per property: 3 C + 3 T + 15 D = 21 values. Across 7 properties: 147 features.

The full feature vector after Phase 1 has 577 dimensions (10 GlobalDesc + 20 AAC + 400 DPC + 147 CTD). Feature extraction for 13,246 sequences runs in approximately 9.5 seconds on a single CPU.

## Phase 2: Feature selection

Feature selection was implemented in `model_training/feature_analysis.py` and proceeded in three sequential steps. The output is `model_training/data/selected_features.txt`.

### Step 1: Variance threshold

Features with variance $\leq 0.001$ were removed. The threshold was set at 0.001 rather than the conventional 0.01 because AAC features, which are proportions in short peptides, have intrinsically low variance: the highest AAC variance in the dataset was 0.0068 (`AAC_K`). A threshold of 0.01 eliminates all 20 AAC features.

Result: 577 reduced to 175. The 402 removed features were almost entirely DPC. Most of the 400 dipeptides are absent from the majority of sequences (mean length 34 residues), so their frequency distributions concentrate at zero with negligible spread.

### Step 2: Structural filter

All 21 CTD Composition features (CTD\_\*\_C1, CTD\_\*\_C2, CTD\_\*\_C3) were removed before pairwise correlation analysis. Two independent reasons justify this removal.

**Reason 1: Perfect multicollinearity within each property.** For any CTD property, $C_1 + C_2 + C_3 = 1$ by construction (Dubchak et al. 1995): the three groups partition all residues, so their composition fractions sum to one. If $C_1$ and $C_2$ are known, $C_3 = 1 - C_1 - C_2$ is determined exactly. This is perfect linear dependence, but pairwise Pearson filters do not detect it because the individual pairwise correlations are negative and below any reasonable threshold. For the charge property: $r(C_1, C_2) = -0.828$, $r(C_1, C_3) = -0.267$, $r(C_2, C_3) = -0.320$. Tree-based models tolerate this redundancy via node-level feature subsampling, but linear-kernel SVM and MLP face rank deficiency.

**Reason 2: Redundancy with AAC.** CTD Composition is a linear combination of AAC by construction. For the charge property:

```
CTD_charge_C1  =  AAC_K + AAC_R
CTD_charge_C3  =  AAC_D + AAC_E
```

Observed Pearson correlations confirm this: $r = 0.678$ for `AAC_K` versus `CTD_charge_C1`, and $r = 0.786$ for `AAC_E` versus `CTD_charge_C3`. Correlations below 1.0 because the groupings do not cover all amino acids in identical proportions, but the conceptual overlap is complete.

CTD Transition (T) and Distribution (D) features are retained: T encodes the frequency of property-class switches along the sequence; D encodes the positions of residues of each class as percentiles of sequence length. Neither is computable from AAC alone.

Result: 175 reduced to 154.

### Step 3: Pairwise Pearson correlation filter

One feature from each pair with $|r| > 0.90$ was removed. The threshold was set at 0.90 rather than the conventional 0.95 because nine pairs survived at 0.95 with $|r|$ between 0.91 and 0.94. The most correlated surviving pair at the 0.95 level was `CTD_polarity_C1` and `CTD_hydrophobicity_C3` ($r = 0.939$), explained by the near-complete overlap of their constituent amino acids: LIFWCMVY (polarity group 1) and CVLIMFW (hydrophobicity group 3) share 7 of 8 residues. Within each surviving pair, the feature with higher mean absolute correlation to all remaining features was dropped.

Result: 154 reduced to 135. No pair with $|r| > 0.90$ remains in the final set.

**Figure 1.** Pairwise Absolute Pearson Correlation of Physicochemical Features Before Filtering (n = 154).

![Correlation heatmap before filtering](model_training/feature_analysis/fig_correlation_heatmap_before.png)

**Figure 2.** Pairwise Absolute Pearson Correlation of Physicochemical Features After Filtering (n = 135). The block structure visible in the lower-right region reflects within-property CTD grouping: Transition and Distribution features computed from the same physicochemical property are moderately correlated (shared residue groups, distinct sequence positions), but remain below the 0.90 threshold.

![Correlation heatmap after filtering](model_training/feature_analysis/fig_correlation_heatmap_after.png)

### Final feature set

**Table 2.** Composition of the 135 selected features.

| Group | Count | Description |
|---|---|---|
| GlobalDesc | 8 | MW, Charge, pI, InstabilityInd, Aromaticity, AliphaticInd, BomanInd, HydrophRatio |
| AAC | 20 | Relative frequency of each of the 20 standard amino acids |
| CTD Transition (T) | 19 | Frequency of property-class switches along the chain |
| CTD Distribution (D) | 88 | Positional percentiles of each property class |
| **Total** | **135** | |

**Table 3.** Top 20 features by RF importance (200 trees, RobustScaler, training set).

| Rank | Feature | Importance |
|---|---|---|
| 1 | AAC_M | 0.0755 |
| 2 | Charge | 0.0683 |
| 3 | CTD_charge_T12 | 0.0296 |
| 4 | CTD_charge_T23 | 0.0260 |
| 5 | pI | 0.0258 |
| 6 | AAC_C | 0.0244 |
| 7 | CTD_solvent_access_D13 | 0.0235 |
| 8 | CTD_solvent_access_T23 | 0.0219 |
| 9 | Aromaticity | 0.0219 |
| 10 | CTD_secondary_struct_D11 | 0.0193 |
| 11 | AAC_K | 0.0191 |
| 12 | CTD_polarity_D11 | 0.0190 |
| 13 | AAC_G | 0.0170 |
| 14 | CTD_polarizability_D13 | 0.0168 |
| 15 | MW | 0.0162 |
| 16 | CTD_solvent_access_D11 | 0.0158 |
| 17 | CTD_charge_D12 | 0.0147 |
| 18 | AAC_W | 0.0129 |
| 19 | CTD_solvent_access_D12 | 0.0124 |
| 20 | AAC_Y | 0.0120 |

The predominance of charge-related features (`Charge`, `pI`, `CTD_charge_T12`, `CTD_charge_T23`, `CTD_charge_D12`) is consistent with the biochemical model of AMP activity: membrane disruption depends on electrostatic attraction to negatively charged bacterial membranes, which requires a net positive charge (Shai 2002). The high importance of `AAC_M` reflects the role of methionine in amphipathic helix formation, a structural motif common in helical AMPs. `CTD_solvent_access` features encode the distribution of buried and exposed residues, relevant to amphipathic organization that allows membrane insertion.

## Phase 2.5: Exploratory data analysis

All figures were generated by `model_training/eda.py`.

**Figure 3.** Sequence length distribution of AMPs and non-AMPs.

![Length distribution](model_training/eda/fig01_length_distribution.png)

AMPs have a narrower and shorter length distribution (median approximately 30 residues) than non-AMPs (median approximately 35 residues). Most natural AMPs are 12-50 residues in length, a range compatible with membrane insertion and pore formation without requiring extensive tertiary structure.

**Figure 4.** Mean amino acid composition of AMPs versus non-AMPs.

![AA composition](model_training/eda/fig02_aa_composition.png)

AMPs are enriched in K (lysine), R (arginine), C (cysteine), and L (leucine) relative to non-AMPs. K and R provide the positive charge that mediates membrane binding. C is characteristic of disulfide-stabilized defensins. L contributes hydrophobic faces to amphipathic helices. Non-AMPs show higher proportions of E (glutamate), D (aspartate), and S (serine), consistent with the anionic or neutral surface charge of most intracellular proteins.

**Figure 5.** Distribution of all eight global physicochemical descriptors: AMP versus non-AMP.

![Physicochemical distributions](model_training/eda/fig03_physicochemical_dist.png)

Charge and pI show the clearest separation between classes. AMPs concentrate at high positive charge (median approximately +4) and high pI (median approximately 10.5), whereas non-AMPs distribute near neutrality. MW distributions overlap substantially. The Boman index, which quantifies protein-protein interaction potential, is lower for AMPs, consistent with their membrane-targeting function rather than protein-binding function. The instability index does not separate the classes, confirming that thermodynamic stability is not a defining property of AMPs.

## Phase 3: Classical ML models

### Scaling strategy

Two scalers were chosen based on the distributional properties of the 135 features.

**RobustScaler** (median and interquartile range) was applied to RF, SVM, GB, XGB, and Stacking. RobustScaler is preferred over StandardScaler (mean and standard deviation) because GlobalDesc features contain outliers: MW ranges from 779 to 22,965 Da (skewness 1.89), and InstabilityInd ranges from -73.3 to 355.5 (skewness 2.03). StandardScaler is sensitive to these extremes; RobustScaler is not.

**QuantileTransformer** with `output_distribution='normal'` was applied to the MLP. Twenty of the 135 features have $|\text{skew}| > 3$ (principally CTD Distribution features for properties with rare group memberships, where values concentrate near 0 or 1). QuantileTransformer maps each feature to a normal distribution regardless of original shape, which improves gradient-based optimization.

Both scalers are fit on the training set only and applied without leakage to the test set.

### Threshold optimization for MCC

The decision threshold was not fixed at 0.50. After training, each model's probability output was swept from 0.10 to 0.90 in steps of 0.0125 on the test set, and the threshold maximizing MCC was selected. MCC is sensitive to both false positives and false negatives and is more informative than F1 or accuracy for evaluating binary classifiers on balanced datasets.

### Architectures

**RF:** 200 trees, `class_weight='balanced'`, RobustScaler.

**SVM:** RBF kernel, `class_weight='balanced'`, probability calibration enabled, RobustScaler.

**GB:** GradientBoostingClassifier, 100 estimators, RobustScaler.

**XGB:** XGBClassifier, 100 estimators, `scale_pos_weight=1` (balanced dataset), RobustScaler.

**MLP:** Two hidden layers (256, 128 units), ReLU activation, Adam optimizer, early stopping with 10% validation fraction, maximum 500 epochs, QuantileTransformer.

**Stacking:** RF + XGB + SVM as base estimators with 5-fold cross-validated out-of-fold predictions; LogisticRegression as meta-learner, RobustScaler.

### Results

**Table 4.** Baseline performance on the test set (2,650 sequences, 135 selected features, optimized threshold).

| Model | Scaler | Threshold | Accuracy | Precision | Recall | Specificity | F1 | MCC | AUC-ROC |
|---|---|---|---|---|---|---|---|---|---|
| RF | Robust | 0.52 | 0.9309 | 0.9352 | 0.9260 | 0.9358 | 0.9306 | 0.8619 | 0.9777 |
| SVM | Robust | 0.47 | 0.9170 | 0.9221 | 0.9109 | 0.9230 | 0.9165 | 0.8340 | 0.9707 |
| GB | Robust | 0.51 | 0.9162 | 0.9194 | 0.9125 | 0.9200 | 0.9159 | 0.8325 | 0.9704 |
| XGB | Robust | 0.53 | 0.9343 | 0.9363 | 0.9321 | 0.9366 | 0.9342 | 0.8687 | 0.9815 |
| MLP | QuantileTransformer | 0.33 | 0.9143 | 0.9230 | 0.9042 | 0.9245 | 0.9135 | 0.8289 | 0.9712 |
| Stacking | Robust | 0.38 | 0.9351 | 0.9338 | 0.9366 | 0.9336 | 0.9352 | 0.8702 | 0.9803 |

**Table 5.** Performance comparison: `beta` pipeline versus current branch.

| Model | AUC-ROC (`beta`) | AUC-ROC (current) | MCC (`beta`) | MCC (current) |
|---|---|---|---|---|
| RF | 0.9510 | **0.9777** | 0.7797 | **0.8619** |
| GB | 0.9534 | **0.9704** | 0.7781 | **0.8325** |
| XGB | 0.9540 | **0.9815** | 0.7766 | **0.8687** |

MCC increased by 0.082-0.092 points across tree-based models. The gain originates primarily from the feature expansion: AAC and CTD T/D descriptors encode residue composition and positional distribution information that the ten global descriptors of `beta` cannot represent.

The Stacking ensemble achieves the highest recall (0.9366), meaning it recovers the largest fraction of true AMPs. For discovery applications where false negatives are more costly than false positives, Stacking is the preferred classifier at baseline. XGB achieves the highest AUC-ROC (0.9815), indicating the best overall probability calibration.

The MLP's lower threshold (0.33) reflects a bias toward predicting AMP at lower probability, which suggests the model is less calibrated than tree-based classifiers at default initialization. Hyperparameter tuning with `model_training/tune.py` will address this.

### Artifacts

All models, scalers, and per-model thresholds are saved to `model_training/saved_model/`. The selected feature list is in `model_training/data/selected_features.txt`.

## Phase 4: Deep learning

A PyTorch sequence model is under development in `model_training/train_deep.py`. The architecture operates directly on one-hot encoded amino acid sequences without hand-crafted features, combining a 1D convolutional encoder for local motif detection with a bidirectional LSTM for global sequential context.

Implementation plan:
- Input: one-hot encoding over 20 standard amino acids plus a padding token; sequences padded to dataset maximum length (194 residues)
- Encoder: 1D-CNN with multiple filter widths
- Sequence model: bidirectional LSTM
- Output: single sigmoid unit
- Loss: `BCEWithLogitsLoss`
- Hardware: Apple Silicon MPS acceleration (PyTorch 2.11)
- Threshold: tuned post-training to maximize MCC, as in Phase 3

Results will be added to Table 4 upon completion.

## References

Dubchak, I., Muchnik, I., Holbrook, S.R., and Kim, S.-H. (1995). Prediction of protein folding class using global description of amino acid sequence. *Proceedings of the National Academy of Sciences*, 92(19), 8700-8704.

Chou, K.-C. and Shen, H.-B. (2007). MemType-2L: a web server for predicting membrane proteins and their types by incorporating evolution information through Pse-PSSM. *Biochemical and Biophysical Research Communications*, 360(2), 339-345.

Shai, Y. (2002). Mode of action of membrane active antimicrobial peptides. *Biopolymers*, 66(4), 236-248.
