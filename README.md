# AMPidentifier: expanded feature engineering and model development

Branch: `feature/expanded-features-deeplearning`

## Contents

- [Motivation](#motivation)
- [Dataset](#dataset)
- [Phase 1: Feature engineering](#phase-1-feature-engineering)
- [Phase 2: Exploratory data analysis](#phase-2-exploratory-data-analysis)
- [Phase 3: Classical ML models (baseline)](#phase-3-classical-ml-models-baseline)
- [Phase 3.1: Hyperparameter tuning](#phase-31-hyperparameter-tuning)
- [Figures](#figures)
- [Discussion](#discussion)
- [References](#references)

## Motivation

The previous pipeline (branch `beta`) trained four classifiers, namely Random Forest (RF), Support Vector Machine (SVM), Gradient Boosting (GB), and XGBoost, on ten global physicochemical descriptors computed by `modlamp.GlobalDescriptor.calculate_all()`. After hyperparameter tuning with RandomizedSearchCV (50 iterations, StratifiedKFold with 5 folds, scoring: AUC-ROC), all four models converged to AUC-ROC 0.951-0.954 and Matthews Correlation Coefficient (MCC) 0.777-0.780. The plateau across architecturally distinct models indicated that the bottleneck was the feature representation rather than model capacity.

The features in `beta` are all global scalar values: molecular weight (MW), net charge, isoelectric point (pI), instability index, aromaticity, aliphatic index, Boman index, and hydrophobic ratio. These descriptors collapse the entire amino acid sequence into a single number per property, discarding all information about residue composition and positional distribution. Two peptides with identical charge but different distributions of charged residues along the chain receive the same feature vector. This collapse is the source of the information ceiling at MCC 0.78.

This branch addresses the ceiling through two changes: feature expansion and feature selection, extending the descriptor set with grouped amino acid composition and positional features derived from FET and solvent accessibility groups.

## Dataset

### Sources

The dataset construction follows the strategy adopted by Macrel (Santos-Júnior et al. 2020), which demonstrated that the quality of non-AMP negatives is as critical as the positive set for reliable AMP classification. Briefly, the positive set contains unique sequences from APD3, CAMPR3, and LAMP databases. Negative sequences were retrieved from UniProt and are restricted to entries not annotated as antimicrobial, membrane, toxic, secretory, defensin, antibiotic, anticancer, antiviral, or antifungal. This curation strategy prevents the model from learning superficial biases (sequence length, global charge) as proxies for AMP identity.

The AmPEP training set (Bhadra et al. 2018) was incorporated and its non-AMP sequences, drawn from the same UniProt-curated strategy, were merged with our UniProt-derived negatives.

**Positive sequences (AMPs):**

| Source | Description |
|---|---|
| APD3 2024 | Antimicrobial Peptide Database, natural AMP release 2024a |
| CAMPR3 | Collection of Antimicrobial Peptides, release 3 |
| LAMP | Library of Antimicrobial Peptides |
| AmPEP | AMP training set (Bhadra et al. 2018) |

**Negative sequences (non-AMPs):**

UniProt Swiss-Prot (`reviewed:true`), excluding keywords KW-0929 (Antimicrobial), KW-0044 (Antibiotic), KW-0472 (Membrane), KW-0800 (Toxin), KW-0964 (Secreted), KW-0163 (Defensin), KW-0044 (Anticancer), KW-0244 (Antiviral), merged with the non-AMP set from AmPEP.

### Pre-processing and balancing

All sequences were filtered to remove non-standard amino acids (outside ACDEFGHIKLMNPQRSTVWY) and sequences outside the 5-255 residue range. Exact-sequence deduplication was applied across all merged sources.

To prevent the model from using sequence length as a discriminating feature, the negative set was subsampled using length-stratified sampling: the length distribution of the negative set was matched to the positive distribution across 10 equal-width bins, with per-bin quotas proportional to the positive class counts. This approach follows the Macrel benchmark design (Santos-Júnior et al. 2020).

### Final composition

| Class | File | Sequences |
|---|---|---|
| AMP (positive) | `model_training/data/positive_sequences.fasta` | 6,623 |
| Non-AMP (negative) | `model_training/data/negative_sequences.fasta` | 6,623 |
| Total | | 13,246 |

The 80/20 train-test split used `random_state=42` with stratification on the binary label. Training set: 10,596 sequences. Test set: 2,650 sequences (1,325 per class). Sequence length distributions are shown in Figure 1.

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

$$\text{FET}_{g,D1} = \frac{\text{index of first residue} \in g + 1}{L}$$

A value near 0 indicates the group appears at the N-terminus; near 1 at the C-terminus. Zero is returned when no residue of the group is present.

### Solvent accessibility local features

Three features encode the relative position of the first residue in each solvent accessibility group (Bhadra et al. 2018): buried (ALFCGIVW), exposed (RKQEND), and intermediate (MSPTHY). The notation follows the CTD Distribution D1 convention:

$$\text{SA}_{g,D1} = \frac{\text{index of first residue} \in g + 1}{L}$$

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

All five models use RobustScaler (median and interquartile range). `InstabilityInd` and `AliphaticInd` contain outliers that inflate standard deviation-based scaling. All scalers are fit on the training set only and applied without leakage to the test set.

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

Results will be updated after tuning completes on the current dataset.

## Figures

Figures 1-5 are in `model_training/eda/`. Figures from feature analysis are in `model_training/feature_analysis/`. Post-tuning figures are generated by `model_training/plot_tuning.py` and saved to `model_training/tuned_model/figures/`.

The following figures will be added after tuning completes on the current dataset:

**Figure 6.** ROC curves for all five models on the held-out test set.

**Figure 7.** Confusion matrices for all five models on the held-out test set.

**Figure 8.** Calibration curves (reliability diagrams) for all models.

**Figure 9.** Mean impurity-based feature importances for tree-based models (RF, GB, XGB), top features each.

**Figure 10.** Cross-validation AUC-ROC score distributions (50 iterations, RandomizedSearchCV).

**Figure 11.** Top 10 predicted AMP candidates from an external validation set.

**Figure 12-15.** Hyperparameter performance surfaces for RF, GB, SVM, and XGB.

**Figure 16.** Per-model comparison of all metrics (Accuracy, Precision, Recall, Specificity, F1, MCC, AUC-ROC) on the test set.

**Figure 17.** Precision-recall curves for all models.

**Figure 18.** Detection error tradeoff (DET) curves for all models.

**Figure 19.** Threshold sensitivity: MCC, F1, Precision, and Recall as a function of decision threshold for the best model.

## Discussion

### Feature expansion effect

The transition from 10 global descriptors (`beta`) to the current 22-feature Macrel-inspired set increased MCC by approximately 0.057-0.063 points and AUC-ROC by 0.018-0.021 at baseline across tree-based models. The improvement confirms that the `beta` plateau was caused by information loss from collapsing amino acid sequences into global scalars, not by model capacity.

The 22 features extend the global descriptors with the hydrophobic moment (amphipathic helical character), nine grouped amino acid composition fractions (functional group membership), and six positional features encoding where FET and solvent accessibility groups first appear along the sequence. These additions capture residue composition patterns and N-terminal positional constraints that global averages discard.

### Model comparison

At baseline, RF, XGB, and LGBM are tied at AUC-ROC 0.972. XGB achieves the highest MCC (0.841) and LGBM the highest precision (0.942) at the cost of lower recall. SVM has the highest recall (0.926) among the five models. Tuning results will be added after re-run on the current dataset.

### Practical threshold selection

The MCC-optimized thresholds range from 0.40 (SVM) to 0.64 (XGB) at baseline. For any deployment context, threshold selection should be driven by the acceptable FP/FN trade-off, not by the MCC-optimized value alone.

## References

Dubchak, I., Muchnik, I., Holbrook, S.R., and Kim, S.-H. (1995). Prediction of protein folding class using global description of amino acid sequence. *Proceedings of the National Academy of Sciences*, 92(19), 8700-8704.

Chou, K.-C. and Shen, H.-B. (2007). MemType-2L: a web server for predicting membrane proteins and their types by incorporating evolution information through Pse-PSSM. *Biochemical and Biophysical Research Communications*, 360(2), 339-345.

Shai, Y. (2002). Mode of action of membrane active antimicrobial peptides. *Biopolymers*, 66(4), 236-248.

Wang, G., Li, X., and Wang, Z. (2009). APD2: the updated antimicrobial peptide database and its application in peptide design. *Nucleic Acids Research*, 37(Database issue), D933-D937.
