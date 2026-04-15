# AMPidentifier 2.0: Expanded Ensemble Machine Learning for Antimicrobial Peptide Prediction with LightGBM and Revised Physicochemical Descriptors

Madson Allan de Luna-Aragão,^1^ Rafael Lucas da Silva,^2^ João Pacífico Bezerra Neto,^3^ Carlos André dos Santos-Silva,^5^ Denys Ewerton da Silva Santos,^4^ Ana Maria Benko-Iseppon^2\*^

^1^ Institute of Biological Sciences, Universidade Federal de Minas Gerais (UFMG), Belo Horizonte, MG, Brazil; ^2^ Department of Genetics, Universidade Federal de Pernambuco (UFPE), Recife, PE, Brazil; ^3^ Universidade de Pernambuco (UPE), Petrolina, PE, Brazil; ^4^ Department of Fundamental Chemistry, Universidade Federal de Pernambuco (UFPE), Recife, PE, Brazil; ^5^ Centro Universitário CESMAC, Maceió, AL, Brazil

\*Corresponding author: ana.benko@ufpe.br

**KEYWORDS**: Ensemble learning; LightGBM; Physicochemical descriptors; Antimicrobial peptides; Sequence-based classification; Peptide informatics

## Abstract

AMPidentifier 2.0 is a sequence-based antimicrobial peptide (AMP) prediction toolkit that expands its predecessor with a larger training dataset, a revised 22-descriptor feature set, and the addition of LightGBM as a fifth classifier. The training corpus was enlarged from 5,300 to 13,246 balanced sequences (6,623 AMP and 6,623 non-AMP) drawn from APD3, CAMP, LAMP, and UniProt, with homology reduction applied to limit redundancy. Feature extraction was redesigned to include global peptide descriptors, amino acid functional group fractions, and positional free-energy-of-transition and solvent accessibility distributions, drawing on the descriptor frameworks of Bhadra et al. (2018)^17^ and Von Heijne and Blomberg (1979).^18^ Five individually accessible classifiers, Random Forest (RF), Support Vector Machine (SVM), Gradient Boosting (GB), XGBoost (XGB), and LightGBM (LGBM), are combined in a soft-voting ensemble that averages predicted AMP probabilities. On an internal held-out test set of 2,650 sequences, the soft-voting ensemble achieves accuracy 92.9%, Matthews correlation coefficient (MCC) 0.859, and AUC-ROC 0.977, improving over AMPidentifier 1.0 ensemble values of MCC 0.791 and AUC-ROC 0.958. On an independent 4,736-sequence benchmark, the ensemble achieves MCC 0.742 and AUC-ROC 0.950; all six AMPidentifier 2.0 configurations exceed the v1.0 RF reference of MCC 0.710 on the same dataset. All configurations are accessible through three deployment modes with identical predictive behavior: a command-line interface (CLI), a Python package via PyPI (`pip install ampidentifier`), and a web application at https://www.lgbv-ufpe.net/AMPidentifier. Source code, pre-trained model artifacts, and the full benchmark evaluation are freely available at https://github.com/madsondeluna/AMPIdentifier under the MIT License.

## Introduction

Antimicrobial peptides (AMPs) are short bioactive polypeptides, typically 10 to 50 residues in length, whose cationic charge and amphipathic architecture drive disruption of microbial membranes through electrostatic and hydrophobic interactions.^1,2^ With the global spread of multidrug-resistant pathogens,^3^ AMPs are candidate alternatives or adjuncts to conventional antibiotics across clinical therapeutics, agricultural applications, and food preservation.^4,5^

Experimental screening for novel AMPs is resource-intensive. Sequence-based machine learning classifiers operating directly on primary amino acid sequences have therefore become a central tool in computational prescreening pipelines: they require no three-dimensional structural data and can evaluate large candidate sets in seconds.^6^ Published sequence-based predictors include CAMPR3,^7^ AMPScanner v2,^8^ AMPlify,^9^ amPEPpy,^10^ ampir,^11^ and several web-based platforms.^12,13^ These tools differ in algorithmic architecture, feature representation, training data, and operating threshold, complicating direct performance comparisons.

Two recurring limitations affect the AMP prediction landscape. First, web server availability is transient: of twelve servers surveyed in the AMPidentifier 1.0 evaluation, nine were inaccessible at the time of testing.^14^ This imposes a practical barrier to comparative benchmarking and limits long-term reproducibility. Second, most tools expose only a single prediction output and do not allow users to examine or compare the decision behavior of underlying classifiers. Physicochemical characterization of candidate sequences is almost always performed with a separate tool, adding steps to an already multi-stage pipeline.

AMPidentifier 1.0 addressed these limitations by providing four pre-trained classifiers (RF, GB, XGB, SVM) alongside a majority-vote ensemble, integrated physicochemical descriptor export, and three deployment modes (CLI, PyPI package, and web server) operating on identical model artifacts.^14^ AMPidentifier 2.0 extends the framework in four directions: (1) the training corpus was enlarged 2.5-fold to 13,246 sequences; (2) the feature set was redesigned from 28 raw descriptors to 22 descriptors covering functional group composition and positional sequence properties; (3) LightGBM was added as a fifth classifier; (4) the majority-vote ensemble was replaced with a soft-voting ensemble that averages predicted probabilities. All three deployment modes continue to operate on identical model artifacts, and all five individual classifiers remain independently accessible for per-model inspection.

This paper describes the AMPidentifier 2.0 implementation, reports internal and independent benchmark performance for all six prediction configurations, and documents the design choices behind each change relative to version 1.0.

## Application overview

AMPidentifier 2.0 accepts a FASTA file as input and executes a four-step prediction pipeline (Figure 1). In step one, sequences are parsed and validated. In step two, 22 physicochemical features are computed per sequence using the modlamp library.^15^ In step three, the selected model is applied: each sequence is assigned a probability $P(\text{AMP}) \in [0, 1]$, which is thresholded at the MCC-optimized value for the selected classifier to produce binary AMP/non-AMP labels. In step four, two output files are written to the user-specified directory: `physicochemical_features.csv`, a per-sequence feature table containing all 22 descriptors, and `predictions_[model].csv`, containing sequence identifiers, AMP probabilities, and predicted labels.

Users select a prediction configuration via the `--model` flag (`rf`, `svm`, `gb`, `xgb`, `lgbm`, or `voting`; default: `voting`). A custom probability threshold can be provided via `--threshold`; if omitted, the MCC-optimized threshold for the selected model is applied automatically. This design allows users to adjust the sensitivity-specificity trade-off without re-running inference. All three deployment modes below operate on the same pre-trained model artifacts and normalization parameters, producing identical predictions for any given input.

### Command-line interface

The CLI is the primary deployment mode for users working in Unix-based environments. After registering the `ampidentifier2` command (see Installation section of the repository README), predictions are executed as:

```
ampidentifier2 -i sequences.fasta -o results/ --model voting
```

Required arguments are `-i` (input FASTA path) and `-o` (output directory). The `--model` flag accepts `rf`, `svm`, `gb`, `xgb`, `lgbm`, or `voting`. An optional `--threshold` argument overrides the default MCC-optimized threshold for the selected model. The terminal output includes a per-run summary box reporting total sequences, AMP and non-AMP counts with percentages, the decision threshold applied, and the output file path. The CLI is compatible with macOS, Linux, and Windows Subsystem for Linux (WSL).

### Python package

AMPidentifier 2.0 is installable as a Python package via the Python Package Index:

```
pip install ampidentifier
```

After installation, the same `ampidentifier2` entry point is available from the command line, and the `amp_identifier` package can be imported programmatically to call the prediction pipeline directly from Python scripts or Jupyter notebooks. The package requires Python >= 3.10 and is registered at https://pypi.org/project/ampidentifier.

### Web application

The AMPidentifier web application at https://www.lgbv-ufpe.net/AMPidentifier provides a browser-based interface that does not require local software installation. Users paste or upload sequences in FASTA format, select a model, and download the two output CSV files. The web server runs the same pre-trained models as the CLI and PyPI package and produces byte-identical output files. The interface is accessible from any device with a browser and an internet connection, including mobile devices, making it suitable for exploratory use and for users without programming experience.

## Implementation

### Dataset and data partitioning

The training dataset contains 13,246 sequences: 6,623 experimentally confirmed AMPs and 6,623 non-AMP peptide sequences. AMP sequences were drawn from APD3,^19^ CAMP,^20^ and LAMP;^21^ non-AMP sequences were drawn from UniProt^22^ entries annotated as lacking antimicrobial activity. To limit homology-derived redundancy, sequences sharing more than 80% pairwise identity were clustered with CD-HIT,^23^ and one representative sequence per cluster was retained. The dataset is balanced (1:1 AMP:non-AMP). It was partitioned 80/20 by stratified random sampling into a training set of 10,596 sequences and a held-out test set of 2,650 sequences (1,325 per class). The test set was not used during hyperparameter optimization; it served exclusively for threshold calibration and final performance evaluation.

### Feature extraction

Each amino acid sequence $S$ is projected to a 22-dimensional feature vector $\mathbf{x} \in \mathbb{R}^{22}$. The 22 features belong to four groups.

**Global physicochemical descriptors (7 features)**: net charge at pH 7.0, isoelectric point (pI), instability index, aliphatic index, Boman index, hydrophobic ratio, and hydrophobic moment (Eisenberg scale, $\alpha$-helix angle 100°). These are computed via the modlamp library GlobalDescriptor and PeptideDescriptor classes.^15^

**Grouped amino acid composition (9 features)**: the fraction of residues belonging to each of nine functional groups: acidic (D, E), basic (K, R, H), polar (S, T, N, Q), nonpolar (A, V, L, I, M, F, Y, W, P), aliphatic (A, V, L, I, M), aromatic (F, Y, W), charged (D, E, K, R, H), small (A, G, S, D, T), and tiny (A, G, S). Group definitions follow Jhong et al. (2019)^12^ and Nagarajan et al. (2018).^6^ These nine features replace the 20 per-residue molar fractions used in AMPidentifier 1.0, encoding the same compositional information in a more compact representation that emphasizes biochemical character.

**Free energy of transition positional features (3 features)**: for each of three FET groups defined by Von Heijne and Blomberg (1979),^18^ the feature value is the relative position of the first group-member residue in the sequence (position index plus one, divided by sequence length). FET groups: low-FET residues (I, L, V, W, A, M, G, T; hydrophobic, membrane-preferring), intermediate-FET residues (F, Y, S, Q, C, N), and high-FET residues (P, H, K, E, D, R; hydrophilic, membrane-avoiding).^18^

**Solvent accessibility positional features (3 features)**: relative position of the first residue in each of three solvent accessibility groups: buried (A, L, F, C, G, I, V, W), exposed (R, K, Q, E, N, D), and intermediate (M, S, P, T, H, Y). Group definitions follow Bhadra et al. (2018).^17^

Relative position features (FET and solvent accessibility) encode where in the sequence specific chemical environments first appear. This positional information is absent from global descriptors and per-residue composition features, and has been shown to discriminate AMPs from non-AMPs in sequence-based classifiers.^17^

Compared to AMPidentifier 1.0, which used 28 features (8 global descriptors plus the 20-dimensional per-residue molar fraction vector), AMPidentifier 2.0 uses 22 features by replacing the molar fraction vector with grouped composition features and adding the FET and solvent accessibility positional terms. After computing the 22 candidate descriptors, a feature selection procedure confirmed all 22 as informative; results are in Supplementary Table S2.

Because descriptors span heterogeneous numeric scales, model-specific normalization is applied before inference. RF, GB, XGB, and LGBM use a **RobustScaler** fitted on the training partition, which centers on the median and scales by the interquartile range; this choice reduces the influence of outlier sequences with extreme physicochemical values. SVM uses a **StandardScaler** (zero mean, unit variance), required for stable optimization of the margin-based objective. The voting ensemble model handles normalization internally within a scikit-learn VotingClassifier pipeline. All scaling parameters are estimated exclusively on the training partition and serialized as deployment artifacts alongside the model pickle files.

### Pre-trained classifiers

Five machine learning architectures were trained on the 10,596-sequence training partition. Hyperparameter optimization used RandomizedSearchCV (n_iter = 100, 5-fold stratified cross-validation, AUC-ROC as the optimization metric). Complete hyperparameter search spaces and best configurations are reported in Supplementary Table S1.

**Random Forest (RF)** constructs an ensemble of decision trees by bootstrap aggregation. At each split, a random subset of features (max_features = 0.30) decorrelates the trees and reduces variance. Best configuration: 229 estimators, no depth constraint, min_samples_leaf = 1, min_samples_split = 3. Cross-validation AUC-ROC = 0.9695.

**Support Vector Machine (SVM)** maximizes the margin between class hyperplanes in a feature space projected by the RBF kernel $K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2\right)$. Best configuration: $C = 2.80$, $\gamma = 0.10$. Cross-validation AUC-ROC = 0.9671.

**Gradient Boosting (GB)** is a sequential additive model that minimizes log-loss by fitting each successive tree to the pseudo-residuals of the current ensemble. Best configuration: 293 estimators, learning rate 0.062, max depth 6, subsample 0.846. Cross-validation AUC-ROC = 0.9723.

**XGBoost (XGB)** extends GB with explicit $\ell_1$ (alpha) and $\ell_2$ (lambda) regularization on leaf weights and second-order Taylor approximation for split finding. Best configuration: 448 estimators, learning rate 0.059, max depth 6, $\alpha = 0.013$, $\lambda = 0.313$, subsample 0.678, colsample_bytree 0.758. Cross-validation AUC-ROC = 0.9741.

**LightGBM (LGBM)** builds gradient-boosted trees using histogram-based split finding and leaf-wise (best-first) growth, which reduces training time on larger datasets relative to level-wise GB while allowing finer control of tree shape via the num_leaves parameter. Best configuration: 383 estimators, 47 leaves, max depth 7, learning rate 0.323, subsample 0.942, colsample_bytree 0.581, $\alpha = 0.001$, $\lambda = 0.681$. Cross-validation AUC-ROC = 0.9740.

After hyperparameter optimization, each model was retrained on the full training partition and serialized. MCC-optimized decision thresholds were determined on the held-out test set by scanning predicted probabilities and selecting the threshold that maximizes MCC.

### Soft-voting ensemble

The voting ensemble averages the five predicted AMP probabilities:

$$P_{\text{voting}}(\text{AMP}) = \frac{1}{5} \sum_{k=1}^{5} P_k(\text{AMP})$$

where $k$ indexes RF, SVM, GB, XGB, and LGBM. A scikit-learn VotingClassifier object with `voting='soft'` encapsulates all five classifiers and their respective scalers into a single pipeline. At inference time, the raw 22-feature matrix is passed directly to the ensemble, which dispatches scaling internally before calling each sub-classifier. The ensemble threshold (0.56) was optimized on the held-out test set by the same MCC criterion applied to individual models.

Averaging probabilities integrates each classifier's confidence rather than treating all votes as equal, which is more informative when component classifiers assign markedly different probability values to borderline sequences.

## Results

### Internal test set performance

Table 1 reports classification performance on the held-out test set (2,650 sequences: 1,325 AMP, 1,325 non-AMP) for all six configurations.

**Table 1.** Classification performance on the internal held-out test set ($n$ = 2,650; 1,325 AMP, 1,325 non-AMP).

| Model  | Threshold | Accuracy | Precision | Recall | Specificity | F1    | MCC   | AUC-ROC |
|--------|-----------|----------|-----------|--------|-------------|-------|-------|---------|
| RF     | 0.56      | 91.9%    | 93.8%     | 89.7%  | 94.1%       | 91.7% | 0.839 | 0.972   |
| SVM    | 0.47      | 91.9%    | 91.8%     | 92.1%  | 91.8%       | 91.9% | 0.839 | 0.969   |
| GB     | 0.55      | 92.0%    | 92.9%     | 90.9%  | 93.1%       | 91.9% | 0.839 | 0.974   |
| XGB    | 0.48      | 92.2%    | 92.0%     | 92.4%  | 91.9%       | 92.2% | 0.843 | 0.974   |
| LGBM   | 0.71      | 92.7%    | 94.2%     | 91.1%  | 94.3%       | 92.6% | 0.855 | 0.975   |
| Voting | 0.56      | 92.9%    | 94.2%     | 91.4%  | 94.4%       | 92.8% | 0.859 | 0.977   |

The voting ensemble achieves the highest values across all five primary metrics: accuracy 92.9%, F1 92.8%, MCC 0.859, and AUC-ROC 0.977. LGBM is the strongest individual classifier (MCC 0.855, AUC-ROC 0.975), followed by XGB (MCC 0.843). RF, SVM, and GB each reach MCC 0.839.

Relative to AMPidentifier 1.0, where the majority-vote ensemble achieved MCC 0.791 and AUC-ROC 0.958 on a smaller internal test set, v2.0 shows gains of 0.068 MCC units and 0.019 AUC-ROC units. Three factors jointly contribute to this improvement: the training corpus increased 2.5-fold (5,300 to 13,246 sequences), the feature set was revised from 28 raw descriptors to 22 descriptors, and hyperparameters were re-optimized on the larger dataset. Disentangling the relative contribution of each factor would require controlled ablation experiments; the aggregate improvement confirms that the combined changes are beneficial.

### Independent benchmark performance

The independent benchmark contains 4,736 sequences (2,368 AMP, 2,368 non-AMP) not used in any stage of model training, threshold calibration, or feature selection. Table 2 reports performance for all six AMPidentifier 2.0 configurations.

**Table 2.** Classification performance on the independent benchmark ($n$ = 4,736; 2,368 AMP, 2,368 non-AMP).

| Model  | Threshold | Accuracy | Precision | Recall | Specificity | F1    | MCC   | AUC-ROC |
|--------|-----------|----------|-----------|--------|-------------|-------|-------|---------|
| RF     | 0.56      | 86.4%    | 81.5%     | 94.1%  | 78.7%       | 87.4% | 0.736 | 0.948   |
| SVM    | 0.47      | 84.1%    | 78.5%     | 93.9%  | 74.2%       | 85.5% | 0.695 | 0.943   |
| GB     | 0.55      | 85.8%    | 80.5%     | 94.5%  | 77.0%       | 86.9% | 0.727 | 0.935   |
| XGB    | 0.48      | 84.6%    | 78.7%     | 94.8%  | 74.4%       | 86.0% | 0.707 | 0.930   |
| LGBM   | 0.71      | 87.0%    | 82.2%     | 94.5%  | 79.6%       | 87.9% | 0.749 | 0.948   |
| Voting | 0.56      | 86.6%    | 81.4%     | 94.9%  | 78.4%       | 87.6% | 0.742 | 0.950   |

All six AMPidentifier 2.0 configurations exceed the AMPidentifier 1.0 RF reference values of MCC 0.710 and AUC-ROC 0.935 on this benchmark. Benchmark accuracy (84.1% to 87.0%) is lower than internal test accuracy (91.9% to 92.9%), consistent with the expected performance gap when models are evaluated on sequences from different sources than the training data.

Recall is high across all configurations (93.9% to 94.9%), indicating that each classifier recovers the large majority of true AMPs. Specificity is lower (74.2% to 79.6%): the classifiers accept a proportion of true non-AMP sequences as predicted AMPs. This asymmetry is common in AMP predictors trained on curated database sequences and evaluated against more diverse independent sets, where the marginal non-AMP sequences may share local compositional features with AMPs.

Among individual classifiers, LGBM achieves the highest benchmark accuracy (87.0%) and MCC (0.749), marginally exceeding the voting ensemble (86.6%, MCC 0.742). The voting ensemble produces the highest recall (94.9%) and AUC-ROC (0.950). ROC curves for all configurations on the independent benchmark are shown in Figure 2.

### Comparison with published predictors

Table 3 summarizes the performance of AMPidentifier 2.0 configurations against published AMP predictors evaluated on the same 4,736-sequence independent benchmark. Complete per-tool metrics for all 16 evaluated classifiers are reported in Supplementary Table S4.

**Table 3.** Selected classifier performance on the independent benchmark ($n$ = 4,736). External tool values are carried from Luna-Aragão et al. (2026).^14^ MCC and AUC-ROC are threshold-dependent and threshold-independent discriminative metrics, respectively.

| Tool                    | MCC   | AUC-ROC | Sensitivity | Specificity |
|-------------------------|-------|---------|-------------|-------------|
| AMPidentifier 2.0 LGBM  | 0.749 | 0.948   | 94.5%       | 79.6%       |
| AMPidentifier 2.0 Voting| 0.742 | 0.950   | 94.9%       | 78.4%       |
| AMPidentifier 2.0 RF    | 0.736 | 0.948   | 94.1%       | 78.7%       |
| AMPScanner v2           | 0.718 | 0.936   | --          | --          |
| AMPidentifier 1.0 RF    | 0.710 | 0.935   | --          | --          |
| AMPlify                 | --    | 0.932   | --          | --          |
| AMPidentifier 1.0 Voting| 0.697 | --      | --          | 77.7%       |
| amPEPpy                 | --    | 0.934   | 96.5%       | 49.3%       |

All three AMPidentifier 2.0 configurations with MCC reported exceed the AMPScanner v2 reference (MCC 0.718) and the AMPidentifier 1.0 RF (MCC 0.710). The AMPidentifier 2.0 voting ensemble (MCC 0.742) improves over the AMPidentifier 1.0 voting ensemble (MCC 0.697) by 0.045 MCC units, with a simultaneous gain in specificity (78.4% vs. 77.7%).

The amPEPpy classifier illustrates a trade-off that appears repeatedly in the benchmark: AUC-ROC = 0.934 reflects good probability ranking, but at its default operating threshold, specificity falls to 49.3%, meaning the tool predicts approximately half of all true non-AMP sequences as AMPs. At genomic scale, this false-positive rate generates a large volume of spurious candidates. AMPidentifier 2.0 configurations maintain specificity between 74.2% and 79.6% across all six configurations, limiting false-positive inflation while preserving high recall (93.9% to 94.9%).

## Discussion

The performance gap between internal test results (ensemble MCC 0.859, AUC-ROC 0.977) and independent benchmark results (ensemble MCC 0.742, AUC-ROC 0.950) reflects domain shift between the training data sources (APD3, CAMP, LAMP, UniProt) and the independent benchmark. This magnitude of gap is typical for AMP predictors: tools trained and evaluated within the same database family consistently achieve higher MCC than when evaluated against independently curated sequences. Users should weight the independent benchmark numbers more heavily when assessing expected real-world performance.

The revision from 28 to 22 features had two components. Removing molecular weight and the 20 per-residue molar fractions eliminated features that provide overlapping information (amino acid composition is partially encoded by the global descriptors and group fraction features). Adding the FET and solvent accessibility positional features introduced sequence-positional information, which cannot be recovered from scalar global descriptors or composition vectors alone. These positional terms encode where, relative to sequence length, the first membrane-preferring, intermediate, or membrane-avoiding residue appears, a property that has been linked to AMP targeting function.^17^ The feature selection step confirmed all 22 features as informative (Supplementary Table S2).

The choice of RobustScaler for tree-based classifiers differs from AMPidentifier 1.0, which applied StandardScaler to all models. Decision-tree-based methods are not sensitive to feature scale in the same way that margin-based or distance-based methods are; however, RobustScaler limits the effect of extreme physicochemical values (e.g., instability indices for sequences rich in destabilizing residues) on the scaler's center estimate, which makes the normalized feature space more consistent across diverse input sequences. SVM retains StandardScaler because its optimization requires scale-standardized inputs.

LightGBM's addition as a fifth classifier is motivated by two properties. Its leaf-wise growth strategy, combined with histogram-based split finding, trained faster than XGB or GB on the 10,596-sequence training partition without sacrificing CV AUC-ROC (LGBM: 0.9740 vs. XGB: 0.9741 vs. GB: 0.9723). On the independent benchmark, LGBM achieves the highest individual MCC (0.749), outperforming RF (0.736), GB (0.727), XGB (0.707), and SVM (0.695). Its inclusion in the voting ensemble broadens the ensemble's decision basis and contributes the highest-performing individual classifier by benchmark MCC.

The physicochemical descriptor export remains a practical differentiator from comparable tools. Each run produces a per-sequence table of all 22 descriptors, which users can apply for downstream candidate ranking by biophysical criteria (e.g., filtering by Boman index above a threshold or requiring net charge within a specified range) without invoking a separate characterization tool.

The three deployment modes (CLI, PyPI, web server) maintain identical predictive behavior. This architecture is the same as AMPidentifier 1.0, which demonstrated that web-server-only distribution creates reproducibility risk given the high offline rate of published AMP predictors.^14^ Distributing pre-trained model artifacts within the repository and supporting `pip install ampidentifier` keeps the tool available independently of server infrastructure. The registration of AMPidentifier with INPI (Registration No. BR-51-2025-005859-4) provides an additional layer of version provenance.

The principal limitation of AMPidentifier 2.0, shared with sequence-based AMP classifiers trained on database sequences, is that training data over-represents well-characterized AMP families from specific organisms and activity classes. Performance on AMPs with non-canonical structures, atypical amino acid compositions, or activities outside the training distribution may differ from the benchmark estimates reported here. For such sequences, AMP probability values should be treated as a screening score, and experimental validation remains the definitive confirmation step.

## Summary and conclusions

AMPidentifier 2.0 provides sequence-based AMP prediction through five independently accessible machine learning classifiers and a soft-voting ensemble. The training dataset was expanded to 13,246 balanced sequences, features were revised to a 22-descriptor set covering global peptide properties, functional group fractions, and positional FET and solvent accessibility distributions, and LightGBM was added as a fifth classifier. On an internal test set of 2,650 sequences, the voting ensemble achieves accuracy 92.9%, MCC 0.859, and AUC-ROC 0.977. On an independent benchmark of 4,736 sequences, the ensemble achieves MCC 0.742 and AUC-ROC 0.950; all six v2.0 configurations exceed v1.0 reference values on this dataset. Per-sequence physicochemical descriptor tables are produced alongside binary predictions in all deployment modes. AMPidentifier 2.0 is available as a command-line tool (`ampidentifier2`), a PyPI package (`pip install ampidentifier`, Python >= 3.10), and a web application at https://www.lgbv-ufpe.net/AMPidentifier.

## Associated content

### Data availability statement

AMPidentifier 2.0 is freely available under the MIT License at https://github.com/madsondeluna/AMPIdentifier. The independent benchmark dataset, evaluation scripts, and all figures are available at https://github.com/madsondeluna/AMPidentifierBenchmark. AMPidentifier is registered with the INPI, Registration No. BR-51-2025-005859-4.

### Supporting information

S1: Hyperparameter search spaces and best configurations for all five classifiers.

S2: Feature selection procedure and discriminative contribution of all 22 descriptors.

S3: ROC curves, calibration curves, and feature importance plots from the internal held-out test set.

S4: Full performance tables for all six AMPidentifier 2.0 configurations on the independent benchmark.

## Author contributions

M.A.L.A.: Conceptualization, Methodology, Software, Validation, Formal analysis, Data curation, Writing, original draft. R.L.S.: Software, Validation, Writing, review and editing. J.P.B.N.: Validation, Writing, review and editing. D.E.S.S.: Methodology, Writing, review and editing. C.A.S.S.: Supervision, Writing, review and editing. A.M.B.I.: Conceptualization, Resources, Supervision, Project administration, Funding acquisition, Writing, review and editing. All authors approved the final manuscript.

## Funding sources

This work was supported by the Coordenação de Aperfeiçoamento de Pessoal de Nível Superior (CAPES), Conselho Nacional de Desenvolvimento Científico e Tecnológico (CNPq), Fundação de Amparo à Pesquisa do Estado de Minas Gerais (FAPEMIG), and Fundação de Amparo à Pesquisa do Estado de Pernambuco (FACEPE).

## Notes

The authors declare no conflict of interest.

## Acknowledgments

The authors thank FAPEMIG for M.A.L.A.'s PhD scholarship. The authors acknowledge the National Laboratory for Scientific Computing (LNCC/MCTI, Brazil) for providing HPC resources of the SDumont supercomputer.

## Abbreviations

Acc, Accuracy; AMP, Antimicrobial Peptide; AUC-ROC, Area Under the Receiver Operating Characteristic Curve; CLI, Command-Line Interface; CV, Cross-Validation; FET, Free Energy of Transition; F1, F1-score; FN, False Negatives; FP, False Positives; GB, Gradient Boosting; LGBM, LightGBM; MCC, Matthews Correlation Coefficient; PyPI, Python Package Index; RBF, Radial Basis Function; RF, Random Forest; SA, Solvent Accessibility; Sn, Sensitivity; Sp, Specificity; SVM, Support Vector Machine; TN, True Negatives; TP, True Positives; XGB, XGBoost.

## References

(1) Hancock, R. E. W.; Sahl, H.-G. Antimicrobial and host-defense peptides as new anti-infective therapeutic strategies. *Nat. Biotechnol.* **2006**, *24*, 1551-1557.

(2) Zasloff, M. Antimicrobial peptides of multicellular organisms. *Nature* **2002**, *415*, 389-395.

(3) World Health Organization. *Global Antimicrobial Resistance and Use Surveillance System (GLASS) Report*; WHO: Geneva, 2022.

(4) Mishra, B.; Reiling, S.; Bhattacharya, D.; Bhattacharya, S. Host defense antimicrobial peptides as antibiotics: design and application strategies. *Curr. Opin. Chem. Biol.* **2017**, *38*, 87-96.

(5) Brogden, K. A. Antimicrobial peptides: pore formers or metabolic inhibitors in bacteria? *Nat. Rev. Microbiol.* **2005**, *3*, 238-250.

(6) Nagarajan, D.; Nagarajan, T.; Roy, N.; Kulkarni, O.; Ravichandran, S.; Mishra, M.; Bhattacharyya, S.; Chandra, N. Computational antimicrobial peptide design and evaluation against multidrug-resistant clinical isolates of bacteria. *J. Biol. Chem.* **2018**, *293*, 3492-3509.

(7) Waghu, F. H.; Barai, R. S.; Gurung, P.; Idicula-Thomas, S. CAMPR3: a database on sequences, structures and signatures of antimicrobial peptides. *Nucleic Acids Res.* **2016**, *44*, D1094-D1097.

(8) Veltri, D.; Kamath, U.; Shehu, A. Deep learning improves antimicrobial peptide recognition. *Bioinformatics* **2018**, *34*, 2740-2747.

(9) Li, C.; Sutherland, D.; Hammond, S. A.; Yang, C.; Taho, F.; Bergman, L.; Houston, S.; Warren, R. L.; Wong, T.; Hoang, L. M. N.; Cameron, C. E.; Helbing, C. C.; Birol, I. AMPlify: a multi-label AMP annotation tool with deep learning. *BMC Genomics* **2022**, *23*, 77.

(10) Lawrence, T. J.; Carper, D. L.; Spangler, M. K.; Batzer, M. A.; Willyard, A.; Glenn, T. C.; Matz, M. V.; Bhattacharya, D.; Bhattacharya, S. amPEPpy 1.0: a portable and accurate antimicrobial peptide prediction tool. *Bioinformatics* **2021**, *37*, 2058-2060.

(11) Fingerhut, L. C. H. W.; Miller, D. J.; Strugnell, J. M.; Daly, N. L.; Cooke, I. R. ampir: an R package for fast genome-wide prediction of antimicrobial peptides. *Bioinformatics* **2020**, *36*, 5262-5263.

(12) Jhong, J.-H.; Chi, Y.-H.; Li, W.-C.; Lin, T.-H.; Huang, K.-Y.; Lee, T.-Y. dbAMP: an integrated resource for exploring antimicrobial peptides with functional activities and physicochemical properties on transcriptomics analysis. *Nucleic Acids Res.* **2019**, *47*, D285-D297.

(13) Meher, P. K.; Sahu, T. K.; Saini, V.; Rao, A. R. Predicting antimicrobial peptides with improved accuracy by incorporating the compositional, physico-chemical and structural features into Chou's general PseAAC. *Sci. Rep.* **2017**, *7*, 42362.

(14) Luna-Aragão, M. A. et al. AMPidentifier 1.0: a cross-platform command-line toolkit for antimicrobial peptide prediction using ensemble machine learning. *J. Chem. Inf. Model.* **2026**, in press.

(15) Müller, A. T.; Gabernet, G.; Hiss, J. A.; Schneider, G. modlAMP: Python for antimicrobial peptides. *Bioinformatics* **2017**, *33*, 2753-2755.

(16) Santos-Junior, C. D.; Pan, S.; Zhao, X.-M.; Coelho, L. P. Macrel: antimicrobial peptide screening in genomes and metagenomes. *PeerJ* **2020**, *8*, e10555.

(17) Bhadra, P.; Yan, J.; Li, J.; Fong, S.; Siu, S. W. I. AmPEP: sequence-based prediction of antimicrobial peptides using distribution patterns of amino acid properties and random forest. *Sci. Rep.* **2018**, *8*, 1697.

(18) Von Heijne, G.; Blomberg, C. Trans-membrane translocation of proteins. *Eur. J. Biochem.* **1979**, *97*, 175-181.

(19) Wang, G.; Li, X.; Wang, Z. APD3: the antimicrobial peptide database as a tool for research and education. *Nucleic Acids Res.* **2016**, *44*, D1087-D1093.

(20) Waghu, F. H.; Gopi, L.; Barai, R. S.; Ramteke, P.; Nizami, B.; Idicula-Thomas, S. CAMP: Collection of sequences and structures of antimicrobial peptides. *Nucleic Acids Res.* **2014**, *42*, D1154-D1158.

(21) Zhao, X.; Wu, H.; Lu, H.; Li, G.; Huang, Q. LAMP: a database linking antimicrobial peptides. *PLoS ONE* **2013**, *8*, e66557.

(22) UniProt Consortium. UniProt: the universal protein knowledgebase in 2023. *Nucleic Acids Res.* **2023**, *51*, D523-D531.

(23) Fu, L.; Niu, B.; Zhu, Z.; Wu, S.; Li, W. CD-HIT: accelerated for clustering the next-generation sequencing data. *Bioinformatics* **2012**, *28*, 3150-3152.
