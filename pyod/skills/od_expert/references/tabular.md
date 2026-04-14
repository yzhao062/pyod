# Tabular anomaly detection reference

PyOD's largest modality (44 of 61 detectors). The agent loads this file when the master decision tree (in SKILL.md) routes to tabular.

## Decision table by data shape (expert heuristics)

These are rules of thumb for reasoning about which detectors a non-expert would reach for by data shape, drawn from ADBench and general tabular OD literature. They are **not** predictions of exact `engine.plan` output. ADEngine's planner has its own routing logic and may return a different triple for the same shape; always read `state.plans` at runtime for the live selection, and use this table only to check whether the planner's choice is plausible.

| Data shape | Heuristic starters | Why |
|---|---|---|
| n < 1k | `ECOD`, `HBOS`, `IForest` | Cheap, robust, do not overfit on small samples |
| 1k ≤ n ≤ 100k | `IForest`, `ECOD`, `LOF` / `KNN` | Classic tabular triple; LOF for local density, KNN for proximity |
| n > 100k | `IForest`, `HBOS`, `COPOD` | Avoid distance-based methods (`LOF`, `KNN`) which scale poorly |
| High dimensional (D > 50) | `COPOD`, `SUOD`, `IForest` | Detector ensembles handle the curse of dimensionality |
| Sparse | `HBOS`, `ECOD` | Histogram and ECDF methods handle sparsity gracefully |
| Mixed (numerical + few categorical) | One-hot encode → `IForest` | After encoding, treat as standard tabular |
| Labeled rare anomalies (< 1%) | `XGBOD`, `DevNet` | Supervised methods exploit even a handful of labels |

## Detectors available in PyOD (KB-derived)

<!-- BEGIN KB-DERIVED: tabular-detector-list -->
- **ABOD** (Angle-Based Outlier Detection) — complexity: time O(n^2 * d) for fast, O(n^3 * d) for default, space O(n^2); best for: High-dimensional datasets where distance-based methods suffer from curse of dimensionality; avoid when: Dataset is very large or low-dimensional where distance-based methods work well; paper: Kriegel et al., KDD 2008
- **AE1SVM** (AutoEncoder with One-Class SVM) — complexity: time O(n * d * h * epochs), space O(d * h + n_sv * h); best for: Datasets benefiting from joint representation learning and one-class classification; avoid when: Simpler pipeline of separate AE + SVM works well, or dataset is small; requires: pyod[torch]; paper: Nguyen and Vien, ECML-PKDD 2019
- **ALAD** (Adversarially Learned Anomaly Detection) — complexity: time O(n * d * h * epochs), space O(d * h); best for: Scenarios where GAN-based detection is desired but fast inference is needed; avoid when: Simpler reconstruction-based deep methods suffice or dataset is small; requires: pyod[torch]; paper: Zenati et al., ICDM 2018
- **AnoGAN** (Anomaly Detection with Generative Adversarial Networks) — complexity: time O(n * d * h * epochs) + O(n * iterations) for inference, space O(d * h); best for: Complex data distributions where GAN-based generation quality is high; avoid when: Fast inference is needed or training instability is a concern; requires: pyod[torch]; paper: Schlegl et al., IPMI 2017
- **AutoEncoder** (Fully Connected AutoEncoder) — complexity: time O(n * d * h * epochs) where h is hidden size, space O(d * h); best for: Datasets with complex nonlinear structure where reconstruction-based scoring is appropriate; avoid when: Dataset is small, tabular and simple methods suffice, or training time is limited; requires: pyod[torch]; paper: Aggarwal, 2017
- **CBLOF** (Cluster-Based Local Outlier Factor) — complexity: time O(n * k * d), space O(n * d); best for: Data with well-separated clusters where outliers deviate from cluster structure; avoid when: Data has no meaningful cluster structure or clusters are heavily overlapping; paper: He et al., 2003
- **CD** (Cook's Distance) — complexity: time O(n * d^2), space O(n * d); best for: Identifying influential observations in regression settings; avoid when: No natural target variable exists or data is purely unsupervised; paper: Cook, 1977
- **COF** (Connectivity-Based Outlier Factor) — complexity: time O(n^2 * d), space O(n^2); best for: Datasets where outliers lie along sparse connectivity paths between clusters; avoid when: Dataset is large or simpler methods like LOF already perform well; paper: Tang et al., 2002
- **COPOD** (Copula-Based Outlier Detection) — complexity: time O(n * d * log(n)), space O(n * d); best for: Large-scale datasets where speed and interpretability matter and features are roughly independent; avoid when: Outliers only appear in joint distributions with strong feature dependencies; paper: Li et al., ICDM 2020
- **DIF** (Deep Isolation Forest) — complexity: time O(n * t * d * log(n)), space O(t * n); best for: Complex datasets where standard Isolation Forest misses anomalies due to axis-aligned splits; avoid when: Standard Isolation Forest performs well or dataset is small and simple; paper: Xu et al., TKDE 2023
- **DeepSVDD** (Deep Support Vector Data Description) — complexity: time O(n * d * h * epochs), space O(d * h); best for: One-class anomaly detection where a compact normal data description is desired; avoid when: Normal data is multi-modal or simpler one-class methods are sufficient; requires: pyod[torch]; paper: Ruff et al., ICML 2018
- **DevNet** (Deep Anomaly Detection with Deviation Networks) — complexity: time O(n * d * h * epochs), space O(d * h); best for: Semi-supervised anomaly detection where a small number of labeled anomalies are available; avoid when: No labeled anomalies are available or dataset is too small for deep learning; requires: pyod[torch]; paper: Pang et al., KDD 2019
- **ECOD** (Empirical Cumulative Distribution Functions) — complexity: time O(n * d * log(n)), space O(n * d); best for: General-purpose outlier detection when speed and interpretability are priorities; avoid when: Features are heavily correlated and outliers only manifest in joint distributions; paper: Li et al., TKDE 2022
- **FeatureBagging** (Feature Bagging Outlier Detection) — complexity: time O(n_estimators * base_detector_time), space O(n_estimators * base_detector_space); best for: High-dimensional data with potentially irrelevant features; avoid when: All features are relevant or a single strong detector suffices; requires: pyod[combo]; paper: Lazarevic and Kumar, KDD 2005
- **GMM** (Gaussian Mixture Model) — complexity: time O(n * k * d^2) per EM iteration, space O(k * d^2); best for: Data with multi-modal distributions that can be approximated by Gaussian mixtures; avoid when: Data does not follow mixture-of-Gaussians assumption or is very high-dimensional; paper: Aggarwal, 2017
- **HBOS** (Histogram-Based Outlier Score) — complexity: time O(n * d), space O(n_bins * d); best for: Large-scale datasets where speed is critical and features are roughly independent; avoid when: Outliers only manifest through feature interactions or correlations; paper: Goldstein and Dengel, KI 2012
- **HDBSCAN** (Hierarchical Density-Based Spatial Clustering of Applications with Noise) — complexity: time O(n * log(n)) to O(n^2), space O(n); best for: Datasets with variable-density clusters where noise points are outliers; avoid when: Data does not have cluster structure or very high-dimensional; paper: Campello et al., PAKDD 2013
- **IForest** (Isolation Forest) — complexity: time O(n * t * log(n)) where t is n_estimators, space O(t * n); best for: General-purpose anomaly detection especially on large or high-dimensional datasets; avoid when: Anomalies are local density deviations or features are strongly correlated; paper: Liu et al., ICDM 2008
- **INNE** (Isolation-based Anomaly Detection Using Nearest-Neighbor Ensembles) — complexity: time O(n * t * s) where t is n_estimators, s is sample size, space O(t * s); best for: Datasets where local density variations matter and Isolation Forest underperforms; avoid when: A simpler method like IForest already works well; paper: Bandaragoda et al., KAIS 2018
- **KDE** (Kernel Density Estimation) — complexity: time O(n^2 * d), space O(n * d); best for: Low-to-moderate dimensional data where non-parametric density estimation is desired; avoid when: Data is high-dimensional or dataset is very large; paper: Latecki et al., SDM 2007
- **KNN** (K-Nearest Neighbors Outlier Detection) — complexity: time O(n^2 * d), space O(n * d); best for: General-purpose distance-based outlier detection on moderate-sized datasets; avoid when: Dataset is very large or has highly variable local densities; paper: Ramaswamy et al., SIGMOD 2000
- **KPCA** (Kernel Principal Component Analysis) — complexity: time O(n^2 * d), space O(n^2); best for: Moderately sized datasets with nonlinear structure; avoid when: Dataset is very large due to quadratic kernel matrix or a linear model suffices; paper: Hoffmann, 2007
- **LLMAD** (LLM-Based Anomaly Detection) — complexity: time varies, space varies; best for: Zero-shot or few-shot anomaly detection leveraging LLM world knowledge; avoid when: Feature is needed before release or LLM API costs are prohibitive; paper: TBD
- **LMDD** (Linear Model Deviation-based Detection) — complexity: time O(n_iter * n * d), space O(n * d); best for: Multivariate data where anomalies are detectable through linear projections; avoid when: Anomalies require nonlinear feature combinations to detect; paper: Arning et al., KDD 1996
- **LOCI** (Local Correlation Integral) — complexity: time O(n^2 * d), space O(n^2); best for: Small to medium datasets where automatic threshold selection is valued; avoid when: Dataset is large or faster LOF-based methods are sufficient; paper: Papadimitriou et al., ICDE 2003
- **LODA** (Lightweight Online Detector of Anomalies) — complexity: time O(n * n_cuts * d), space O(n_bins * n_cuts); best for: Streaming or online anomaly detection with limited computational resources; avoid when: Batch setting with enough time for more powerful methods; paper: Pevny, 2016
- **LOF** (Local Outlier Factor) — complexity: time O(n^2 * d), space O(n * d); best for: Datasets with clusters of varying densities where local anomalies are of interest; avoid when: Dataset is very large (>50K) or data is uniformly distributed; paper: Breunig et al., SIGMOD 2000
- **LSCP** (Locally Selective Combination of Parallel Outlier Ensembles) — complexity: time O(n * n_detectors * base_cost), space O(n * n_detectors); best for: Scenarios where diverse base detectors are available and local performance varies; avoid when: Only one detector type is appropriate or computational budget is limited; paper: Zhao et al., SDM 2019
- **LUNAR** (Learnable Unified Neighborhood-based Anomaly Ranking) — complexity: time O(n * k * d + n * h * epochs), space O(n * k + d * h); best for: Datasets where learned neighborhood-based scoring outperforms handcrafted rules; avoid when: PyTorch is not available or simpler KNN/LOF methods work well; requires: pyod[torch]; paper: Goodge et al., AAAI 2022
- **MAD** (Median Absolute Deviation) — complexity: time O(n * log(n)), space O(n); best for: Univariate outlier detection with a robust central tendency measure; avoid when: Data is multivariate or relationships between features are important; paper: Iglewicz and Hoaglin, 1993
- **MCD** (Minimum Covariance Determinant) — complexity: time O(n * d^2), space O(d^2); best for: Multivariate Gaussian-like data requiring robust covariance estimation; avoid when: Data is non-Gaussian, very high-dimensional, or strongly nonlinear; paper: Rousseeuw and Driessen, 1999
- **MO_GAAL** (Multiple-Objective Generative Adversarial Active Learning) — complexity: time O(k * n * d * h * epochs) where k is number of generators, space O(k * d * h); best for: Complex datasets where diverse generated outlier references improve detection; avoid when: Computational resources are limited or simpler GAN approaches suffice; requires: pyod[torch]; paper: Liu et al., 2019
- **OCSVM** (One-Class Support Vector Machine) — complexity: time O(n^2 * d) to O(n^3), space O(n * d); best for: Medium-sized datasets where a flexible decision boundary is needed; avoid when: Dataset is very large or real-time training is required; paper: Scholkopf et al., 2001
- **PCA** (Principal Component Analysis) — complexity: time O(n * d^2), space O(d^2); best for: Datasets with linear structure where outliers deviate from main variance directions; avoid when: Data has strong nonlinear structure or outliers align with principal components; paper: Shyu et al., 2003
- **QMCD** (Quasi-Monte Carlo Discrepancy) — complexity: time O(n^2 * d), space O(n * d); best for: Detecting anomalies as deviations from uniform space-filling in moderate-dimensional data; avoid when: Dataset is very large or high-dimensional; paper: Fang et al., 2001
- **RGraph** (R-Graph Outlier Detection) — complexity: time O(n^2 * d + transition_steps * n^2), space O(n^2); best for: Datasets where graph connectivity and neighborhood structure reveal anomalies; avoid when: Dataset is very large or a simpler proximity method suffices; paper: You et al., AAAI 2017
- **ROD** (Rotation-Based Outlier Detection) — complexity: time O(n * d^2), space O(n * d); best for: Low-dimensional data where rotation-invariant outlier detection is desired; avoid when: Data is high-dimensional or a well-tuned alternative is available; paper: Almardeny et al., 2020
- **SOD** (Subspace Outlier Detection) — complexity: time O(n^2 * d), space O(n * d); best for: High-dimensional data where outliers deviate in axis-parallel subspaces; avoid when: Data is low-dimensional or anomalies require oblique subspaces to detect; paper: Kriegel et al., PAKDD 2009
- **SOS** (Stochastic Outlier Selection) — complexity: time O(n^2 * d), space O(n^2); best for: Medium-sized datasets where probability-calibrated outlier scores are valuable; avoid when: Dataset is very large (>10K samples) due to quadratic complexity; paper: Janssens et al., 2012
- **SO_GAAL** (Single-Objective Generative Adversarial Active Learning) — complexity: time O(n * d * h * epochs), space O(d * h); best for: Exploratory anomaly detection with GAN-generated reference outliers; avoid when: Stable and fast results are required, or dataset is small; requires: pyod[torch]; paper: Liu et al., 2019
- **SUOD** (Scalable Unsupervised Outlier Detection) — complexity: time varies (depends on base estimators), space varies; best for: Large-scale datasets where running multiple detectors is desired but time is limited; avoid when: Exact results from a single well-chosen detector are preferred; requires: pyod[suod]; paper: Zhao et al., MLSys 2021
- **Sampling** (Rapid Distance-Based Outlier Detection via Sampling) — complexity: time O(n * s * d) where s is subset size, space O(n * d); best for: Large-scale datasets requiring fast distance-based outlier detection; avoid when: Precise and deterministic results are required or dataset is small enough for exact methods; paper: Sugiyama and Borgwardt, 2013
- **VAE** (Variational AutoEncoder) — complexity: time O(n * d * h * epochs), space O(d * h); best for: Datasets where probabilistic reconstruction scoring and smooth latent spaces are beneficial; avoid when: Simpler autoencoder or non-deep methods work well, or dataset is very small; requires: pyod[torch]; paper: Kingma and Welling, 2014
- **XGBOD** (Extreme Gradient Boosting Outlier Detection) — complexity: time O(n * d * n_estimators * log(n)), space O(n * d); best for: Semi-supervised settings where some labeled anomalies are available; avoid when: No labeled data is available or a purely unsupervised approach is needed; requires: pyod[xgboost]; paper: Zhao and Hryniewicki, IJCNN 2018
<!-- END KB-DERIVED: tabular-detector-list -->

## Worked example: simple tabular fraud detection

### Setup

User provides 5,000 credit card transactions with 12 features (amount, time, merchant category, etc.). No labels. Wants to find suspicious transactions.

### Agent flow

```python
from pyod.utils.ad_engine import ADEngine
import pandas as pd

df = pd.read_csv("transactions.csv")
# 1 categorical column → one-hot encode
df = pd.get_dummies(df, columns=["merchant_category"])
X = df.values

engine = ADEngine()
state = engine.start(X)
# state.profile: {'data_type': 'tabular', 'n_samples': 5000, 'n_features': 27,
#                 'dimensionality_class': 'medium', 'has_nan': False, 'dtype': 'float64'}
# Modality: tabular ✓
# n in [1k, 100k] → top-3 chosen by the planner (see below)
# No triggers fire (no DL detectors, no missing extras)

state = engine.plan(state)
# [p['detector_name'] for p in state.plans[:3]] == ['IForest', 'ECOD', 'KNN']
# (Captured from a live probe on 2026-04-14. The planner may return a
# different triple as it evolves; always read state.plans at runtime.)

state = engine.run(state)
state = engine.analyze(state)
# state.quality: {agreement: 0.71, separation: 0.28, stability: 0.66}
# best_detector: ECOD

# No triggers fire. Proceed to report.

report = engine.report(state)
```

### What the agent reports

> "Found 200 of 5,000 transactions (4%) flagged as suspicious. The three detectors I ran (`IForest`, `ECOD`, `KNN`) agreed well (consensus 0.71). `ECOD` performed best.
>
> Top-5 most suspicious: rows [42, 77, 153, 891, 1240]. The dominant features driving their flagging were transaction amount (4 of 5) and time-of-day (3 of 5).
>
> Confidence: medium. Separation is moderate (0.28), so I recommend manually checking the top-5 against your fraud database.
>
> **What I assumed**:
> - Tabular, unlabeled
> - Contamination ~4% (estimated from the data)
> - Detectors: `IForest`, `ECOD`, `KNN` — the top-3 returned by the current planner for mid-size tabular data
> - Best detector: `ECOD`"

## Tabular-specific pitfalls

### T1. Treating ID columns as features

User CSVs often have a row ID column (sequential integers). Including it as a feature breaks distance-based detectors and adds noise to ensemble methods. Mitigation: detect monotonically increasing integer columns and drop them automatically. If unsure, ask the user.

### T2. Mixed-scale numeric features

Income ($) ranges from 1k to 1M; age ranges from 18 to 100. Without scaling, income dominates every distance computation. Mitigation: always `RobustScaler` before `LOF`, `KNN`, `OCSVM`, `CBLOF`.

### T3. Per-column contamination

Some datasets have outliers concentrated in 1-2 columns (e.g., latency monitoring with most outliers in a latency column). Detecting per-column instead of jointly can be more interpretable. Mitigation: when the user mentions "I want to know which feature is anomalous", run per-column detection (1D arrays) instead of joint.

### T4. Ensemble combination defaults

The default consensus combination is `mean`. For very imbalanced detector quality, `weighted_mean` (weighted by separation) can be better. Mitigation: ADEngine handles this if you use `engine.run` directly. Do not bypass.

### T5. ID-only embeddings without numerics

When the data is mostly identifiers (user IDs, product IDs), one-hot encoding creates a very sparse high-D matrix. Distance-based detectors give nonsense on such data. Mitigation: use `HBOS` or `ECOD` (frequency-based methods that handle sparsity gracefully), or learn embeddings via `EmbeddingOD` first.

## Tabular-specific escalation triggers

In addition to the global triggers in SKILL.md, watch for these tabular-specific cases:

- **Very wide tables (D > n)**: dimensionality exceeds samples → escalate, recommend PCA or feature selection first
- **Single-column tabular**: a 1-feature dataset is a 1D outlier problem → use simple statistics (z-score, IQR) instead of OD
- **All-binary features**: detection on one-hot data → escalate, recommend `HBOS` or categorical-aware methods

## See also

- `pitfalls.md` — extended pitfalls library (preprocessing → detection → analysis → reporting)
- `workflow.md` — the autonomous loop pattern
- `SKILL.md` — top-10 critical pitfalls and master decision tree
