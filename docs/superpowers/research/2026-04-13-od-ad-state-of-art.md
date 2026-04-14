# OD / AD State of the Art — Research Notes (2026-04-13)

> Compiled for PyOD v3.2.0 od-expert skill deepening.
> All findings filtered for compatibility with PyOD's ADEngine inventory (60 detectors across tabular / time_series / graph / text / image / multimodal).
> Search window: 2024-01 through 2026-04. Time-boxed ~40 min of web search.

## Survey papers (arxiv ID + year + venue)

- **Deep Learning Advancements in Anomaly Detection: A Comprehensive Survey** — Huang et al., arxiv:2503.13195 (2025-03, IEEE IoT Journal 2025). Reviews 180+ deep AD studies, categorizes reconstruction-based vs prediction-based, discusses hybrid traditional+deep approaches. https://arxiv.org/abs/2503.13195
- **Deep Graph Anomaly Detection: A Survey and New Perspectives** — arxiv:2409.09957 (2024-09, TKDE 2025). Taxonomy of deep GAD across 13 fine-grained categories organized by GNN backbone design, proxy task design, and graph anomaly measures. Companion repo at github.com/mala-lab/Awesome-Deep-Graph-Anomaly-Detection. https://arxiv.org/abs/2409.09957
- **Deep Learning for Time Series Anomaly Detection: A Survey** — Darban et al., arxiv:2211.05244 (published ACM Computing Surveys 2024, DOI 10.1145/3691338). Benchmark / dataset compendium across univariate & multivariate. https://dl.acm.org/doi/10.1145/3691338
- **A Survey on Diffusion Models for Anomaly Detection** — arxiv:2501.11430 (2025-01, IJCAI 2025 submission). Reviews diffusion-based AD methods; interpretability flagged as key challenge. https://arxiv.org/abs/2501.11430
- **Dive into Time-Series Anomaly Detection: A Decade Review** — arxiv:2412.20512 (2024-12). Ten-year retrospective on TSAD, taxonomy and evaluation pitfalls. https://arxiv.org/abs/2412.20512
- **Open Challenges in Time Series Anomaly Detection: An Industry Perspective** — Paparrizos et al., arxiv:2502.05392 (2025-02). Position paper from a cloud-deployed system arguing academic definitions miss streaming, HITL, point-process, conditional, and population-scale concerns. https://arxiv.org/abs/2502.05392

## Top method papers by modality

### Tabular
- **Unsupervised Anomaly Detection Algorithms on Real-world Data: How Many Do We Need?** — NeurIPS 2024 (poster 98305). Largest-ever comparison: 33 unsupervised algos × 52 tabular datasets. Finding: Extended Isolation Forest dominates global anomalies, k-NN dominates local. Maps to PyOD IForest/KNN. https://neurips.cc/virtual/2024/poster/98305
- **AnoLLM: Large Language Models for Tabular Anomaly Detection** — Tsai et al., ICLR 2025. Uses LLMs as tabular anomaly detectors. Maps to PyOD LLMAD (tabular+text). https://assets.amazon.science/f3/e4/9033ae94402eb468072da852f55c/anollm-large-language-models-for-tabular-anomaly-detection.pdf
- **Unsupervised Anomaly Detection for Tabular Data Using Noise Evaluation** — AAAI 2025 (DOI 10.1609/aaai.v39i11.33257). Learns decision boundary using clean + synthetically noised datasets. https://dl.acm.org/doi/10.1609/aaai.v39i11.33257
- **Dense Projection for Anomaly Detection** — AAAI 2024. Projection-based boundary estimation; a conceptual cousin of LMDD/PCA. https://ojs.aaai.org/index.php/AAAI/article/view/
- **Data-Efficient and Interpretable Tabular Anomaly Detection** — KDD 2023 (DOI 10.1145/3580305.3599294). Interpretable tabular AD with minimal labels; still a frequently-cited reference through 2024-2025.
- **NLP-ADBench** — Li et al., arxiv:2412.04784 (2024-12, EMNLP Findings 2025). Established baseline: two-step OpenAI-embedding + classical OD beats specialized end-to-end text AD. Maps to PyOD EmbeddingOD. https://arxiv.org/abs/2412.04784
- **Impact of Inaccurate Contamination Ratio on Robust Unsupervised Anomaly Detection** — arxiv:2408.07718 (2024-08). Shows robust AD models are surprisingly insensitive (and sometimes improved) when contamination ratio is misspecified. Directly relevant to PyOD's `contamination=` hyperparameter. https://arxiv.org/abs/2408.07718
- **Deep Positive-Unlabeled Anomaly Detection for Contaminated Unlabeled Data** — arxiv:2405.18929 (2024-05). Combines PU learning with deep AD when unlabeled pool is contaminated. https://arxiv.org/abs/2405.18929
- **Unsupervised Anomaly Detection in the Presence of Missing Values** — NeurIPS 2024. Handles missingness natively rather than via imputation. https://proceedings.neurips.cc/paper_files/paper/2024/file/f99f7b22ad47fa6ce151730cf8d17911-Paper-Conference.pdf
- **Robust Conformal Outlier Detection under Contaminated Reference Data** — ICML 2025 (poster 43852). Conformal framework with guarantees when reference set has anomalies. https://icml.cc/virtual/2025/poster/43852

### Time series
- **The Elephant in the Room: Towards A Reliable Time-Series Anomaly Detection Benchmark (TSB-AD)** — Liu, Paparrizos et al., NeurIPS 2024 Datasets & Benchmarks. 1070 curated series / 40 datasets / 40 algorithms. Key finding: simple stat methods beat SOTA transformers; VUS-PR is the most reliable metric. https://openreview.net/forum?id=R6kJtWsTGy
- **TSB-AutoAD: Towards Automated Solutions for Time-Series Anomaly Detection** — Liu et al., VLDB 2025 (pvldb/vol18/p4364). AutoML layer over TSB-AD for detector selection. https://www.vldb.org/pvldb/vol18/p4364-liu.pdf
- **CATCH: Channel-Aware Multivariate TSAD via Frequency Patching** — ICLR 2025. Patch-wise masked-attention on frequency spectra for multivariate series. https://openreview.net/forum?id=m08aK3xxdJ
- **Multi-Resolution Decomposable Diffusion Model for TSAD** — ICLR 2025. Coarse-to-fine diffusion to handle non-stationarity. https://proceedings.iclr.cc/paper_files/paper/2025/file/9f7f063144103bf6debb09a3f15e00fb-Paper-Conference.pdf
- **GCAD: Anomaly Detection in Multivariate Time Series from the Perspective of Granger Causality** — AAAI 2025. Gradient-based Granger graph + sparsification. https://ojs.aaai.org/index.php/AAAI/article/view/34096
- **Towards Unbiased Evaluation of Time-series Anomaly Detector** — arxiv:2409.13053 (NeurIPS 2024). Proposes "Balanced Point Adjustment" to fix F1 inflation from the classical PA protocol. https://arxiv.org/abs/2409.13053
- **THEMIS** — arxiv:2510.03911 (2025-10). Uses frozen Chronos encoder embeddings + unsupervised OD, SOTA on MSL. https://arxiv.org/html/2510.03911
- **When Foundation Models are One-Liners** (OpenReview 2025). Shows MOMENT, Chronos, TimesFM, Time-MoE, TSPulse often tied with moving-window variance baselines for TSAD. https://openreview.net/forum?id=H27kvyG4qf

### Graph
- **UniGAD: Unifying Multi-level Graph Anomaly Detection** — NeurIPS 2024. Shared model across node/edge/graph-level anomaly tasks. https://proceedings.neurips.cc/paper_files/paper/2024/file/f57de20ab7bb1540bcac55266ebb5401-Paper-Conference.pdf
- **Generative Semi-supervised Graph Anomaly Detection (GGAD)** — NeurIPS 2024. Generates outlier nodes for one-class training. https://proceedings.neurips.cc/paper_files/paper/2024/file/085b4b5d1f81ad9e057ad2b3de922ad4-Paper-Conference.pdf
- **ARC: A Generalist Graph Anomaly Detector with In-Context Learning** — NeurIPS 2024. In-context learning for GAD. https://proceedings.neurips.cc/paper_files/paper/2024/file/5acb720a361eecb34ee62d356859d246-Paper-Conference.pdf
- **How to Use Graph Data in the Wild to Help Graph Anomaly Detection?** — Cao et al., KDD 2025. Leverages external graphs to augment training.
- **Graph Anomaly Detection with Few Labels: A Data-Centric Approach** — Ma et al., KDD 2024. Data-centric framing of few-shot GAD.
- **SpaceGNN: Multi-Space GNN for Node AD with Extremely Limited Labels** — ICLR 2025. Works with 1-5 labels per class.
- **Kernel-Aware Graph Prompt Learning for Few-Shot Anomaly Detection** — AAAI 2025. Graph prompt learning for few-shot GAD.
- **A Generalizable Anomaly Detection Method in Dynamic Graphs** — AAAI 2025 (view 35508). Dynamic graph AD.
- **Dynamic Spectral Graph Anomaly Detection** — AAAI 2025. Spectral features over time.
- **TGTOD: Global Temporal Graph Transformer for Outlier Detection at Scale** — Liu et al., 2024. Temporal graph transformer for OD.
- **UniGOF / UB-GOLD: Unifying Unsupervised Graph-Level OD and AD** — ICLR 2025. Benchmark library for graph-level OD/AD.

### Text / image / multimodal
- **Text-ADBench: Text Anomaly Detection Benchmark based on LLMs Embedding** — arxiv:2507.12295 (2025-07). Companion benchmark to NLP-ADBench, focuses on LLM embedding backbones. https://arxiv.org/abs/2507.12295
- **TAD-Bench** — arxiv:2501.11960 (2025-01). Embedding-based text AD benchmark. https://arxiv.org/html/2501.11960v1
- **Advancing Video Anomaly Detection: A Concise Review and a New Dataset** — NeurIPS 2024 Datasets & Benchmarks. https://proceedings.neurips.cc/paper_files/paper/2024/file/a3c5af1f56fc73eef1ba0f442739f5ca-Paper-Datasets_and_Benchmarks_Track.pdf
- **Towards Multi-Domain Learning for Generalizable Video Anomaly Detection** — NeurIPS 2024.
- **One-for-All Few-Shot Anomaly Detection via Prompt Generation** — ICLR 2025. Class-shared prompt generator + text-guided normality descriptions. https://proceedings.iclr.cc/paper_files/paper/2025/file/9f7f2f57d8eaf44b2f09020f64ff6d96-Paper-Conference.pdf
- **COBRA: Adversarially Robust AD** — ICLR 2025. +26.1 AUROC improvement on adversarial settings.
- **AD-LLM: Benchmarking Large Language Models for Anomaly Detection** — arxiv:2412.11142 (2024-12). Evaluates zero-shot, data-augmentation, and model-selection use cases for LLMs in AD. https://arxiv.org/abs/2412.11142
- **LogLLM: Log-based Anomaly Detection Using LLMs** — arxiv:2411.08561 (2024-11). BERT-based extractor + Llama classifier. https://arxiv.org/abs/2411.08561
- **Large Language Models for Forecasting and Anomaly Detection: A Systematic Literature Review** — arxiv:2402.10350 (2024-02). https://arxiv.org/abs/2402.10350

## Recent benchmarks

- **ADBench** (NeurIPS 2022, still current reference) — Han et al., arxiv:2206.09426. 57 datasets, 30 algorithms, 98k experiments, covers unsupervised/semi-supervised/supervised. Actively maintained at github.com/Minqi824/ADBench. No full 2024-2025 successor paper, but the dataset/algo set is still the de-facto benchmark. https://arxiv.org/abs/2206.09426
- **TSB-AD** (NeurIPS 2024) — Liu/Paparrizos. 1070 series / 40 datasets / 40 algos. Key deliverable is VUS-PR as the reliable metric and a ranked leaderboard showing statistical methods often beat deep models. Tutorial + notebooks released 2025-10. https://thedatumorg.github.io/TSB-AD/
- **TSB-AutoAD** (VLDB 2025) — Automated detector selection over TSB-AD. https://www.vldb.org/pvldb/vol18/p4364-liu.pdf
- **BOND** — Liu et al. 2022, still the unsupervised node GAD benchmark referenced through 2025 in the mala-lab GAD survey. The 2025 TKDE survey explicitly builds on BOND's structural/contextual split. https://github.com/pygod-team/pygod
- **UB-GOLD** — ICLR 2025. Unified benchmark for unsupervised graph-level OD / AD. https://proceedings.iclr.cc/paper_files/paper/2025/file/e58fa42d4b7a798eef8d0d75098f87ad-Paper-Conference.pdf
- **NLP-ADBench** — arxiv:2412.04784 (2024-12, EMNLP Findings 2025). 8 datasets, 19 algorithms (3 end-to-end, 16 two-step). Key finding: two-step (embedding + classical OD) consistently beats specialized end-to-end. https://arxiv.org/abs/2412.04784
- **Text-ADBench / TAD-Bench** — 2025 text-specific benchmarks on LLM embeddings. Complementary to NLP-ADBench. https://arxiv.org/abs/2507.12295
- **DGraph / DGraphFin** — NeurIPS 2022, still the benchmark for large-scale financial GAD (3M nodes / 4M edges, leaderboard at dgraph.xinye.com/leaderboards/dgraphfin). Used by KDD 2025 GAD papers. https://arxiv.org/abs/2207.03579

## Industry case studies

### Fraud detection
- **Stripe Radar — How we built it** — stripe.dev/blog 2024. ML fraud prevention trained on hundreds of billions in payments; claims 38% average fraud reduction. https://stripe.dev/blog/how-we-built-it-stripe-radar
- **Stripe — Using AI to create dynamic, risk-based Radar rules** — stripe.com/blog 2024. Combines ML scores with issuer CVC / postal-code responses in real time, delivered 1.3 pp payment success lift. https://stripe.com/blog/using-ai-dynamic-radar-rules
- **Stripe Payments Foundation Model** — 2025. Reported card-testing attack detection 59%→97% after foundation-model upgrade. https://stripe.com/blog/using-ai-optimize-payments-performance-payments-intelligence-suite
- **Uber — Project RADAR: Early Fraud Detection with Humans in the Loop** — uber.com/blog. Hybrid ML + human review on streaming time-series fraud signals. https://www.uber.com/blog/project-radar-intelligent-early-fraud-detection/
- **Uber — Risk Entity Watch** — uber.com/blog. Templated anomaly-model pipeline; time-normalized entity-level features (analogous to PyOD's `contamination`/threshold workflow). https://www.uber.com/blog/risk-entity-watch/
- **Uber — Fraud Detection: Using Relational Graph Learning to Detect Collusion** — uber.com/blog. Graph-based collusion detection. https://www.uber.com/blog/fraud-detection/
- **Netflix — Machine Learning for Fraud Detection in Streaming Services** — netflixtechblog.com. Framework for unexpected-streaming-behavior detection via model + data-driven methods. https://netflixtechblog.com/machine-learning-for-fraud-detection-in-streaming-services-b0b4ef3be3f6

### Observability / monitoring
- **Datadog — Watchdog: Auto-detect performance anomalies without setting alerts** — datadoghq.com/blog. Automatic statistical baseline + multivariate alerts since 2018; extended through 2024-2025 to Infra + Kubernetes + Database Monitoring. https://www.datadoghq.com/blog/watchdog/
- **Datadog — Detect anomalies before they become incidents with Datadog AIOps** — 2025. Early anomaly detection framing for incident prevention. https://www.datadoghq.com/blog/early-anomaly-detection-datadog-aiops/
- **Datadog — Toto time-series foundation model** — 2025. Domain-tuned foundation model for multivariate observability AD; reported top-ranked on internal benchmark at 5.495. https://www.datadoghq.com/blog/datadog-time-series-foundation-model/
- **Netflix — Title Launch Observability at Netflix Scale** — netflixtechblog.com 2025-01. Anomaly detection on title-display logs. https://netflixtechblog.com/title-launch-observability-at-netflix-scale-c88c586629eb
- **Netflix — How and Why Netflix Built a Real-Time Distributed Graph** — netflixtechblog.com 2025-10. Graph DB supports pattern + anomaly detection use cases. https://netflixtechblog.com/how-and-why-netflix-built-a-real-time-distributed-graph-part-1-ingesting-and-processing-data-80113e124acc
- **Meta — HawkEye: AI debugging at Meta** — engineering.fb.com 2023-12 (still the canonical Meta ML observability reference in 2024-2025). Data drift + label anomaly diagnostics; order-of-magnitude debug-time reduction. https://engineering.fb.com/2023/12/19/data-infrastructure/hawkeye-ai-debugging-meta/

### Cloud-vendor platforms & other
- **Databricks — Training 10,000 Anomaly Detection Models on 1B Records with Explainable Predictions** — databricks.com/blog 2024-10. Explicitly uses PyOD's ECOD algorithm as the core detector for manufacturing fault detection at scale ("DAXS"). Direct PyOD reference. https://www.databricks.com/blog/training-10000-anomaly-detection-models-one-billion-records-explainable-predictions
- **Databricks — Building robust IoT streaming anomaly detection with Delta Live Tables** — databricks.com/blog 2024-10. https://www.databricks.com/blog/near-real-time-iot-robust-anomaly-detection-framework
- **Databricks — Anomaly Detection using Embeddings and GenAI** — community.databricks.com 2026-02. Embedding-based OD pattern for fraud. https://community.databricks.com/t5/technical-blog/anomaly-detection-using-embeddings-and-genai/ba-p/95564
- **AWS — Efficiently build and tune custom log anomaly detection models with Amazon SageMaker** — aws.amazon.com/blogs/machine-learning 2025-01. Full log-AD pipeline with model registry. https://aws.amazon.com/blogs/machine-learning/efficiently-build-and-tune-custom-log-anomaly-detection-models-with-amazon-sagemaker/
- **AWS — Real-Time Anomaly Detection in Streaming Time Series with Apache Flink** — 2024-09. https://aws.amazon.com/blogs/machine-learning/anomaly-detection-in-streaming-time-series-data-with-online-learning-using-amazon-managed-service-for-apache-flink/
- **Google Cloud — Unsupervised anomaly detection for time series and non-time series with BigQuery ML** — cloud.google.com/blog. AI.DETECT_ANOMALIES / ML.DETECT_ANOMALIES SQL interface. https://cloud.google.com/blog/products/data-analytics/bigquery-ml-unsupervised-anomaly-detection
- **Google — SPADE: semi-supervised AD under distribution mismatch** — opensource.googleblog.com 2024-05. Open-sourced semi-supervised detector for handful-of-labels scenarios. https://opensource.googleblog.com/2024/05/semi-supervised-anomaly-detection-under-distribution-mismatch.html
- **Snowflake — ML-Based Forecasting and Anomaly Detection in Cortex** — snowflake.com/blog 2024. SQL-native GBM + ARIMA-style AD functions (GA). https://www.snowflake.com/en/blog/ml-based-forecast-anomaly-detection-cortex/

## Data science blogs (lighter touch)
- **Towards Data Science — Introducing Anomaly/Outlier Detection in Python with PyOD** — 2025-01-29. Reintroduction of PyOD for the TDS audience. https://towardsdatascience.com/introducing-anomaly-outlier-detection-in-python-with-pyod-40afcccee9ff/
- **Towards Data Science — A Practical Toolkit for Time Series Anomaly Detection, Using Python** — 2025-12. Mixed toolkit article. https://towardsdatascience.com/a-practical-toolkit-for-time-series-anomaly-detection-using-python/
- **Towards Data Science — Boosting Your Anomaly Detection With LLMs** — 2025-09-04. Covers LLM-driven model selection in PyOD 2. https://towardsdatascience.com/boosting-your-anomaly-detection-with-llms/
- **KDnuggets — We Used 5 Outlier Detection Methods on a Real Dataset: They Disagreed on 96% of Flagged Samples** — 2026-03. Practical pitfall demo; 5-method ensemble on wine dataset. https://www.kdnuggets.com/we-used-5-outlier-detection-methods-on-a-real-dataset-they-disagreed-on-96-of-flagged-samples
- **PyCharm Blog — Anomaly Detection in Machine Learning Using Python** — blog.jetbrains.com 2025-01. PyOD-walk-through article for developer audience. https://blog.jetbrains.com/pycharm/2025/01/anomaly-detection-in-machine-learning/

(Sebastian Raschka has not published OD-specific pieces in 2024-2025 — his focus has been LLM/architecture research. Distill has not published OD content. Omitted rather than fabricated.)

## Newly recognized pitfalls (2024-2025 literature)

1. **Point-adjustment F1 inflation on time series** — Towards Unbiased Evaluation of TSAD, arxiv:2409.13053 (NeurIPS 2024) — The classical PA protocol inflates F1 from 0.32 to 0.85 on random scores; use Balanced Point Adjustment or VUS-PR instead. Applies when benchmarking PyOD's time_series detectors (TimeSeriesOD, MatrixProfile, LSTMAD, AnomalyTransformer, SpectralResidual, KShape, SAND).
2. **Foundation-model TSAD frequently ties with one-liner baselines** — "When Foundation Models are One-Liners", OpenReview 2025 — MOMENT / Chronos / TimesFM / Time-MoE / TSPulse do not reliably beat moving-window variance or squared-difference on standard TSB-AD tasks. Reinforces PyOD docs note that AnomalyTransformer is "research use only".
3. **Simple statistical TSAD methods beat SOTA transformers on TSB-AD** — Liu et al., NeurIPS 2024 — Leaderboard finding that simpler methods often dominate. Maps to PyOD recommendation to try SpectralResidual / MatrixProfile / SAND before LSTMAD / AnomalyTransformer.
4. **Detector disagreement is the norm, not the exception** — KDnuggets 2026-03 — 5 classic OD methods disagreed on 96% of flagged wine-dataset samples. Only points flagged by 3+ methods are trustworthy. Directly motivates PyOD's ensemble detectors (SUOD, LSCP, FeatureBagging).
5. **Misspecified contamination ratios are surprisingly harmless (for robust detectors)** — arxiv:2408.07718 (2024-08) — Robust unsupervised detectors are not adversely affected by wrong contamination hyper-parameters and sometimes improve. Good news for PyOD users who can't estimate contamination.
6. **Training data contamination silently degrades one-class models** — Deep PU-AD, arxiv:2405.18929 (2024-05) — Unlabeled pools often contain anomalies; naive one-class training degrades. Argues for semi-supervised variants (PyOD: DevNet, XGBOD) when even weak labels exist.
7. **Controlled-lab AD benchmarks do not transfer to production** — "Beyond Academic Benchmarks" (arxiv:2503.23451, 2025-03); "Open Challenges in TSAD" (arxiv:2502.05392, 2025-02) — Academic benchmarks use artificial anomalies; real production data shows significant performance degradation and needs streaming / HITL / conditional AD that academic setups ignore.
8. **Concept drift / non-stationarity is the dominant production failure mode** — Uber Risk Entity Watch blog + industrial AD reviews 2025 — Fixed thresholds and static models erode as distributions evolve; online recalibration is essential. PyOD tooling: SAND (designed for non-stationary series), LODA (streaming-friendly).
9. **Two-step OD (embedding + classical detector) dominates specialized end-to-end for text** — NLP-ADBench, arxiv:2412.04784 — OpenAI embeddings + classical OD > end-to-end text AD. Directly validates PyOD's EmbeddingOD wrapper.
10. **Fraud patterns manifest as time-series anomalies on populations of entities, not point anomalies** — Uber Project RADAR — Fraud attacks appear as anomalous shifts in aggregated entity metrics. Use TimeSeriesOD over tabular PyOD detectors for fraud-bursts; use graph detectors (DOMINANT/CoLA/CONAD) for collusion rings.

## Best-practice patterns

1. **Ensemble / multi-detector agreement for trustworthy flags** — KDnuggets 2026-03, ADBench 2022. Use PyOD's SUOD, LSCP, FeatureBagging to run multiple base detectors and only trust points flagged by the majority.
2. **Two-step architecture for text/image/multimodal** — NLP-ADBench 2024-12. Embed with a frozen foundation model, then run a classical PyOD detector on the embeddings. Maps to PyOD's EmbeddingOD and MultiModalOD.
3. **Prefer robust + interpretable detectors as production baselines** — Databricks DAXS blog 2024-10 + arxiv:2503.13195 survey. ECOD, COPOD, HBOS, IForest, PCA are the reliable production backbones; deep methods add value mainly when data structure justifies it.
4. **Use VUS-PR (or similar range-based) metrics for time-series evaluation** — TSB-AD NeurIPS 2024. Replace AUROC/point-F1 with VUS-PR on time_series detectors.
5. **Start with simple stat methods on time series before reaching for transformers** — TSB-AD 2024. Order of attack: SpectralResidual / MatrixProfile > KShape / SAND > LSTMAD > AnomalyTransformer (last resort).
6. **Isolation Forest as default for tabular, LOF when local density matters** — NeurIPS 2024 "How Many Do We Need?". IForest dominates global anomalies, LOF / KNN dominate local. Maps to PyOD IForest (global) vs LOF / KNN / COF (local).
7. **Use labeled anomalies when you have any — even a handful** — DevNet / SPADE / ADBench 2022. Even 5-10 labeled anomalies beat pure unsupervised. Maps to PyOD DevNet and XGBOD.
8. **Add human-in-the-loop review before rules promote to production** — Uber Project RADAR blog. Analyst validation of flagged rules reduces FP and protects UX.
9. **Monitor detector drift, not just data drift** — Meta HawkEye + Databricks 2025 data-quality monitor. Track AUROC / alert rate on a holdout set and retrain when it degrades.
10. **Auto-select the detector per dataset rather than universal-model search** — TSB-AutoAD (VLDB 2025), NLP-ADBench (EMNLP Findings 2025). "No single model dominates" is the consistent finding. Directly aligned with PyOD ADEngine's planned detector routing.

## Worked example raw material

Short list of concrete real-world OD scenarios that could become worked examples in the od-expert skill:

1. **Credit-card fraud point detection** — tabular — IForest + ECOD + LOF ensemble — dataset: Kaggle CC fraud / Stripe Radar analogue — outcome: production-grade baseline (Stripe reports 38% fraud reduction on real Radar). Source: Stripe Radar blogs 2024.
2. **Manufacturing sensor fault detection at scale** — tabular — ECOD per-asset (10k models / 1B rows) — outcome: explainable per-sensor anomaly scores on IoT data. Source: Databricks "DAXS / 10,000 models" blog 2024-10.
3. **Cloud-infrastructure metric anomaly** — time_series multivariate — SpectralResidual / MatrixProfile baseline, Toto-style foundation model for advanced case — outcome: Datadog Watchdog-style auto-baselining. Source: Datadog Watchdog / Toto blogs 2024-2025.
4. **Streaming IoT anomaly detection** — time_series streaming — SAND or LODA — outcome: near-real-time detection on Delta Live Tables. Source: Databricks IoT AD blog 2024-10.
5. **Document / log NLP anomaly** — text — EmbeddingOD with OpenAI/BERT embeddings + IForest — outcome: NLP-ADBench validated two-step pattern. Source: NLP-ADBench arxiv:2412.04784.
6. **Financial transaction graph fraud** — graph — DOMINANT / CoLA / CONAD on DGraphFin-style data — outcome: catches collusion rings unreachable by tabular methods. Source: DGraph benchmark + Uber relational-graph blog.
7. **User-session fraud bursts** — time_series over entity aggregates — TimeSeriesOD wrapping IForest — outcome: Uber Risk Entity Watch pattern, time-normalized entity features. Source: Uber Risk Entity Watch blog.
8. **Title-launch observability** — time_series on business KPI counters — MatrixProfile for seasonality + KShape for subsequence — outcome: Netflix Title Launch pattern. Source: Netflix Tech Blog 2025-01.
9. **Supply-chain energy-loss detection** — tabular — autoencoder / VAE + HBOS ensemble — outcome: energy-loss prevention. Source: Databricks energy-loss blog 2024.
10. **Log anomaly detection** — text over parsed logs — EmbeddingOD + LLMAD — outcome: SageMaker custom log-AD pipeline. Source: AWS SageMaker log AD 2025-01.

## PyOD gap list (literature recommends but PyOD does not yet ship)

1. **Extended Isolation Forest (EIF)** — NeurIPS 2024 "How Many Do We Need?" — named as the strongest global-anomaly detector across 52 tabular datasets. PyOD ships IForest and DIF but not EIF specifically.
2. **Time-series foundation model encoder wrappers (Chronos / MOMENT / TimesFM / Toto)** — THEMIS 2025-10, Datadog Toto 2025 — Encode series with a frozen TSFM then run classical OD on embeddings. PyOD's TimeSeriesOD does not yet wrap TSFM encoders. This is a direct analogue of EmbeddingOD for time_series.
3. **Balanced Point Adjustment (BA) evaluation metric** — arxiv:2409.13053 — Evaluation-side utility, not a detector; PyOD's time_series evaluation tooling could ship VUS-PR + BA implementations.
4. **Deep PU-learning AD for contaminated unlabeled data** — arxiv:2405.18929 — PyOD has DevNet/XGBOD for labeled-anomaly semi-supervised, but no PU-specific framework for the "unlabeled pool is contaminated" case.
5. **SPADE (Google semi-supervised under distribution mismatch)** — opensource.googleblog.com 2024-05 — Open-source semi-supervised detector not currently in PyOD.
6. **AnoLLM (LLM-as-tabular-detector)** — ICLR 2025 — PyOD's LLMAD exists, but a dedicated tabular-focused implementation following the AnoLLM paper design could be a gap worth closing.
7. **CATCH (frequency-patching multivariate TSAD)** — ICLR 2025 — Channel-aware frequency-patching detector for multivariate series. PyOD's multivariate TSAD coverage (LSTMAD, AnomalyTransformer) does not include this frequency-patching family.
8. **Multi-Resolution Decomposable Diffusion Model for TSAD** — ICLR 2025 — Diffusion-based TSAD. PyOD has no diffusion-based time-series detector.
9. **GGAD / ARC (generative / in-context GAD)** — NeurIPS 2024 — PyOD's graph detectors (DOMINANT / CoLA / CONAD / AnomalyDAE) are static; no generative or in-context learning variant.
10. **UniGAD (multi-level GAD — node + edge + graph)** — NeurIPS 2024 — PyOD's graph detectors are node-level only. Edge-level and graph-level AD is a structural gap.
11. **Conformal outlier detection with coverage guarantees** — Robust Conformal OD ICML 2025 — Conformal wrappers are increasingly standard; PyOD does not yet ship a conformal-prediction layer around detectors.
12. **AutoAD / automated detector selection** — TSB-AutoAD VLDB 2025, AD-AGENT arxiv:2505.12594 — Though PyOD ADEngine is heading this way, neither TSB-AutoAD's automated per-dataset detector-selection nor a fully agentic pipeline is shipped as a first-class PyOD API yet.
13. **Dynamic/adaptive thresholding layer** — multiple 2024-2025 papers — Online threshold recalibration using recent score statistics is a recurring pattern in production blogs; PyOD detectors currently expose only static `contamination` / `threshold_` at fit time.

## Open questions

- Does PyOD's ADEngine routing agree with TSB-AD's finding that simple stat methods outperform deep TSAD? Worth validating programmatically.
- Which time-series foundation-model encoder (Chronos vs MOMENT vs Toto) produces the best embeddings for a PyOD-style two-step TSAD pipeline? THEMIS says Chronos, but this has not been independently reproduced.
- How should ADEngine combine TSB-AD (NeurIPS 2024) ranking signals with ADBench (2022) ranking signals when the user's data spans both tabular and time_series modalities?
- What is the correct failure mode for PyOD detectors when `contamination=` is clearly wrong? Current literature (2408.07718) suggests "don't worry" but this is detector-dependent.
- Should PyOD publish a VUS-PR + Balanced-Point-Adjustment evaluation utility as part of v3.2.0? The TSB-AD leaderboard implies yes.
- Is there a consensus detector for "collective anomalies on entity populations" (Uber RADAR framing)? This sits between tabular, time_series, and graph and has no clean academic framing.
