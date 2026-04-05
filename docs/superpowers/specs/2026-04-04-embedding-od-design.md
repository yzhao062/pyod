# PyOD Multi-Modal Expansion: Foundation Model Bridge

**Date:** 2026-04-04
**Status:** Draft
**Version:** 2 (comprehensive)

---

## 1. Vision

Expand PyOD from a tabular-only outlier detection library into the unified interface for anomaly detection across modalities (text, image, tabular, mixed) by leveraging foundation model embeddings and LLM-based detection. This is delivered through three layers:

| Layer | Class | Mechanism | Paper backing |
|-------|-------|-----------|---------------|
| 1. EmbeddingOD | `EmbeddingOD` | Two-step: encode raw data + run any PyOD detector on embeddings | NLP-ADBench, TAD-Bench, Text-ADBench |
| 2. Presets | `EmbeddingOD.for_text()`, `.for_image()` | Benchmark-informed default configurations | NLP-ADBench, TAD-Bench |
| 3. LLM-based AD | `LLMAD` | Direct LLM prompting for zero-shot anomaly scoring | AD-LLM |

Each layer is independent and can ship separately.

---

## 2. Research Backing

### 2.1 The core finding: two-step beats end-to-end

**NLP-ADBench** (EMNLP Findings 2025; Li, Li, Xiao, Yang, Nian, Hu, Zhao) tested 19 algorithms on 8 text datasets. 16 were two-step methods (embedding + classical OD via PyOD); 3 were end-to-end. Results:

| Rank | Method | Avg AUROC | Type |
|------|--------|-----------|------|
| 1 | **OpenAI + LUNAR** | **0.8633** | Two-step |
| 2 | OpenAI + LOF | 0.8198 | Two-step |
| 3 | OpenAI + AE | 0.7480 | Two-step |
| 4 | DATE | 0.7663 | End-to-end |

Two-step methods with OpenAI embeddings dominated. The end-to-end method DATE only won on binary spam tasks.

**Embedding quality is decisive.** OpenAI embeddings (3072-dim) outperformed BERT (768-dim) by 20-80%:
- Emotion dataset: OpenAI+LUNAR 0.9328 vs BERT+LUNAR 0.5186 (+80%)
- YelpReview: OpenAI+LUNAR 0.9452 vs BERT+LUNAR 0.6522 (+45%)

**TAD-Bench** (Cao et al., Jan 2025) independently confirmed this with 8 embedding models x 8 detectors on 6 datasets. OpenAI family dominated. kNN and INNE were top detectors (LUNAR was not tested).

**Text-ADBench** (Jul 2025) stated it directly: "embedding quality significantly governs anomaly detection efficacy, and deep learning-based approaches demonstrate no performance advantage over conventional shallow algorithms (e.g., KNN, Isolation Forest) when leveraging LLM-derived embeddings."

**Implication for PyOD:** The detector side is already strong. The missing piece is the encoder bridge. EmbeddingOD fills exactly this gap.

### 2.2 Best detector per benchmark

| Benchmark | Best detector (with OpenAI embeddings) | Notes |
|-----------|----------------------------------------|-------|
| NLP-ADBench | LUNAR (0.86), then LOF (0.82) | LUNAR uses GNN-enhanced scoring |
| TAD-Bench | kNN, then INNE, then ECOD | LUNAR not tested |
| Both | OpenAI family embeddings dominant | Consistent across all tested datasets |

Detectors that performed poorly on text embeddings: IForest, SO-GAAL, DeepSVDD. These may suffer from high dimensionality of embeddings (3072-dim).

### 2.3 LLM-based zero-shot detection

**AD-LLM** (ACL Findings 2025; Yang, Nian, Li, Xu, Li, Li, Xiao, Hu, Rossi, Ding, Hu, Zhao) evaluated three LLM tasks for AD:

**Zero-shot detection** -- LLM directly scores text samples:
- GPT-4o achieved 0.9293 AUROC on AG News (vs best baseline 0.9226)
- GPT-4o achieved 0.9668 on IMDB Reviews (vs best baseline 0.7366 -- a 30% improvement)
- GPT-4o outperformed all two-step baselines on 4 of 5 datasets
- Llama 3.1-8B significantly weaker than GPT-4o

**Data augmentation** -- LLM generates synthetic normal data:
- Improved flexible models: LUNAR +11.3%, AE +8.5%
- Harmed rigid models: LOF -7.9%, IForest -5.2%

**Model selection** -- LLM recommends detector given dataset description:
- o1-preview most often picked OpenAI+LUNAR (13/25 queries)
- Recommendations approached top baselines but explanations were generic
- Conceptually related to PyOD's AutoModelSelector but uses paper abstracts rather than meta-learning

### 2.4 Image AD with pretrained embeddings

**AnomalyDINO** (WACV 2025): Frozen DINOv2 ViT-S features + memory bank distance scoring achieved 96.6% AUROC on MVTec-AD (1-shot), outperforming PatchCore (83.4%) and WinCLIP+ (93.1%).

**Key pattern:** Extract patch-level features from DINOv2, compute distance to normal memory bank, use top-k% distances as anomaly score. This is equivalent to EmbeddingOD with DINOv2 encoder + KNN detector.

### 2.5 Dimensionality reduction before OD

Empirical studies show:
- IForest + PCA is robust: equal or higher detection in 17/18 datasets when reduced to 2-3 dims
- Distance-based methods (LOF, KNN) benefit most from DR due to curse of dimensionality
- For high-dim embeddings (>100d), PCA before UMAP/detection is recommended
- Training time reduction of ~300% at half dimensionality

This justifies the `reduce_dim` parameter in EmbeddingOD.

### 2.6 Tabular AD with LLMs

**AnoLLM** (ICLR 2025, Amazon): Serializes tabular rows to text ("feature_name is value, ..."), fine-tunes SmolLM-135M, uses negative log-likelihood as anomaly score. Best on 6 mixed-type benchmarks (avg 0.803 AUROC). First method handling raw text columns without preprocessing.

This represents a complementary approach (generative, not embedding-based) that could be added later.

---

## 3. Layer 1: EmbeddingOD

### 3.1 Architecture

```
EmbeddingOD(BaseDetector)
    |
    |-- encoder: BaseEncoder (or str/callable resolved to BaseEncoder)
    |     |-- SentenceTransformerEncoder
    |     |-- HuggingFaceEncoder  
    |     |-- OpenAIEncoder
    |     |-- CallableEncoder
    |
    |-- detector: BaseDetector (or str resolved to BaseDetector)
    |     |-- Any of PyOD's 50+ detectors
    |
    |-- [optional] PCA for dimensionality reduction
```

### 3.2 BaseEncoder

```python
class BaseEncoder:
    """Abstract base for embedding encoders."""
    
    def encode(self, X, batch_size=32, show_progress=True) -> np.ndarray:
        """Convert raw input to 2D numpy array (n_samples, n_features).
        
        Parameters
        ----------
        X : list or array-like
            Raw input data (text strings, PIL images, file paths, etc.)
        batch_size : int
            Batch size for encoding.
        show_progress : bool
            Show progress bar during encoding.
            
        Returns
        -------
        embeddings : np.ndarray of shape (n_samples, n_features)
        """
        raise NotImplementedError
    
    def _validate_output(self, embeddings):
        """Ensure output is 2D float64 numpy array."""
        embeddings = np.asarray(embeddings, dtype=np.float64)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        check_array(embeddings)
        return embeddings
```

### 3.3 Encoder Implementations

#### SentenceTransformerEncoder

Wraps `sentence_transformers.SentenceTransformer`. This is the most common text embedding library and covers most HuggingFace text embedding models.

```python
class SentenceTransformerEncoder(BaseEncoder):
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None, 
                 normalize=False, truncate_dim=None):
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self.truncate_dim = truncate_dim
    
    def encode(self, X, batch_size=32, show_progress=True):
        if SentenceTransformer is None:
            raise ImportError(
                "SentenceTransformerEncoder requires 'sentence-transformers'. "
                "Install with: pip install sentence-transformers")
        if not hasattr(self, 'model_'):
            self.model_ = SentenceTransformer(
                self.model_name, device=self.device)
        embeddings = self.model_.encode(
            X, batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            truncate_dim=self.truncate_dim)
        return self._validate_output(embeddings)
```

**Supported models** (via sentence-transformers, all work with same API):
- `all-MiniLM-L6-v2` (384-dim, 22M params, fast)
- `all-mpnet-base-v2` (768-dim, 110M params, higher quality)
- `BAAI/bge-small-en-v1.5` (384-dim)
- `intfloat/multilingual-e5-large` (1024-dim, multilingual)
- Any model on HuggingFace with sentence-transformers tag

#### HuggingFaceEncoder

For models not compatible with sentence-transformers (vision models, custom architectures).

```python
class HuggingFaceEncoder(BaseEncoder):
    def __init__(self, model_name, device=None, 
                 pooling='cls', modality='auto'):
        self.model_name = model_name
        self.device = device
        self.pooling = pooling       # 'cls', 'mean'
        self.modality = modality     # 'text', 'image', 'auto'
```

**Text mode:** Uses `AutoTokenizer` + `AutoModel`, extracts [CLS] token or mean-pooled hidden states.

**Image mode:** Uses `AutoImageProcessor` + `AutoModel`, extracts [CLS] token from ViT output.

**Supported models:**
- `facebook/dinov2-small` (384-dim), `dinov2-base` (768-dim), `dinov2-large` (1024-dim)
- `openai/clip-vit-base-patch32` (512-dim)
- `google/vit-base-patch16-224` (768-dim)
- `bert-base-uncased` (768-dim) -- matches NLP-ADBench BERT baseline
- Any HuggingFace model with compatible architecture

#### OpenAIEncoder

For OpenAI API-based embeddings. These dominated in NLP-ADBench and TAD-Bench.

```python
class OpenAIEncoder(BaseEncoder):
    def __init__(self, model_name='text-embedding-3-small', 
                 dimensions=None, api_key=None):
        self.model_name = model_name
        self.dimensions = dimensions   # Matryoshka truncation
        self.api_key = api_key         # Falls back to OPENAI_API_KEY env var
```

**Batching:** OpenAI API accepts up to 2048 items per request. Encoder handles chunking internally.

**Supported models:**
- `text-embedding-3-small` (1536-dim, cheap)
- `text-embedding-3-large` (3072-dim, used in NLP-ADBench)
- Both support `dimensions` parameter for Matryoshka-style truncation

**Cost consideration:** Embedding via API costs money. The `cache_embeddings` parameter in EmbeddingOD stores training embeddings to avoid re-encoding. The OpenAI Batch API (50% cheaper, async) is out of scope for v1 but could be added later.

#### CallableEncoder

Escape hatch for any custom embedding function.

```python
class CallableEncoder(BaseEncoder):
    def __init__(self, fn):
        self.fn = fn
    
    def encode(self, X, batch_size=32, show_progress=True):
        embeddings = self.fn(X)
        return self._validate_output(embeddings)
```

Use cases: user's own fine-tuned model, Anthropic API embeddings, local Ollama models, pre-computed embeddings loaded from disk.

### 3.4 Encoder Registry and Resolution

```python
_ENCODER_REGISTRY = {
    # Sentence Transformers
    'all-MiniLM-L6-v2': ('sentence_transformer', 
        {'model_name': 'all-MiniLM-L6-v2'}),
    'all-mpnet-base-v2': ('sentence_transformer', 
        {'model_name': 'all-mpnet-base-v2'}),
    
    # OpenAI
    'text-embedding-3-small': ('openai', 
        {'model_name': 'text-embedding-3-small'}),
    'text-embedding-3-large': ('openai', 
        {'model_name': 'text-embedding-3-large'}),
    
    # HuggingFace Vision
    'dinov2-small': ('huggingface', 
        {'model_name': 'facebook/dinov2-small', 'modality': 'image'}),
    'dinov2-base': ('huggingface', 
        {'model_name': 'facebook/dinov2-base', 'modality': 'image'}),
    'clip-vit-base': ('huggingface', 
        {'model_name': 'openai/clip-vit-base-patch32', 'modality': 'image'}),
    
    # HuggingFace Text (non-sentence-transformers)
    'bert-base-uncased': ('huggingface', 
        {'model_name': 'bert-base-uncased', 'modality': 'text'}),
}
```

**Resolution logic:**
1. If `encoder` is a `BaseEncoder` instance, use it directly
2. If `encoder` is callable, wrap in `CallableEncoder`
3. If `encoder` is a string:
   a. Check `_ENCODER_REGISTRY` for exact match
   b. If string contains `/` (looks like HuggingFace model ID), try sentence-transformers first, then HuggingFace AutoModel
   c. Otherwise try sentence-transformers (most text embedding models are compatible)
   d. Raise ValueError with helpful message if nothing works

### 3.5 Detector Resolution

```python
_DETECTOR_SHORTCUTS = {
    # Distance-based
    'KNN': ('pyod.models.knn', 'KNN', {}),
    'LOF': ('pyod.models.lof', 'LOF', {}),
    
    # Density-based
    'ECOD': ('pyod.models.ecod', 'ECOD', {}),
    'COPOD': ('pyod.models.copod', 'COPOD', {}),
    'HBOS': ('pyod.models.hbos', 'HBOS', {}),
    
    # Linear model
    'PCA': ('pyod.models.pca', 'PCA', {}),
    'OCSVM': ('pyod.models.ocsvm', 'OCSVM', {}),
    'MCD': ('pyod.models.mcd', 'MCD', {}),
    
    # Ensemble
    'IForest': ('pyod.models.iforest', 'IForest', {}),
    'INNE': ('pyod.models.inne', 'INNE', {}),
    
    # Deep learning
    'AutoEncoder': ('pyod.models.auto_encoder', 'AutoEncoder', {}),
    'VAE': ('pyod.models.vae', 'VAE', {}),
    'DeepSVDD': ('pyod.models.deep_svdd', 'DeepSVDD', {}),
    'LUNAR': ('pyod.models.lunar', 'LUNAR', {}),
    
    # ... all other PyOD detectors
}

def resolve_detector(detector, contamination=0.1):
    """Resolve detector from string or BaseDetector instance."""
    if isinstance(detector, BaseDetector):
        return detector
    if isinstance(detector, str):
        if detector in _DETECTOR_SHORTCUTS:
            module_path, class_name, kwargs = _DETECTOR_SHORTCUTS[detector]
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            return cls(contamination=contamination, **kwargs)
        raise ValueError(f"Unknown detector: {detector}. "
                        f"Available: {list(_DETECTOR_SHORTCUTS.keys())}")
    raise TypeError(f"detector must be str or BaseDetector, got {type(detector)}")
```

### 3.6 EmbeddingOD Class

```python
class EmbeddingOD(BaseDetector):
    """Anomaly detection on raw data via embedding + detector pipeline.
    
    Chains any embedding encoder with any PyOD detector. Encode raw data
    (text, images, or other modalities) into numeric embeddings, then
    apply outlier detection in the embedding space.
    
    This implements the two-step approach shown to outperform end-to-end
    methods in NLP-ADBench (Li et al., EMNLP 2025) and TAD-Bench 
    (Cao et al., 2025).
    
    Parameters
    ----------
    encoder : str, BaseEncoder, or callable
        Embedding encoder. Accepts:
        - Registry shortcut: 'all-MiniLM-L6-v2', 'text-embedding-3-small',
          'dinov2-base'
        - HuggingFace model ID: 'sentence-transformers/all-MiniLM-L6-v2'
        - BaseEncoder instance
        - Callable: fn(X) -> np.ndarray of shape (n_samples, n_features)
    
    detector : str or BaseDetector, default='LUNAR'
        Any PyOD detector. String resolves to default-configured instance.
        Default is LUNAR (best performer in NLP-ADBench).
    
    contamination : float, default=0.1
        Expected proportion of outliers in the dataset.
    
    batch_size : int, default=32
        Batch size for encoding.
    
    cache_embeddings : bool, default=False
        Cache training embeddings. Recommended for API-based encoders.
    
    reduce_dim : int or None, default=None
        If set, apply PCA to reduce embedding dimensionality before
        detection. Recommended for embeddings >1000 dims with 
        distance-based detectors (KNN, LOF).
    
    standardize : bool, default=True
        Apply StandardScaler to embeddings before detection.
        Matches NLP-ADBench preprocessing pipeline.
    
    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples,)
        Outlier scores of the training data.
    threshold_ : float
        The threshold based on contamination.
    labels_ : numpy array of shape (n_samples,)
        Binary labels of the training data (0: inlier, 1: outlier).
    encoder_ : BaseEncoder
        The resolved encoder instance.
    detector_ : BaseDetector
        The resolved detector instance.
    
    Examples
    --------
    >>> from pyod.models.embedding import EmbeddingOD
    >>> clf = EmbeddingOD(encoder='all-MiniLM-L6-v2', detector='KNN')
    >>> clf.fit(train_texts)
    >>> scores = clf.decision_function(test_texts)
    >>> labels = clf.predict(test_texts)
    """
    
    def __init__(self, encoder, detector='LUNAR', contamination=0.1,
                 batch_size=32, cache_embeddings=False, 
                 reduce_dim=None, standardize=True):
        super(EmbeddingOD, self).__init__(contamination=contamination)
        self.encoder = encoder
        self.detector = detector
        self.batch_size = batch_size
        self.cache_embeddings = cache_embeddings
        self.reduce_dim = reduce_dim
        self.standardize = standardize
    
    def fit(self, X, y=None):
        # Resolve encoder and detector
        self.encoder_ = resolve_encoder(self.encoder)
        self.detector_ = resolve_detector(self.detector, self.contamination)
        
        # Encode raw data to embeddings
        X_emb = self.encoder_.encode(
            X, batch_size=self.batch_size, show_progress=True)
        
        # Clean embeddings (matches NLP-ADBench pipeline)
        X_emb = np.nan_to_num(X_emb)
        X_emb = np.clip(X_emb, np.finfo(np.float32).min, 
                        np.finfo(np.float32).max)
        
        # Standardize
        if self.standardize:
            self.scaler_ = StandardScaler()
            X_emb = self.scaler_.fit_transform(X_emb)
        
        # Dimensionality reduction
        if self.reduce_dim is not None:
            self.pca_ = PCA(n_components=self.reduce_dim)
            X_emb = self.pca_.fit_transform(X_emb)
        
        X_emb = X_emb.astype(np.float32)
        
        # Cache if requested
        if self.cache_embeddings:
            self.train_embeddings_ = X_emb
        
        # Fit detector
        self._set_n_classes(y)
        self.detector_.fit(X_emb, y)
        self.decision_scores_ = self.detector_.decision_scores_
        self._process_decision_scores()
        return self
    
    def decision_function(self, X):
        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])
        
        X_emb = self.encoder_.encode(
            X, batch_size=self.batch_size, show_progress=False)
        X_emb = np.nan_to_num(X_emb)
        X_emb = np.clip(X_emb, np.finfo(np.float32).min, 
                        np.finfo(np.float32).max)
        
        if self.standardize:
            X_emb = self.scaler_.transform(X_emb)
        if self.reduce_dim is not None:
            X_emb = self.pca_.transform(X_emb)
        
        X_emb = X_emb.astype(np.float32)
        return self.detector_.decision_function(X_emb)
```

### 3.7 Input Validation

`check_array(X)` from sklearn rejects non-numeric data. For EmbeddingOD:

- **Before encoding:** Validate type only (list or array-like, non-empty). Each encoder validates modality-specific constraints (e.g., SentenceTransformerEncoder checks for list of strings).
- **After encoding:** `_validate_output()` calls `check_array()` on the numeric embedding matrix.
- **After preprocessing:** NaN/Inf cleaning (nan_to_num, clip) ensures detector receives clean input. This matches the NLP-ADBench preprocessing pipeline (feature_select.py and benchmark scripts).

### 3.8 Preprocessing Pipeline

Matches the pipeline from NLP-ADBench benchmark scripts:
1. `np.nan_to_num()` -- replace NaN/Inf
2. `np.clip()` to float32 range -- prevent overflow
3. `StandardScaler()` -- normalize features
4. Cast to float32 -- memory efficiency
5. (Optional) PCA dimensionality reduction

This is important because the benchmark results were obtained with this exact pipeline. Deviating from it may not reproduce the published numbers.

---

## 4. Layer 2: Benchmark-Informed Presets

### 4.1 Design

Class methods on EmbeddingOD that return pre-configured instances based on benchmark winners.

```python
@classmethod
def for_text(cls, quality='balanced', **kwargs):
    """Create an EmbeddingOD optimized for text anomaly detection.
    
    Configurations informed by NLP-ADBench (EMNLP 2025)
    and TAD-Bench (2025).
    
    Parameters
    ----------
    quality : str, default='balanced'
        'fast'     -- MiniLM (384d) + KNN, no API needed
        'balanced' -- mpnet (768d) + LUNAR, no API needed
        'best'     -- OpenAI large (3072d) + LUNAR, requires API key
    **kwargs
        Override any EmbeddingOD parameter.
    """
    presets = {
        'fast': {
            'encoder': 'all-MiniLM-L6-v2',
            'detector': 'KNN',
            'reduce_dim': None,
        },
        'balanced': {
            'encoder': 'all-mpnet-base-v2',
            'detector': 'LUNAR',
            'reduce_dim': None,
        },
        'best': {
            'encoder': 'text-embedding-3-large',
            'detector': 'LUNAR',
            'cache_embeddings': True,
            'reduce_dim': None,
        },
    }
    config = {**presets[quality], **kwargs}
    return cls(**config)

@classmethod
def for_image(cls, quality='balanced', **kwargs):
    """Create an EmbeddingOD optimized for image anomaly detection.
    
    Configurations informed by AnomalyDINO (WACV 2025).
    
    Parameters
    ----------
    quality : str, default='balanced'
        'fast'     -- DINOv2-small (384d) + KNN
        'balanced' -- DINOv2-base (768d) + LOF
        'best'     -- DINOv2-large (1024d) + KNN
    """
    presets = {
        'fast': {
            'encoder': 'dinov2-small',
            'detector': 'KNN',
        },
        'balanced': {
            'encoder': 'dinov2-base',
            'detector': 'LOF',
        },
        'best': {
            'encoder': 'dinov2-large',
            'detector': 'KNN',
        },
    }
    config = {**presets[quality], **kwargs}
    return cls(**config)
```

### 4.2 Preset Justification

**Text presets** -- based on NLP-ADBench rankings:

| Preset | Encoder | Dims | Detector | Expected AUROC | Cost |
|--------|---------|------|----------|---------------|------|
| fast | all-MiniLM-L6-v2 | 384 | KNN | ~0.70-0.75 | Free, local |
| balanced | all-mpnet-base-v2 | 768 | LUNAR | ~0.75-0.80 | Free, local |
| best | text-embedding-3-large | 3072 | LUNAR | ~0.86 (NLP-ADBench avg) | API cost |

LUNAR is default because it was the best detector in NLP-ADBench (avg 0.8633 with OpenAI). KNN is the fallback for the fast preset because it was the best in TAD-Bench and has no torch dependency.

**Image presets** -- based on AnomalyDINO results:
- DINOv2 outperformed CLIP and PatchCore for image AD
- KNN on DINOv2 features is the simplest effective approach

### 4.3 Override Pattern

Presets are starting points, not rigid. Users can override any parameter:

```python
# Start from text preset, but use IForest instead of LUNAR
clf = EmbeddingOD.for_text(quality='best', detector='IForest')

# Start from image preset, add dimensionality reduction
clf = EmbeddingOD.for_image(quality='balanced', reduce_dim=50)
```

---

## 5. Layer 3: LLMAD (LLM-Based Zero-Shot Anomaly Detection)

### 5.1 Motivation

AD-LLM showed that GPT-4o achieves 0.9668 AUROC on IMDB Reviews (vs 0.7366 for best two-step baseline -- a 30% improvement). For some datasets, direct LLM scoring is strictly better than embedding + detector.

This is a different mechanism from EmbeddingOD: instead of embedding data and running a classical detector, the LLM directly reasons about whether each sample is anomalous.

### 5.2 Design

```python
class LLMAD(BaseDetector):
    """LLM-based zero-shot anomaly detection.
    
    Uses an LLM to directly score samples as anomalous or normal.
    Implements the zero-shot detection approach from AD-LLM
    (Yang et al., ACL 2025).
    
    Parameters
    ----------
    model : str, default='gpt-4o-mini'
        LLM model name. Supports OpenAI models.
    
    normal_categories : list of str or None
        Names of normal categories. If None, inferred from 
        training data during fit().
    
    anomaly_categories : list of str or None
        Names of anomaly categories. If provided, uses the
        "Normal + Anomaly" setting from AD-LLM (higher accuracy).
    
    n_examples : int, default=5
        Number of training examples to include in prompt context.
    
    api_key : str or None
        OpenAI API key. Falls back to OPENAI_API_KEY env var.
    
    contamination : float, default=0.1
        Expected proportion of outliers.
    """
```

### 5.3 Fit and Scoring

```python
def fit(self, X, y=None):
    """Build prompt context from training data.
    
    Unlike EmbeddingOD, this does not train a model.
    It constructs the system prompt with category descriptions
    and example samples for in-context learning.
    """
    self.context_ = self._build_context(X, y)
    # Generate synthetic decision_scores_ for BaseDetector compatibility
    # by scoring training data (expensive for large datasets)
    if len(X) <= self.max_fit_samples:
        self.decision_scores_ = self._score_batch(X)
    else:
        # Sample subset for threshold estimation
        idx = np.random.choice(len(X), self.max_fit_samples, replace=False)
        subset_scores = self._score_batch([X[i] for i in idx])
        self.decision_scores_ = np.zeros(len(X))
        self.decision_scores_[idx] = subset_scores
    self._process_decision_scores()
    return self

def decision_function(self, X):
    """Score each sample using LLM zero-shot detection."""
    check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])
    return self._score_batch(X)
```

### 5.4 Prompt Structure

Based on AD-LLM's prompt templates:

```
System: You are an intelligent assistant that detects anomalies in text data.
Normal categories: [list]
Anomaly categories: [list, if provided]

Rules:
1. Read the text carefully
2. Compare to category descriptions  
3. Determine closest category alignment
4. Assign anomaly confidence score 0-1

Respond in JSON: {"reason": "...", "anomaly_score": 0.0-1.0}

User: [text sample]
```

### 5.5 Scope and Limitations

- **Cost:** Each sample requires an LLM API call. Suitable for small datasets or high-value detection.
- **Latency:** Much slower than embedding-based detection.
- **Reproducibility:** LLM outputs are non-deterministic (temperature=0 helps but does not guarantee).
- **Not for images:** Text-only in v1. Image support via vision LLMs is future work.

### 5.6 Ship Independently

LLMAD has no dependency on EmbeddingOD. It can be developed and released as a separate module. The only shared infrastructure is BaseDetector inheritance.

---

## 6. Integration with Existing PyOD

### 6.1 BaseDetector Compliance

Both EmbeddingOD and LLMAD inherit from BaseDetector (pyod/models/base.py). This gives them:
- `predict(X)` -- binary labels via threshold_
- `predict_proba(X)` -- probability estimates ('linear' or 'unify' methods)
- `predict_confidence(X)` -- Bayesian confidence estimation
- `predict_with_rejection(X)` -- prediction with uncertainty rejection
- `get_params()` / `set_params()` -- sklearn compatibility

No changes to BaseDetector are needed.

### 6.2 Wrapper Pattern

EmbeddingOD follows the same delegation pattern as existing PyOD wrappers:
- LOF wraps sklearn's LocalOutlierFactor
- IForest wraps sklearn's IsolationForest
- FeatureBagging wraps multiple detectors

The pattern: fit inner detector, copy `decision_scores_`, call `_process_decision_scores()`.

### 6.3 AutoModelSelector Integration

Not in v1 scope, but the path is clear:
- Add EmbeddingOD configurations to `_MODEL_REGISTRY`
- Add JSON analysis files for embedding-based models
- The LLM selection prompt in AutoModelSelector already asks about data type (images, text, tabular). With EmbeddingOD available, the selector can recommend it for non-tabular data.

### 6.4 Dependency Management

**Required dependencies** (no change): numpy, scipy, scikit-learn, joblib, matplotlib, numba

**New optional dependencies** for EmbeddingOD:
- `sentence-transformers` -- for SentenceTransformerEncoder
- `transformers` + `torch` -- for HuggingFaceEncoder
- `openai` -- for OpenAIEncoder

**New optional dependency** for LLMAD:
- `openai` -- for LLM API calls

Pattern: lazy import at module level with try/except, clear error message at instantiation if missing. Matches existing PyOD pattern (torch for deep learning models).

**setup.py extras:**
```python
extras_require={
    'embedding': ['sentence-transformers>=2.0'],
    'openai': ['openai>=1.0'],
    'all': ['sentence-transformers>=2.0', 'openai>=1.0', 'transformers>=4.0', 'torch>=2.0'],
}
```

---

## 7. File Structure

```
pyod/
  models/
    embedding.py              # EmbeddingOD, presets, resolve_detector()
    llm_ad.py                 # LLMAD (Layer 3, ships separately)
  utils/
    encoders/
      __init__.py             # BaseEncoder, resolve_encoder(), registry
      sentence_transformer.py # SentenceTransformerEncoder
      huggingface.py          # HuggingFaceEncoder
      openai_encoder.py       # OpenAIEncoder
      callable_encoder.py     # CallableEncoder
  test/
    test_embedding.py         # EmbeddingOD unit tests (CallableEncoder mock)
    test_embedding_st.py      # Integration tests with sentence-transformers
    test_encoders.py          # Individual encoder unit tests
    test_llm_ad.py            # LLMAD tests (mocked API)
```

---

## 8. Testing Strategy

### 8.1 Unit Tests (no external dependencies)

**test_embedding.py** -- Tests EmbeddingOD wrapper logic using CallableEncoder:

```python
class TestEmbeddingOD(unittest.TestCase):
    def setUp(self):
        # Generate synthetic data as text-like input
        self.n_train, self.n_test = 200, 100
        self.X_train = [f"sample_{i}" for i in range(self.n_train)]
        self.X_test = [f"test_{i}" for i in range(self.n_test)]
        
        # Mock encoder: maps string to random but deterministic embedding
        def mock_encode(X):
            rng = np.random.RandomState(42)
            return rng.randn(len(X), 50)
        
        self.clf = EmbeddingOD(
            encoder=mock_encode, detector='KNN', contamination=0.1)
    
    def test_fit_sets_attributes(self):
        self.clf.fit(self.X_train)
        assert hasattr(self.clf, 'decision_scores_')
        assert hasattr(self.clf, 'labels_')
        assert hasattr(self.clf, 'threshold_')
        assert len(self.clf.decision_scores_) == self.n_train
    
    def test_decision_function_shape(self):
        self.clf.fit(self.X_train)
        scores = self.clf.decision_function(self.X_test)
        assert scores.shape == (self.n_test,)
    
    def test_predict_labels(self):
        self.clf.fit(self.X_train)
        labels = self.clf.predict(self.X_test)
        assert set(labels) <= {0, 1}
    
    def test_predict_proba_range(self):
        self.clf.fit(self.X_train)
        proba = self.clf.predict_proba(self.X_test)
        assert proba.min() >= 0 and proba.max() <= 1
    
    def test_detector_string_resolution(self):
        for name in ['KNN', 'LOF', 'ECOD', 'IForest']:
            clf = EmbeddingOD(encoder=lambda X: np.random.randn(len(X), 10),
                             detector=name)
            clf.fit(self.X_train)
    
    def test_standardize(self):
        self.clf.standardize = True
        self.clf.fit(self.X_train)
        assert hasattr(self.clf, 'scaler_')
    
    def test_reduce_dim(self):
        self.clf.reduce_dim = 10
        self.clf.fit(self.X_train)
        assert hasattr(self.clf, 'pca_')
    
    def test_cache_embeddings(self):
        self.clf.cache_embeddings = True
        self.clf.fit(self.X_train)
        assert hasattr(self.clf, 'train_embeddings_')
```

### 8.2 Integration Tests (require optional dependencies)

```python
@unittest.skipUnless(
    importlib.util.find_spec('sentence_transformers') is not None,
    "sentence-transformers not installed")
class TestEmbeddingODSentenceTransformers(unittest.TestCase):
    def test_text_anomaly_detection(self):
        normal = ["The stock market rose today"] * 50
        anomaly = ["alien spacecraft landed in park"] * 5
        train = normal + anomaly
        
        clf = EmbeddingOD(encoder='all-MiniLM-L6-v2', detector='KNN')
        clf.fit(train)
        scores = clf.decision_function(train)
        assert scores.shape == (55,)
```

### 8.3 LLMAD Tests (mocked API)

```python
class TestLLMAD(unittest.TestCase):
    @patch('pyod.models.llm_ad.OpenAI')
    def test_fit_and_predict(self, mock_openai):
        # Mock API response
        mock_openai.return_value.chat.completions.create.return_value = ...
        
        clf = LLMAD(model='gpt-4o-mini')
        clf.fit(train_texts)
        labels = clf.predict(test_texts)
```

---

## 9. Delivery Order

### Phase 1: EmbeddingOD + Encoders (Layer 1)

**Files:**
- `pyod/utils/encoders/__init__.py`
- `pyod/utils/encoders/sentence_transformer.py`
- `pyod/utils/encoders/huggingface.py`
- `pyod/utils/encoders/openai_encoder.py`
- `pyod/utils/encoders/callable_encoder.py`
- `pyod/models/embedding.py`
- `pyod/test/test_embedding.py`
- `pyod/test/test_encoders.py`

**Dependencies:** None new (CallableEncoder works with base PyOD). Optional: sentence-transformers, openai, transformers+torch.

### Phase 2: Presets (Layer 2)

**Files:** Added to `pyod/models/embedding.py` as class methods.

**Dependencies:** Same as Phase 1.

### Phase 3: LLMAD (Layer 3)

**Files:**
- `pyod/models/llm_ad.py`
- `pyod/test/test_llm_ad.py`

**Dependencies:** openai.

### Phase 4: Documentation and Examples

**Files:**
- `examples/embedding_od_example.py`
- `examples/llm_ad_example.py`
- README update
- API docs update

---

## 10. Related Work and References

### Papers backing this design

| Paper | Venue | Key finding for PyOD |
|-------|-------|---------------------|
| NLP-ADBench (Li et al.) | EMNLP Findings 2025 | Two-step (embedding + PyOD detector) beats end-to-end; OpenAI+LUNAR best combo |
| TAD-Bench (Cao et al.) | arXiv 2501.11960 | Confirmed across 8 embeddings x 8 detectors; kNN/INNE top detectors |
| Text-ADBench | arXiv 2507.12295 | Shallow OD methods match deep methods when embeddings are good |
| AD-LLM (Yang et al.) | ACL Findings 2025 | GPT-4o zero-shot beats two-step on some datasets; data augmentation helps |
| AnoLLM (Tsai et al.) | ICLR 2025 | Tabular-to-text + NLL scoring; complementary approach |
| AnomalyDINO | WACV 2025 | DINOv2 + distance scoring for image AD |
| ADBench (Han et al.) | NeurIPS 2022 | Original tabular AD benchmark; PyOD baseline |

### Existing PyOD issues addressed

- #443: "How to use PyOD with Image and Text data" -- directly solved
- #387: "CNN-based AutoEncoder for Image Outlier Detection" -- solved via DINOv2 encoder
- #334: "Could PyOD be implemented in CV datasets" -- solved via image encoders
- #427: "Which algorithm for unsupervised image outlier detection" -- solved via presets

### Positioning

"ADBench (NeurIPS 2022) benchmarked tabular AD. NLP-ADBench (EMNLP 2025) benchmarked text AD. AD-LLM (ACL 2025) benchmarked LLM-based AD. PyOD unifies all of these into one library with a consistent API."

---

## 11. Scope Boundaries

**In scope:**
- EmbeddingOD wrapper (Layer 1)
- Four encoder backends (SentenceTransformer, HuggingFace, OpenAI, Callable)
- Encoder/detector registries with string resolution
- Benchmark-informed presets (Layer 2)
- LLMAD zero-shot detector (Layer 3)
- StandardScaler + PCA preprocessing
- Embedding caching
- Unit and integration tests
- setup.py optional dependency extras

**Out of scope (future work):**
- Fine-tuning encoders during fit
- Multi-modal fusion (combining text + image + tabular)
- AnoLLM-style generative scoring (tabular-to-text + NLL)
- AD-LLM data augmentation task
- AD-LLM model selection integration into AutoModelSelector
- OpenAI Batch API for cheaper encoding
- Streaming / online encoding
- Time-series encoders (separate expansion)
