# Text and image anomaly detection reference

PyOD covers text and image anomaly detection through the `EmbeddingOD` and `MultiModalOD` wrappers — a two-step "embed + classical OD" pattern that consistently beats specialized end-to-end methods on standard benchmarks. The agent loads this file when the master decision tree (in SKILL.md) routes to text or image.

## When to use this modality

- **Text anomaly detection**: novel-product detection from descriptions, log anomaly detection, document outlier detection, malicious-text detection.
- **Image anomaly detection**: novel-image detection in retail/manufacturing, OOD detection on classification pipelines, defect detection.
- **Multimodal**: detecting samples whose text and image disagree (e.g., a product description that doesn't match its image).

The two-step pattern (embed first, then run a classical detector) won the NLP-ADBench benchmark (EMNLP Findings 2025): OpenAI / sentence-transformer embeddings plus classical OD consistently outperformed specialized end-to-end text AD models.

## Detectors available in PyOD (KB-derived)

<!-- BEGIN KB-DERIVED: text-image-detector-list -->
- **EmbeddingOD** (Embedding-Based Outlier Detection) — complexity: time O(n * embedding_cost + detector_cost), space O(n * embedding_dim); best for: Anomaly detection on unstructured data (text, images) via foundation model representations; avoid when: Data is already tabular or a suitable encoder is not available; requires: pyod[torch]; paper: Zhao et al., 2025
- **LLMAD** (LLM-Based Anomaly Detection) — complexity: time varies, space varies; best for: Zero-shot or few-shot anomaly detection leveraging LLM world knowledge; avoid when: Feature is needed before release or LLM API costs are prohibitive; paper: TBD
- **MultiModalOD** (Multi-Modal Outlier Detection) — complexity: time O(n * n_modalities * embedding_cost + detector_cost), space O(n * n_modalities * embedding_dim); best for: Anomaly detection on multi-modal data combining text, image, or other modalities; avoid when: Only a single modality is available or data is purely tabular; requires: pyod[torch]; paper: Zhao et al., 2025
<!-- END KB-DERIVED: text-image-detector-list -->

## EmbeddingOD usage

```python
from pyod.utils.ad_engine import ADEngine

# Text input: a list of strings
texts = ["product description 1", "product description 2", ...]
engine = ADEngine()
state = engine.start(texts)
# state.profile['data_type'] == 'text'
# Encoder choice is made by the planner and is visible later on the
# fit detector (state.results[i]['plan']['params']), not in state.profile.

state = engine.plan(state)
# On a live probe: [p['detector_name'] for p in state.plans] == ['EmbeddingOD']
# In v3.2.x the text/image planner returns a single EmbeddingOD entry rather
# than separate base-detector variants; the wrapper picks its base detector
# internally from its own parameters.

state = engine.run(state)
state = engine.analyze(state)
report = engine.report(state)
```

For image data, pass a list of `PIL.Image` objects or numpy arrays:

```python
from PIL import Image
images = [Image.open(p) for p in image_paths]
state = engine.start(images)
# state.profile['data_type'] == 'image'
# Default encoder: HuggingFace ViT (vit-base-patch16-224)
```

## Encoder choice

`EmbeddingOD` supports three encoder backends, configurable via the `encoder` parameter:

1. **sentence-transformers** (default for text). Light, fast, no API call. Install: `pip install pyod[embedding]`.
2. **OpenAI** (best embedding quality for text). Requires `OPENAI_API_KEY` and pays per token. Install: `pip install pyod[openai]`.
3. **HuggingFace** (default for image; usable for text). Requires `transformers` + `torch` and ~2 GB of model weights on first run. Install: `pip install pyod[huggingface]`.

**When to switch from default**: if the user mentions privacy / no external API, force `sentence-transformers`. If the user wants top quality and is willing to pay, use `OpenAI`. For image, `HuggingFace` is the only practical default; sentence-transformers is text-only.

## What the planner returns on text / image

In v3.2.x, `engine.plan` on text or image data returns a single `EmbeddingOD` plan entry, not a spread across multiple base detectors. The wrapper then picks its own base detector internally. If you want to compare base detectors on the embeddings (LOF vs KNN vs IForest), do it outside the session:

- Call `engine.start` / `engine.plan` / `engine.run` / `engine.analyze` to get the embedding-based anomaly scores via `EmbeddingOD`.
- To ensemble across different base detectors on the embeddings, embed once, then fit `LOF` / `KNN` / `IForest` on the embedding matrix via the classic fit / predict path. Future versions of the planner may return a multi-detector spread natively.

## Worked example: novel-product detection from descriptions

### Setup

A retailer wants to flag product descriptions that look unlike anything in their existing catalog (potential listing errors, fraudulent listings, or mislabeled categories). They have 50,000 product descriptions.

### Agent flow

```python
import pandas as pd
from pyod.utils.ad_engine import ADEngine

df = pd.read_csv("products.csv")
texts = df["description"].tolist()

engine = ADEngine()
state = engine.start(texts)
# state.profile: {'data_type': 'text', 'n_samples': 50000, ...}
# Modality: text ✓
# Encoder (sentence-transformers default) is selected in plan(), not profile.

state = engine.plan(state)
# [p['detector_name'] for p in state.plans] == ['EmbeddingOD']
# (Live probe, 2026-04-14. The planner returns a single EmbeddingOD entry;
# base-detector selection happens inside the wrapper.)

state = engine.run(state)
state = engine.analyze(state)
# state.quality: {'agreement': None, 'separation': 0.24, 'stability': 0.71, ...}
# (Agreement is None when only one detector ran.)
# state.analysis['best_detector']: 'EmbeddingOD'

report = engine.report(state)
```

### What the agent reports

> "Found 1,200 product descriptions of 50,000 (2.4%) flagged as unusual via `EmbeddingOD` over sentence-transformer embeddings. Only one detector ran (the current planner returns `EmbeddingOD` as a single entry on text data), so agreement is undefined; separation is 0.24 and stability is 0.71.
>
> Top-5 most unusual descriptions: rows [42, 815, 2200, 14501, 31200]. The dominant pattern was descriptions in non-English languages mixed into the catalog, plus descriptions consisting only of model numbers without prose.
>
> Confidence: medium. Separation is 0.24 (above the 0.1 floor, but moderate), so I recommend manually reviewing the top-50 to validate the categories of anomaly.
>
> **What I assumed**:
> - Text data, English (auto-detected from sample)
> - Encoder: sentence-transformers (default, no API key required)
> - Detector: `EmbeddingOD` (single-entry plan; the wrapper picks its base detector internally)"

## Lightweight pitfalls

### TI1. Embedding model mismatch with data domain

Sentence-transformers trained on general English embeds biomedical or legal text poorly. Mitigation: use a domain-tuned model (e.g., a biomedical sentence-transformer) when the data is technical.

### TI2. Dimension mismatch between text and image

`EmbeddingOD` text embeddings are 384-768 dims; image embeddings are 512-2048 dims. Combining them naively fails. Mitigation: use `MultiModalOD` which handles concatenation and per-modality scaling.

### TI3. Batch size for inference

Large image inference with default batch size will run out of GPU memory. Mitigation: pass `batch_size=8` or `batch_size=16` to the encoder.

### TI4. OOV tokens drop signal

If the input text contains a lot of OOV tokens (slang, code, custom IDs), the embedding loses those signals. Mitigation: preprocess to remove or tokenize the special content before embedding, OR use a tokenizer-flexible model.

## See also

- `pitfalls.md` — extended pitfalls library (preprocessing → detection → analysis → reporting)
- `workflow.md` — the autonomous loop pattern
- SKILL.md — top-10 critical pitfalls and master decision tree
