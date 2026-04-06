# -*- coding: utf-8 -*-
"""Example of using EmbeddingOD for text anomaly detection.

EmbeddingOD chains a foundation model encoder with any PyOD detector,
enabling anomaly detection on text, image, and other non-tabular data.

This implements the two-step approach shown to outperform end-to-end
methods in NLP-ADBench (Li et al., EMNLP 2025).

Requirements:
    pip install pyod sentence-transformers
"""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

from pyod.models.embedding import EmbeddingOD

# Training data: normal samples (consistent topic)
train_texts = [
    "Quarterly revenue exceeded expectations by 12 percent",
    "The company announced a new product line for Q3",
    "Stock price remained stable after the earnings report",
    "Board of directors approved the annual dividend",
    "Operating costs decreased due to efficiency improvements",
    "Market analysts upgraded the company rating to buy",
    "New partnership expected to drive growth next quarter",
    "Employee headcount grew by 5 percent this year",
] * 20  # 160 training samples

# Test data: mix of normal and anomalous
test_texts = [
    "Annual report shows strong financial performance",    # normal
    "Cost reduction strategy yielded positive results",     # normal
    "The volcano erupted covering the island in ash",       # anomaly
    "Alien signals detected by deep space telescope",       # anomaly
    "Profit margins improved across all business units",    # normal
    "A rare species of deep-sea fish was discovered",       # anomaly
]

# ---- Method 1: Manual configuration ----
print("Method 1: Manual configuration")
clf = EmbeddingOD(encoder='all-MiniLM-L6-v2', detector='KNN',
                  contamination=0.1)
clf.fit(train_texts)

scores = clf.decision_function(test_texts)
labels = clf.predict(test_texts)
proba = clf.predict_proba(test_texts)

for i, text in enumerate(test_texts):
    print(f"  [{labels[i]}] score={scores[i]:.3f}  "
          f"prob={proba[i, 1]:.3f}  {text[:50]}")

# ---- Method 2: Use a preset ----
print("\nMethod 2: Preset (fast text)")
clf2 = EmbeddingOD.for_text(quality='fast')
clf2.fit(train_texts)

labels2 = clf2.predict(test_texts)
for i, text in enumerate(test_texts):
    tag = "ANOMALY" if labels2[i] == 1 else "normal "
    print(f"  {tag}  {text[:50]}")

# ---- Method 3: Custom encoder function ----
print("\nMethod 3: Custom encoder (random projection demo)")
import numpy as np


def hash_encoder(texts):
    """Toy encoder: hash-based random projection."""
    rng = np.random.RandomState(42)
    vocab = {}
    dim = 50
    result = np.zeros((len(texts), dim))
    for i, text in enumerate(texts):
        for word in text.lower().split():
            if word not in vocab:
                vocab[word] = rng.randn(dim)
            result[i] += vocab[word]
    return result


clf3 = EmbeddingOD(encoder=hash_encoder, detector='LOF')
clf3.fit(train_texts)
labels3 = clf3.predict(test_texts)
print(f"  Predictions: {labels3}")
