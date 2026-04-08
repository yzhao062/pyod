# PyOD Documentation Cleanup Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update all PyOD documentation to reflect EmbeddingOD, fix inconsistencies between README.rst and index.rst, clean up outdated content, and ensure the docs build correctly.

**Architecture:** Two parallel documentation surfaces (README.rst for GitHub, index.rst for ReadTheDocs) must stay synchronized. API docs are auto-generated via Sphinx autodoc. All changes must preserve existing RST formatting and citation styles.

**Tech Stack:** reStructuredText, Sphinx autodoc, sphinxcontrib-bibtex, furo theme

---

## Current State

- EmbeddingOD code is committed (`39898e3` on `development`)
- Doc updates from yesterday are stashed (`git stash pop` to restore)
- README.rst and index.rst have diverged in structure and content
- Several outdated sections (FAQ, benchmark hardware, GPU support claims)

## Key Issues to Address

1. **EmbeddingOD not documented** in README.rst, index.rst, API docs
2. **README.rst and index.rst are out of sync** (different algorithm counts, different section structure, different citation formats)
3. **API docs missing** EmbeddingOD module and encoder modules
4. **Outdated content** in FAQ (GPU support, Docker, Gitter), benchmark (2018 hardware)
5. **Reference/citation** for NLP-ADBench missing from both zreferences.bib and README.rst footnotes
6. **Optional dependencies** not listed for EmbeddingOD
7. **Example** not referenced from docs

---

### Task 1: Pop stash and reconcile with current state

**Files:**
- Modify: `README.rst`, `docs/pyod.models.rst`, `docs/pyod.utils.rst`

- [ ] **Step 1: Pop the stashed doc changes**

Run: `git -C C:/Users/yuezh/PycharmProjects/pyod stash pop`

- [ ] **Step 2: Check for conflicts and review what was stashed**

Run: `git -C C:/Users/yuezh/PycharmProjects/pyod diff`

Review the stashed changes. They include:
- README.rst: EmbeddingOD in features, algorithm table, optional deps, code example, NLP-ADBench reference
- pyod.models.rst: EmbeddingOD autodoc entry
- pyod.utils.rst: Encoders autodoc entries

- [ ] **Step 3: Verify the stashed changes still apply cleanly**

Check that no code changes conflict with the stashed doc changes. If conflicts exist, resolve them.

---

### Task 2: Update README.rst with EmbeddingOD

**Files:**
- Modify: `README.rst`

The stashed changes should already include most of this. Verify and refine:

- [ ] **Step 1: Verify EmbeddingOD is in PyOD V2 features list**

After the LLM-based Model Selection bullet, there should be a bullet about EmbeddingOD:

```rst
* **Multi-Modal Detection via EmbeddingOD**: Chain foundation model encoders (sentence-transformers, OpenAI, HuggingFace) with any PyOD detector for text and image anomaly detection. See `EmbeddingOD example <https://github.com/yzhao062/pyod/blob/master/examples/embedding_od_example.py>`_.
```

- [ ] **Step 2: Verify EmbeddingOD code example in About section**

After the 5-line ECOD example, there should be an EmbeddingOD example:

```rst
**Text/Image Anomaly Detection with EmbeddingOD** (``pip install pyod[embedding]``):

.. code-block:: python

    from pyod.models.embedding import EmbeddingOD
    clf = EmbeddingOD(encoder='all-MiniLM-L6-v2', detector='KNN')
    clf.fit(train_texts)                          # list of strings
    scores = clf.decision_function(test_texts)    # anomaly scores
    labels = clf.predict(test_texts)              # binary labels

    # Or use a preset:
    clf = EmbeddingOD.for_text(quality='fast')    # MiniLM + KNN
    clf = EmbeddingOD.for_image(quality='balanced')  # DINOv2 + LOF
```

- [ ] **Step 3: Verify EmbeddingOD row in algorithm table**

In the "(i) Individual Detection Algorithms" table, after the LUNAR row:

```rst
Embedding-based      EmbeddingOD         Multi-modal anomaly detection via foundation model embeddings (text, image)                            2025   [#Li2024NLPADBench]_
```

- [ ] **Step 4: Verify optional dependencies include EmbeddingOD**

In the "Optional Dependencies" list:

```rst
* sentence-transformers (optional, required for EmbeddingOD text detection; install via ``pip install pyod[embedding]``)
* openai (optional, required for EmbeddingOD with OpenAI embeddings; install via ``pip install pyod[openai]``)
```

- [ ] **Step 5: Verify NLP-ADBench reference footnote**

In the Reference section (alphabetical order, after Li2019MADGAN):

```rst
.. [#Li2024NLPADBench] Li, Y., Li, J., Xiao, Z., Yang, T., Nian, Y., Hu, X. and Zhao, Y., 2025. NLP-ADBench: NLP Anomaly Detection Benchmark. In *Findings of the Association for Computational Linguistics: EMNLP 2025*.
```

- [ ] **Step 6: Update algorithm count**

The README says "more than 50 detection algorithms". Verify this is still accurate (was 45 in V2, now 46 with EmbeddingOD). Update if needed.

---

### Task 3: Update index.rst (ReadTheDocs) to match README.rst

**Files:**
- Modify: `docs/index.rst`

index.rst and README.rst must be synchronized. index.rst uses `:cite:` and `:class:` roles instead of `[#footnote]_` references.

- [ ] **Step 1: Add EmbeddingOD to V2 features list**

After the LLM-based Model Selection bullet (around line 76):

```rst
* **Multi-Modal Detection via EmbeddingOD**: Chain foundation model encoders (sentence-transformers, OpenAI, HuggingFace) with any PyOD detector for text and image anomaly detection :cite:`a-li2024nlp`.
```

- [ ] **Step 2: Add EmbeddingOD code example**

After the 5-line ECOD example (around line 115), add the same EmbeddingOD example as in README.rst.

- [ ] **Step 3: Add EmbeddingOD row to algorithm table**

In the "(i) Individual Detection Algorithms" table (after LUNAR, around line 256):

```rst
Embedding-based      EmbeddingOD       Multi-modal anomaly detection via foundation model embeddings (text, image)                            2025   :class:`pyod.models.embedding.EmbeddingOD`           :cite:`a-li2024nlp`
```

- [ ] **Step 4: Verify the algorithm group count**

Line 202 says "three major functional groups". This is still correct (Individual, Ensembles/Combinations, Utilities). EmbeddingOD falls under Individual.

- [ ] **Step 5: Add NLP-ADBench to zreferences.bib**

Add to `docs/zreferences.bib` (alphabetical by key):

```bibtex
@inproceedings{li2024nlp,
    title={NLP-ADBench: NLP Anomaly Detection Benchmark},
    author={Li, Yuangang and Li, Jiaqi and Xiao, Zhuo and Yang, Tiankai and Nian, Yi and Hu, Xiyang and Zhao, Yue},
    booktitle={Findings of the Association for Computational Linguistics: EMNLP 2025},
    year={2025}
}
```

---

### Task 4: Update API docs (pyod.models.rst and pyod.utils.rst)

**Files:**
- Modify: `docs/pyod.models.rst`
- Modify: `docs/pyod.utils.rst`

The stashed changes should already include these. Verify:

- [ ] **Step 1: Verify pyod.models.embedding entry in pyod.models.rst**

Between ecod and feature_bagging (alphabetical), verify:

```rst
pyod.models.embedding module
-----------------------------

.. automodule:: pyod.models.embedding
    :members:
    :exclude-members: get_params, set_params, resolve_detector, _DETECTOR_SHORTCUTS
    :undoc-members:
    :show-inheritance:
    :inherited-members:
```

- [ ] **Step 2: Verify encoders entries in pyod.utils.rst**

Before pyod.utils.data, verify encoder module documentation:

```rst
pyod.utils.encoders module
--------------------------

.. automodule:: pyod.utils.encoders
    :members:
    :exclude-members: _ENCODER_REGISTRY, _ENCODER_BACKENDS, _INSTALL_HINTS, _create_encoder
    :undoc-members:
    :show-inheritance:

pyod.utils.encoders.sentence\_transformer module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pyod.utils.encoders.sentence_transformer
    :members:
    :undoc-members:
    :show-inheritance:

pyod.utils.encoders.openai\_encoder module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pyod.utils.encoders.openai_encoder
    :members:
    :undoc-members:
    :show-inheritance:

pyod.utils.encoders.huggingface module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pyod.utils.encoders.huggingface
    :members:
    :undoc-members:
    :show-inheritance:
```

---

### Task 5: Clean up outdated content in FAQ and benchmark

**Files:**
- Modify: `docs/faq.rst`
- Modify: `docs/benchmark.rst`

- [ ] **Step 1: Read current faq.rst**

Read `docs/faq.rst` to identify outdated items.

- [ ] **Step 2: Update FAQ items**

Known outdated items:
- "GPU support" listed as future work (PyTorch models already support GPU)
- Docker mentioned as unimplemented
- Gitter contact channel never existed

Update or remove these entries as appropriate. Do not add new content that was not there before; just fix what is clearly wrong.

- [ ] **Step 3: Review benchmark.rst**

Read `docs/benchmark.rst`. The 2018 hardware spec is outdated but the benchmark results are still valid. Add a note that these are historical results if not already present. Do not delete the results.

---

### Task 6: Verify docs build

- [ ] **Step 1: Check that Sphinx can parse the RST files**

Run: `C:/Users/yuezh/miniforge3/envs/py312/python.exe -m sphinx -b html docs/ docs/_build/html 2>&1 | tail -20`

Check for errors or warnings. Fix any RST syntax issues.

- [ ] **Step 2: Verify no broken cross-references**

Check the Sphinx output for:
- Missing citation keys (e.g., `a-li2024nlp` must exist in zreferences.bib)
- Broken `:class:` references (e.g., `pyod.models.embedding.EmbeddingOD` must be importable)
- Missing `:func:` references

---

### Task 7: Stage, review, and commit

- [ ] **Step 1: Stage all documentation changes**

```
git add README.rst docs/index.rst docs/pyod.models.rst docs/pyod.utils.rst docs/zreferences.bib docs/faq.rst docs/benchmark.rst
```

- [ ] **Step 2: Run implement-review with Codex**

Use the implement-review skill with the `general` lens (documentation changes).

- [ ] **Step 3: Commit and push**

```
git commit -m "docs: add EmbeddingOD documentation, sync README and index.rst, clean up FAQ"
git push origin development
```

---

## Summary

| Task | What it delivers | Files |
|------|-----------------|-------|
| 1 | Pop stash, reconcile | README.rst, pyod.models.rst, pyod.utils.rst |
| 2 | README.rst with EmbeddingOD | README.rst |
| 3 | index.rst synced with README | docs/index.rst, docs/zreferences.bib |
| 4 | API docs for EmbeddingOD | docs/pyod.models.rst, docs/pyod.utils.rst |
| 5 | Clean up outdated FAQ/benchmark | docs/faq.rst, docs/benchmark.rst |
| 6 | Verify docs build | - |
| 7 | Stage, review, commit | All |
