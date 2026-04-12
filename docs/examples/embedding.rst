Layer 1: Text and Image Anomaly Detection
===========================================

Full example: `embedding_od_example.py <https://github.com/yzhao062/pyod/blob/master/examples/embedding_od_example.py>`_

EmbeddingOD chains a foundation model encoder with any PyOD detector, enabling anomaly detection on text, image, and other non-tabular data. This implements the two-step approach (pretrained embeddings + classical OD) shown to outperform end-to-end methods in NLP-ADBench :cite:`b-li2024nlp`.

1. Install the optional dependency and import.

    .. code-block:: bash

        pip install sentence-transformers

    .. code-block:: python

        from pyod.models.embedding import EmbeddingOD

2. Prepare text data. Training data should represent the "normal" distribution. Anomalies are texts that differ in topic or style.

    .. code-block:: python

        train_texts = [
            "Quarterly revenue exceeded expectations by 12 percent",
            "The company announced a new product line for Q3",
            "Stock price remained stable after the earnings report",
            "Board of directors approved the annual dividend",
        ] * 40  # 160 training samples

        test_texts = [
            "Annual report shows strong financial performance",    # normal
            "Cost reduction strategy yielded positive results",     # normal
            "The volcano erupted covering the island in ash",       # anomaly
            "Alien signals detected by deep space telescope",       # anomaly
        ]

3. Initialize an :class:`pyod.models.embedding.EmbeddingOD` detector, fit, and predict. The encoder converts text to embeddings; the detector finds outliers in the embedding space.

    .. code-block:: python

        clf = EmbeddingOD(encoder='all-MiniLM-L6-v2', detector='KNN',
                          contamination=0.1)
        clf.fit(train_texts)

        labels = clf.predict(test_texts)           # binary labels (0 or 1)
        scores = clf.decision_function(test_texts)  # anomaly scores
        proba = clf.predict_proba(test_texts)       # probability estimates

4. Alternatively, use a preset for one-line setup.

    .. code-block:: python

        # Fast text detection (MiniLM + KNN, no API key needed):
        clf = EmbeddingOD.for_text(quality='fast')

    Additional presets require extra packages:

    .. code-block:: python

        # Best text detection (requires: pip install openai, plus OPENAI_API_KEY):
        clf = EmbeddingOD.for_text(quality='best')

        # Image anomaly detection (requires: pip install transformers torch):
        clf = EmbeddingOD.for_image(quality='balanced')

5. Use a custom encoder function for any embedding source.

    .. code-block:: python

        import numpy as np

        def my_encoder(texts):
            # Your custom embedding logic here
            return np.random.randn(len(texts), 128)

        clf = EmbeddingOD(encoder=my_encoder, detector='LOF')
        clf.fit(train_texts)

.. rubric:: References

.. bibliography::
   :cited:
   :labelprefix: B
   :keyprefix: b-
