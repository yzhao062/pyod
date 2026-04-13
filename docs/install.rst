Installation
^^^^^^^^^^^^

PyOD is designed for easy installation using either **pip** or **conda**. We recommend using the latest version of PyOD due to frequent updates and enhancements:

.. code-block:: bash

   pip install pyod            # normal install
   pip install --upgrade pyod  # or update if needed

.. code-block:: bash

   conda install -c conda-forge pyod

Alternatively, you can clone and run the setup.py file:

.. code-block:: bash

   git clone https://github.com/yzhao062/pyod.git
   cd pyod
   pip install .

Modality extras are available for heavier modalities:

.. code-block:: bash

   pip install pyod[graph]     # PyG-based graph detectors (DOMINANT, CoLA, SCAN, etc.)

**Required Dependencies**:

* Python 3.9 or higher
* ``joblib``
* ``matplotlib``
* ``numpy>=1.19``
* ``numba>=0.51``
* ``scipy>=1.5.1``
* ``scikit-learn>=0.22.0``

**Optional Dependencies** (install only what you need):

.. list-table::
   :widths: 33 33 34
   :header-rows: 0

   * - ``pytorch``: deep learning models (AutoEncoder, VAE, DeepSVDD)
     - ``suod``: SUOD acceleration framework
     - ``xgboost``: XGBOD supervised detector
   * - ``combo``: model combination, FeatureBagging
     - ``pythresh``: data-driven thresholding
     - ``sentence-transformers``: EmbeddingOD text
   * - ``openai``: EmbeddingOD with OpenAI embeddings
     - ``transformers``, ``torch``: EmbeddingOD image, HuggingFace encoder
     - ``torch_geometric``: graph detectors (``pip install pyod[graph]``)

.. warning::

    PyOD includes several neural-network-based models, including AutoEncoders, VAE, DeepSVDD, and the graph detectors (DOMINANT, CoLA, etc.), all implemented in PyTorch. These deep learning libraries are not automatically installed by PyOD to avoid conflicts with existing installations. If you plan to use neural-net-based or graph detectors, install PyTorch (and ``torch_geometric`` for graph models) separately. Similarly, ``xgboost`` is not installed by default but is required for XGBOD.
