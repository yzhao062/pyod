Installation
^^^^^^^^^^^^

PyOD 3 ships as a single pip-installable library plus optional agent activation paths. This guide covers every install variant, from a minimal core install to the full agentic stack.

Quickstart
----------

Core library (required for every activation path):

.. code-block:: bash

    pip install pyod

Then pick the activation path that matches your agent stack:

.. code-block:: bash

    # 1. Claude Code / Claude Desktop — enables the od-expert skill
    pyod install skill

    # 2. Any MCP-compatible LLM — requires the optional mcp extra
    pip install pyod[mcp]
    pyod mcp serve                 # alias for `python -m pyod.mcp_server`

    # 3. Pure Python — no extra step
    #    from pyod.utils.ad_engine import ADEngine

Run ``pyod info`` at any time to see version, detector counts, and the install state of each activation path.

Core library install
--------------------

PyOD is distributed through both **pip** (PyPI) and **conda** (conda-forge). We recommend the latest version due to frequent updates:

.. code-block:: bash

    pip install pyod            # normal install
    pip install --upgrade pyod  # upgrade if already installed

conda users can install from conda-forge:

.. code-block:: bash

    conda install -c conda-forge pyod

To install from source (useful for development):

.. code-block:: bash

    git clone https://github.com/yzhao062/pyod.git
    cd pyod
    pip install .

Agentic activation paths
------------------------

PyOD 3 supports three activation paths for AI agents. Pick the one that matches your agent stack; you can enable more than one in the same environment.

**Claude Code / Claude Desktop**
    The ``od-expert`` skill ships as package data inside the pyod wheel and is copied into Claude Code's skill directory via the ``pyod install skill`` command:

    .. code-block:: bash

        pip install pyod
        pyod install skill                  # user-global → ~/.claude/skills/od-expert/
        pyod install skill --project        # project-local → ./skills/od-expert/
        pyod install skill --list           # list available packaged skills
        pyod install skill --target <path>  # custom destination

    After installing, run ``pyod info`` to confirm the skill is detected. The legacy ``pyod-install-skill`` command from v3.0.0 is kept as a backward-compat alias and shares a single code path with ``pyod install skill``.

**Codex users**
    Codex does not have a user-global skill directory like Claude Code. It reads shared skills from ``./skills/<skill-name>/`` in the project root, which is exactly the path ``pyod install skill --project`` writes to. From a project directory, run:

    .. code-block:: bash

        pyod install skill --project

    Codex picks up ``od-expert`` in that project automatically. ``pyod info`` detects ``~/.codex/`` and reports Codex alongside Claude Code in its output.

**MCP-compatible agents**
    The MCP server exposes PyOD tools to any MCP-compatible LLM (e.g., Claude Desktop via MCP, other agent frameworks). It requires the optional ``mcp`` extra:

    .. code-block:: bash

        pip install pyod[mcp]
        pyod mcp serve              # alias for ``python -m pyod.mcp_server``

    The server registers seven tools: ``profile_data``, ``plan_detection``, ``build_detector``, ``list_detectors``, ``explain_detector``, ``compare_detectors``, and ``get_benchmarks``.

**Python apps / custom agents**
    Import and call PyOD's orchestration layer directly:

    .. code-block:: python

        from pyod.utils.ad_engine import ADEngine
        engine = ADEngine()
        state = engine.investigate(X_train)

    No extra install step beyond ``pip install pyod``. See the :doc:`examples/agentic` walkthrough for a full conversation example.

Verifying your install
----------------------

Run ``pyod info`` to check version, detector counts, and the install state of every activation path:

.. code-block:: bash

    pyod info

Example output:

.. code-block:: text

    PyOD version:          3.1.0
    Detectors (ADEngine):  61 total (44 tabular, 7 time-series, 8 graph, 3 text, 2 image, 1 multimodal)
    Classic API:           OK
    ADEngine (Layer 2):    OK
    MCP extra:             OK (run: pyod mcp serve)
    od-expert skill:       INSTALLED (user-global) at /Users/you/.claude/skills/od-expert/SKILL.md

If the od-expert skill line reads ``NOT INSTALLED`` but Claude Code is detected, run ``pyod install skill``. If the MCP extra shows ``NOT INSTALLED`` and you want MCP access, run ``pip install pyod[mcp]``.

Required dependencies
---------------------

* Python 3.9 or higher
* ``joblib``
* ``matplotlib``
* ``numpy>=1.19``
* ``numba>=0.51``
* ``scipy>=1.5.1``
* ``scikit-learn>=0.22.0``

Optional dependencies
---------------------

Install only what you need:

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
