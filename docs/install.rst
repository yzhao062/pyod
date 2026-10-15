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

    # 1. Claude Code / Codex — enables the od-expert skill
    pyod install skill             # Claude Code: installs to ~/.claude/skills/
    pyod install skill --project   # Codex: installs to ./skills/ in the project

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

**Claude Code**
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

    The server registers ten stateless tools: ``profile_data``, ``plan_detection``, ``build_detector``, ``list_detectors``, ``explain_detector``, ``compare_detectors``, ``get_benchmarks``, ``run_detection``, ``analyze_results``, and ``explain_findings``.

    Claude Desktop connects through this path: it reads MCP servers and has no skill directory, so ``pyod install skill`` does not reach it.

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
    Detectors (ADEngine):  61 total (43 tabular, 7 time-series, 8 graph, 2 text, 2 image, 1 multimodal, 3 audio)
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

Every optional feature ships as a pip extra. Install only what you need, or take the whole stack at once:

.. code-block:: bash

    pip install pyod[torch]        # one extra
    pip install pyod[torch,graph]  # several at once
    pip install pyod[all]          # every optional dependency

The extra names in the first column below are the only valid ones, and they are matched exactly. pip treats an unrecognized extra as a warning rather than an error, so ``pip install pyod[pytorch]`` exits successfully having installed PyOD itself but none of the PyTorch stack the name suggests, and the mistake only surfaces later as an ``ImportError``. The extra that carries PyTorch is ``torch``. On zsh, quote the argument (``pip install 'pyod[all]'``) so the shell does not expand the brackets.

.. list-table::
   :widths: 16 34 50
   :header-rows: 1

   * - Extra
     - Installs
     - Enables
   * - ``torch``
     - ``torch>=2.0``
     - Neural detectors: AutoEncoder, VAE, DeepSVDD
   * - ``suod``
     - ``suod``
     - SUOD acceleration framework
   * - ``xgboost``
     - ``xgboost``
     - XGBOD supervised detector
   * - ``combo``
     - ``combo``
     - Model combination, FeatureBagging
   * - ``pythresh``
     - ``pythresh``
     - Data-driven thresholding
   * - ``embedding``
     - ``sentence-transformers>=5.0.0``
     - EmbeddingOD text detection
   * - ``openai``
     - ``openai>=1.0``
     - EmbeddingOD with OpenAI embeddings
   * - ``huggingface``
     - ``transformers>=4.25.1``, ``torch>=2.0``, ``Pillow``
     - EmbeddingOD image, HuggingFace encoder
   * - ``graph``
     - ``torch>=2.0``, ``torch_geometric>=2.0``
     - Graph detectors (DOMINANT, CoLA, and the rest)
   * - ``mcp``
     - ``mcp>=1.0``
     - MCP server for MCP-compatible agents
   * - ``audio``
     - ``librosa>=0.10``, ``soundfile``
     - ``EmbeddingOD.for_audio()``; ``AudioAE`` also needs ``torch``
   * - ``all``
     - Every package listed above
     - The full stack in one command

.. warning::

    PyOD includes several neural-network-based models, including AutoEncoders, VAE, DeepSVDD, and the graph detectors (DOMINANT, CoLA, etc.), all implemented in PyTorch. These deep learning libraries are not installed with the core package, so installing PyOD without an extra leaves an existing PyTorch installation untouched. For most users the extras are the shortest path: ``pip install pyod[torch]`` for the neural detectors and ``pip install pyod[graph]`` for the graph models, both of which pull a default PyTorch build from PyPI. Installing PyTorch separately still matters when you need a build other than the PyPI default, such as a CPU-only wheel or a particular CUDA or ROCm version. Those variants come from PyTorch's own package index, whose URL the selector at `pytorch.org <https://pytorch.org/get-started/locally/>`__ generates for you. In that case install PyTorch first; a later ``pip install pyod[torch]`` leaves an already-satisfied ``torch>=2.0`` untouched. Similarly, ``xgboost`` is not installed by default but is required for XGBOD (``pip install pyod[xgboost]``).
