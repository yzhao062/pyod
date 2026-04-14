Skill Maintenance Methodology
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PyOD ships agent skills (currently ``od-expert``) as packaged Markdown that Claude Code reads at session start. This document describes the methodology for building and maintaining "real" expert skills, distillations that drive the agent autonomously rather than just documenting the API.

What makes a skill "real"
-------------------------

A "real" skill encodes domain expertise so a non-expert user gets expert-quality results without driving every decision. The four criteria:

1. **Drives the agent autonomously through a complete workflow.** From data to profile to detector selection to run to analyze to iterate to report, the agent makes informed decisions on the user's behalf and only pauses when uncertain (adaptive escalation).
2. **Encodes domain knowledge a non-expert lacks.** Decision rules, pitfalls, result interpretation patterns, and worked examples, all distilled from real literature and practice.
3. **Maps every recommendation to an agent-callable method.** A skill that mentions a method the agent cannot actually call is broken. For PyOD's od-expert, every detector and method must exist in ``pyod.utils.knowledge.algorithms``.
4. **Curates expertise from current literature.** A skill written from memory ages fast. Real skills incorporate continuous research input: papers, industry case studies, benchmark updates.

Skill anatomy
-------------

The pattern PyOD follows for ``od-expert`` (and recommends for any future skill):

.. code-block:: text

    pyod/skills/<skill-name>/
        SKILL.md                # always loaded by Claude Code
        references/             # on-demand, agent loads as needed
            workflow.md         # autonomous loop + escalation triggers + canonical example
            pitfalls.md         # extended pitfalls library
            <modality>.md       # one per deep modality
        __init__.py             # data-only subpackage marker

The split is deliberate. ``SKILL.md`` is everything the agent must always have in working memory: activation rules, master decision tree, top-10 critical pitfalls, escalation trigger summary. The ``references/`` files are deeper content the agent loads only when it determines they apply, keeping ``SKILL.md`` focused while allowing real depth.

The ADEngine compatibility constraint
-------------------------------------

Every detector name in the skill must map to a real entry in ``pyod.utils.knowledge.algorithms``. This is non-negotiable: the agent only knows what the skill says it can call, and a skill that mentions a non-existent detector produces broken conversations.

The CI safety net (``pyod/test/test_skill_kb_consistency.py``) enforces this for **detector names** specifically. It extracts every backtick-wrapped, capitalised, single-token from the skill files (e.g. ``\`IForest\```) and asserts each one is either:

- a key in ``pyod.utils.knowledge.algorithms``, or
- on the small allowlist of legitimate non-detector tokens (orchestration objects like ``ADEngine``, CLI commands, file paths, benchmark names).

ADEngine **method names** (e.g. ``\`engine.start\```) and **parameter names** (e.g. ``\`contamination\```) are NOT mechanically validated. The current safety net leaves those to human review. If the maintainer mistypes a method name in the skill, only a careful reader will catch it. Future work could extend the validator to also walk ``ADEngine`` via introspection.

Detector-name convention: detector names in skill prose must be wrapped in backticks (inline code style). Free prose like "Isolation Forest" is unvalidated; ``\`IForest\``` is validated against the KB. This lets maintainers write expository prose freely while keeping canonical detector references type-checked.

Hybrid content: hand-written prose + KB-derived facts
-----------------------------------------------------

PyOD's skill content has two sources:

**Hand-written prose** (the maintainer's editorial voice):

- Workflow patterns and the autonomous loop logic
- Decision rules in the master tree
- Adaptive escalation triggers (when the agent should pause)
- Worked examples (narrative + transcript format)
- Pitfalls phrased as expert advice
- Result interpretation patterns

**KB-derived facts** (auto-generated from ``pyod.utils.knowledge`` at build time):

- Per-modality detector lists, one bullet per detector, containing: name, full_name, complexity, best_for, avoid_when, requires (as ``pyod[extra]`` install hints), and paper reference.
- Total detector counts (one-line summary by modality)
- Deduplicated benchmark references

Other KB fields (``strengths``, ``weaknesses``, ``default_params``) are NOT auto-generated into markdown in v3.2.0. The agent can still query them at runtime via ``engine.explain_detector(name)``; promoting them to static markdown would duplicate the runtime query and risk going stale.

KB-derived sections are demarcated with HTML-comment markers so the generator can update them in place without touching hand-written content:

.. code-block:: markdown

    <!-- BEGIN KB-DERIVED: tabular-detector-list -->
    ... auto-generated content ...
    <!-- END KB-DERIVED: tabular-detector-list -->

The generator parses these markers, regenerates the content between them, and leaves the rest of the file alone.

Manual update workflow
----------------------

When you read a new paper, blog post, or case study and want to reflect it in the skill:

1. **Read and distill.** What is the actionable insight? A new pitfall? A revised decision rule? A new escalation trigger? A worked example?
2. **Filter for ADEngine compatibility.** If the insight requires a method PyOD does not have, log it to ``docs/v3.3-backlog.md`` instead of adding it to the skill.
3. **Locate the right file.** Top-10 pitfalls go in ``SKILL.md``. Modality-specific content goes in the modality file. Cross-cutting workflow content goes in ``references/workflow.md``. The extended pitfalls library is ``references/pitfalls.md``.
4. **Add the content.** Cite the source in a markdown comment so future maintainers can trace the provenance. Example::

       <!-- Source: arxiv:2024.12345, "Foo et al. 2024" -->
       **Pitfall**: Running KNN on > 1M points with default parameters hits a memory wall ...

5. **Run the safety net.** From the repo root::

       pytest pyod/test/test_skill_kb_consistency.py -v

6. **Open a PR.** Reviewer checks the citation, confirms the change is ADEngine-compatible, and merges.

Automatic update workflow (when ``pyod.utils.knowledge`` changes)
-----------------------------------------------------------------

When PyOD adds a new detector, removes one, or updates a benchmark:

1. **Run the generator** from the repo root::

       python scripts/regen_skill.py

   This rewrites the KB-derived sections in every ``references/*.md`` file.

2. **Review the diff.** Only KB-DERIVED blocks should change. If a hand-written section was modified, it is a bug in the generator.

3. **Run the safety net**::

       pytest pyod/test/test_skill_kb_consistency.py -v

4. **Commit and PR**::

       git add pyod/skills/od_expert/
       git commit -m "skill(od-expert): regenerate KB-derived sections after KB update"

Adding a new skill (template)
-----------------------------

To create a new packaged skill for PyOD (e.g., ``model-saver-expert``):

1. **Create the directory structure**::

       pyod/skills/<new_skill_name>/
           SKILL.md
           references/
               workflow.md
               <modality-or-topic>.md
           __init__.py

2. **Add the data-only subpackage marker** in ``__init__.py``::

       """The <new-skill-name> Claude Code skill, packaged as data-only subpackage."""

3. **Update** ``pyproject.toml`` package data::

       [tool.setuptools.package-data]
       "pyod.skills.<new_skill_name>" = ["*.md", "references/*.md"]

4. **Update** ``pyod/skills/__init__.py`` ``_INSTALL_DIRNAME_MAP``::

       _INSTALL_DIRNAME_MAP = {
           "od_expert": "od-expert",
           "<new_skill_name>": "<new-skill-name>",
       }

5. **Add the CI safety net coverage** in ``pyod/test/test_skill_kb_consistency.py`` so the new skill is also validated against the KB.

6. **Document the new skill** in this file (add a section under "Currently shipped skills" below).

Currently shipped skills
------------------------

**od-expert** (since v3.0.0, deepened in v3.2.0). Anomaly detection expert. Drives ADEngine for tabular, time series, graph, text, and image data. Covers the full session lifecycle (start, plan, run, analyze, iterate, report) with adaptive escalation. Default install location: ``~/.claude/skills/od-expert/``.

Versioning
----------

The skill version follows the PyOD library version. There is no separate skill version field in v3.2.0: if PyOD ships v3.2.0, the bundled ``od-expert`` skill is the v3.2.0 version. This keeps the install-time wiring simple.

If a future skill change is significant enough to warrant a separate version (a breaking change in the agent's expected behavior), introduce a ``version`` field in the SKILL.md frontmatter at that point. Until then, PyOD library version is the source of truth.
