# pyod/utils/investigation.py
# -*- coding: utf-8 -*-
"""Investigation state for ADEngine session workflow."""

import time
from dataclasses import dataclass, field

PHASES = ('profiled', 'planned', 'detected', 'analyzed')

ACTION_TYPES = (
    'plan',
    'run',
    'analyze',
    'report_to_user',
    'confirm_with_user',
    'iterate',
    'done',
)


@dataclass
class InvestigationState:
    """Typed state object for an ADEngine investigation session.

    Tracks the full workflow: profiling, planning, detection,
    analysis, and iteration. Each session method updates the state
    and sets ``next_action`` to guide the agent.

    Attributes
    ----------
    phase : str
        One of ``PHASES``: 'profiled', 'planned', 'detected', 'analyzed'.
    iteration : int
        Current iteration (0 = first run).
    history : list
        List of HistoryEntry dicts.
    data : object
        Reference to input data (not copied).
    profile : dict
        Output of ``profile_data()``.
    plans : list
        List of DetectionPlan dicts (top-N).
    results : list
        List of DetectorResult dicts.
    consensus : dict or None
        ConsensusResult dict.
    analysis : dict or None
        InvestigationAnalysis dict.
    quality : dict or None
        QualityAssessment dict.
    next_action : dict
        NextAction dict guiding the agent.
    """
    phase: str
    iteration: int = 0
    history: list = field(default_factory=list)
    data: object = None
    profile: dict = field(default_factory=dict)
    plans: list = field(default_factory=list)
    results: list = field(default_factory=list)
    consensus: dict = None
    analysis: dict = None
    quality: dict = None
    next_action: dict = field(default_factory=dict)


def _make_history_entry(phase, action, iteration, detail=''):
    """Create a HistoryEntry dict."""
    return {
        'phase': phase,
        'action': action,
        'iteration': iteration,
        'timestamp': time.time(),
        'detail': detail,
    }
