"""Detector construction from ADEngine plans.

Extracted from `pyod.utils.ad_engine.ADEngine` in 2026-05.
Not part of the public API.
"""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyod.utils.knowledge import KnowledgeBase

logger = logging.getLogger(__name__)


def build_detector_from_plan(plan: dict, kb: 'KnowledgeBase') -> object:
    """Build and return an unfitted detector from a plan.

    Parameters
    ----------
    plan : dict (DetectionPlan)
        Output of plan_detection().
    kb : KnowledgeBase
        Knowledge base used to look up algorithm metadata.

    Returns
    -------
    detector : BaseDetector
    """
    name = plan['detector_name']
    algo = kb.get_algorithm(name)
    if algo is None:
        raise ValueError("Unknown detector '%s'" % name)
    if algo.get('status') not in ('shipped', 'experimental'):
        raise ValueError(
            "Detector '%s' has status '%s' and cannot be built"
            % (name, algo.get('status', 'unknown')))

    preset = plan.get('preset')
    if preset:
        return build_from_preset(name, preset,
                                 plan.get('params', {}))

    class_path = algo['class_path']
    module_path, class_name = class_path.rsplit('.', 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    params = plan.get('params', {})
    return cls(**params)


def build_from_preset(detector_name: str, preset: str,
                      extra_params: dict) -> object:
    """Build a detector using a factory preset.

    Presets are class-method factories that wire common defaults for
    a modality (e.g., text or image). Currently only `EmbeddingOD`
    exposes presets (``'for_text'``, ``'for_image'``).

    Parameters
    ----------
    detector_name : str
        Class name of the detector. Currently only ``'EmbeddingOD'``
        is recognized.
    preset : str
        Preset name. For ``'EmbeddingOD'``, one of ``'for_text'`` or
        ``'for_image'``.
    extra_params : dict
        Additional kwargs forwarded to the preset class method.

    Returns
    -------
    BaseDetector
        Unfitted detector instance.

    Raises
    ------
    ValueError
        If the (detector_name, preset) pair is not recognized.
    """
    if detector_name == 'EmbeddingOD':
        from pyod.models.embedding import EmbeddingOD
        if preset == 'for_text':
            return EmbeddingOD.for_text(**extra_params)
        elif preset == 'for_image':
            return EmbeddingOD.for_image(**extra_params)
    raise ValueError("Unknown preset '%s' for '%s'"
                     % (preset, detector_name))
