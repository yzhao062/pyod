# -*- coding: utf-8 -*-
"""Knowledge base for PyOD's intelligent agent layer.

Loads structured JSON files containing algorithm metadata,
benchmark results, routing rules, and paper citations.
"""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

import json
import os


class KnowledgeBase:
    """Loader and accessor for PyOD's structured knowledge base.

    Reads JSON files from the knowledge directory and provides
    query methods for algorithm metadata, benchmarks, and routing.

    Parameters
    ----------
    knowledge_dir : str or None
        Path to knowledge directory. If None, uses the bundled
        directory shipped with PyOD.
    """

    def __init__(self, knowledge_dir=None):
        if knowledge_dir is None:
            knowledge_dir = os.path.dirname(__file__)
        self._dir = knowledge_dir
        self._algorithms = None
        self._benchmarks = None
        self._routing_rules = None
        self._papers = None

    def _load_json(self, filename):
        path = os.path.join(self._dir, filename)
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @property
    def algorithms(self):
        if self._algorithms is None:
            self._algorithms = self._load_json('algorithms.json')
        return self._algorithms

    @property
    def benchmarks(self):
        if self._benchmarks is None:
            self._benchmarks = self._load_json('benchmarks.json')
        return self._benchmarks

    @property
    def routing_rules(self):
        if self._routing_rules is None:
            self._routing_rules = self._load_json('routing_rules.json')
        return self._routing_rules

    @property
    def papers(self):
        if self._papers is None:
            self._papers = self._load_json('papers.json')
        return self._papers

    def get_algorithm(self, name):
        """Get algorithm metadata by name. Returns None if not found."""
        return self.algorithms.get(name)

    def list_by_data_type(self, data_type, status='shipped'):
        """List algorithms supporting a given data type."""
        results = []
        for name, entry in self.algorithms.items():
            if data_type in entry.get('data_types', []):
                if status == 'all' or entry.get('status') == status:
                    results.append({'name': name, **entry})
        return results

    def list_by_status(self, status):
        """List algorithms with a given status."""
        results = []
        for name, entry in self.algorithms.items():
            if entry.get('status') == status:
                results.append({'name': name, **entry})
        return results
