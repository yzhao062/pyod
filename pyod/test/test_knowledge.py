# -*- coding: utf-8 -*-

import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.utils.knowledge import KnowledgeBase


class TestKnowledgeBase(unittest.TestCase):
    def setUp(self):
        self.kb = KnowledgeBase()

    def test_loads_algorithms(self):
        algos = self.kb.algorithms
        assert isinstance(algos, dict)
        assert len(algos) > 40

    def test_algorithm_has_required_fields(self):
        algos = self.kb.algorithms
        required = {'class_path', 'full_name', 'status', 'data_types',
                    'category', 'strengths', 'weaknesses'}
        for name, entry in algos.items():
            for field in required:
                assert field in entry, \
                    f"Algorithm '{name}' missing field '{field}'"

    def test_algorithm_status_values(self):
        for name, entry in self.kb.algorithms.items():
            assert entry['status'] in ('shipped', 'experimental', 'planned'), \
                f"Algorithm '{name}' has invalid status '{entry['status']}'"

    def test_loads_benchmarks(self):
        benchmarks = self.kb.benchmarks
        assert isinstance(benchmarks, dict)
        assert 'ADBench' in benchmarks
        assert 'NLP_ADBench' in benchmarks

    def test_loads_routing_rules(self):
        rules = self.kb.routing_rules
        assert isinstance(rules, dict)
        assert 'rules' in rules
        assert len(rules['rules']) > 0

    def test_routing_rule_has_required_fields(self):
        for rule in self.kb.routing_rules['rules']:
            assert 'id' in rule
            assert 'conditions' in rule
            assert 'recommendations' in rule
            for cond in rule['conditions']:
                assert 'field' in cond
                assert 'op' in cond
                assert 'value' in cond

    def test_loads_papers(self):
        papers = self.kb.papers
        assert isinstance(papers, dict)
        assert 'pyod' in papers

    def test_get_algorithm(self):
        algo = self.kb.get_algorithm('ECOD')
        assert algo is not None
        assert algo['status'] == 'shipped'
        assert 'tabular' in algo['data_types']

    def test_get_algorithm_missing_returns_none(self):
        assert self.kb.get_algorithm('NonExistent') is None

    def test_list_by_data_type(self):
        tabular = self.kb.list_by_data_type('tabular')
        assert len(tabular) > 30
        text = self.kb.list_by_data_type('text')
        assert 'EmbeddingOD' in [a['name'] for a in text]

    def test_list_by_status(self):
        shipped = self.kb.list_by_status('shipped')
        assert len(shipped) >= 46
        planned = self.kb.list_by_status('planned')
        names = [a['name'] for a in planned]
        assert 'TimeSeriesOD' in names

    def test_caching(self):
        a1 = self.kb.algorithms
        a2 = self.kb.algorithms
        assert a1 is a2


if __name__ == '__main__':
    unittest.main()
