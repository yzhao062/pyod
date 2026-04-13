# -*- coding: utf-8 -*-
"""Validate that README.rst parses without errors.

Catches malformed RST tables, broken references, and other
formatting issues before they reach PyPI or GitHub rendering.
"""

import os
import unittest


class TestReadmeRST(unittest.TestCase):
    def test_readme_parses_without_errors(self):
        """README.rst should parse with zero RST errors."""
        readme_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'README.rst')
        if not os.path.exists(readme_path):
            self.skipTest("README.rst not found (installed package)")

        try:
            import docutils.parsers.rst
            import docutils.utils
            import docutils.frontend
        except ImportError:
            self.skipTest("docutils not installed")

        with open(readme_path, encoding='utf-8') as f:
            content = f.read()

        parser = docutils.parsers.rst.Parser()
        settings = docutils.frontend.get_default_settings(
            docutils.parsers.rst.Parser)
        doc = docutils.utils.new_document('README.rst', settings)
        parser.parse(content, doc)

        errors = [m for m in doc.parse_messages if m['level'] >= 3]
        if errors:
            error_msgs = [m.astext()[:200] for m in errors[:5]]
            self.fail(
                "README.rst has %d RST error(s):\n%s"
                % (len(errors), '\n'.join(error_msgs)))


if __name__ == '__main__':
    unittest.main()
