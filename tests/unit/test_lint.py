""" Lint tests """
import os
import unittest

import pycodestyle  # formerly known as pep8


class TestLint(unittest.TestCase):

    def test_pep8_conformance(self):
        """Test that we conform to PEP-8."""

        check_paths = [
            'hermesv3_bu',
            'tests',
        ]
        exclude_paths = [

        ]

        print("PEP8 check of directories: {}\n".format(', '.join(check_paths)))

        # Get paths wrt package root
        package_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        for paths in (check_paths, exclude_paths):
            for i, path in enumerate(paths):
                paths[i] = os.path.join(package_root, path)

        style = pycodestyle.StyleGuide()
        style.options.exclude.extend(exclude_paths)
        style.options.max_line_length = 120

        self.assertEqual(style.check_files(check_paths).total_errors, 0)
