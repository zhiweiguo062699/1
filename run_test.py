# coding=utf-8
"""Script to run the tests for EarthDiagnostics and generate the code coverage report"""

import os
import sys
import pytest


work_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
os.chdir(work_path)
print(work_path)


version = sys.version_info[0]
report_dir = 'tests/report/python{}'.format(version)
errno = pytest.main([
    'tests',
    '--ignore=tests/report',
    '--cov=hermesv3_bu',
    '--cov-report=term',
    '--cov-report=html:{}/coverage_html'.format(report_dir),
    '--cov-report=xml:{}/coverage.xml'.format(report_dir),
])
sys.exit(errno)
