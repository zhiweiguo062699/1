#!/usr/bin/env python

import sys
import os
from timeit import default_timer as gettime
from warnings import warn


class Io(object):
    def __init__(self, comm=None):
        self.comm = comm