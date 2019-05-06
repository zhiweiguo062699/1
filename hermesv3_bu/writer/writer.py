#!/usr/bin/env python

import os
import numpy as np


def select_writer(comm, arguments):
    pass


class Writer(object):
    def __init__(self, comm, netcdf_path):

        self.comm = comm
        self.netcdf_path = netcdf_path
