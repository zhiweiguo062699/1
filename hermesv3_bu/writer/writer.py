#!/usr/bin/env python

import os
import numpy as np
from mpi4py import MPI


def select_writer(arguments, grid):

    comm = MPI.COMM_WORLD
    if arguments.output_model == 'DEFAULT':
        from hermesv3_bu.writer.default_writer import DefaultWriter
        writer = DefaultWriter(comm, arguments.output_path)

    return writer


class Writer(object):
    def __init__(self, comm, netcdf_path):

        self.comm = comm
        self.netcdf_path = netcdf_path
