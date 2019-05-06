#!/usr/bin/env python

import os
import numpy as np
from mpi4py import MPI
from warnings import warn


def select_writer(arguments, grid):
    print grid.shapefile
    print grid.shape
    comm = MPI.COMM_WORLD

    if arguments.writing_processors > min((comm.Get_size(), grid.shape[2])):
        warn('Exceeded maximum of writing processors. Setting it to {0}'.format(min((comm.Get_size(), grid.shape[2]))))
        arguments.writing_processors = min((comm.Get_size(), grid.shape[2]))

    if arguments.output_model == 'DEFAULT':
        from hermesv3_bu.writer.default_writer import DefaultWriter
        writer = DefaultWriter(comm, arguments.output_name, grid)

    return writer


class Writer(object):
    def __init__(self, comm, netcdf_path, grid):

        self.comm = comm
        self.netcdf_path = netcdf_path
        self.grid = grid
