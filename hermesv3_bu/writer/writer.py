#!/usr/bin/env python

import os
import numpy as np
from mpi4py import MPI
from warnings import warn

CHUNKING = True


def select_writer(arguments, grid):

    comm_world = MPI.COMM_WORLD

    if grid.shape[2] % 2 == 0:
        max_procs = grid.shape[2] // 2
    else:
        max_procs = (grid.shape[2] // 2) + 1

    if arguments.writing_processors > min((comm_world.Get_size(), max_procs)):
        warn('Exceeded maximum of writing processors. Setting it to {0}'.format(
            min((comm_world.Get_size(), max_procs))))

        arguments.writing_processors = min((comm_world.Get_size(), max_procs))

    rank_distribution = get_distribution(arguments.writing_processors, grid.shape)

    if comm_world.Get_rank() < arguments.writing_processors:
        color = 99
    else:
        color = 0

    comm_write = comm_world.Split(color, comm_world.Get_rank())

    if arguments.output_model == 'DEFAULT':
        from hermesv3_bu.writer.default_writer import DefaultWriter
        writer = DefaultWriter(comm_world, comm_write, arguments.output_name, rank_distribution)

    return writer


def get_distribution(processors, shape):
    fid_dist = {}
    total_rows = shape[2]

    aux_rows = total_rows // processors
    if total_rows % processors > 0:
        aux_rows += 1

    rows_sum = 0
    for proc in xrange(processors):
        total_rows -= aux_rows
        if total_rows < 0:
            rows = total_rows + aux_rows
        else:
            rows = aux_rows

        min_fid = proc * aux_rows * shape[3]
        max_fid = (proc + 1) * aux_rows * shape[3]

        fid_dist[proc] = {
            'y_min': rows_sum,
            'y_max': rows_sum + rows,
            'x_min': 0,
            'x_max': shape[3],
            'fid_min': min_fid,
            'fid_max': max_fid,
            'shape': (shape[0], shape[1], rows, shape[3]),
        }

        rows_sum += rows

    return fid_dist


class Writer(object):
    def __init__(self, comm_world, comm_write, netcdf_path, rank_distribution):

        self.comm_world = comm_world
        self.comm_write = comm_write
        self.netcdf_path = netcdf_path
        self.rank_distribution = rank_distribution
