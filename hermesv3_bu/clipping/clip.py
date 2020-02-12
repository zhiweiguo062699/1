#!/usr/bin/env python

import os
import sys
import timeit
from hermesv3_bu.logger.log import Log


def select_clip(comm, logger, auxiliary_path, clipping, grid):
    """
    Create and initialise the clip.

    :param comm: MPI communicator.

    :param logger: Logger
    :type logger: Log

    :param auxiliary_path: Path to the folder to store all the needed auxiliary files.
    :type auxiliary_path: str

    :param clipping: String (or None) with the path to the shapefile clip or a list of points to make the clip.
    :type clipping: str

    :param grid: Desired output grid
    :type grid: Grid

    :return: Clip
    :rtype: Clip
    """
    spent_time = timeit.default_timer()
    if comm.Get_rank() == 0:
        if clipping is None:
            from hermesv3_bu.clipping.default_clip import DefaultClip
            clip = DefaultClip(logger, auxiliary_path, grid)
        elif clipping[0] == os.path.sep:
            from hermesv3_bu.clipping.shapefile_clip import ShapefileClip
            clip = ShapefileClip(logger, auxiliary_path, clipping, grid)
        else:
            from hermesv3_bu.clipping.custom_clip import CustomClip
            clip = CustomClip(logger, auxiliary_path, clipping, grid)
    else:
        clip = None

    clip = comm.bcast(clip, root=0)

    logger.write_time_log('Clip', 'select_clip', timeit.default_timer() - spent_time)
    return clip


class Clip(object):

    def __init__(self, logger, auxiliary_path, grid):
        spent_time = timeit.default_timer()
        self.logger = logger
        self.grid = grid
        self.shapefile = None
        self.shapefile_path = os.path.join(auxiliary_path, 'clip', 'clip.shp')

        self.logger.write_time_log('Clip', '__init__', timeit.default_timer() - spent_time)
