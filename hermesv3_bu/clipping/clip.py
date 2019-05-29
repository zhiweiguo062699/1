#!/usr/bin/env python

import os
import sys


def select_clip(comm, auxiliary_path, clipping, grid):
    """
    Create and initialise the clip.

    :param comm: MPI communicator.

    :param auxiliary_path: Path to the folder to store all the needed auxiliary files.
    :type auxiliary_path: str

    :param clipping: String (or None) with the path to the shapefile clip or a list of points to make the clip.
    :type clipping: str

    :param grid_shp: Desired output grid
    :type grid_shp: Grid

    :return: Clip
    :rtype: Clip
    """
    if comm.Get_rank() == 0:
        if clipping is None:
            from hermesv3_bu.clipping.default_clip import DefaultClip
            clip = DefaultClip(auxiliary_path, grid)
        elif clipping[0] == os.path.sep:
            from hermesv3_bu.clipping.shapefile_clip import ShapefileClip
            clip = ShapefileClip(auxiliary_path, clipping)
        else:
            from hermesv3_bu.clipping.custom_clip import CustomClip
            clip = CustomClip(auxiliary_path, clipping)
    else:
        clip = None

    clip = comm.bcast(clip, root=0)
    return clip


class Clip(object):

    def __init__(self, auxiliary_path):
        self.clip_type = None
        self.shapefile = None
        self.shapefile_path = os.path.join(auxiliary_path, 'clip', 'clip.shp')

    def __str__(self):
        text = "I'm a {0}. \n\tShapefile path: {1}\n\tClip polygon: {2}".format(
            self.clip_type, self.shapefile_path, self.shapefile.loc[0, 'geometry'])
        return text
