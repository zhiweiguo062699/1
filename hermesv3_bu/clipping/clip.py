#!/usr/bin/env python

import os
import sys


def select_clip(auxiliary_path, clipping, grid_shp):
    """
    Create and initialise the clip.

    :param auxiliary_path: Path to the folder to store all the needed auxiliary files.
    :type auxiliary_path: str

    :param clipping: String (or None) with the path to the shapefile clip or a list of points to make the clip.
    :type clipping: str

    :param grid_shp: Shapefile with the desired output grid
    :type grid_shp: geopandas.GeoDataFrame

    :return: Clip
    :rtype: Clip
    """
    if clipping is None:
        from hermesv3_bu.clipping.default_clip import DefaultClip
        clip = DefaultClip(auxiliary_path, grid_shp)
    return clip


class Clip(object):

    def __init__(self, auxiliary_path):
        self.shapefile = None
        self.shapefile_path = os.path.join(auxiliary_path, 'clip', 'clip.shp')
