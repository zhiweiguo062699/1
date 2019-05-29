#!/usr/bin/env python

import sys
import os
import geopandas as gpd
from hermesv3_bu.clipping.clip import Clip
from hermesv3_bu.logger.log import Log


class DefaultClip(Clip):
    def __init__(self, logger, auxiliary_path, grid):
        """
        Initialise the Custom Clip class

        :param logger: Logger
        :type logger: Log

        :param auxiliary_path: Path to the auxiliary directory.
        :type auxiliary_path: str

        :param grid: Grid object
        :type grid: Grid
        """
        super(DefaultClip, self).__init__(logger, auxiliary_path)
        self.clip_type = 'Default clip'
        self.shapefile = self.create_clip(grid)

    def create_clip(self, grid):
        """
        Create a clip using the unary union of the desired output grid.

        :param grid: Desired output grid
        :type grid: Grid

        :return: Clip shapefile
        :rtype: geopandas.GeoDataFrame
        """
        if not os.path.exists(self.shapefile_path):
            if not os.path.exists(os.path.dirname(self.shapefile_path)):
                os.makedirs(os.path.dirname(self.shapefile_path))

            clip = gpd.GeoDataFrame(geometry=[grid.shapefile.unary_union], crs=grid.shapefile.crs)

            clip.to_file(self.shapefile_path)
        else:
            clip = gpd.read_file(self.shapefile_path)
        return clip
