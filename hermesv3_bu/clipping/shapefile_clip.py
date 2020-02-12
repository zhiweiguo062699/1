#!/usr/bin/env python

import sys
import os
import timeit
import geopandas as gpd
from hermesv3_bu.clipping.clip import Clip
from hermesv3_bu.logger.log import Log
from hermesv3_bu.tools.checker import error_exit


class ShapefileClip(Clip):
    def __init__(self, logger, auxiliary_path, clip_input_path, grid):
        """
        Initialise the Shapefile Clip class

        :param logger: Logger
        :type logger: Log

        :param auxiliary_path: Path to the auxiliary directory.
        :type auxiliary_path: str

        :param clip_input_path: Path to the shapefile.
        :type clip_input_path: str
        """
        spent_time = timeit.default_timer()
        logger.write_log('Shapefile clip selected')
        super(ShapefileClip, self).__init__(logger, auxiliary_path, grid)
        self.clip_type = 'Shapefile clip'
        self.shapefile = self.create_clip(clip_input_path)
        self.logger.write_time_log('ShapefileClip', '__init__', timeit.default_timer() - spent_time)

    def create_clip(self, clip_path):
        """
        Create a clip using the unary union of the desired output grid.

        :param clip_path: Path to the shapefile that contains the clip
        :type clip_path: str

        :return: Clip shapefile
        :rtype: GeoDataFrame
        """
        spent_time = timeit.default_timer()
        if not os.path.exists(self.shapefile_path):
            if os.path.exists(clip_path):
                if not os.path.exists(os.path.dirname(self.shapefile_path)):
                    os.makedirs(os.path.dirname(self.shapefile_path))
                clip = gpd.read_file(clip_path)
                border = gpd.GeoDataFrame(geometry=[self.grid.shapefile.unary_union], crs=self.grid.shapefile.crs)
                geom = gpd.overlay(clip, border.to_crs(clip.crs), how='intersection').unary_union
                clip = gpd.GeoDataFrame(geometry=[geom], crs=clip.crs)
                clip.to_file(self.shapefile_path)
            else:
                error_exit(" Clip shapefile {0} not found.")
        else:
            clip = gpd.read_file(self.shapefile_path)
        self.logger.write_log("\tClip created at '{0}'".format(self.shapefile_path), 3)
        self.logger.write_time_log('ShapefileClip', 'create_clip', timeit.default_timer() - spent_time)
        return clip
