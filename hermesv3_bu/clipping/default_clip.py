#!/usr/bin/env python

import sys
import os
import geopandas as gpd
from hermesv3_bu.clipping.clip import Clip


class DefaultClip(Clip):
    def __init__(self, auxiliary_path, grid_shp):
        self.clip_type = 'Default clip'
        super(DefaultClip, self).__init__(auxiliary_path)
        self.shapefile = self.create_clip(grid_shp)

    def create_clip(self, grid_shp):
        """
        Create a clip using the unary union of the desired output grid.

        :param grid_shp: Desired output grid shapefile
        :type grid_shp: geopandas.GeoDataFrame

        :return: Clip shapefile
        :rtype: geopandas.GeoDataFrame
        """
        if not os.path.exists(self.shapefile_path):
            if not os.path.exists(os.path.dirname(self.shapefile_path)):
                os.makedirs(os.path.dirname(self.shapefile_path))

            clip = gpd.GeoDataFrame(geometry=[grid_shp.unary_union], crs=grid_shp.crs)
            clip.to_file(self.shapefile_path)
        else:
            clip = gpd.read_file(self.shapefile_path)
        return clip
