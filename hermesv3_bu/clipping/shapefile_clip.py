#!/usr/bin/env python

import sys
import os
import geopandas as gpd
from hermesv3_bu.clipping.clip import Clip


class ShapefileClip(Clip):
    def __init__(self, auxiliary_path, clip_input_path):
        super(ShapefileClip, self).__init__(auxiliary_path)
        self.clip_type = 'Shapefile clip'
        self.shapefile = self.create_clip(clip_input_path)

    def create_clip(self, clip_path):
        """
        Create a clip using the unary union of the desired output grid.

        :param clip_path: Path to the shapefile that contains the clip
        :type clip_path: str

        :return: Clip shapefile
        :rtype: geopandas.GeoDataFrame
        """
        if not os.path.exists(self.shapefile_path):
            if os.path.exists(clip_path):
                if not os.path.exists(os.path.dirname(self.shapefile_path)):
                    os.makedirs(os.path.dirname(self.shapefile_path))
                clip = gpd.read_file(clip_path)
                clip = gpd.GeoDataFrame(geometry=[clip.unary_union], crs=clip.crs)
                clip.to_file(self.shapefile_path)
            else:
                raise IOError(" Clip shapefile {0} not found.")
        else:
            clip = gpd.read_file(self.shapefile_path)
        return clip
