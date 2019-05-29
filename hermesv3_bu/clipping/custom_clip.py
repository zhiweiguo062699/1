#!/usr/bin/env python

import sys
import os
import geopandas as gpd
from hermesv3_bu.clipping.clip import Clip
from hermesv3_bu.logger.log import Log


class CustomClip(Clip):
    def __init__(self, logger, auxiliary_path, points_str):
        """
        Initialise the Custom Clip class

        :param logger: Logger
        :type logger: Log

        :param auxiliary_path: Path to the auxiliary directory.
        :type auxiliary_path: str

        :param points_str: List of points in string format.
        :type points_str: str
        """
        super(CustomClip, self).__init__(logger, auxiliary_path)
        self.clip_type = 'Custom clip'
        self.shapefile = self.create_clip(points_str)

    def create_clip(self, points_str):
        """
        Create a clip using the unary union of the desired output grid.

        :param points_str: List of points (lat, lon)
        :type points_str: str

        :return: Clip shapefile
        :rtype: geopandas.GeoDataFrame
        """
        import re
        from shapely.geometry import Point, Polygon

        if not os.path.exists(self.shapefile_path):
            if not os.path.exists(os.path.dirname(self.shapefile_path)):
                os.makedirs(os.path.dirname(self.shapefile_path))
            str_clip = re.split(' , | ,|, |,', points_str)
            lon_list = []
            lat_list = []
            for components in str_clip:
                components = re.split(' ', components)
                lon_list.append(float(components[0]))
                lat_list.append(float(components[1]))

            if not ((lon_list[0] == lon_list[-1]) and (lat_list[0] == lat_list[-1])):
                lon_list.append(lon_list[0])
                lat_list.append(lat_list[0])

            clip = gpd.GeoDataFrame(
                geometry=[Polygon([[p.x, p.y] for p in [Point(xy) for xy in zip(lon_list, lat_list)]])],
                crs={'init': 'epsg:4326'})

            clip.to_file(self.shapefile_path)
        else:
            clip = gpd.read_file(self.shapefile_path)
        return clip
