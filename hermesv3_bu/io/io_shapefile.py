#!/usr/bin/env python

import sys
import os
from timeit import default_timer as gettime
from warnings import warn
import numpy as np
import pandas as pd
import geopandas as gpd

import IN.src.config.settings as settings
from IN.src.modules.io.io import Io


class IoShapefile(Io):
    def __init__(self):
        super(IoShapefile, self).__init__()

    def write_serial_shapefile(self, data, path):
        """

        :param data: GeoDataset to be written
        :type data: geopandas.GeoDataFrame

        :param path:

        :return: True when the writing is finished.
        :rtype: bool
        """
        if settings.log_level_3:
            st_time = gettime()
        else:
            st_time = None

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        data.to_file(path)
        print 'TIME -> IoShapefile.write_serial_shapefile: Rank {0} {1} s'.format(settings.rank, round(gettime() - st_time, 2))

        return True

    def read_serial_shapefile(self, path):
        if settings.log_level_3:
            st_time = gettime()
        else:
            st_time = None

        gdf = gpd.read_file(path)
        print 'TIME -> IoShapefile.read_serial_shapefile: Rank {0} {1} s'.format(settings.rank, round(gettime() - st_time, 2))

        return gdf

    def write_shapefile(self, data, path):
        """

        :param data: GeoDataset to be written
        :type data: geopandas.GeoDataFrame

        :param path:

        :return: True when the writing is finished.
        :rtype: bool
        """
        if settings.log_level_3:
            st_time = gettime()
        else:
            st_time = None
        data = self.comm.gather(data, root=0)
        if self.rank == 0:
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            data = pd.concat(data)
            data.to_file(path)

        self.comm.Barrier()
        print 'TIME -> IoShapefile.write_shapefile: Rank {0} {1} s'.format(settings.rank, round(gettime() - st_time, 2))

        return True

    def read_shapefile(self, path):
        if settings.log_level_3:
            st_time = gettime()
        else:
            st_time = None

        if self.rank == 0:
            gdf = gpd.read_file(path)
            gdf = np.array_split(gdf, self.size)
        else:
            gdf = None

        gdf = self.comm.scatter(gdf, root=0)
        print 'TIME -> IoShapefile.read_shapefile: Rank {0} {1} s'.format(settings.rank, round(gettime() - st_time, 2))

        return gdf

    def split_shapefile(self, data):
        if settings.log_level_3:
            st_time = gettime()
        else:
            st_time = None

        if self.size == 1:
            print 'TIME -> IoShapefile.split_shapefile: Rank {0} {1} s'.format(settings.rank, round(gettime() - st_time, 2))
            return data

        if self.rank == 0:
            data = np.array_split(data, self.size)
        else:
            data = None

        data = self.comm.scatter(data, root=0)

        print 'TIME -> IoShapefile.split_shapefile: Rank {0} {1} s'.format(settings.rank, round(gettime() - st_time, 2))

        return data

    def balance(self, data):
        if settings.log_level_3:
            st_time = gettime()
        else:
            st_time = None

        data = self.comm.gather(data, root=0)
        if self.rank == 0:
            data = pd.concat(data)
            data = np.array_split(data, self.size)
        else:
            data = None

        data = self.comm.scatter(data, root=0)
        print 'TIME -> IoShapefile.balance: Rank {0} {1} s'.format(settings.rank, round(gettime() - st_time, 2))

        return data

