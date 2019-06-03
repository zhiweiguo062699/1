#!/usr/bin/env python

import sys
import os
from timeit import default_timer as gettime
from warnings import warn
import numpy as np
import pandas as pd
import geopandas as gpd
from mpi4py import MPI

from hermesv3_bu.io_server.io_server import IoServer


class IoShapefile(IoServer):
    def __init__(self, comm=None):
        if comm is None:
            comm = MPI.COMM_WORLD

        super(IoShapefile, self).__init__(comm)

    def write_shapefile(self, data, path):
        """

        :param data: GeoDataset to be written
        :type data: geopandas.GeoDataFrame

        :param path:

        :return: True when the writing is finished.
        :rtype: bool
        """
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        data.to_file(path)

        return True

    def read_serial_shapefile(self, path):

        gdf = gpd.read_file(path)

        return gdf

    def write_parallel_shapefile(self, data, path, rank):
        """

        :param data: GeoDataset to be written
        :type data: geopandas.GeoDataFrame

        :param path:

        :return: True when the writing is finished.
        :rtype: bool
        """
        data = self.comm.gather(data, root=rank)
        if self.comm.Get_rank() == 0:
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            data = pd.concat(data)
            data.to_file(path)

        self.comm.Barrier()

        return True

    def read_shapefile(self, path):
        if self.comm.Get_rank() == 0:
            gdf = gpd.read_file(path)
            gdf = np.array_split(gdf, self.comm.Get_size())
        else:
            gdf = None

        gdf = self.comm.scatter(gdf, root=0)

        return gdf

    def read_parallel_shapefile(self, path):
        if self.comm.Get_rank() == 0:
            data = self.read_serial_shapefile(path)
        else:
            data = None

        data = self.split_shapefile(data)

        return data

    def split_shapefile(self, data):

        if self.comm.Get_size() == 1:
            data = data
        else:
            if self.comm.Get_rank() == 0:
                data = np.array_split(data, self.comm.Get_size())
            else:
                data = None
            data = self.comm.scatter(data, root=0)

        return data

    def balance(self, data):

        data = self.comm.gather(data, root=0)
        if self.comm.Get_rank() == 0:
            data = pd.concat(data)
            data = np.array_split(data, self.comm.Get_size())
        else:
            data = None

        data = self.comm.scatter(data, root=0)

        return data
