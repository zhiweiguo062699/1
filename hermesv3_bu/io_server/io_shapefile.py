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
from hermesv3_bu.tools.checker import check_files


class IoShapefile(IoServer):
    def __init__(self, comm=None):
        if comm is None:
            comm = MPI.COMM_WORLD

        super(IoShapefile, self).__init__(comm)

    def write_shapefile_serial(self, data, path):
        """

        :param data: GeoDataset to be written
        :type data: GeoDataFrame

        :param path:

        :return: True when the writing is finished.
        :rtype: bool
        """
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        data.to_file(path)

        return True

    def write_shapefile_parallel(self, data, path, rank=0):
        """

        :param data: GeoDataset to be written
        :type data: GeoDataFrame

        :param path:

        :return: True when the writing is finished.
        :rtype: bool
        """
        data = self.comm.gather(data, root=rank)
        if self.comm.Get_rank() == rank:
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            data = pd.concat(data)
            data.to_file(path)

        self.comm.Barrier()

        return True

    def read_shapefile_serial(self, path):
        check_files(path)
        gdf = gpd.read_file(path)

        return gdf

    def read_shapefile(self, path, rank=0):
        if self.comm.Get_rank() == rank:
            check_files(path)
            gdf = gpd.read_file(path)
            gdf = np.array_split(gdf, self.comm.Get_size())
        else:
            gdf = None

        gdf = self.comm.scatter(gdf, root=rank)

        return gdf

    def read_shapefile_parallel(self, path, rank=0):
        if self.comm.Get_rank() == rank:
            data = self.read_shapefile_serial(path)
        else:
            data = None

        data = self.split_shapefile(data, rank)

        return data

    def split_shapefile(self, data, rank=0):

        if self.comm.Get_size() == 1:
            data = data
        else:
            if self.comm.Get_rank() == rank:
                data = np.array_split(data, self.comm.Get_size())
            else:
                data = None
            data = self.comm.scatter(data, root=rank)

        return data

    def balance(self, data, rank=0):

        data = self.comm.gather(data, root=rank)
        if self.comm.Get_rank() == rank:
            data = pd.concat(data)
            data = np.array_split(data, self.comm.Get_size())
        else:
            data = None

        data = self.comm.scatter(data, root=rank)

        return data
