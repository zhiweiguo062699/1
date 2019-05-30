#!/usr/bin/env python

import os
import timeit
import numpy as np
from pyproj import Proj
from grid import Grid

from hermesv3_bu.logger.log import Log


class LccGrid(Grid):

    def __init__(self, logger, auxiliary_path, tstep_num, vertical_description_path, lat_1, lat_2, lon_0, lat_0,  nx,
                 ny, inc_x, inc_y, x_0, y_0, earth_radius=6370000.000):
        """
        Lambert Conformal Conic (LCC) grid object that contains all the information to do a lcc output.

        :param logger: Logger.
        :type logger: Log

        :param auxiliary_path: Path to the folder to store all the needed auxiliary files.
        :type auxiliary_path: str

        :param tstep_num: Number of time steps.
        :type tstep_num: int

        :param vertical_description_path: Path to the file that describes the vertical resolution
        :type vertical_description_path: str

        :param lat_1: Value of the Lat1 for the LCC grid type.
        :type lat_1: float

        :param lat_2: Value of the Lat2 for the LCC grid type.
        :type lat_2: float

        :param lon_0: Value of the Lon0 for the LCC grid type.
        :type lon_0: float

        :param lat_0: Value of the Lat0 for the LCC grid type.
        :type lat_0: float

        :param nx: Number of cells on the x dimension.
        :type nx: int

        :param ny: Number of cells on the y dimension.
        :type ny: int

        :param inc_x: Increment between x dimensions cell centroids (metres).
        :type inc_x: int

        :param inc_y: Increment between y dimensions cell centroids (metres).
        :type inc_y: int

        :param x_0: Value of the X0 for the LCC grid type.
        :type x_0: float

        :param y_0: Value of the Y0 for the LCC grid type.
        :type y_0: float

        :param earth_radius: Radius of the Earth (metres).
        Default = 6370000.000
        :type earth_radius: float
        """
        spent_time = timeit.default_timer()
        logger.write_log('Lambert Conformal Conic grid selected.')
        self.grid_type = 'Lambert Conformal Conic'

        # UTM coordinates
        self.x = None
        self.y = None

        attributes = {'lat_1': lat_1, 'lat_2': lat_2, 'lon_0': lon_0, 'lat_0': lat_0, 'nx': nx, 'ny': ny,
                      'inc_x': inc_x, 'inc_y': inc_y, 'x_0': x_0, 'y_0': y_0, 'earth_radius': earth_radius,
                      'crs': "+proj=lcc +lat_1={0} +lat_2={1} +lat_0={2} +lon_0={3} +x_0={4} +y_0={5} ".format(
                          lat_1, lat_2, lat_0, lon_0, 0, 0) + "+datum=WGS84 +units=m"}

        # Initialises with parent class
        super(LccGrid, self).__init__(logger, attributes, auxiliary_path, vertical_description_path)
        self.shape = (tstep_num, len(self.vertical_desctiption), ny, nx)
        self.logger.write_time_log('LccGrid', '__init__', timeit.default_timer() - spent_time)

    def write_netcdf(self):
        """
        Write a lambert conformal conic grid NetCDF with empty data
        """
        from hermesv3_bu.io_server.io_netcdf import write_coords_netcdf
        spent_time = timeit.default_timer()

        if not os.path.exists(self.netcdf_path):
            if not os.path.exists(os.path.dirname(self.netcdf_path)):
                os.makedirs(os.path.dirname(self.netcdf_path))
            # Writes an auxiliary empty NetCDF only with the coordinates and an empty variable.
            write_coords_netcdf(self.netcdf_path, self.center_latitudes, self.center_longitudes,
                                [{'name': 'var_aux', 'units': '', 'data': 0}],
                                boundary_latitudes=self.boundary_latitudes,
                                boundary_longitudes=self.boundary_longitudes, lcc=True, lcc_x=self.x, lcc_y=self.y,
                                lat_1_2="{0}, {1}".format(self.attributes['lat_1'], self.attributes['lat_2']),
                                lon_0=self.attributes['lon_0'], lat_0=self.attributes['lat_0'])

        self.logger.write_log("\tGrid created at '{0}'".format(self.netcdf_path), 3)
        self.logger.write_time_log('LccGrid', 'write_netcdf', timeit.default_timer() - spent_time, 3)
        return True

    def create_coords(self):
        """
        Create the coordinates for a lambert conformal conic domain.
        """
        spent_time = timeit.default_timer()
        # Create a regular grid in metres (Two 1D arrays)
        self.x = np.arange(self.attributes['x_0'], self.attributes['x_0'] + self.attributes['inc_x'] *
                           self.attributes['nx'], self.attributes['inc_x'], dtype=np.float)
        self.y = np.arange(self.attributes['y_0'], self.attributes['y_0'] + self.attributes['inc_y'] *
                           self.attributes['ny'], self.attributes['inc_y'], dtype=np.float)

        # 1D to 2D
        x = np.array([self.x] * len(self.y))
        y = np.array([self.y] * len(self.x)).T

        # Create UTM bounds
        y_b = self.create_bounds(y, self.attributes['inc_y'], number_vertices=4, inverse=True)
        x_b = self.create_bounds(x, self.attributes['inc_x'], number_vertices=4)

        # Create the LCC projection
        projection = Proj(
            proj='lcc',
            ellps='WGS84',
            R=self.attributes['earth_radius'],
            lat_1=self.attributes['lat_1'],
            lat_2=self.attributes['lat_2'],
            lon_0=self.attributes['lon_0'],
            lat_0=self.attributes['lat_0'],
            to_meter=1,
            x_0=0,
            y_0=0,
            a=self.attributes['earth_radius'],
            k_0=1.0)

        # UTM to LCC
        self.center_longitudes, self.center_latitudes = projection(x, y, inverse=True)
        self.boundary_longitudes, self.boundary_latitudes = projection(x_b, y_b, inverse=True)

        self.logger.write_time_log('LccGrid', 'create_coords', timeit.default_timer() - spent_time, 2)
        return True
