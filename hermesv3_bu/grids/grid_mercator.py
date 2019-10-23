#!/usr/bin/env python

import os
import timeit
import numpy as np
from pyproj import Proj
from hermesv3_bu.grids.grid import Grid
from hermesv3_bu.logger.log import Log


class MercatorGrid(Grid):

    def __init__(self, logger, auxiliary_path, tstep_num, vertical_description_path, lat_ts, lon_0, nx, ny, inc_x,
                 inc_y, x_0, y_0, earth_radius=6370000.000):
        """
        Mercator grid object that contains all the information to do a mercator output.

        :param logger: Logger.
        :type logger: Log

        :param auxiliary_path: Path to the folder to store all the needed auxiliary files.
        :type auxiliary_path: str

        :param tstep_num: Number of time steps.
        :type tstep_num: int

        :param vertical_description_path: Path to the file that describes the vertical resolution
        :type vertical_description_path: str

        :param lon_0: Value of the Lon0 for the LCC grid type.
        :type lon_0: float

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

        logger.write_log('Mercator grid selected.')
        self.grid_type = 'Mercator'
        attributes = {'lat_ts': lat_ts, 'lon_0': lon_0, 'nx': nx, 'ny': ny, 'inc_x': inc_x, 'inc_y': inc_y,
                      'x_0': x_0 + (inc_x / 2), 'y_0': y_0 + (inc_y / 2), 'earth_radius': earth_radius,
                      'crs': "+proj=merc +a={2} +b={2} +lat_ts={0} +lon_0={1}".format(
                          lat_ts, lon_0, earth_radius)}

        # UTM coordinates
        self.x = None
        self.y = None

        # Initialises with parent class
        super(MercatorGrid, self).__init__(logger, attributes, auxiliary_path, vertical_description_path)

        self.shape = (tstep_num, len(self.vertical_desctiption), ny, nx)
        self.__logger.write_time_log('MercatorGrid', '__init__', timeit.default_timer() - spent_time, 3)

    def write_netcdf(self):
        """
        Write a mercator grid NetCDF with empty data
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
                                boundary_longitudes=self.boundary_longitudes,
                                mercator=True, lcc_x=self.x, lcc_y=self.y, lon_0=self.attributes['lon_0'],
                                lat_ts=self.attributes['lat_ts'])
        self.__logger.write_log("\tGrid created at '{0}'".format(self.netcdf_path), 3)
        self.__logger.write_time_log('MercatorGrid', 'write_netcdf', timeit.default_timer() - spent_time, 3)
        return True

    def create_coords(self):
        """
        Create the coordinates for a mercator domain.
        """
        spent_time = timeit.default_timer()
        # Create a regular grid in metres (Two 1D arrays)
        self.x = np.linspace(self.attributes['x_0'], self.attributes['x_0'] +
                             (self.attributes['inc_x'] * (self.attributes['nx'] - 1)), self.attributes['nx'],
                             dtype=np.float)
        self.y = np.arange(self.attributes['y_0'], self.attributes['y_0'] +
                           (self.attributes['inc_y'] * (self.attributes['ny'] - 1)), self.attributes['ny'],
                           dtype=np.float)

        # 1D to 2D
        x = np.array([self.x] * len(self.y))
        y = np.array([self.y] * len(self.x)).T

        # Create UTM bounds
        y_b = self.create_bounds(y, self.attributes['inc_y'], number_vertices=4, inverse=True)
        x_b = self.create_bounds(x, self.attributes['inc_x'], number_vertices=4)

        # Create the LCC projection
        projection = Proj(self.attributes['crs'])

        # UTM to Mercator
        self.center_longitudes, self.center_latitudes = projection(x, y, inverse=True)
        self.boundary_longitudes, self.boundary_latitudes = projection(x_b, y_b, inverse=True)

        self.__logger.write_time_log('MercatorGrid', 'create_coords', timeit.default_timer() - spent_time, 3)

        return True
