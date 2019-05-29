#!/usr/bin/env python

# Copyright 2018 Earth Sciences Department, BSC-CNS
#
# This file is part of HERMESv3_GR.
#
# HERMESv3_GR is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HERMESv3_GR is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HERMESv3_GR. If not, see <http://www.gnu.org/licenses/>.


import os
import timeit

import numpy as np
from hermesv3_bu.grids.grid import Grid
from hermesv3_bu.io_server.io_netcdf import write_coords_netcdf
from hermesv3_bu.logger.log import Log


class LatLonGrid(Grid):

    def __init__(self, logger, auxiliary_path, tstep_num, vertical_description_path, inc_lat, inc_lon, lat_orig,
                 lon_orig, n_lat, n_lon):
        """
        Regional regular lat-lon grid object that contains all the information to do a global output.

        :param logger: Logger.
        :type logger: Log

        :param auxiliary_path: Path to the folder to store all the needed auxiliary files.
        :type auxiliary_path: str

        :param tstep_num: Number of time steps.
        :type tstep_num: int

        :param vertical_description_path: Path to the file that describes the vertical resolution
        :type vertical_description_path: str

        :param inc_lat: Increment between latitude centroids.
        :type inc_lat: float

        :param inc_lon: Increment between longitude centroids.
        :type inc_lon: float

        :param lat_orig: Location of the latitude of the corner of the first cell (down left).
        :type lat_orig: float

        :param lon_orig: Location of the longitude of the corner of the first cell (down left).
        :type lon_orig: float

        :param n_lat: Number of cells on the latitude direction.
        :type n_lat = int

        :param n_lon: Number of cells on the latitude direction.
        :type n_lon = int
        """
        spent_time = timeit.default_timer()
        logger.write_log('Regular Lat-Lon grid selected.')
        self.grid_type = 'Regular Lat-Lon'
        attributes = {'inc_lat': inc_lat, 'inc_lon': inc_lon, 'lat_orig': lat_orig, 'lon_orig': lon_orig,
                      'n_lat': n_lat, 'n_lon': n_lon, 'crs': {'init': 'epsg:4326'}}
        # Initialize the class using parent
        super(LatLonGrid, self).__init__(logger, attributes, auxiliary_path, vertical_description_path)

        self.shape = (tstep_num, len(self.vertical_desctiption), n_lat, n_lon)

        self.logger.write_time_log('LatLonGrid', '__init__', timeit.default_timer() - spent_time)

    def create_coords(self):
        """
        Create the coordinates for a global domain.
        """
        spent_time = timeit.default_timer()
        # From corner latitude /longitude to center ones
        lat_c_orig = self.attributes['lat_orig'] + (self.attributes['inc_lat'] / 2)
        self.center_latitudes = np.linspace(
            lat_c_orig, lat_c_orig + (self.attributes['inc_lat'] * (self.attributes['n_lat'] - 1)),
            self.attributes['n_lat'], dtype=np.float)
        self.boundary_latitudes = self.create_bounds(self.center_latitudes, self.attributes['inc_lat'])

        # ===== Longitudes =====
        lon_c_orig = self.attributes['lon_orig'] + (self.attributes['inc_lon'] / 2)
        self.center_longitudes = np.linspace(
            lon_c_orig, lon_c_orig + (self.attributes['inc_lon'] * (self.attributes['n_lon'] - 1)),
            self.attributes['n_lon'], dtype=np.float)

        self.boundary_longitudes = self.create_bounds(self.center_longitudes, self.attributes['inc_lon'])

        self.boundary_latitudes = self.boundary_latitudes.reshape((1,) + self.boundary_latitudes.shape)
        self.boundary_longitudes = self.boundary_longitudes.reshape((1,) + self.boundary_longitudes.shape)

        self.logger.write_time_log('LatLonGrid', 'create_coords', timeit.default_timer() - spent_time, 2)

    def write_netcdf(self):
        """
        Write a regular lat-lon grid NetCDF with empty data
        """
        spent_time = timeit.default_timer()
        if not os.path.exists(self.netcdf_path):
            if not os.path.exists(os.path.dirname(self.netcdf_path)):
                os.makedirs(os.path.dirname(self.netcdf_path))
            # Writes an auxiliary empty NetCDF only with the coordinates and an empty variable.
            write_coords_netcdf(self.netcdf_path, self.center_latitudes, self.center_longitudes,
                                [{'name': 'var_aux', 'units': '', 'data': 0}],
                                boundary_latitudes=self.boundary_latitudes,
                                boundary_longitudes=self.boundary_longitudes,
                                regular_latlon=True)

        self.logger.write_log("\tRegular Lat-Lon grid write at '{0}'".format(self.netcdf_path), 3)
        self.logger.write_time_log('LatLonGrid', 'write_netcdf', timeit.default_timer() - spent_time, 3)
