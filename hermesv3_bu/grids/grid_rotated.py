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


import sys
import os
from grid import Grid
import numpy as np
import math


class RotatedGrid(Grid):
    def __init__(self, grid_type, temporal_path, centre_lat, centre_lon, west_boundary, south_boundary,
                 inc_rlat, inc_rlon):

        # Initialises with parent class
        super(RotatedGrid, self).__init__(grid_type, temporal_path)

        # Setting parameters
        self.new_pole_longitude_degrees = -180 + centre_lon
        self.new_pole_latitude_degrees = centre_lat
        self.centre_lat = centre_lat
        self.centre_lon = centre_lon
        self.west_boundary = west_boundary
        self.south_boundary = south_boundary
        self.inc_rlat = inc_rlat
        self.inc_rlon = inc_rlon
        self.n_lat = int((abs(south_boundary) / inc_rlat) * 2 + 1)
        self.n_lon = int((abs(west_boundary) / inc_rlon) * 2 + 1)

        # Rotated coordinates
        self.rlat = None
        self.rlon = None

        # Create coordinates
        self.crs = {'init': 'epsg:4326'}
        self.create_coords()

    def create_regular_rotated(self, lat_origin, lon_origin, lat_inc, lon_inc, n_lat, n_lon):
        center_latitudes = np.arange(lat_origin, lat_origin + (n_lat * lat_inc), lat_inc, dtype=np.float)
        center_longitudes = np.arange(lon_origin, lon_origin + (n_lon * lon_inc), lon_inc, dtype=np.float)

        corner_latitudes = self.create_bounds(center_latitudes, self.inc_rlat, number_vertices=4, inverse=True)
        corner_longitudes = self.create_bounds(center_longitudes, self.inc_rlon, number_vertices=4)

        return center_latitudes, center_longitudes, corner_latitudes, corner_longitudes

    def create_coords(self):
        """
        Create the coordinates for a rotated domain.
        """
        # Create rotated coordinates
        (self.rlat, self.rlon, br_lats_single, br_lons_single) = self.create_regular_rotated(
            self.south_boundary, self.west_boundary, self.inc_rlat, self.inc_rlon, self.n_lat, self.n_lon)

        # 1D to 2D
        c_lats = np.array([self.rlat] * len(self.rlon)).T
        c_lons = np.array([self.rlon] * len(self.rlat))

        # Create rotated boundary coordinates
        b_lats = super(RotatedGrid, self).create_bounds(c_lats, self.inc_rlat, number_vertices=4, inverse=True)
        b_lons = super(RotatedGrid, self).create_bounds(c_lons, self.inc_rlon, number_vertices=4)

        # Rotated to Lat-Lon
        self.boundary_longitudes, self.boundary_latitudes = self.rotated2latlon(b_lons, b_lats)
        self.center_longitudes, self.center_latitudes = self.rotated2latlon(c_lons, c_lats)

    def rotated2latlon(self, lon_deg, lat_deg, lon_min=-180):
        """
        Calculate the unrotated coordinates using the rotated ones.

        :param lon_deg: Rotated longitude coordinate.
        :type lon_deg: numpy.array

        :param lat_deg: Rotated latitude coordinate.
        :type lat_deg: numpy.array

        :param lon_min: Minimum value for the longitudes: -180 (-180 to 180) or 0 (0 to 360)
        :type lon_min: float

        :return: Unrotated coordinates. Longitudes, Latitudes
        :rtype: tuple(numpy.array, numpy.array)
        """
        degrees_to_radians = math.pi / 180.
        # radians_to_degrees = 180. / math.pi

        # Positive east to negative east
        # self.new_pole_longitude_degrees -= 180

        tph0 = self.new_pole_latitude_degrees * degrees_to_radians
        tlm = lon_deg * degrees_to_radians
        tph = lat_deg * degrees_to_radians
        tlm0d = self.new_pole_longitude_degrees
        ctph0 = np.cos(tph0)
        stph0 = np.sin(tph0)

        stlm = np.sin(tlm)
        ctlm = np.cos(tlm)
        stph = np.sin(tph)
        ctph = np.cos(tph)

        # Latitude
        sph = (ctph0 * stph) + (stph0 * ctph * ctlm)
        # if sph > 1.:
        #     sph = 1.
        # if sph < -1.:
        #     sph = -1.
        # print type(sph)
        sph[sph > 1.] = 1.
        sph[sph < -1.] = -1.

        aph = np.arcsin(sph)
        aphd = aph / degrees_to_radians

        # Longitude
        anum = ctph * stlm
        denom = (ctlm * ctph - stph0 * sph) / ctph0
        relm = np.arctan2(anum, denom) - math.pi
        almd = relm / degrees_to_radians + tlm0d

        # if almd < min_lon:
        #     almd += 360
        # elif almd > max_lon:
        #     almd -= 360
        almd[almd > (lon_min + 360)] -= 360
        almd[almd < lon_min] += 360

        return almd, aphd

    def write_coords_netcdf(self):
        from netcdf_tools import write_netcdf

        if not self.chech_coords_file():

            # Writes an auxiliary empty NetCDF only with the coordinates and an empty variable.
            write_netcdf(self.netcdf_file, self.center_latitudes, self.center_longitudes,
                         [{'name': 'var_aux', 'units': '', 'data': 0}],
                         boundary_latitudes=self.boundary_latitudes,
                         boundary_longitudes=self.boundary_longitudes,
                         rotated=True,
                         rotated_lats=self.rlat,
                         rotated_lons=self.rlon,
                         north_pole_lat=90 - self.new_pole_latitude_degrees,
                         north_pole_lon=self.new_pole_longitude_degrees)


if __name__ == '__main__':
    grid = RotatedGrid('rotated', '/home/Earth/ctena/temp', centre_lat=35., centre_lon=20., west_boundary=-51,
                       south_boundary=-35, inc_rlat=0.1, inc_rlon=0.1)
    grid.write_coords_netcdf()
