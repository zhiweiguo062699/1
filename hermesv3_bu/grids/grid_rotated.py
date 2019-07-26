#!/usr/bin/env python

import os
import timeit
from grid import Grid
import numpy as np
import math

from hermesv3_bu.logger.log import Log


class RotatedGrid(Grid):
    def __init__(self, logger, auxiliary_path, tstep_num, vertical_description_path, centre_lat, centre_lon,
                 west_boundary, south_boundary, inc_rlat, inc_rlon):
        """

        :param logger: Logger.
        :type logger: Log

        :param auxiliary_path:
        :param tstep_num:
        :param vertical_description_path:
        :param centre_lat:
        :param centre_lon:
        :param west_boundary:
        :param south_boundary:
        :param inc_rlat:
        :param inc_rlon:
        """
        spent_time = timeit.default_timer()

        self.rlat = None
        self.rlon = None

        logger.write_log('Rotated grid selected.')
        self.grid_type = 'Rotated'
        attributes = {'new_pole_longitude_degrees': -180 + centre_lon, 'new_pole_latitude_degrees': centre_lat,
                      'centre_lat': centre_lat, 'centre_lon': centre_lon, 'west_boundary': west_boundary,
                      'south_boundary': south_boundary, 'inc_rlat': inc_rlat, 'inc_rlon': inc_rlon,
                      'n_lat': int((abs(south_boundary) / inc_rlat) * 2 + 1),
                      'n_lon': int((abs(west_boundary) / inc_rlon) * 2 + 1), 'crs': {'init': 'epsg:4326'}}

        # Initialises with parent class
        super(RotatedGrid, self).__init__(logger, attributes, auxiliary_path, vertical_description_path)

        self.shape = (tstep_num, len(self.vertical_desctiption), len(attributes['rlat']), len(attributes['rlon']))
        self.logger.write_time_log('RotatedGrid', '__init__', timeit.default_timer() - spent_time, 3)

    def create_regular_rotated(self):
        """
        Create a regular grid on the rotated domain.

        :return: center_latitudes, center_longitudes, corner_latitudes, corner_longitudes
        :rtype: tuple
        """
        spent_time = timeit.default_timer()

        center_latitudes = np.linspace(self.attributes['south_boundary'], self.attributes['south_boundary'] +
                                       (self.attributes['inc_rlat'] * (self.attributes['n_lat'] - 1)),
                                       self.attributes['n_lat'], dtype=np.float)
        center_longitudes = np.linspace(self.attributes['west_boundary'], self.attributes['west_boundary'] +
                                        (self.attributes['inc_rlon'] * (self.attributes['n_lon'] - 1)),
                                        self.attributes['n_lon'], dtype=np.float)

        corner_latitudes = self.create_bounds(center_latitudes, self.attributes['inc_rlat'], number_vertices=4,
                                              inverse=True)
        corner_longitudes = self.create_bounds(center_longitudes, self.attributes['inc_rlon'], number_vertices=4)

        self.logger.write_time_log('RotatedGrid', 'create_regular_rotated', timeit.default_timer() - spent_time, 3)
        return center_latitudes, center_longitudes, corner_latitudes, corner_longitudes

    def create_coords(self):
        """
        Create the coordinates for a rotated domain.
        """
        spent_time = timeit.default_timer()
        # Create rotated coordinates
        (self.rlat, self.rlon, br_lats_single, br_lons_single) = self.create_regular_rotated()

        # 1D to 2D
        c_lats = np.array([self.rlat] * len(self.rlon)).T
        c_lons = np.array([self.rlon] * len(self.rlat))

        # Create rotated boundary coordinates
        b_lats = super(RotatedGrid, self).create_bounds(c_lats, self.attributes['inc_rlat'], number_vertices=4,
                                                        inverse=True)
        b_lons = super(RotatedGrid, self).create_bounds(c_lons, self.attributes['inc_rlon'], number_vertices=4)

        # Rotated to Lat-Lon
        self.boundary_longitudes, self.boundary_latitudes = self.rotated2latlon(b_lons, b_lats)
        self.center_longitudes, self.center_latitudes = self.rotated2latlon(c_lons, c_lats)

        self.logger.write_time_log('RotatedGrid', 'create_coords', timeit.default_timer() - spent_time, 3)
        return True

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
        spent_time = timeit.default_timer()
        degrees_to_radians = math.pi / 180.
        # radians_to_degrees = 180. / math.pi

        # Positive east to negative east
        # self.new_pole_longitude_degrees -= 180

        tph0 = self.attributes['new_pole_latitude_degrees'] * degrees_to_radians
        tlm = lon_deg * degrees_to_radians
        tph = lat_deg * degrees_to_radians
        tlm0d = self.attributes['new_pole_longitude_degrees']
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

        self.logger.write_time_log('RotatedGrid', 'rotated2latlon', timeit.default_timer() - spent_time, 3)

        return almd, aphd

    def write_netcdf(self):
        """
        Write a rotated grid NetCDF with empty data
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
                                rotated=True, rotated_lats=self.rlat, rotated_lons=self.rlon,
                                north_pole_lat=90 - self.attributes['new_pole_latitude_degrees'],
                                north_pole_lon=self.attributes['new_pole_longitude_degrees'])
        self.logger.write_log("\tGrid created at '{0}'".format(self.netcdf_path), 3)
        self.logger.write_time_log('RotatedGrid', 'write_netcdf', timeit.default_timer() - spent_time, 3)
        return True
