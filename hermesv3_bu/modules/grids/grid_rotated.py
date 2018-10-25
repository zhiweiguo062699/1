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
import timeit
import hermesv3_gr.config.settings as settings
from grid import Grid


class RotatedGrid(Grid):
    # TODO Rotated options description
    """
    :param grid_type: Type of the output grid [global, rotated, lcc, mercator].
    :type grid_type: str

    :param vertical_description_path: Path to the file that contains the vertical description.
    :type vertical_description_path: str


    :param timestep_num: Number of timesteps.
    :type timestep_num: int
    """

    def __init__(self, grid_type, vertical_description_path, timestep_num, temporal_path, centre_lat, centre_lon,
                 west_boundary, south_boundary, inc_rlat, inc_rlon):
        import ESMF

        st_time = timeit.default_timer()
        settings.write_log('\tCreating Rotated grid.', level=2)

        # Initialises with parent class
        super(RotatedGrid, self).__init__(grid_type, vertical_description_path, temporal_path)

        # Setting parameters
        self.new_pole_longitude_degrees = -180 + centre_lon
        self.new_pole_latitude_degrees = centre_lat  # 90 - centre_lat
        self.centre_lat = centre_lat
        self.centre_lon = centre_lon
        self.west_boundary = west_boundary  # + inc_rlon #/ 2
        self.south_boundary = south_boundary  # + inc_rlat #/ 2
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

        if not os.path.exists(self.coords_netcdf_file):
            if settings.rank == 0:
                # super(RotatedGrid, self).write_coords_netcdf()
                self.write_coords_netcdf()
            settings.comm.Barrier()

        # self.write_coords_netcdf()

        self.esmf_grid = super(RotatedGrid, self).create_esmf_grid_from_file(self.coords_netcdf_file, sphere=False)

        self.x_lower_bound = self.esmf_grid.lower_bounds[ESMF.StaggerLoc.CENTER][1]
        self.x_upper_bound = self.esmf_grid.upper_bounds[ESMF.StaggerLoc.CENTER][1]
        self.y_lower_bound = self.esmf_grid.lower_bounds[ESMF.StaggerLoc.CENTER][0]
        self.y_upper_bound = self.esmf_grid.upper_bounds[ESMF.StaggerLoc.CENTER][0]

        self.shape = (timestep_num, len(self.vertical_description), self.x_upper_bound-self.x_lower_bound,
                      self.y_upper_bound-self.y_lower_bound)

        self.cell_area = self.get_cell_area()[self.x_lower_bound:self.x_upper_bound,
                                              self.y_lower_bound:self.y_upper_bound]

        settings.write_time('RotatedGrid', 'Init', timeit.default_timer() - st_time, level=1)

    def create_coords(self):
        """
        Create the coordinates for a rotated domain.
        """
        from hermesv3_gr.tools.coordinates_tools import create_regular_rotated
        import numpy as np

        st_time = timeit.default_timer()
        settings.write_log('\t\tCreating rotated coordinates.', level=3)

        # Create rotated coordinates
        (self.rlat, self.rlon, br_lats_single, br_lons_single) = create_regular_rotated(
            self.south_boundary, self.west_boundary, self.inc_rlat, self.inc_rlon, self.n_lat, self.n_lon)
        if len(self.rlon)//2 < settings.size:
            settings.write_log('ERROR: Check the .err file to get more info.')
            if settings.rank == 0:
                raise AttributeError("ERROR: Maximum number of processors exceeded. " +
                                     "It has to be less or equal than {0}.".format(len(self.rlon)//2))
            sys.exit(1)
        # 1D to 2D
        c_lats = np.array([self.rlat] * len(self.rlon)).T
        c_lons = np.array([self.rlon] * len(self.rlat))

        # Create rotated boundary coordinates
        b_lats = super(RotatedGrid, self).create_bounds(c_lats, self.inc_rlat, number_vertices=4, inverse=True)
        b_lons = super(RotatedGrid, self).create_bounds(c_lons, self.inc_rlon, number_vertices=4)

        # Rotated to Lat-Lon
        self.boundary_longitudes, self.boundary_latitudes = self.rotated2latlon(b_lons, b_lats)
        self.center_longitudes, self.center_latitudes = self.rotated2latlon(c_lons, c_lats)

        settings.write_time('RotatedGrid', 'create_coords', timeit.default_timer() - st_time, level=2)

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
        import numpy as np
        import math

        st_time = timeit.default_timer()
        settings.write_log('\t\t\tTransforming rotated coordinates to latitude, longitude coordinates.', level=3)

        # TODO Document this function
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

        settings.write_time('RotatedGrid', 'rotated2latlon', timeit.default_timer() - st_time, level=3)

        return almd, aphd

    def write_coords_netcdf(self):
        """
        Writes the temporal file with the coordinates of the output needed to generate the weight matrix.
        If it is already well created it will only add the cell_area parameter.
        """
        from hermesv3_gr.modules.writing.writer import Writer

        st_time = timeit.default_timer()
        settings.write_log('\tWriting {0} file.'.format(self.coords_netcdf_file), level=3)

        if not self.chech_coords_file():
            # Writes an auxiliary empty NetCDF only with the coordinates and an empty variable.
            Writer.write_netcdf(self.coords_netcdf_file, self.center_latitudes, self.center_longitudes,
                                [{'name': 'var_aux', 'units': '', 'data': 0}],
                                boundary_latitudes=self.boundary_latitudes,
                                boundary_longitudes=self.boundary_longitudes,
                                roated=True, rotated_lats=self.rlat, rotated_lons=self.rlon,
                                north_pole_lat=self.new_pole_latitude_degrees,
                                north_pole_lon=self.new_pole_longitude_degrees)

            # Calculate the cell area of the auxiliary NetCDF file
            self.cell_area = self.get_cell_area()

            # Re-writes the NetCDF adding the cell area
            Writer.write_netcdf(self.coords_netcdf_file, self.center_latitudes, self.center_longitudes,
                                [{'name': 'var_aux', 'units': '', 'data': 0}],
                                boundary_latitudes=self.boundary_latitudes,
                                boundary_longitudes=self.boundary_longitudes, cell_area=self.cell_area,
                                roated=True, rotated_lats=self.rlat, rotated_lons=self.rlon,
                                north_pole_lat=self.new_pole_latitude_degrees,
                                north_pole_lon=self.new_pole_longitude_degrees)
        else:
            self.cell_area = self.get_cell_area()

        settings.write_time('RotatedGrid', 'write_coords_netcdf', timeit.default_timer() - st_time, level=3)


if __name__ == '__main__':
    pass
