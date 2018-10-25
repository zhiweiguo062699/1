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
import sys
import timeit

import hermesv3_gr.config.settings as settings
from grid import Grid


class GlobalGrid(Grid):
    """
    Global grid object that contains all the information to do a global output.

    :param grid_type: Type of the output grid [global, rotated, lcc, mercator].
    :type grid_type: str

    :param vertical_description_path: Path to the file that contains the vertical description.
    :type vertical_description_path: str

    :param timestep_num: Number of timesteps.
    :type timestep_num: int

    :param temporal_path: Path to the temporal folder.
    :type temporal_path: str

    :param inc_lat: Increment between latitude centroids.
    :type inc_lat: float

    :param inc_lon: Increment between longitude centroids.
    :type inc_lon: float

    :param center_longitude: Location of the longitude of the center cell.
    Default = 0
    :type center_longitude: float
    """

    def __init__(self, grid_type, vertical_description_path, timestep_num, temporal_path, inc_lat, inc_lon,
                 center_longitude=float(0)):
        import ESMF

        st_time = timeit.default_timer()
        settings.write_log('\tCreating Global grid.', level=2)

        # Initialize the class using parent
        super(GlobalGrid, self).__init__(grid_type, vertical_description_path, temporal_path)

        self.center_lat = float(0)
        self.center_lon = center_longitude
        self.inc_lat = inc_lat
        self.inc_lon = inc_lon

        self.crs = {'init': 'epsg:4326'}
        self.create_coords()

        if not os.path.exists(self.coords_netcdf_file):
            if settings.rank == 0:
                super(GlobalGrid, self).write_coords_netcdf()
            settings.comm.Barrier()

        self.esmf_grid = super(GlobalGrid, self).create_esmf_grid_from_file(self.coords_netcdf_file)

        self.x_lower_bound = self.esmf_grid.lower_bounds[ESMF.StaggerLoc.CENTER][1]
        self.x_upper_bound = self.esmf_grid.upper_bounds[ESMF.StaggerLoc.CENTER][1]
        self.y_lower_bound = self.esmf_grid.lower_bounds[ESMF.StaggerLoc.CENTER][0]
        self.y_upper_bound = self.esmf_grid.upper_bounds[ESMF.StaggerLoc.CENTER][0]

        self.shape = (timestep_num, len(self.vertical_description), self.x_upper_bound-self.x_lower_bound,
                      self.y_upper_bound-self.y_lower_bound)

        self.cell_area = self.get_cell_area()[self.x_lower_bound:self.x_upper_bound,
                                              self.y_lower_bound:self.y_upper_bound]

        settings.write_time('GlobalGrid', 'Init', timeit.default_timer() - st_time, level=1)

    def create_coords(self):
        """
        Create the coordinates for a global domain.
        """
        import numpy as np

        st_time = timeit.default_timer()
        settings.write_log('\t\tCreating global coordinates', level=3)

        self.center_latitudes = self.create_regular_grid_1d_array(self.center_lat, self.inc_lat, -90)
        self.boundary_latitudes = self.create_bounds(self.center_latitudes, self.inc_lat)

        # ===== Longitudes =====
        self.center_longitudes = self.create_regular_grid_1d_array(self.center_lon, self.inc_lon, -180)
        if len(self.center_longitudes)//2 < settings.size:
            settings.write_log('ERROR: Check the .err file to get more info.')
            if settings.rank == 0:
                raise AttributeError("ERROR: Maximum number of processors exceeded. " +
                                     "It has to be less or equal than {0}.".format(len(self.center_longitudes)//2))
            sys.exit(1)
        self.boundary_longitudes = self.create_bounds(self.center_longitudes, self.inc_lon)

        # Creating special cells with half cell on le left and right border
        lat_origin = self.center_lat - abs(-90)
        lon_origin = self.center_lon - abs(-180)
        n_lat = (abs(-90) / self.inc_lat) * 2
        n_lon = (abs(-180) / self.inc_lon) * 2
        self.center_latitudes = np.concatenate([
            [lat_origin + self.inc_lat / 2 - self.inc_lat / 4], self.center_latitudes,
            [lat_origin + (n_lat * self.inc_lat) - self.inc_lat / 2 + self.inc_lat / 4]])

        self.center_longitudes = np.concatenate([
            [lon_origin + self.inc_lon / 2 - self.inc_lon / 4], self.center_longitudes,
            [lon_origin + (n_lon * self.inc_lon) - self.inc_lon / 2 + self.inc_lon / 4]])

        self.boundary_latitudes = np.concatenate([
            [[lat_origin, lat_origin + self.inc_lat / 2]], self.boundary_latitudes,
            [[lat_origin + (n_lat * self.inc_lat) - self.inc_lat / 2, lat_origin + (n_lat * self.inc_lat)]]])

        self.boundary_longitudes = np.concatenate([
            [[lon_origin, lon_origin + self.inc_lon / 2]], self.boundary_longitudes,
            [[lon_origin + (n_lon * self.inc_lon) - self.inc_lon / 2, lon_origin + (n_lon * self.inc_lon)]]],)

        self.boundary_latitudes = self.boundary_latitudes.reshape((1,) + self.boundary_latitudes.shape)
        self.boundary_longitudes = self.boundary_longitudes.reshape((1,) + self.boundary_longitudes.shape)

        settings.write_time('GlobalGrid', 'create_coords', timeit.default_timer() - st_time, level=2)


if __name__ == '__main__':
    pass
