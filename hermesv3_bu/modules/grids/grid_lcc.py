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


class LccGrid(Grid):
    """
    Lambert Conformal Conic (LCC) grid object that contains all the information to do a lcc output.

    :param grid_type: Type of the output grid [global, rotated, lcc, mercator].
    :type grid_type: str

    :param vertical_description_path: Path to the file that contains the vertical description.
    :type vertical_description_path: str

    :param timestep_num: Number of timesteps.
    :type timestep_num: int

    :param temporal_path: Path to the temporal folder.
    :type temporal_path: str

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

    def __init__(self, grid_type, vertical_description_path, timestep_num, temporal_path, lat_1, lat_2, lon_0, lat_0,
                 nx, ny, inc_x, inc_y, x_0, y_0, earth_radius=6370000.000):
        import ESMF
        st_time = timeit.default_timer()
        settings.write_log('\tCreating Lambert Conformal Conic (LCC) grid.', level=2)

        # Initialises with parent class
        super(LccGrid, self).__init__(grid_type, vertical_description_path, temporal_path)

        # Setting parameters
        self.lat_1 = lat_1
        self.lat_2 = lat_2
        self.lon_0 = lon_0
        self.lat_0 = lat_0
        self.nx = nx
        self.ny = ny
        self.inc_x = inc_x
        self.inc_y = inc_y
        self.x_0 = x_0 + (inc_x / 2)
        self.y_0 = y_0 + (inc_y / 2)
        self.earth_radius = earth_radius

        # UTM coordinates
        self.x = None
        self.y = None

        # Creating coordinates
        self.crs = "+proj=lcc +lat_1={0} +lat_2={1} +lat_0={2} +lon_0={3} +x_0={4} +y_0={5} +datum=WGS84".format(
            self.lat_1, self.lat_2, self.lat_0, self.lon_0, 0, 0) + " +units=m"
        self.create_coords()

        if not os.path.exists(self.coords_netcdf_file):
            if settings.rank == 0:
                # super(LccGrid, self).write_coords_netcdf()
                self.write_coords_netcdf()
            settings.comm.Barrier()

        self.esmf_grid = super(LccGrid, self).create_esmf_grid_from_file(self.coords_netcdf_file, sphere=False)
        #
        self.x_lower_bound = self.esmf_grid.lower_bounds[ESMF.StaggerLoc.CENTER][1]
        self.x_upper_bound = self.esmf_grid.upper_bounds[ESMF.StaggerLoc.CENTER][1]
        self.y_lower_bound = self.esmf_grid.lower_bounds[ESMF.StaggerLoc.CENTER][0]
        self.y_upper_bound = self.esmf_grid.upper_bounds[ESMF.StaggerLoc.CENTER][0]

        self.shape = (timestep_num, len(self.vertical_description), self.x_upper_bound-self.x_lower_bound,
                      self.y_upper_bound-self.y_lower_bound)
        # print 'Rank {0} _3_\n'.format(settings.rank)
        settings.comm.Barrier()
        # print 'Rank {0} _4_\n'.format(settings.rank)
        self.cell_area = self.get_cell_area()[self.x_lower_bound:self.x_upper_bound,
                                              self.y_lower_bound:self.y_upper_bound]

        settings.write_time('LccGrid', 'Init', timeit.default_timer() - st_time, level=1)

    def write_coords_netcdf(self):
        """
        Writes the temporal file with the coordinates of the output needed to generate the weight matrix.
        If it is already well created it will only add the cell_area parameter.
        """
        from hermesv3_gr.tools.netcdf_tools import write_netcdf

        st_time = timeit.default_timer()
        settings.write_log('\tWriting {0} file.'.format(self.coords_netcdf_file), level=3)

        if not self.chech_coords_file():
            # Writes an auxiliary empty NetCDF only with the coordinates and an empty variable.
            write_netcdf(self.coords_netcdf_file, self.center_latitudes, self.center_longitudes,
                         [{'name': 'var_aux', 'units': '', 'data': 0}],
                         boundary_latitudes=self.boundary_latitudes, boundary_longitudes=self.boundary_longitudes,
                         lcc=True, lcc_x=self.x, lcc_y=self.y,
                         lat_1_2="{0}, {1}".format(self.lat_1, self.lat_2), lon_0=self.lon_0, lat_0=self.lat_0)

            # Calculate the cell area of the auxiliary NetCDF file
            self.cell_area = self.get_cell_area()

            # Re-writes the NetCDF adding the cell area
            write_netcdf(self.coords_netcdf_file, self.center_latitudes, self.center_longitudes,
                         [{'name': 'var_aux', 'units': '', 'data': 0}],
                         boundary_latitudes=self.boundary_latitudes, boundary_longitudes=self.boundary_longitudes,
                         cell_area=self.cell_area,
                         lcc=True, lcc_x=self.x, lcc_y=self.y,
                         lat_1_2="{0}, {1}".format(self.lat_1, self.lat_2), lon_0=self.lon_0, lat_0=self.lat_0)
        else:
            self.cell_area = self.get_cell_area()

        settings.write_time('LccGrid', 'write_coords_netcdf', timeit.default_timer() - st_time, level=3)

    def create_coords(self):
        """
        Create the coordinates for a lambert conformal conic domain.
        """
        import numpy as np
        from pyproj import Proj

        st_time = timeit.default_timer()
        settings.write_log('\t\tCreating lcc coordinates', level=3)

        # Create a regular grid in metres (Two 1D arrays)
        self.x = np.arange(self.x_0, self.x_0 + self.inc_x * self.nx, self.inc_x, dtype=np.float)
        if len(self.x)//2 < settings.size:
            settings.write_log('ERROR: Check the .err file to get more info.')
            if settings.rank == 0:
                raise AttributeError("ERROR: Maximum number of processors exceeded. " +
                                     "It has to be less or equal than {0}.".format(len(self.x)//2))
            sys.exit(1)
        self.y = np.arange(self.y_0, self.y_0 + self.inc_y * self.ny, self.inc_y, dtype=np.float)

        # 1D to 2D
        x = np.array([self.x] * len(self.y))
        y = np.array([self.y] * len(self.x)).T

        # Create UTM bounds
        y_b = super(LccGrid, self).create_bounds(y, self.inc_y, number_vertices=4, inverse=True)
        x_b = super(LccGrid, self).create_bounds(x, self.inc_x, number_vertices=4)

        # Create the LCC projection
        projection = Proj(
            proj='lcc',
            ellps='WGS84',
            R=self.earth_radius,
            lat_1=self.lat_1,
            lat_2=self.lat_2,
            lon_0=self.lon_0,
            lat_0=self.lat_0,
            to_meter=1,
            x_0=0,
            y_0=0,
            a=self.earth_radius,
            k_0=1.0)

        # UTM to LCC
        self.center_longitudes, self.center_latitudes = projection(x, y, inverse=True)
        self.boundary_longitudes, self.boundary_latitudes = projection(x_b, y_b, inverse=True)

        settings.write_time('LccGrid', 'create_coords', timeit.default_timer() - st_time, level=2)


if __name__ == '__main__':
    pass
