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
import numpy as np
import ESMF
import hermesv3_gr.config.settings as settings


class Grid(object):
    """
    Grid object that contains the information of the output grid.

    :param grid_type: Type of the output grid [global, rotated, lcc, mercator].
    :type grid_type: str

    :param vertical_description_path: Path to the file that contains the vertical description.
    :type vertical_description_path: str

    :param temporal_path: Path to the temporal folder.
    :type temporal_path: str
    """

    def __init__(self, grid_type, vertical_description_path, temporal_path):
        st_time = timeit.default_timer()
        # settings.write_log('Creating Grid...', level=1)

        # Defining class atributes
        self.procs_array = None
        self.nrows = 0
        self.ncols = 0

        self.grid_type = grid_type
        self.vertical_description = self.set_vertical_levels(vertical_description_path)
        self.center_latitudes = None
        self.center_longitudes = None
        self.boundary_latitudes = None
        self.boundary_longitudes = None

        self.cell_area = None
        if settings.rank == 0:
            if not os.path.exists(os.path.join(temporal_path)):
                os.makedirs(os.path.join(temporal_path))
        settings.comm.Barrier()

        self.coords_netcdf_file = os.path.join(temporal_path, 'temporal_coords.nc')
        self.temporal_path = temporal_path
        self.shapefile_path = None

        self.esmf_grid = None
        self.x_lower_bound = None
        self.x_upper_bound = None
        self.y_lower_bound = None
        self.y_upper_bound = None
        self.shape = None

        self.crs = None

        settings.write_time('Grid', 'Init', timeit.default_timer() - st_time, level=1)

    @staticmethod
    def create_esmf_grid_from_file(file_name, sphere=True):
        import ESMF

        st_time = timeit.default_timer()
        settings.write_log('\t\tCreating ESMF grid from file {0}'.format(file_name), level=3)

        # ESMF.Manager(debug=True)

        grid = ESMF.Grid(filename=file_name, filetype=ESMF.FileFormat.GRIDSPEC, is_sphere=sphere,
                         add_corner_stagger=True)

        settings.write_time('Grid', 'create_esmf_grid_from_file', timeit.default_timer() - st_time, level=3)
        return grid

    @staticmethod
    def select_grid(grid_type, vertical_description_path, timestep_num, temporal_path, inc_lat, inc_lon,
                    centre_lat, centre_lon, west_boundary, south_boundary, inc_rlat, inc_rlon,
                    lat_1, lat_2, lon_0, lat_0, nx, ny, inc_x, inc_y, x_0, y_0, lat_ts):
        # TODO describe better the rotated parameters
        """
        Create a Grid object depending on the grid type.

        :param grid_type: type of grid to create [global, rotated, lcc, mercator]
        :type grid_type: str

        :param vertical_description_path: Path to the file that contains the vertical description.
        :type vertical_description_path: str

        :param timestep_num: Number of timesteps.
        :type timestep_num: int

        :param temporal_path: Path to the temporal folder.
        :type temporal_path: str

        :param inc_lat: [global] Increment between latitude centroids (degrees).
        :type inc_lat: float

        :param inc_lon: [global] Increment between longitude centroids (degrees).
        :type inc_lon: float

        :param centre_lat: [rotated]
        :type centre_lat: float

        :param centre_lon: [rotated]
        :type centre_lon: float

        :param west_boundary: [rotated]
        :type west_boundary: float

        :param south_boundary: [rotated]
        :type south_boundary: float

        :param inc_rlat: [rotated] Increment between rotated latitude centroids (degrees).
        :type inc_rlat: float

        :param inc_rlon: [rotated] Increment between rotated longitude centroids (degrees).
        :type inc_rlon: float

        :param lat_ts: [mercator]
        :type lat_ts: float

        :param lat_1: [lcc] Value of the Lat1 for the LCC grid type.
        :type lat_1: float

        :param lat_2: [lcc] Value of the Lat2 for the LCC grid type.
        :type lat_2: float

        :param lon_0: [lcc, mercator] Value of the Lon0 for the LCC grid type.
        :type lon_0: float

        :param lat_0: [lcc] Value of the Lat0 for the LCC grid type.
        :type lat_0: float

        :param nx: [lcc, mercator] Number of cells on the x dimension.
        :type nx: int

        :param ny: [lcc, mercator] Number of cells on the y dimension.
        :type ny: int

        :param inc_x: [lcc, mercator] Increment between x dimensions cell centroids (metres).
        :type inc_x: int

        :param inc_y: [lcc, mercator] Increment between y dimensions cell centroids (metres).
        :type inc_y: int

        :param x_0: [lcc, mercator] Value of the X0 for the LCC grid type.
        :type x_0: float

        :param y_0: [lcc, mercator] Value of the Y0 for the LCC grid type.
        :type y_0: float

        :return: Grid object. It will return a GlobalGrid, RotatedGrid or LccGrid depending on the type.
        :rtype: Grid
        """

        st_time = timeit.default_timer()
        settings.write_log('Selecting grid', level=1)

        # Creating a different object depending on the grid type
        if grid_type == 'global':
            from hermesv3_gr.modules.grids.grid_global import GlobalGrid
            grid = GlobalGrid(grid_type, vertical_description_path, timestep_num, temporal_path, inc_lat, inc_lon)

        elif grid_type == 'rotated':
            from hermesv3_gr.modules.grids.grid_rotated import RotatedGrid
            grid = RotatedGrid(grid_type, vertical_description_path, timestep_num, temporal_path,
                               centre_lat, centre_lon, west_boundary, south_boundary, inc_rlat, inc_rlon)

        elif grid_type == 'lcc':
            from hermesv3_gr.modules.grids.grid_lcc import LccGrid
            grid = LccGrid(grid_type, vertical_description_path, timestep_num, temporal_path, lat_1, lat_2, lon_0,
                           lat_0, nx, ny, inc_x, inc_y, x_0, y_0)

        elif grid_type == 'mercator':
            from hermesv3_gr.modules.grids.grid_mercator import MercatorGrid
            grid = MercatorGrid(grid_type, vertical_description_path, timestep_num, temporal_path, lat_ts, lon_0,
                                nx, ny, inc_x, inc_y, x_0, y_0)
        else:
            settings.write_log('ERROR: Check the .err file to get more info.')
            if settings.rank == 0:
                raise NotImplementedError("The grid type {0} is not implemented.".format(grid_type)
                                          + " Use 'global', 'rotated' or 'lcc'.")
            sys.exit(1)

        settings.write_time('Grid', 'select_grid', timeit.default_timer() - st_time, level=3)

        return grid

    @staticmethod
    def set_vertical_levels(vertical_description_path):
        """
        Extract the vertical levels.

        :param vertical_description_path: path to the file that contain the vertical description of the required output
        file.
        :type vertical_description_path: str

        :return: Vertical levels.
        :rtype: list of int
        """
        import pandas as pd

        st_time = timeit.default_timer()
        settings.write_log('\t\tSetting vertical levels', level=3)

        df = pd.read_csv(vertical_description_path, sep=';')

        heights = df.height_magl.values

        settings.write_time('Grid', 'set_vertical_levels', timeit.default_timer() - st_time, level=3)

        return heights

    def write_coords_netcdf(self):
        """
        Writes the temporal file with the coordinates of the output needed to generate the weight matrix.
        If it is already well created it will only add the cell_area parameter.
        """
        # TODO Not to write two NetCDF. Open one and modify it.
        from hermesv3_gr.tools.netcdf_tools import write_netcdf

        st_time = timeit.default_timer()
        settings.write_log('\twrite_coords_netcdf', level=3)

        if not self.chech_coords_file():
            # Writes an auxiliary empty NetCDF only with the coordinates and an empty variable.
            write_netcdf(self.coords_netcdf_file, self.center_latitudes, self.center_longitudes,
                         [{'name': 'var_aux', 'units': '', 'data': 0}],
                         boundary_latitudes=self.boundary_latitudes, boundary_longitudes=self.boundary_longitudes,
                         regular_latlon=True)

            # Calculate the cell area of the auxiliary NetCDF file
            self.cell_area = self.get_cell_area()

            # Re-writes the NetCDF adding the cell area
            write_netcdf(self.coords_netcdf_file, self.center_latitudes, self.center_longitudes,
                         [{'name': 'var_aux', 'units': '', 'data': 0}],
                         cell_area=self.cell_area, boundary_latitudes=self.boundary_latitudes,
                         boundary_longitudes=self.boundary_longitudes, regular_latlon=True)
        else:
            self.cell_area = self.get_cell_area()

        settings.write_time('Grid', 'write_coords_netcdf', timeit.default_timer() - st_time, level=3)

    def get_cell_area(self):
        """
        Calculate the cell area of the grid.

        :return: Area of each cell of the grid.
        :rtype: numpy.array
        """
        from cdo import Cdo
        from netCDF4 import Dataset

        st_time = timeit.default_timer()
        settings.write_log('\t\tGetting cell area from {0}'.format(self.coords_netcdf_file), level=3)

        # Initialises the CDO
        cdo = Cdo()
        # Create a temporal file 's' with the cell area
        s = cdo.gridarea(input=self.coords_netcdf_file)
        # Get the cell area of the temporal file
        nc_aux = Dataset(s, mode='r')
        cell_area = nc_aux.variables['cell_area'][:]
        nc_aux.close()

        settings.write_time('Grid', 'get_cell_area', timeit.default_timer() - st_time, level=3)

        return cell_area

    @staticmethod
    def create_regular_grid_1d_array(center, inc, boundary):
        """
        Create a regular grid giving the center, boundary and increment.

        :param center: Center of the coordinates.
        :type center: float

        :param inc: Resolution: Increment between cells.
        :type inc: float

        :param boundary: Limit of the coordinates: Distance between the first cell and the center.
        :type boundary: float

        :return: 1D array with the coordinates.
        :rtype: numpy.array
        """

        st_time = timeit.default_timer()

        # Calculate first center point.
        origin = center - abs(boundary)
        # Calculate the quantity of cells.
        n = (abs(boundary) / inc) * 2
        # Calculate all the values
        values = np.arange(origin + inc, origin + (n * inc) - inc + inc / 2, inc, dtype=np.float)

        settings.write_time('Grid', 'create_regular_grid_1d_array', timeit.default_timer() - st_time, level=3)

        return values

    @staticmethod
    def create_bounds(coords, inc, number_vertices=2, inverse=False):
        """
        Calculate the vertices coordinates.

        :param coords: Coordinates in degrees (latitude or longitude)
        :type coords: numpy.array

        :param inc: Increment between center values.
        :type inc: float

        :param number_vertices: Non mandatory parameter that informs the number of vertices that must have the
                boundaries (by default 2).
        :type number_vertices: int

        :param inverse: For some grid latitudes.
        :type inverse: bool

        :return: Array with as many elements as vertices for each value of coords.
        :rtype: numpy.array
        """
        st_time = timeit.default_timer()
        settings.write_log('\t\t\tCreating boundaries.', level=3)

        # Create new arrays moving the centers half increment less and more.
        coords_left = coords - inc / 2
        coords_right = coords + inc / 2

        # Defining the number of corners needed. 2 to regular grids and 4 for irregular ones.
        if number_vertices == 2:
            # Create an array of N arrays of 2 elements to store the floor and the ceil values for each cell
            bound_coords = np.dstack((coords_left, coords_right))
            bound_coords = bound_coords.reshape((len(coords), number_vertices))
        elif number_vertices == 4:
            # Create an array of N arrays of 4 elements to store the corner values for each cell
            # It can be stored in clockwise starting form the left-top element, or in inverse mode.
            if inverse:
                bound_coords = np.dstack((coords_left, coords_left, coords_right, coords_right))

            else:
                bound_coords = np.dstack((coords_left, coords_right, coords_right, coords_left))
        else:
            if settings.rank == 0:
                raise ValueError('ERROR: The number of vertices of the boundaries must be 2 or 4.')
            settings.write_log('ERROR: Check the .err file to get more info.')
            sys.exit(1)

        settings.write_time('Grid', 'create_bounds', timeit.default_timer() - st_time, level=3)

        return bound_coords

    def get_coordinates_2d(self):
        """
        Returns the coordinates but in a 2D format.

        A regular grid only needs two 1D arrays (latitudes and longitudes) to define a grid.
        This method is to convert this two 1D arrays into 2D arrays replicating the info of each value.

        :return: Tuple with 2 fields, the first the 2D latitude coordinate, and the second for the 2D longitude
        coordinate.
        :rtype: tuple
        """
        st_time = timeit.default_timer()
        settings.write_log('\t\tGetting 2D coordinates from ESMPy Grid', level=3)

        lat = self.esmf_grid.get_coords(1, ESMF.StaggerLoc.CENTER).T
        lon = self.esmf_grid.get_coords(0, ESMF.StaggerLoc.CENTER).T

        settings.write_time('Grid', 'get_coordinates_2d', timeit.default_timer() - st_time, level=3)

        return lat, lon

    def is_shapefile(self):
        return os.path.exists(self.shapefile_path)

    def to_shapefile(self, full_grid=True):
        import geopandas as gpd
        import pandas as pd
        from shapely.geometry import Polygon

        st_time = timeit.default_timer()
        # settings.write_log('\t\tGetting grid shapefile', level=3)

        if full_grid:
            self.shapefile_path = os.path.join(self.temporal_path, 'shapefile')
        else:
            self.shapefile_path = os.path.join(self.temporal_path, 'shapefiles_n{0}'.format(settings.size))

        if settings.rank == 0:
            if not os.path.exists(self.shapefile_path):
                os.makedirs(self.shapefile_path)
        if full_grid:
            self.shapefile_path = os.path.join(self.shapefile_path, 'grid_shapefile.shp')
        else:
            self.shapefile_path = os.path.join(self.shapefile_path, 'grid_shapefile_{0}.shp'.format(settings.rank))

        done = self.is_shapefile()

        if not done:
            settings.write_log('\t\tGrid shapefile not done. Lets try to create it.', level=3)
            # Create Shapefile

            # Use the meters coordiantes to create the shapefile

            y = self.boundary_latitudes
            x = self.boundary_longitudes
            # sys.exit()

            if self.grid_type == 'global':
                x = x.reshape((x.shape[1], x.shape[2]))
                y = y.reshape((y.shape[1], y.shape[2]))

                # x_aux = np.empty((x.shape[0], y.shape[0], 4))
                # x_aux[:, :, 0] = x[:, np.newaxis, 0]
                # x_aux[:, :, 1] = x[:, np.newaxis, 1]
                # x_aux[:, :, 2] = x[:, np.newaxis, 1]
                # x_aux[:, :, 3] = x[:, np.newaxis, 0]
                aux_shape = (y.shape[0], x.shape[0], 4)
                x_aux = np.empty(aux_shape)
                x_aux[:, :, 0] = x[np.newaxis, :, 0]
                x_aux[:, :, 1] = x[np.newaxis, :, 1]
                x_aux[:, :, 2] = x[np.newaxis, :, 1]
                x_aux[:, :, 3] = x[np.newaxis, :, 0]

                x = x_aux
                # print x
                del x_aux

                # y_aux = np.empty((x.shape[0], y.shape[0], 4))
                # y_aux[:, :, 0] = y[np.newaxis, :, 0]
                # y_aux[:, :, 1] = y[np.newaxis, :, 0]
                # y_aux[:, :, 2] = y[np.newaxis, :, 1]
                # y_aux[:, :, 3] = y[np.newaxis, :, 1]

                y_aux = np.empty(aux_shape)
                y_aux[:, :, 0] = y[:, np.newaxis, 0]
                y_aux[:, :, 1] = y[:, np.newaxis, 0]
                y_aux[:, :, 2] = y[:, np.newaxis, 1]
                y_aux[:, :, 3] = y[:, np.newaxis, 1]

                # print y_aux
                y = y_aux
                del y_aux

                # exit()

            if not full_grid:
                y = y[self.x_lower_bound:self.x_upper_bound, self.y_lower_bound:self.y_upper_bound, :]
                x = x[self.x_lower_bound:self.x_upper_bound, self.y_lower_bound:self.y_upper_bound, :]

            aux_b_lats = y.reshape((y.shape[0] * y.shape[1], y.shape[2]))
            aux_b_lons = x.reshape((x.shape[0] * x.shape[1], x.shape[2]))

            # The regular lat-lon projection has only 2 (laterals) points for each cell instead of 4 (corners)
            # if aux_b_lats.shape[1] == 2:
            #     aux_b = np.empty((aux_b_lats.shape[0], 4))
            #     aux_b[:, 0] = aux_b_lats[:, 0]
            #     aux_b[:, 1] = aux_b_lats[:, 0]
            #     aux_b[:, 2] = aux_b_lats[:, 1]
            #     aux_b[:, 3] = aux_b_lats[:, 1]
            #     aux_b_lats = aux_b
            #
            # if aux_b_lons.shape[1] == 2:
            #     aux_b = np.empty((aux_b_lons.shape[0], 4))
            #     aux_b[:, 0] = aux_b_lons[:, 0]
            #     aux_b[:, 1] = aux_b_lons[:, 1]
            #     aux_b[:, 2] = aux_b_lons[:, 1]
            #     aux_b[:, 3] = aux_b_lons[:, 0]
            #     aux_b_lons = aux_b

            # Create one dataframe with 8 columns, 4 points with two coordinates each one
            df_lats = pd.DataFrame(aux_b_lats, columns=['b_lat_1', 'b_lat_2', 'b_lat_3', 'b_lat_4'])
            df_lons = pd.DataFrame(aux_b_lons, columns=['b_lon_1', 'b_lon_2', 'b_lon_3', 'b_lon_4'])
            df = pd.concat([df_lats, df_lons], axis=1)

            # Substituate 8 columns by 4 with the two coordinates
            df['p1'] = zip(df.b_lon_1, df.b_lat_1)
            del df['b_lat_1'], df['b_lon_1']
            df['p2'] = zip(df.b_lon_2, df.b_lat_2)
            del df['b_lat_2'], df['b_lon_2']
            df['p3'] = zip(df.b_lon_3, df.b_lat_3)
            del df['b_lat_3'], df['b_lon_3']
            df['p4'] = zip(df.b_lon_4, df.b_lat_4)
            del df['b_lat_4'], df['b_lon_4']

            # Make a list of list of tuples
            # [[(point_1.1), (point_1.2), (point_1.3), (point_1.4)],
            # [(point_2.1), (point_2.2), (point_2.3), (point_2.4)], ...]
            list_points = df.as_matrix()
            del df['p1'], df['p2'], df['p3'], df['p4']

            # List of polygons from the list of points
            geometry = [Polygon(list(points)) for points in list_points]
            # geometry = []
            # for point in list_points:
            #     print point
            #     geometry.append(Polygon(list(point)))
            # print geometry[0]
            # sys.exit()
            # print len(geometry), len(df),

            gdf = gpd.GeoDataFrame(df, crs={'init': 'epsg:4326'}, geometry=geometry)
            gdf = gdf.to_crs(self.crs)

            gdf['FID'] = gdf.index

            gdf.to_file(self.shapefile_path)
        else:
            settings.write_log('\t\tGrid shapefile already done. Lets try to read it.', level=3)
            gdf = gpd.read_file(self.shapefile_path)

        settings.write_time('Grid', 'to_shapefile', timeit.default_timer() - st_time, level=1)

        return gdf

    def chech_coords_file(self):
        """
        Checks if the auxiliary coordinates file is created well.

        :return: True: if it is well created.
        :rtype: bool
        """
        # TODO better check by partition size
        return os.path.exists(self.coords_netcdf_file)


if __name__ == '__main__':
    pass
