#!/usr/bin/env python

import os
import timeit
import numpy as np

from hermesv3_bu.logger.log import Log


def select_grid(comm, logger, arguments):
    """
    Create and initialise the output grid.

    :param comm: MPI communicator.

    :param logger: Logger
    :type logger: Log

    :param arguments: Dictionary with all the necessary arguments to initialise the grid
    :type arguments: namespace

    :return: Desired output grid.
    :rtype: Grid
    """
    spent_time = timeit.default_timer()
    if comm.Get_rank() == 0:
        if arguments.domain_type == 'regular':
            from hermesv3_bu.grids.grid_latlon import LatLonGrid
            grid = LatLonGrid(
                logger, arguments.auxiliary_files_path, arguments.output_timestep_num,
                arguments.vertical_description, arguments.inc_lat, arguments.inc_lon, arguments.lat_orig,
                arguments.lon_orig, arguments.n_lat, arguments.n_lon)

        elif arguments.domain_type == 'lcc':
            from hermesv3_bu.grids.grid_lcc import LccGrid
            grid = LccGrid(
                logger, arguments.auxiliary_files_path, arguments.output_timestep_num,
                arguments.vertical_description, arguments.lat_1, arguments.lat_2, arguments.lon_0, arguments.lat_0,
                arguments.nx, arguments.ny, arguments.inc_x, arguments.inc_y, arguments.x_0, arguments.y_0)

        elif arguments.domain_type == 'rotated':
            from hermesv3_bu.grids.grid_rotated import RotatedGrid
            grid = RotatedGrid(
                logger, arguments.auxiliary_files_path, arguments.output_timestep_num,
                arguments.vertical_description, arguments.centre_lat, arguments.centre_lon, arguments.west_boundary,
                arguments.south_boundary, arguments.inc_rlat, arguments.inc_rlon)

        elif arguments.domain_type == 'mercator':
            from hermesv3_bu.grids.grid_mercator import MercatorGrid
            grid = MercatorGrid(
                logger, arguments.auxiliary_files_path, arguments.output_timestep_num,
                arguments.vertical_description, arguments.lat_ts, arguments.lon_0, arguments.nx, arguments.ny,
                arguments.inc_x, arguments.inc_y, arguments.x_0, arguments.y_0)

        else:
            raise NameError('Unknown grid type {0}'.format(arguments.domain_type))
    else:
        grid = None

    grid = comm.bcast(grid, root=0)

    logger.write_time_log('Grid', 'select_grid', timeit.default_timer() - spent_time)
    return grid


class Grid(object):

    def __init__(self, logger, attributes, auxiliary_path, vertical_description_path):
        """
        Initialise the Grid class

        :param logger: Logger
        :type logger: Log

        :param attributes: Attributes to define the grid
        :type attributes: dict

        :param auxiliary_path: Path to the folder to store all the needed auxiliary files.
        :type auxiliary_path: str

        :param vertical_description_path: Path to the file that describes the vertical resolution
        :type vertical_description_path: str
        """
        spent_time = timeit.default_timer()
        self.logger = logger
        self.logger.write_log('\tGrid specifications: {0}'.format(attributes), 3)
        self.attributes = attributes
        self.netcdf_path = os.path.join(auxiliary_path, 'grid', 'grid.nc')
        self.shapefile_path = os.path.join(auxiliary_path, 'grid', 'grid.shp')

        self.center_latitudes = None
        self.center_longitudes = None
        self.boundary_latitudes = None
        self.boundary_longitudes = None
        self.shape = None
        self.create_coords()
        self.write_netcdf()

        self.vertical_desctiption = self.get_vertical_description(vertical_description_path)
        self.shapefile = self.create_shapefile()

        logger.write_time_log('Grid', '__init__', timeit.default_timer() - spent_time)

    def get_vertical_description(self, path):
        """
        Extract the vertical description of the desired output.

        :param path: Path to the file that contains the output vertical description.
        :type path: str

        :return: Heights of the output vertical layers.
        :rtype: list
        """
        import pandas as pd
        spent_time = timeit.default_timer()
        df = pd.read_csv(path, sep=',')

        heights = df.height_magl.values
        self.logger.write_time_log('Grid', 'get_vertical_description', timeit.default_timer() - spent_time, 3)
        return heights

    def write_netcdf(self):
        """
        Implemented on inner classes
        """
        pass

    def create_coords(self):
        """
        Implemented on inner classes
        """
        pass

    def create_bounds(self, coordinates, inc, number_vertices=2, inverse=False):
        """
        Calculate the vertices coordinates.

        :param coordinates: Coordinates in degrees (latitude or longitude)
        :type coordinates: numpy.array

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
        spent_time = timeit.default_timer()
        # Create new arrays moving the centers half increment less and more.
        coords_left = coordinates - inc / 2
        coords_right = coordinates + inc / 2

        # Defining the number of corners needed. 2 to regular grids and 4 for irregular ones.
        if number_vertices == 2:
            # Create an array of N arrays of 2 elements to store the floor and the ceil values for each cell
            bound_coords = np.dstack((coords_left, coords_right))
            bound_coords = bound_coords.reshape((len(coordinates), number_vertices))
        elif number_vertices == 4:
            # Create an array of N arrays of 4 elements to store the corner values for each cell
            # It can be stored in clockwise starting form the left-top element, or in inverse mode.
            if inverse:
                bound_coords = np.dstack((coords_left, coords_left, coords_right, coords_right))
            else:
                bound_coords = np.dstack((coords_left, coords_right, coords_right, coords_left))
        else:
            raise ValueError('ERROR: The number of vertices of the boundaries must be 2 or 4.')
        self.logger.write_time_log('Grid', 'create_bounds', timeit.default_timer() - spent_time, 3)
        return bound_coords

    def create_shapefile(self):
        """
        Create a shapefile with the grid.

        :return: Grid shapefile
        :rtype: GeoDataFrame
        """
        import geopandas as gpd
        import pandas as pd
        from shapely.geometry import Polygon, Point
        spent_time = timeit.default_timer()

        if not os.path.exists(self.shapefile_path):
            if not os.path.exists(os.path.dirname(self.shapefile_path)):
                os.makedirs(os.path.dirname(self.shapefile_path))

            y = self.boundary_latitudes
            x = self.boundary_longitudes

            if self.grid_type == 'Regular Lat-Lon':
                x = x.reshape((x.shape[1], x.shape[2]))
                y = y.reshape((y.shape[1], y.shape[2]))

                aux_shape = (y.shape[0], x.shape[0], 4)
                x_aux = np.empty(aux_shape)
                x_aux[:, :, 0] = x[np.newaxis, :, 0]
                x_aux[:, :, 1] = x[np.newaxis, :, 1]
                x_aux[:, :, 2] = x[np.newaxis, :, 1]
                x_aux[:, :, 3] = x[np.newaxis, :, 0]

                x = x_aux
                del x_aux

                y_aux = np.empty(aux_shape)
                y_aux[:, :, 0] = y[:, np.newaxis, 0]
                y_aux[:, :, 1] = y[:, np.newaxis, 0]
                y_aux[:, :, 2] = y[:, np.newaxis, 1]
                y_aux[:, :, 3] = y[:, np.newaxis, 1]

                y = y_aux
                del y_aux

            aux_b_lats = y.reshape((y.shape[0] * y.shape[1], y.shape[2]))
            aux_b_lons = x.reshape((x.shape[0] * x.shape[1], x.shape[2]))
            gdf = gpd.GeoDataFrame(index=range(aux_b_lons.shape[0]), crs={'init': 'epsg:4326'})
            gdf['geometry'] = None
            # Create one dataframe with 8 columns, 4 points with two coordinates each one
            for i in range(aux_b_lons.shape[0]):
                gdf.loc[i, 'geometry'] = Polygon([(aux_b_lons[i, 0], aux_b_lats[i, 0]),
                                                  (aux_b_lons[i, 1], aux_b_lats[i, 1]),
                                                  (aux_b_lons[i, 2], aux_b_lats[i, 2]),
                                                  (aux_b_lons[i, 3], aux_b_lats[i, 3]),
                                                  (aux_b_lons[i, 0], aux_b_lats[i, 0])])

            gdf.to_crs(self.attributes['crs'], inplace=True)
            gdf['FID'] = gdf.index
            gdf.to_file(self.shapefile_path)

        else:
            gdf = gpd.read_file(self.shapefile_path)

        # gdf.set_index('FID', inplace=True, drop=False)
        self.logger.write_time_log('Grid', 'create_shapefile', timeit.default_timer() - spent_time, 2)

        return gdf

    def add_cell_area(self):
        from cdo import Cdo
        # spent_time = timeit.default_timer()

        # Initialises the CDO
        cdo = Cdo()
        cell_area = cdo.gridarea(input=self.netcdf_path, returnArray='cell_area')
        self.shapefile['cell_area'] = cell_area.flatten()

        # self.logger.write_time_log('Grid', 'add_cell_area', timeit.default_timer() - spent_time)
