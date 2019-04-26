#!/usr/bin/env python

import os
import sys
import timeit
import numpy as np


def select_grid(arguments):
    if arguments.domain_type == 'regular':
        from hermesv3_bu.grids.grid_latlon import LatLonGrid

        grid = LatLonGrid(arguments.auxiliar_files_path, arguments.vertical_description, arguments.inc_lat,
                          arguments.inc_lon, arguments.lat_orig, arguments.lon_orig, arguments.n_lat, arguments.n_lon)
    elif arguments.domain_type == 'lcc':
        from hermesv3_bu.grids.grid_lcc import LccGrid

        grid = LccGrid(arguments.auxiliar_files_path, arguments.vertical_description,arguments.lat_1, arguments.lat_2,
                       arguments.lon_0, arguments.lat_0, arguments.nx, arguments.ny,arguments.inc_x, arguments.inc_y,
                       arguments.x_0, arguments.y_0)
    else:
        raise NameError('Unknown grid type {0}'.format(arguments.domain_type))

    return grid

class Grid(object):
    """
    Grid object that contains the information of the output grid.

    :param attributes: Attributes to define the grid
    :type attributes: dict


    """
    def __init__(self, attributes, auxiliary_path, vertical_description_path):
        self.attributes = attributes
        self.netcdf_path = os.path.join(auxiliary_path, 'grid', 'grid.nc')
        self.shapefile_path = os.path.join(auxiliary_path, 'grid', 'grid.shp')

        self.center_latitudes = None
        self.center_longitudes = None
        self.boundary_latitudes = None
        self.boundary_longitudes = None
        self.create_coords()
        self.write_netcdf()

        self.vertical_desctiption = None
        self.shapefile = self.create_shapefile()

    def write_netcdf(self):
        # implemented on inner classes
        pass

    def create_coords(self):
        # implemented on inner classes
        pass

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
            raise ValueError('ERROR: The number of vertices of the boundaries must be 2 or 4.')

        return bound_coords

    def create_shapefile(self):
        import geopandas as gpd
        import pandas as pd
        from shapely.geometry import Polygon

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
            list_points = df.values
            del df['p1'], df['p2'], df['p3'], df['p4']

            # List of polygons from the list of points
            geometry = [Polygon(list(points)) for points in list_points]

            gdf = gpd.GeoDataFrame(index=df.index, crs={'init': 'epsg:4326'}, geometry=geometry)
            gdf = gdf.to_crs(self.attributes['crs'])
            gdf['FID'] = gdf.index
            gdf.to_file(self.shapefile_path)

        else:
            gdf = gpd.read_file(self.shapefile_path)

        return gdf