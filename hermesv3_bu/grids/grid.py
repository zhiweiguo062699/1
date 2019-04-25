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
    else:
        raise NameError('Unknown grid type {0}'.format(arguments.domain_type))


class Grid(object):
    """
    Grid object that contains the information of the output grid.

    :param attributes: Attributes to define the grid
    :type attributes: dict


    """
    def __init__(self, attributes, auxiliary_path, vertical_description_path):
        self.attributes = attributes
        self.vertical_desctiption = None
        self.shapefile = None
        self.shape = None
        self.netcdf_path = os.path.join(auxiliary_path, 'grid', 'grid.nc')
        self.shapefile_path = os.path.join(auxiliary_path, 'grid', 'grid.shp')

        self.center_latitudes = None
        self.center_longitudes = None
        self.boundary_latitudes = None
        self.boundary_longitudes = None

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
