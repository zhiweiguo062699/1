#!/usr/bin/env python

import os
import sys
import numpy as np
from pyproj import Proj
from grid import Grid
from netcdf_tools import write_netcdf


class LccGrid(Grid):
    """
    Lambert Conformal Conic (LCC) grid object that contains all the information to do a lcc output.

    :param grid_type: Type of the output grid [global, rotated, lcc, mercator].
    :type grid_type: str

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

    def __init__(self, grid_type, temporal_path, lat_1, lat_2, lon_0, lat_0,  nx, ny, inc_x, inc_y, x_0, y_0,
                 earth_radius=6370000.000):

        # Initialises with parent class
        super(LccGrid, self).__init__(grid_type, temporal_path)

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

    def write_coords_netcdf(self):
        if not self.chech_coords_file():
            # Writes an auxiliary empty NetCDF only with the coordinates and an empty variable.
            write_netcdf(self.netcdf_file, self.center_latitudes, self.center_longitudes,
                         [{'name': 'var_aux', 'units': '', 'data': 0}],
                         boundary_latitudes=self.boundary_latitudes, boundary_longitudes=self.boundary_longitudes,
                         lcc=True, lcc_x=self.x, lcc_y=self.y,
                         lat_1_2="{0}, {1}".format(self.lat_1, self.lat_2), lon_0=self.lon_0, lat_0=self.lat_0)

    def create_coords(self):
        """
        Create the coordinates for a lambert conformal conic domain.
        """

        # Create a regular grid in metres (Two 1D arrays)
        self.x = np.arange(self.x_0, self.x_0 + self.inc_x * self.nx, self.inc_x, dtype=np.float)
        self.y = np.arange(self.y_0, self.y_0 + self.inc_y * self.ny, self.inc_y, dtype=np.float)

        # 1D to 2D
        x = np.array([self.x] * len(self.y))
        y = np.array([self.y] * len(self.x)).T

        # Create UTM bounds
        y_b = self.create_bounds(y, self.inc_y, number_vertices=4, inverse=True)
        x_b = self.create_bounds(x, self.inc_x, number_vertices=4)

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


if __name__ == '__main__':
    grid = LccGrid('lambert', '/home/Earth/ctena/temp', lat_1=37, lat_2=43, lon_0=-3, lat_0=40, nx=397, ny=397,
                   inc_x=4000, inc_y=4000, x_0=-807847.688, y_0=-797137.125)
    grid.write_coords_netcdf()
