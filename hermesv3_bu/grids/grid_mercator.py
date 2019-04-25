#!/usr/bin/env python

import os
import sys
import numpy as np
from pyproj import Proj
from grid import Grid
from netcdf_tools import write_netcdf


class MercatorGrid(Grid):
    """
    Mercator grid object that contains all the information to do a mercator output.

    :param grid_type: Type of the output grid [global, rotated, lcc, mercator].
    :type grid_type: str

    :param temporal_path: Path to the temporal folder.
    :type temporal_path: str

    :param lon_0: Value of the Lon0 for the LCC grid type.
    :type lon_0: float

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

    def __init__(self, grid_type, temporal_path, lat_ts, lon_0, nx, ny, inc_x, inc_y, x_0, y_0,
                 earth_radius=6370000.000):
        # Initialises with parent class
        super(MercatorGrid, self).__init__(grid_type, temporal_path)

        # Setting parameters
        self.lat_ts = lat_ts
        self.lon_0 = lon_0
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
        self.crs = "+proj=merc +a={2} +b={2} +lat_ts={0} +lon_0={1}".format(self.lat_ts, self.lon_0, earth_radius)

        self.create_coords()

    def write_coords_netcdf(self):
        if not self.chech_coords_file():
            # Writes an auxiliary empty NetCDF only with the coordinates and an empty variable.
            write_netcdf(self.netcdf_file, self.center_latitudes, self.center_longitudes,
                         [{'name': 'var_aux', 'units': '', 'data': 0}],
                         boundary_latitudes=self.boundary_latitudes, boundary_longitudes=self.boundary_longitudes,
                         mercator=True, lcc_x=self.x, lcc_y=self.y, lon_0=self.lon_0, lat_ts=self.lat_ts)

    def create_coords(self):
        """
        Create the coordinates for a mercator domain.
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
        projection = Proj(self.crs)

        # UTM to Mercator
        self.center_longitudes, self.center_latitudes = projection(x, y, inverse=True)
        self.boundary_longitudes, self.boundary_latitudes = projection(x_b, y_b, inverse=True)


if __name__ == '__main__':
    grid = MercatorGrid('mercator', '/home/Earth/ctena/temp', lat_ts=-1.5, lon_0=-18, nx=210, ny=236, inc_x=50000,
                        inc_y=50000, x_0=-126017.5, y_0=-5407460)
    grid.write_coords_netcdf()

