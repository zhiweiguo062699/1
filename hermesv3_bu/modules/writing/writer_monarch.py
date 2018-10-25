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
import timeit
import numpy as np
from netCDF4 import Dataset
from mpi4py import MPI
from hermesv3_gr.modules.writing.writer import Writer
from hermesv3_gr.config import settings


class WriterMonarch(Writer):
    """
   Class to Write the output file in CF-1.6 conventions.

   :param path: Path to the destination file.
   :type path: str

   :param grid: Grid of the destination file.
   :type grid: Grid

   :param levels: List with the levels of the grid.
   :type levels: list

   :param date: Date of the output file
   :type date: datetime.datetime

   :param hours: List with the timestamp hours.
   :type hours: list.

   :param global_attributes_path: Path to the file that contains the static global attributes.
   :type global_attributes_path: str

   :param compress: Indicates if you want to compress the netCDF variable data.
   :type compress: bool

   :param parallel: Indicates if you want to write in parallel mode.
   :type parallel. bool
   """

    def __init__(self, path, grid, levels, date, hours, global_attributes_path, compress=True, parallel=False):
        super(WriterMonarch, self).__init__(path, grid, levels, date, hours, global_attributes_path, compress, parallel)

        # self.global_attributes = {
        #     'nom_attribut': 'value_attribut'
        # }

    def unit_change(self, variable, data):
        """
        Do the unit conversions of the data.

        :param variable: Variable to convert.
        :type variable: dict

        :param data: Data to change.
        :type data: numpy.array

        :return: Data with the new units.
        :rtype: numpy.array
        """
        from cf_units import Unit
        st_time = timeit.default_timer()

        if data is not None:
            units = None
            for var_name in self.variables_attributes:
                if var_name == variable:
                    units = self.variables_attributes[var_name]['units']
                    break

            if Unit(units).symbol == Unit('mol.s-1.m-2').symbol:
                data = data * 1000
            elif Unit(units).symbol == Unit('kg.s-1.m-2').symbol:
                pass
            else:
                settings.write_log('ERROR: Check the .err file to get more info.')
                if settings.rank == 0:
                    raise TypeError("The unit '{0}' of specie {1} is not defined correctly. ".format(units, variable) +
                                    "Should be 'mol.s-1.m-2' or 'kg.s-1.m-2'")
                sys.exit(1)
        settings.write_time('WriterMonarch', 'unit_change', timeit.default_timer() - st_time, level=3)
        return data

    def change_variable_attributes(self):
        """
        Modify the emission list to be consistent to use the output as input for CMAQ model.

        :return: Emission list ready for CMAQ
        :rtype: dict
        """
        new_variable_dict = {}
        for variable in self.variables_attributes:
            new_variable_dict[variable['name']] = variable
            del new_variable_dict[variable['name']]['name']

        self.variables_attributes = new_variable_dict

    def create_parallel_netcdf(self):
        """
        Create an empty netCDF4.

        :return: True at end.
        :rtype: bool
        """
        from cf_units import Unit, encode_time

        st_time = timeit.default_timer()

        RegularLatLon = False
        Rotated = False
        LambertConformalConic = False
        if self.grid.grid_type == 'global':
            RegularLatLon = True
        elif self.grid.grid_type == 'rotated':
            Rotated = True
        elif self.grid.grid_type == 'lcc':
            LambertConformalConic = True

        settings.write_log("\tCreating parallel NetCDF file.", level=2)
        # netcdf = Dataset(netcdf_path, mode='w', format="NETCDF4", parallel=True, comm=settings.comm, info=MPI.Info())
        netcdf = Dataset(self.path, mode='w', format="NETCDF4")
        # print 'NETCDF PATH: {0}'.format(netcdf_path)

        settings.write_log("\t\tCreating NetCDF dimensions.", level=2)
        # ===== Dimensions =====
        if RegularLatLon:
            var_dim = ('lat', 'lon',)

            # Latitude
            if len(self.grid.center_latitudes.shape) == 1:
                netcdf.createDimension('lat', self.grid.center_latitudes.shape[0])
                settings.write_log("\t\t\t'lat' dimension: {0}".format(self.grid.center_latitudes.shape[0]), level=3)
                lat_dim = ('lat',)
            elif len(self.grid.center_latitudes.shape) == 2:
                netcdf.createDimension('lat', self.grid.center_latitudes.shape[0])
                settings.write_log("\t\t\t'lat' dimension: {0}".format(self.grid.center_latitudes.shape[0]), level=3)
                lat_dim = ('lon', 'lat', )
            else:
                print 'ERROR: Latitudes must be on a 1D or 2D array instead of {0}'.format(
                    len(self.grid.center_latitudes.shape))
                sys.exit(1)

            # Longitude
            if len(self.grid.center_longitudes.shape) == 1:
                netcdf.createDimension('lon', self.grid.center_longitudes.shape[0])
                settings.write_log("\t\t\t'lon' dimension: {0}".format(self.grid.center_longitudes.shape[0]), level=3)
                lon_dim = ('lon',)
            elif len(self.grid.center_longitudes.shape) == 2:
                netcdf.createDimension('lon', self.grid.center_longitudes.shape[1])
                settings.write_log("\t\t\t'lon' dimension: {0}".format(self.grid.center_longitudes.shape[1]), level=3)
                lon_dim = ('lon', 'lat', )
            else:
                print 'ERROR: Longitudes must be on a 1D or 2D array instead of {0}'.format(
                    len(self.grid.center_longitudes.shape))
                sys.exit(1)
        elif Rotated:
            var_dim = ('rlat', 'rlon',)

            # Rotated Latitude
            if self.grid.rlat is None:
                print 'ERROR: For rotated grids is needed the rotated latitudes.'
                sys.exit(1)
            netcdf.createDimension('rlat', len(self.grid.rlat))
            settings.write_log("\t\t\t'rlat' dimension: {0}".format(len(self.grid.rlat)), level=3)
            lat_dim = ('rlat', 'rlon',)

            # Rotated Longitude
            if self.grid.rlon is None:
                print 'ERROR: For rotated grids is needed the rotated longitudes.'
                sys.exit(1)
            netcdf.createDimension('rlon', len(self.grid.rlon))
            settings.write_log("\t\t\t'rlon' dimension: {0}".format(len(self.grid.rlon)), level=3)
            lon_dim = ('rlat', 'rlon',)

        elif LambertConformalConic:
            var_dim = ('y', 'x',)

            netcdf.createDimension('y', len(self.grid.y))
            settings.write_log("\t\t\t'y' dimension: {0}".format(len(self.grid.y)), level=3)
            lat_dim = ('y', 'x', )

            netcdf.createDimension('x', len(self.grid.x))
            settings.write_log("\t\t\t'x' dimension: {0}".format(len(self.grid.x)), level=3)
            lon_dim = ('y', 'x', )
        else:
            lat_dim = None
            lon_dim = None
            var_dim = None

        # Levels
        if self.levels is not None:
            netcdf.createDimension('lev', len(self.levels))
            settings.write_log("\t\t\t'lev' dimension: {0}".format(len(self.levels)), level=3)

        # Bounds
        if self.grid.boundary_latitudes is not None:
            # print boundary_latitudes.shape
            # print len(boundary_latitudes[0, 0])
            netcdf.createDimension('nv', len(self.grid.boundary_latitudes[0, 0]))
            settings.write_log("\t\t\t'nv' dimension: {0}".format(len(self.grid.boundary_latitudes[0, 0])), level=3)
            # sys.exit()

        # Time
        # netcdf.createDimension('time', None)
        netcdf.createDimension('time', len(self.hours))
        settings.write_log("\t\t\t'time' dimension: {0}".format(len(self.hours)), level=3)

        # ===== Variables =====
        settings.write_log("\t\tCreating NetCDF variables.", level=2)
        # Time
        if self.date is None:
            time = netcdf.createVariable('time', 'd', ('time',))
            time.units = "months since 2000-01-01 00:00:00"
            time.standard_name = "time"
            time.calendar = "gregorian"
            time.long_name = "time"
            time[:] = [0.]
        else:
            time = netcdf.createVariable('time', 'd', ('time',))
            time.units = str(Unit('hours').offset_by_time(encode_time(self.date.year, self.date.month, self.date.day,
                                                          self.date.hour, self.date.minute, self.date.second)))
            time.standard_name = "time"
            time.calendar = "gregorian"
            time.long_name = "time"
            if settings.rank == 0:
                time[:] = self.hours
            settings.write_log("\t\t\t'time' variable created with size: {0}".format(time[:].shape), level=3)

        # Latitude
        lats = netcdf.createVariable('lat', 'f', lat_dim, zlib=self.compress)
        lats.units = "degrees_north"
        lats.axis = "Y"
        lats.long_name = "latitude coordinate"
        lats.standard_name = "latitude"
        if settings.rank == 0:
            lats[:] = self.grid.center_latitudes
        settings.write_log("\t\t\t'lat' variable created with size: {0}".format(lats[:].shape), level=3)

        if self.grid.boundary_latitudes is not None:
            lats.bounds = "lat_bnds"
            lat_bnds = netcdf.createVariable('lat_bnds', 'f', lat_dim + ('nv',), zlib=self.compress)
            # print lat_bnds[:].shape, boundary_latitudes.shape
            if settings.rank == 0:
                lat_bnds[:] = self.grid.boundary_latitudes
            settings.write_log("\t\t\t'lat_bnds' variable created with size: {0}".format(lat_bnds[:].shape), level=3)

        # Longitude
        lons = netcdf.createVariable('lon', 'f', lon_dim, zlib=self.compress)
        lons.units = "degrees_east"
        lons.axis = "X"
        lons.long_name = "longitude coordinate"
        lons.standard_name = "longitude"
        if settings.rank == 0:
            lons[:] = self.grid.center_longitudes
        settings.write_log("\t\t\t'lon' variable created with size: {0}".format(lons[:].shape), level=3)

        if self.grid.boundary_longitudes is not None:
            lons.bounds = "lon_bnds"
            lon_bnds = netcdf.createVariable('lon_bnds', 'f', lon_dim + ('nv',), zlib=self.compress)
            # print lon_bnds[:].shape, boundary_longitudes.shape
            if settings.rank == 0:
                lon_bnds[:] = self.grid.boundary_longitudes
            settings.write_log("\t\t\t'lon_bnds' variable created with size: {0}".format(lon_bnds[:].shape), level=3)

        if Rotated:
            # Rotated Latitude
            rlat = netcdf.createVariable('rlat', 'f', ('rlat',), zlib=self.compress)
            rlat.long_name = "latitude in rotated pole grid"
            rlat.units = Unit("degrees").symbol
            rlat.standard_name = "grid_latitude"
            if settings.rank == 0:
                rlat[:] = self.grid.rlat
            settings.write_log("\t\t\t'rlat' variable created with size: {0}".format(rlat[:].shape), level=3)

            # Rotated Longitude
            rlon = netcdf.createVariable('rlon', 'f', ('rlon',), zlib=self.compress)
            rlon.long_name = "longitude in rotated pole grid"
            rlon.units = Unit("degrees").symbol
            rlon.standard_name = "grid_longitude"
            if settings.rank == 0:
                rlon[:] = self.grid.rlon
            settings.write_log("\t\t\t'rlon' variable created with size: {0}".format(rlon[:].shape), level=3)
        if LambertConformalConic:
            x_var = netcdf.createVariable('x', 'd', ('x',), zlib=self.compress)
            x_var.units = Unit("km").symbol
            x_var.long_name = "x coordinate of projection"
            x_var.standard_name = "projection_x_coordinate"
            if settings.rank == 0:
                x_var[:] = self.grid.x
            settings.write_log("\t\t\t'x' variable created with size: {0}".format(x_var[:].shape), level=3)

            y_var = netcdf.createVariable('y', 'd', ('y',), zlib=self.compress)
            y_var.units = Unit("km").symbol
            y_var.long_name = "y coordinate of projection"
            y_var.standard_name = "projection_y_coordinate"
            if settings.rank == 0:
                y_var[:] = self.grid.y
            settings.write_log("\t\t\t'y' variable created with size: {0}".format(y_var[:].shape), level=3)

        cell_area_dim = var_dim
        # Levels
        if self.levels is not None:
            var_dim = ('lev',) + var_dim
            lev = netcdf.createVariable('lev', 'f', ('lev',), zlib=self.compress)
            lev.units = Unit("m").symbol
            lev.positive = 'up'
            if settings.rank == 0:
                lev[:] = self.levels
            settings.write_log("\t\t\t'lev' variable created with size: {0}".format(lev[:].shape), level=3)
        # print 'DATA LIIIIST {0}'.format(data_list)
    #     # All variables
        if len(self.variables_attributes) is 0:
            var = netcdf.createVariable('aux_var', 'f', ('time',) + var_dim, zlib=self.compress)
            if settings.rank == 0:
                var[:] = 0

        index = 0
        for var_name, variable in self.variables_attributes.iteritems():
            index += 1

            var = netcdf.createVariable(var_name, 'f', ('time',) + var_dim, zlib=self.compress)

            var.units = Unit(variable['units']).symbol
            if 'long_name' in variable:
                var.long_name = str(variable['long_name'])
            if 'standard_name' in variable:
                var.standard_name = str(variable['standard_name'])
            if 'cell_method' in variable:
                var.cell_method = str(variable['cell_method'])
            var.coordinates = "lat lon"
            if self.grid.cell_area is not None:
                var.cell_measures = 'area: cell_area'
            if RegularLatLon:
                var.grid_mapping = 'crs'
            elif Rotated:
                var.grid_mapping = 'rotated_pole'
            elif LambertConformalConic:
                var.grid_mapping = 'Lambert_conformal'
            settings.write_log("\t\t\t'{0}' variable created with size: {1}".format(var_name, var[:].shape) +
                               "\n\t\t\t\t'{0}' variable will be filled later.".format(var_name), level=3)

        settings.write_log("\t\tCreating NetCDF metadata.", level=2)
        # Grid mapping
        if RegularLatLon:
            # CRS
            mapping = netcdf.createVariable('crs', 'i')
            mapping.grid_mapping_name = "latitude_longitude"
            mapping.semi_major_axis = 6371000.0
            mapping.inverse_flattening = 0
        elif Rotated:
            # Rotated pole
            mapping = netcdf.createVariable('rotated_pole', 'c')
            mapping.grid_mapping_name = 'rotated_latitude_longitude'
            mapping.grid_north_pole_latitude = self.grid.new_pole_latitude_degrees
            mapping.grid_north_pole_longitude = 90 - self.grid.new_pole_longitude_degrees
        elif LambertConformalConic:
            # CRS
            mapping = netcdf.createVariable('Lambert_conformal', 'i')
            mapping.grid_mapping_name = "lambert_conformal_conic"
            mapping.standard_parallel = "{0}, {1}".format(self.grid.lat_1, self.grid.lat_2)
            mapping.longitude_of_central_meridian = self.grid.lon_0
            mapping.latitude_of_projection_origin = self.grid.lat_0

        # Cell area
        if self.grid.cell_area is not None:
            c_area = netcdf.createVariable('cell_area', 'f', cell_area_dim)
            c_area.long_name = "area of the grid cell"
            c_area.standard_name = "cell_area"
            c_area.units = Unit("m2").symbol
            # print c_area[:].shape, cell_area.shape
            # c_area[grid.x_lower_bound:grid.x_upper_bound, grid.y_lower_bound:grid.y_upper_bound] = cell_area

        if self.global_attributes is not None:
            netcdf.setncatts(self.global_attributes)

        netcdf.close()

        settings.write_time('WriterMonarch', 'create_parallel_netcdf', timeit.default_timer() - st_time, level=3)
        return True

    def write_serial_netcdf(self, emission_list,):
        """
        Write the netCDF4 file in serial mode.

        :param emission_list: Data to append.
        :type emission_list: list

        :return: True at end.
        :rtype: bool
        """
        from cf_units import Unit, encode_time

        st_time = timeit.default_timer()

        mpi_numpy = False
        mpi_vector = True

        # Gathering the index
        if mpi_numpy or mpi_vector:
            rank_position = np.array([self.grid.x_lower_bound, self.grid.x_upper_bound, self.grid.y_lower_bound,
                                      self.grid.y_upper_bound], dtype='i')
            full_position = None
            if settings.rank == 0:
                full_position = np.empty([settings.size, 4], dtype='i')
            settings.comm.Gather(rank_position, full_position, root=0)

        if settings.rank == 0:

            regular_latlon = False
            rotated = False
            lcc = False

            if self.grid.grid_type == 'global':
                regular_latlon = True
            elif self.grid.grid_type == 'rotated':
                rotated = True
            elif self.grid.grid_type == 'lcc':
                lcc = True
            settings.write_log("\tCreating NetCDF file.", level=2)
            netcdf = Dataset(self.path, mode='w', format="NETCDF4")

            # ===== Dimensions =====
            settings.write_log("\t\tCreating NetCDF dimensions.", level=2)
            if regular_latlon:
                var_dim = ('lat', 'lon',)

                # Latitude
                if len(self.grid.center_latitudes.shape) == 1:
                    settings.write_log("\t\t\t'lat' dimension: {0}".format(self.grid.center_latitudes.shape[0]),
                                       level=3)
                    netcdf.createDimension('lat', self.grid.center_latitudes.shape[0])
                    lat_dim = ('lat',)
                elif len(self.grid.center_latitudes.shape) == 2:
                    settings.write_log("\t\t\t'lat' dimension: {0}".format(self.grid.center_latitudes.shape[0]),
                                       level=3)
                    netcdf.createDimension('lat', self.grid.center_latitudes.shape[0])
                    lat_dim = ('lon', 'lat', )
                else:
                    settings.write_log('ERROR: Check the .err file to get more info.')
                    if settings.rank == 0:
                        raise TypeError(
                            'ERROR: Latitudes must be on a 1D or 2D array instead of {0} shape.'.format(
                                len(self.grid.center_latitudes.shape)))
                    sys.exit(1)

                # Longitude
                if len(self.grid.center_longitudes.shape) == 1:
                    settings.write_log("\t\t\t'lon' dimension: {0}".format(self.grid.center_longitudes.shape[0]),
                                       level=3)
                    netcdf.createDimension('lon', self.grid.center_longitudes.shape[0])
                    lon_dim = ('lon',)
                elif len(self.grid.center_longitudes.shape) == 2:
                    settings.write_log("\t\t\t'lon' dimension: {0}".format(self.grid.center_longitudes.shape[0]),
                                       level=3)
                    netcdf.createDimension('lon', self.grid.center_longitudes.shape[1])
                    lon_dim = ('lon', 'lat', )
                else:
                    settings.write_log('ERROR: Check the .err file to get more info.')
                    if settings.rank == 0:
                        raise TypeError(
                            'ERROR: Longitudes must be on a 1D or 2D array instead of {0} shape.'.format(
                                len(self.grid.center_longitudes.shape)))
                    sys.exit(1)
            elif rotated:
                var_dim = ('rlat', 'rlon',)

                # rotated Latitude
                if self.grid.rlat is None:
                    settings.write_log('ERROR: Check the .err file to get more info.')
                    if settings.rank == 0:
                        raise TypeError('ERROR: For rotated grids is needed the rotated latitudes.')
                    sys.exit(1)
                settings.write_log("\t\t'rlat' dimension: {0}".format(len(self.grid.rlat)), level=2)
                netcdf.createDimension('rlat', len(self.grid.rlat))
                lat_dim = ('rlat', 'rlon',)

                # rotated Longitude
                if self.grid.rlon is None:
                    settings.write_log('ERROR: Check the .err file to get more info.')
                    if settings.rank == 0:
                        raise TypeError('ERROR: For rotated grids is needed the rotated longitudes.')
                    sys.exit(1)
                settings.write_log("\t\t\t'rlon' dimension: {0}".format(len(self.grid.rlon)), level=3)
                netcdf.createDimension('rlon', len(self.grid.rlon))
                lon_dim = ('rlat', 'rlon',)

            elif lcc:
                var_dim = ('y', 'x',)
                settings.write_log("\t\t\t'y' dimension: {0}".format(len(self.grid.y)), level=3)
                netcdf.createDimension('y', len(self.grid.y))
                lat_dim = ('y', 'x', )
                settings.write_log("\t\t\t'x' dimension: {0}".format(len(self.grid.x)), level=3)
                netcdf.createDimension('x', len(self.grid.x))
                lon_dim = ('y', 'x', )
            else:
                lat_dim = None
                lon_dim = None
                var_dim = None

            # Levels
            if self.levels is not None:
                settings.write_log("\t\t\t'lev' dimension: {0}".format(len(self.levels)), level=3)
                netcdf.createDimension('lev', len(self.levels))

            # Bounds
            if self.grid.boundary_latitudes is not None:
                settings.write_log("\t\t\t'nv' dimension: {0}".format(len(self.grid.boundary_latitudes[0, 0])), level=3)
                netcdf.createDimension('nv', len(self.grid.boundary_latitudes[0, 0]))

            # Time
            settings.write_log("\t\t\t'time' dimension: {0}".format(len(self.hours)), level=3)
            netcdf.createDimension('time', len(self.hours))

            # ===== Variables =====
            settings.write_log("\t\tCreating NetCDF variables.", level=2)
            # Time
            if self.date is None:
                time = netcdf.createVariable('time', 'd', ('time',))
                time.units = "months since 2000-01-01 00:00:00"
                time.standard_name = "time"
                time.calendar = "gregorian"
                time.long_name = "time"
                time[:] = [0.]
            else:
                time = netcdf.createVariable('time', 'd', ('time',))
                time.units = str(Unit('hours').offset_by_time(encode_time(
                    self.date.year, self.date.month, self.date.day, self.date.hour, self.date.minute,
                    self.date.second)))
                time.standard_name = "time"
                time.calendar = "gregorian"
                time.long_name = "time"
                time[:] = self.hours
            settings.write_log("\t\t\t'time' variable created with size: {0}".format(time[:].shape), level=3)

            # Latitude
            lats = netcdf.createVariable('lat', 'f', lat_dim, zlib=self.compress)
            lats.units = "degrees_north"
            lats.axis = "Y"
            lats.long_name = "latitude coordinate"
            lats.standard_name = "latitude"
            lats[:] = self.grid.center_latitudes
            settings.write_log("\t\t\t'lat' variable created with size: {0}".format(lats[:].shape), level=3)

            if self.grid.boundary_latitudes is not None:
                lats.bounds = "lat_bnds"
                lat_bnds = netcdf.createVariable('lat_bnds', 'f', lat_dim + ('nv',), zlib=self.compress)
                # print lat_bnds[:].shape, boundary_latitudes.shape
                lat_bnds[:] = self.grid.boundary_latitudes
                settings.write_log(
                    "\t\t\t'lat_bnds' variable created with size: {0}".format(lat_bnds[:].shape), level=3)

            # Longitude
            lons = netcdf.createVariable('lon', 'f', lon_dim, zlib=self.compress)
            lons.units = "degrees_east"
            lons.axis = "X"
            lons.long_name = "longitude coordinate"
            lons.standard_name = "longitude"
            lons[:] = self.grid.center_longitudes
            settings.write_log("\t\t\t'lon' variable created with size: {0}".format(lons[:].shape),
                               level=3)

            if self.grid.boundary_longitudes is not None:
                lons.bounds = "lon_bnds"
                lon_bnds = netcdf.createVariable('lon_bnds', 'f', lon_dim + ('nv',), zlib=self.compress)
                # print lon_bnds[:].shape, boundary_longitudes.shape
                lon_bnds[:] = self.grid.boundary_longitudes
                settings.write_log(
                    "\t\t\t'lon_bnds' variable created with size: {0}".format(lon_bnds[:].shape), level=3)

            if rotated:
                # rotated Latitude
                rlat = netcdf.createVariable('rlat', 'f', ('rlat',), zlib=self.compress)
                rlat.long_name = "latitude in rotated pole grid"
                rlat.units = Unit("degrees").symbol
                rlat.standard_name = "grid_latitude"
                rlat[:] = self.grid.rlat
                settings.write_log("\t\t\t'rlat' variable created with size: {0}".format(rlat[:].shape), level=3)

                # rotated Longitude
                rlon = netcdf.createVariable('rlon', 'f', ('rlon',), zlib=self.compress)
                rlon.long_name = "longitude in rotated pole grid"
                rlon.units = Unit("degrees").symbol
                rlon.standard_name = "grid_longitude"
                rlon[:] = self.grid.rlon
                settings.write_log("\t\t\t'rlon' variable created with size: {0}".format(rlon[:].shape), level=3)
            if lcc:
                x_var = netcdf.createVariable('x', 'd', ('x',), zlib=self.compress)
                x_var.units = Unit("km").symbol
                x_var.long_name = "x coordinate of projection"
                x_var.standard_name = "projection_x_coordinate"
                x_var[:] = self.grid.x
                settings.write_log("\t\t\t'x' variable created with size: {0}".format(x_var[:].shape), level=3)

                y_var = netcdf.createVariable('y', 'd', ('y',), zlib=self.compress)
                y_var.units = Unit("km").symbol
                y_var.long_name = "y coordinate of projection"
                y_var.standard_name = "projection_y_coordinate"
                y_var[:] = self.grid.y
                settings.write_log("\t\t\t'y' variable created with size: {0}".format(y_var[:].shape), level=3)

            cell_area_dim = var_dim
            # Levels
            if self.levels is not None:
                var_dim = ('lev',) + var_dim
                lev = netcdf.createVariable('lev', 'f', ('lev',), zlib=self.compress)
                lev.units = Unit("m").symbol
                lev.positive = 'up'
                lev[:] = self.levels
                settings.write_log("\t\t\t'lev' variable created with size: {0}".format(lev[:].shape), level=3)

            if len(self.variables_attributes) is 0:
                var = netcdf.createVariable('aux_var', 'f', ('time',) + var_dim, zlib=self.compress)
                var[:] = 0

        full_shape = None
        index = 0
        for var_name in self.variables_attributes.iterkeys():
            if settings.size != 1:
                settings.write_log("\t\t\tGathering {0} data.".format(var_name), level=3)
            rank_data = self.calculate_data_by_var(var_name, emission_list, self.grid.shape)
            if mpi_numpy or mpi_vector:
                if rank_data is not None:
                    root_shape = settings.comm.bcast(rank_data.shape, root=0)
                    if full_shape is None:
                        full_shape = settings.comm.allgather(rank_data.shape)
                        # print 'Rank {0} full_shape: {1}\n'.format(settings.rank, full_shape)

            if mpi_numpy:
                if settings.size != 1:
                    if settings.rank == 0:
                        recvbuf = np.empty((settings.size,) + rank_data.shape)
                    else:
                        recvbuf = None
                    if root_shape != rank_data.shape:
                        rank_data_aux = np.empty(root_shape)
                        rank_data_aux[:, :, :, :-1] = rank_data
                        rank_data = rank_data_aux
                    # print 'Rank {0} data.shape {1}'.format(settings.rank, rank_data.shape)
                    settings.comm.Gather(rank_data, recvbuf, root=0)
                else:
                    recvbuf = rank_data
            elif mpi_vector:
                if rank_data is not None:
                    counts_i = self.tuple_to_index(full_shape)
                    rank_buff = [rank_data, counts_i[settings.rank]]
                    if settings.rank == 0:
                        displacements = self.calculate_displacements(counts_i)
                        recvdata = np.empty(sum(counts_i), dtype=settings.precision)
                    else:
                        displacements = None
                        recvdata = None
                    if settings.precision == np.float32:
                        recvbuf = [recvdata, counts_i, displacements, MPI.FLOAT]
                    elif settings.precision == np.float64:
                        recvbuf = [recvdata, counts_i, displacements, MPI.DOUBLE]
                    else:
                        settings.write_log('ERROR: Check the .err file to get more info.')
                        if settings.rank == 0:
                            raise TypeError('ERROR: precision {0} unknown'.format(settings.precision))
                        sys.exit(1)

                    settings.comm.Gatherv(rank_buff, recvbuf, root=0)

            else:
                if settings.size != 1:
                    data = settings.comm.gather(rank_data, root=0)
                else:
                    data = rank_data

            if settings.rank == 0:
                if not (mpi_numpy or mpi_vector):
                    if settings.size != 1:
                        try:
                            data = np.concatenate(data, axis=3)
                        except (UnboundLocalError, TypeError, IndexError):
                            data = 0
                index += 1
                var = netcdf.createVariable(var_name, 'f', ('time',) + var_dim, zlib=self.compress)

                var.units = Unit(self.variables_attributes[var_name]['units']).symbol

                if 'long_name' in self.variables_attributes[var_name]:
                    var.long_name = str(self.variables_attributes[var_name]['long_name'])

                if 'standard_name' in self.variables_attributes[var_name]:
                    var.standard_name = str(self.variables_attributes[var_name]['standard_name'])

                if 'cell_method' in self.variables_attributes[var_name]:
                    var.cell_method = str(self.variables_attributes[var_name]['cell_method'])

                var.coordinates = "lat lon"

                if self.grid.cell_area is not None:
                    var.cell_measures = 'area: cell_area'
                if regular_latlon:
                    var.grid_mapping = 'crs'
                elif rotated:
                    var.grid_mapping = 'rotated_pole'
                elif lcc:
                    var.grid_mapping = 'Lambert_conformal'

                if mpi_numpy:
                    data = np.ones(var[:].shape, dtype=settings.precision) * 100
                    for i in xrange(settings.size):
                        try:
                            if i == 0:
                                var[:, :, :, :full_position[i][3]] = recvbuf[i]
                            elif i == settings.size - 1:
                                var[:, :, :, full_position[i][2]:] = recvbuf[i, :, :, :, :-1]
                            else:
                                var[:, :, :, full_position[i][2]:full_position[i][3]] = \
                                    recvbuf[i, :, :, :, : full_shape[i][-1]]
                        except ValueError:
                            settings.write_log('ERROR: Check the .err file to get more info.')
                            if settings.rank == 0:
                                raise TypeError("ERROR on i {0} ".format(i) +
                                                "data shape: {0} ".format(data[:, :, :, full_position[i][2]:].shape) +
                                                "recvbuf shape {0}".format(recvbuf[i].shape))
                            sys.exit(1)

                elif mpi_vector:
                    if rank_data is not None:
                        data = np.empty(var[:].shape, dtype=settings.precision)
                        for i in xrange(settings.size):
                            if not i == settings.size - 1:
                                data[:, :, full_position[i][0]:full_position[i][1],
                                     full_position[i][2]:full_position[i][3]] = \
                                    np.array(recvbuf[0][displacements[i]: displacements[i + 1]]).reshape(full_shape[i])
                            else:
                                data[:, :, full_position[i][0]:full_position[i][1],
                                     full_position[i][2]:full_position[i][3]] = \
                                    np.array(recvbuf[0][displacements[i]:]).reshape(full_shape[i])
                    else:
                        data = 0
                    var[:] = data
                else:
                    var[:] = data
                settings.write_log("\t\t\t'{0}' variable created with size: {1}".format(var_name, var[:].shape),
                                   level=3)
        settings.write_log("\t\tCreating NetCDF metadata.", level=2)
        if settings.rank == 0:
            # Grid mapping
            if regular_latlon:
                # CRS
                mapping = netcdf.createVariable('crs', 'i')
                mapping.grid_mapping_name = "latitude_longitude"
                mapping.semi_major_axis = 6371000.0
                mapping.inverse_flattening = 0
            elif rotated:
                # rotated pole
                mapping = netcdf.createVariable('rotated_pole', 'c')
                mapping.grid_mapping_name = 'rotated_latitude_longitude'
                mapping.grid_north_pole_latitude = 90 - self.grid.new_pole_latitude_degrees
                mapping.grid_north_pole_longitude = self.grid.new_pole_longitude_degrees
            elif lcc:
                # CRS
                mapping = netcdf.createVariable('Lambert_conformal', 'i')
                mapping.grid_mapping_name = "lambert_conformal_conic"
                mapping.standard_parallel = "{0}, {1}".format(self.grid.lat_1, self.grid.lat_2)
                mapping.longitude_of_central_meridian = self.grid.lon_0
                mapping.latitude_of_projection_origin = self.grid.lat_0

        if self.grid.cell_area is not None:
            cell_area = settings.comm.gather(self.grid.cell_area, root=0)
            if settings.rank == 0:
                # Cell area
                if self.grid.cell_area is not None:
                    c_area = netcdf.createVariable('cell_area', 'f', cell_area_dim)
                    c_area.long_name = "area of the grid cell"
                    c_area.standard_name = "cell_area"
                    c_area.units = Unit("m2").symbol

                    cell_area = np.concatenate(cell_area, axis=1)

                    c_area[:] = cell_area

        if settings.rank == 0:
            if self.global_attributes is not None:
                netcdf.setncatts(self.global_attributes)
        if settings.rank == 0:
            netcdf.close()
        settings.write_time('WriterMonarch', 'write_serial_netcdf', timeit.default_timer() - st_time, level=3)
