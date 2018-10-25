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
from mpi4py import MPI
from netCDF4 import Dataset
from hermesv3_gr.config import settings


class Writer(object):
    """
    Class to Write the output file.

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

        self.path = path
        self.grid = grid
        self.compress = compress
        self.parallel = parallel

        self.variables_attributes = None
        self.levels = levels
        self.date = date
        self.hours = hours

        self.global_attributes = None

        self.global_attributes_path = global_attributes_path

    def write(self, inventory_list):
        """
        Write the netCDF4 file with the pollutants of the given list of inventories.

        :param inventory_list: List of inventories.
        :type inventory_list: list

        :return: True at end
        :rtype: bool
        """
        st_time = timeit.default_timer()
        settings.write_log('')
        settings.write_log("Writing netCDF output file {0} .".format(self.path))

        self.set_variable_attributes(inventory_list)
        self.change_variable_attributes()
        if self.parallel:
            if settings.rank == 0:
                self.create_parallel_netcdf()
            settings.comm.Barrier()
            self.write_parallel_netcdf(inventory_list)
        else:
            self.write_serial_netcdf(inventory_list)

        settings.write_time('Writer', 'write', timeit.default_timer() - st_time)
        return True

    def change_variable_attributes(self):
        pass

    def create_parallel_netcdf(self):
        """
        Implemented on inner class.
        """
        return None

    def write_parallel_netcdf(self, emission_list):
        """
        Append the data to the netCDF4 file already created in parallel mode.

        :param emission_list: Data to append.
        :type emission_list: list

        :return: True at end.
        :rtype: bool
        """

        st_time = timeit.default_timer()

        settings.write_log("\tAppending data to parallel NetCDF file.", level=2)
        if settings.size > 1:
            netcdf = Dataset(self.path, mode='a', format="NETCDF4", parallel=True, comm=settings.comm, info=MPI.Info())
        else:
            netcdf = Dataset(self.path, mode='a', format="NETCDF4")
        settings.write_log("\t\tParallel NetCDF file ready to write.", level=2)
        index = 0
        # print "Rank {0} 2".format(rank)
        for var_name in self.variables_attributes.iterkeys():

            data = self.calculate_data_by_var(var_name, emission_list, self.grid.shape)
            st_time = timeit.default_timer()
            index += 1

            var = netcdf.variables[var_name]
            if settings.size > 1:
                var.set_collective(True)
            # Correcting NAN
            if data is None:
                data = 0
            var[:, :, self.grid.x_lower_bound:self.grid.x_upper_bound,
                self.grid.y_lower_bound:self.grid.y_upper_bound] = data

            settings.write_log("\t\t\t'{0}' variable filled".format(var_name))

        if 'cell_area' in netcdf.variables:
            c_area = netcdf.variables['cell_area']
            c_area[self.grid.x_lower_bound:self.grid.x_upper_bound,
                   self.grid.y_lower_bound:self.grid.y_upper_bound] = self.grid.cell_area

        netcdf.close()
        settings.write_time('Writer', 'write_parallel_netcdf', timeit.default_timer() - st_time, level=3)
        return True

    def write_serial_netcdf(self, emission_list):
        """
        Implemented on inner class.
        """
        return None

    def set_variable_attributes(self, inventory_list):
        """
        Change the variables_attribute parameter of the Writer class.

        :param inventory_list: list of invenotries.
        :type inventory_list: list

        :return: True at end.
        :rtype: bool
        """
        st_time = timeit.default_timer()
        empty_dict = {}
        for inventory in inventory_list:
            for emi in inventory.emissions:
                if not emi['name'] in empty_dict:
                    dict_aux = emi.copy()
                    dict_aux['data'] = None
                    empty_dict[emi['name']] = dict_aux

        self.variables_attributes = empty_dict.values()

        settings.write_time('Writer', 'set_variable_attributes', timeit.default_timer() - st_time, level=3)

        return True

    def calculate_data_by_var(self, variable, inventory_list, shape):
        """
        Calculate the date of the given variable throw the inventory list.

        :param variable: Variable to calculate.
        :type variable: str

        :param inventory_list: Inventory list
        :type inventory_list: list

        :param shape: Output desired shape.
        :type shape: tuple

        :return: Data of the given variable.
        :rtype: numpy.array
        """
        st_time = timeit.default_timer()
        settings.write_log("\t\t\t\tGetting data for '{0}' pollutant.".format(variable), level=3)

        data = None

        for ei in inventory_list:
            for emission in ei.emissions:
                if emission['name'] == variable:
                    if emission['data'] is not 0:
                        vertical_time = timeit.default_timer()
                        if ei.source_type == 'area':
                            if ei.vertical_factors is not None:
                                aux_data = emission['data'][np.newaxis, :, :] * ei.vertical_factors[:, np.newaxis,
                                                                                                    np.newaxis]
                            else:
                                if len(emission['data'].shape) != 3:
                                    aux_data = np.zeros((shape[1], shape[2], shape[3]))
                                    aux_data[0, :, :] = emission['data']
                                else:
                                    aux_data = emission['data']
                        elif ei.source_type == 'point':
                            aux_data = np.zeros((shape[1], shape[2] * shape[3]))
                            aux_data[ei.location['layer'], ei.location['FID']] = emission['data']
                            aux_data = aux_data.reshape((shape[1], shape[2], shape[3]))
                        else:
                            aux_data = None

                        settings.write_time('VerticalDistribution', 'calculate_data_by_var',
                                            timeit.default_timer() - vertical_time, level=2)
                        del emission['data']

                        temporal_time = timeit.default_timer()
                        if data is None:
                            data = np.zeros(shape)
                        if ei.temporal_factors is not None:
                            data += aux_data[np.newaxis, :, :, :] * ei.temporal_factors[:, np.newaxis, :, :]
                        else:
                            data += aux_data[np.newaxis, :, :, :]
                        settings.write_time('TemporalDistribution', 'calculate_data_by_var',
                                            timeit.default_timer() - temporal_time, level=2)
        # Unit changes
        data = self.unit_change(variable, data)
        if data is not None:
            data[data < 0] = 0
        settings.write_time('Writer', 'calculate_data_by_var', timeit.default_timer() - st_time, level=3)
        return data

    def unit_change(self, variable, data):
        """
        Implement on inner class
        """
        return np.array([0])

    @staticmethod
    def calculate_displacements(counts):
        """
        Calculate the index position of all the ranks.

        :param counts: Number of elements for rank
        :type counts: list

        :return: Displacements
        :rtype: list
        """
        st_time = timeit.default_timer()

        new_list = [0]
        accum = 0
        for counter in counts[:-1]:
            accum += counter
            new_list.append(accum)

        settings.write_time('Writer', 'calculate_displacements', timeit.default_timer() - st_time, level=3)
        return new_list

    @staticmethod
    def tuple_to_index(tuple_list, bidimensional=False):
        """
        Get the index for a list of shapes.

        :param tuple_list: List os shapes.
        :type tuple_list: list

        :param bidimensional: Indicates if the tuple is bidimensional.
        :type bidimensional: bool

        :return: List of index
        :rtype: list
        """
        from operator import mul
        st_time = timeit.default_timer()

        new_list = []
        for my_tuple in tuple_list:
            if bidimensional:
                new_list.append(my_tuple[-1] * my_tuple[-2])
            else:
                new_list.append(reduce(mul, my_tuple))
        settings.write_time('Writer', 'tuple_to_index', timeit.default_timer() - st_time, level=3)
        return new_list

    @staticmethod
    def get_writer(output_model, path, grid, levels, date, hours, global_attributes_path, compress, parallel):
        """
        Choose between the different writers depending on the desired output model.

        :param output_model: Name of the output model. Only accepted 'MONARCH, CMAQ or WRF_CHEM.
        :type output_model: str

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

        :return: Writing object of the desired output model.
        :rtype: Writer
        """
        from hermesv3_gr.modules.writing.writer_cmaq import WriterCmaq
        from hermesv3_gr.modules.writing.writer_monarch import WriterMonarch
        from hermesv3_gr.modules.writing.writer_wrf_chem import WriterWrfChem

        settings.write_log('Selecting writing output type for {0}.'.format(output_model))
        if output_model.lower() == 'monarch':
            return WriterMonarch(path, grid, levels, date, hours, global_attributes_path, compress, parallel)
        elif output_model.lower() == 'cmaq':
            return WriterCmaq(path, grid, levels, date, hours, global_attributes_path, compress, parallel)
        elif output_model.lower() == 'wrf_chem':
            return WriterWrfChem(path, grid, levels, date, hours, global_attributes_path, compress, parallel)
        else:
            settings.write_log('ERROR: Check the .err file to get more info.')
            if settings.rank == 0:
                raise AttributeError("The desired '{0}' output model is not available. ".format(output_model) +
                                     "Only accepted 'MONARCH, CMAQ or WRF_CHEM.")
            sys.exit(1)

    @staticmethod
    def write_netcdf(netcdf_path, center_latitudes, center_longitudes, data_list,
                     levels=None, date=None, hours=None,
                     boundary_latitudes=None, boundary_longitudes=None, cell_area=None, global_attributes=None,
                     regular_latlon=False,
                     roated=False, rotated_lats=None, rotated_lons=None, north_pole_lat=None, north_pole_lon=None,
                     lcc=False, lcc_x=None, lcc_y=None, lat_1_2=None, lon_0=None, lat_0=None,
                     mercator=False, lat_ts=None):
        # TODO Deprecate
        """
        Will be deprecated
        """
        from netCDF4 import Dataset
        from cf_units import Unit, encode_time

        if not (regular_latlon or lcc or roated or mercator):
            regular_latlon = True
        netcdf = Dataset(netcdf_path, mode='w', format="NETCDF4")

        # ===== Dimensions =====
        if regular_latlon:
            var_dim = ('lat', 'lon',)

            # Latitude
            if len(center_latitudes.shape) == 1:
                netcdf.createDimension('lat', center_latitudes.shape[0])
                lat_dim = ('lat',)
            elif len(center_latitudes.shape) == 2:
                netcdf.createDimension('lat', center_latitudes.shape[0])
                lat_dim = ('lon', 'lat',)
            else:
                print 'ERROR: Latitudes must be on a 1D or 2D array instead of {0}'.format(len(center_latitudes.shape))
                sys.exit(1)

            # Longitude
            if len(center_longitudes.shape) == 1:
                netcdf.createDimension('lon', center_longitudes.shape[0])
                lon_dim = ('lon',)
            elif len(center_longitudes.shape) == 2:
                netcdf.createDimension('lon', center_longitudes.shape[1])
                lon_dim = ('lon', 'lat',)
            else:
                print 'ERROR: Longitudes must be on a 1D or 2D array instead of {0}'.format(
                    len(center_longitudes.shape))
                sys.exit(1)
        elif roated:
            var_dim = ('rlat', 'rlon',)

            # Rotated Latitude
            if rotated_lats is None:
                print 'ERROR: For rotated grids is needed the rotated latitudes.'
                sys.exit(1)
            netcdf.createDimension('rlat', len(rotated_lats))
            lat_dim = ('rlat', 'rlon',)

            # Rotated Longitude
            if rotated_lons is None:
                print 'ERROR: For rotated grids is needed the rotated longitudes.'
                sys.exit(1)
            netcdf.createDimension('rlon', len(rotated_lons))
            lon_dim = ('rlat', 'rlon',)
        elif lcc or mercator:
            var_dim = ('y', 'x',)

            netcdf.createDimension('y', len(lcc_y))
            lat_dim = ('y', 'x',)

            netcdf.createDimension('x', len(lcc_x))
            lon_dim = ('y', 'x',)
        else:
            lat_dim = None
            lon_dim = None
            var_dim = None

        # Levels
        if levels is not None:
            netcdf.createDimension('lev', len(levels))

        # Bounds
        if boundary_latitudes is not None:
            # print boundary_latitudes.shape
            # print len(boundary_latitudes[0, 0])
            try:
                netcdf.createDimension('nv', len(boundary_latitudes[0, 0]))
            except TypeError:
                netcdf.createDimension('nv', boundary_latitudes.shape[1])

            # sys.exit()

        # Time
        netcdf.createDimension('time', None)

        # ===== Variables =====
        # Time
        if date is None:
            time = netcdf.createVariable('time', 'd', ('time',), zlib=True)
            time.units = "months since 2000-01-01 00:00:00"
            time.standard_name = "time"
            time.calendar = "gregorian"
            time.long_name = "time"
            time[:] = [0.]
        else:
            time = netcdf.createVariable('time', 'd', ('time',), zlib=True)
            # print u.offset_by_time(encode_time(date.year, date.month, date.day, date.hour, date.minute, date.second))
            # Unit('hour since 1970-01-01 00:00:00.0000000 UTC')
            time.units = str(Unit('hours').offset_by_time(
                encode_time(date.year, date.month, date.day, date.hour, date.minute, date.second)))
            time.standard_name = "time"
            time.calendar = "gregorian"
            time.long_name = "time"
            time[:] = hours

        # Latitude
        lats = netcdf.createVariable('lat', 'f', lat_dim, zlib=True)
        lats.units = "degrees_north"
        lats.axis = "Y"
        lats.long_name = "latitude coordinate"
        lats.standard_name = "latitude"
        lats[:] = center_latitudes

        if boundary_latitudes is not None:
            lats.bounds = "lat_bnds"
            lat_bnds = netcdf.createVariable('lat_bnds', 'f', lat_dim + ('nv',), zlib=True)
            # print lat_bnds[:].shape, boundary_latitudes.shape
            lat_bnds[:] = boundary_latitudes

        # Longitude
        lons = netcdf.createVariable('lon', 'f', lon_dim, zlib=True)

        lons.units = "degrees_east"
        lons.axis = "X"
        lons.long_name = "longitude coordinate"
        lons.standard_name = "longitude"
        # print 'lons:', lons[:].shape, center_longitudes.shape
        lons[:] = center_longitudes
        if boundary_longitudes is not None:
            lons.bounds = "lon_bnds"
            lon_bnds = netcdf.createVariable('lon_bnds', 'f', lon_dim + ('nv',), zlib=True)
            # print lon_bnds[:].shape, boundary_longitudes.shape
            lon_bnds[:] = boundary_longitudes

        if roated:
            # Rotated Latitude
            rlat = netcdf.createVariable('rlat', 'f', ('rlat',), zlib=True)
            rlat.long_name = "latitude in rotated pole grid"
            rlat.units = Unit("degrees").symbol
            rlat.standard_name = "grid_latitude"
            rlat[:] = rotated_lats

            # Rotated Longitude
            rlon = netcdf.createVariable('rlon', 'f', ('rlon',), zlib=True)
            rlon.long_name = "longitude in rotated pole grid"
            rlon.units = Unit("degrees").symbol
            rlon.standard_name = "grid_longitude"
            rlon[:] = rotated_lons
        if lcc or mercator:
            x_var = netcdf.createVariable('x', 'd', ('x',), zlib=True)
            x_var.units = Unit("km").symbol
            x_var.long_name = "x coordinate of projection"
            x_var.standard_name = "projection_x_coordinate"
            x_var[:] = lcc_x

            y_var = netcdf.createVariable('y', 'd', ('y',), zlib=True)
            y_var.units = Unit("km").symbol
            y_var.long_name = "y coordinate of projection"
            y_var.standard_name = "projection_y_coordinate"
            y_var[:] = lcc_y

        cell_area_dim = var_dim
        # Levels
        if levels is not None:
            var_dim = ('lev',) + var_dim
            lev = netcdf.createVariable('lev', 'f', ('lev',), zlib=True)
            lev.units = Unit("m").symbol
            lev.positive = 'up'
            lev[:] = levels

        # All variables
        if len(data_list) is 0:
            var = netcdf.createVariable('aux_var', 'f', ('time',) + var_dim, zlib=True)
            var[:] = 0
        for variable in data_list:
            # print ('time',) + var_dim
            var = netcdf.createVariable(variable['name'], 'f', ('time',) + var_dim, zlib=True)
            var.units = Unit(variable['units']).symbol
            if 'long_name' in variable:
                var.long_name = str(variable['long_name'])
            if 'standard_name' in variable:
                var.standard_name = str(variable['standard_name'])
            if 'cell_method' in variable:
                var.cell_method = str(variable['cell_method'])
            var.coordinates = "lat lon"
            if cell_area is not None:
                var.cell_measures = 'area: cell_area'
            if regular_latlon:
                var.grid_mapping = 'crs'
            elif roated:
                var.grid_mapping = 'rotated_pole'
            elif lcc:
                var.grid_mapping = 'Lambert_conformal'
            elif mercator:
                var.grid_mapping = 'mercator'
            try:
                var[:] = variable['data']
            except ValueError:
                print 'VAR ERROR, netcdf shape: {0}, variable shape: {1}'.format(var[:].shape, variable['data'].shape)

        # Grid mapping
        if regular_latlon:
            # CRS
            mapping = netcdf.createVariable('crs', 'i')
            mapping.grid_mapping_name = "latitude_longitude"
            mapping.semi_major_axis = 6371000.0
            mapping.inverse_flattening = 0
        elif roated:
            # Rotated pole
            mapping = netcdf.createVariable('rotated_pole', 'c')
            mapping.grid_mapping_name = 'rotated_latitude_longitude'
            mapping.grid_north_pole_latitude = north_pole_lat
            mapping.grid_north_pole_longitude = north_pole_lon
        elif lcc:
            # CRS
            mapping = netcdf.createVariable('Lambert_conformal', 'i')
            mapping.grid_mapping_name = "lambert_conformal_conic"
            mapping.standard_parallel = lat_1_2
            mapping.longitude_of_central_meridian = lon_0
            mapping.latitude_of_projection_origin = lat_0
        elif mercator:
            # Mercator
            mapping = netcdf.createVariable('mercator', 'i')
            mapping.grid_mapping_name = "mercator"
            mapping.longitude_of_projection_origin = lon_0
            mapping.standard_parallel = lat_ts

        # Cell area
        if cell_area is not None:
            c_area = netcdf.createVariable('cell_area', 'f', cell_area_dim)
            c_area.long_name = "area of the grid cell"
            c_area.standard_name = "cell_area"
            c_area.units = Unit("m2").symbol
            # print c_area[:].shape, cell_area.shape
            c_area[:] = cell_area

        if global_attributes is not None:
            netcdf.setncatts(global_attributes)

        netcdf.close()
