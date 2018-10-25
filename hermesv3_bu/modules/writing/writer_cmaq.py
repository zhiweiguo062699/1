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


class WriterCmaq(Writer):
    """
   Class to Write the output file for CMAQ Chemical Transport Model CCTM.

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
        super(WriterCmaq, self).__init__(path, grid, levels, date, hours, global_attributes_path, compress, parallel)

        self.global_attributes_order = [
            'IOAPI_VERSION', 'EXEC_ID', 'FTYPE', 'CDATE', 'CTIME', 'WDATE', 'WTIME', 'SDATE', 'STIME', 'TSTEP', 'NTHIK',
            'NCOLS', 'NROWS', 'NLAYS', 'NVARS', 'GDTYP', 'P_ALP', 'P_BET', 'P_GAM', 'XCENT', 'YCENT', 'XORIG', 'YORIG',
            'XCELL', 'YCELL', 'VGTYP', 'VGTOP', 'VGLVLS', 'GDNAM', 'UPNAM', 'FILEDESC', 'HISTORY', 'VAR-LIST']

    def unit_change(self, variable, data):
        # TODO Documentation
        """

        :param variable:
        :param data:
        :return:
        """
        from cf_units import Unit

        if data is not None:
            units = None
            for var_name in self.variables_attributes:
                if var_name == variable:
                    units = self.variables_attributes[var_name]['units']
                    break

            if Unit(units).symbol == Unit('mol.s-1').symbol:
                data = data * 1000 * self.grid.cell_area
            elif Unit(units).symbol == Unit('g.s-1').symbol:
                data = data * 1000 * self.grid.cell_area
            else:
                settings.write_log('ERROR: Check the .err file to get more info.')
                if settings.rank == 0:
                    raise TypeError("The unit '{0}' of specie {1} is not defined correctly. ".format(units, variable) +
                                    "Should be 'mol.s-1.m-2' or 'kg.s-1.m-2'")
                sys.exit(1)
        return data

    def change_variable_attributes(self):
        """
        Modify the emission list to be consistent to use the output as input for CMAQ model.

        :return: Emission list ready for CMAQ
        :rtype: dict
        """
        from cf_units import Unit

        new_variable_dict = {}
        for variable in self.variables_attributes:
            if Unit(variable['units']).symbol == Unit('mol.s-1').symbol:
                new_variable_dict[variable['name']] = {
                    'units': "{:<16}".format('mole/s'),
                    'var_desc': "{:<80}".format(variable['long_name']),
                    'long_name': "{:<16}".format(variable['name']),
                }
            elif Unit(variable['units']).symbol == Unit('g.s-1').symbol:
                new_variable_dict[variable['name']] = {
                    'units': "{:<16}".format('g/s'),
                    'var_desc': "{:<80}".format(variable['long_name']),
                    'long_name': "{:<16}".format(variable['name']),
                }
            else:
                settings.write_log('ERROR: Check the .err file to get more info.')
                if settings.rank == 0:
                    raise TypeError("The unit '{0}' of specie {1} is not ".format(variable['units'], variable['name']) +
                                    "defined correctly. Should be 'mol.s-1' or 'g.s-1'")
                sys.exit(1)

        self.variables_attributes = new_variable_dict

    @staticmethod
    def create_tflag(st_date, hours_array, num_vars):
        """
        Create the content of the CMAQ variable TFLAG

        :param st_date: Starting date
        :type st_date: datetime.datetime

        :param hours_array: Array with as elements as time steps. Each element has the delta hours from the starting
                date.
        :type hours_array: numpy.array

        :param num_vars: Number of variables that will contain the NetCDF.
        :type num_vars: int

        :return: Array with the content of TFLAG
        :rtype: numpy.array
        """
        from datetime import timedelta

        a = np.array([[[]]])

        for inc_hours in hours_array:
            date = st_date + timedelta(hours=inc_hours)
            b = np.array([[int(date.strftime('%Y%j'))], [int(date.strftime('%H%M%S'))]] * num_vars)
            a = np.append(a, b)

        a.shape = (len(hours_array), 2, num_vars)
        return a

    @staticmethod
    def str_var_list(var_list):
        """
        Transform a list to a string with the elements with 16 white spaces.

        :param var_list: List of variables.
        :type var_list: list

        :return: List transformed on string.
        :rtype: str
        """
        str_var_list = ""
        for var in var_list:
            str_var_list += "{:<16}".format(var)

        return str_var_list

    def read_global_attributes(self):
        # TODO Documentation
        """

        :return:
        """
        import pandas as pd
        from warnings import warn as warning
        float_atts = ['VGTOP']
        int_atts = ['FTYPE', 'NTHIK', 'VGTYP']
        str_atts = ['EXEC_ID', 'GDNAM']
        list_float_atts = ['VGLVLS']

        atts_dict = {
            'EXEC_ID': "{:<80}".format('0.1alpha'),
            'FTYPE': np.int32(1),
            'NTHIK': np.int32(1),
            'VGTYP': np.int32(7),
            'VGTOP': np.float32(5000.),
            'VGLVLS': np.array([1., 0.], dtype=np.float32),
            'GDNAM': "{:<16}".format(''),
        }

        if self.global_attributes_path is not None:
            df = pd.read_csv(self.global_attributes_path)

            for att in atts_dict.iterkeys():
                try:
                    if att in int_atts:
                        atts_dict[att] = np.int32(df.loc[df['attribute'] == att, 'value'].item())
                    elif att in float_atts:
                        atts_dict[att] = np.float32(df.loc[df['attribute'] == att, 'value'].item())
                    elif att in str_atts:
                        atts_dict[att] = str(df.loc[df['attribute'] == att, 'value'].item())
                    elif att in list_float_atts:
                        atts_dict[att] = np.array(df.loc[df['attribute'] == att, 'value'].item().split(),
                                                  dtype=np.float32)
                except ValueError:
                    settings.write_log('WARNING: The global attribute {0} is not defined;'.format(att) +
                                       ' Using default value {0}'.format(atts_dict[att]))
                    if settings.rank == 0:
                        warning('WARNING: The global attribute {0} is not defined; Using default value {1}'.format(
                            att, atts_dict[att]))

        else:
            settings.write_log('WARNING: Check the .err file to get more information.')
            message = 'WARNING: No output attributes defined, check the output_attributes'
            message += ' parameter of the configuration file.\nUsing default values:'
            for key, value in atts_dict.iteritems():
                message += '\n\t{0} = {1}'.format(key, value)
            if settings.rank == 0:
                warning(message)

        return atts_dict

    def create_global_attributes(self, var_list):
        """
        Create the global attributes and the order that they have to be filled.

        :param var_list: List of variables
        :type var_list: list

        :return: Dict of global attributes and a list with the keys ordered.
        :rtype: tuple
        """
        from datetime import datetime

        global_attributes = self.read_global_attributes()

        if len(self.hours) > 1:
            tstep = (self.hours[1] - self.hours[0]) * 10000
        else:
            tstep = 1 * 10000

        now = datetime.now()
        global_attributes['IOAPI_VERSION'] = 'None: made only with NetCDF libraries'
        global_attributes['CDATE'] = np.int32(now.strftime('%Y%j'))
        global_attributes['CTIME'] = np.int32(now.strftime('%H%M%S'))
        global_attributes['WDATE'] = np.int32(now.strftime('%Y%j'))
        global_attributes['WTIME'] = np.int32(now.strftime('%H%M%S'))
        global_attributes['SDATE'] = np.int32(self.date.strftime('%Y%j'))
        global_attributes['STIME'] = np.int32(self.date.strftime('%H%M%S'))
        global_attributes['TSTEP'] = np.int32(tstep)
        global_attributes['NLAYS'] = np.int32(len(self.levels))
        global_attributes['NVARS'] = np.int32(len(var_list))
        global_attributes['UPNAM'] = "{:<16}".format('HERMESv3')
        global_attributes['FILEDESC'] = 'Emissions generated by HERMESv3_GR.'
        global_attributes['HISTORY'] = \
            'Code developed by Barcelona Supercomputing Center (BSC, https://www.bsc.es/).' + \
            'Developer: Carles Tena Medina (carles.tena@bsc.es)' + \
            'Reference: Guevara et al., 2018, GMD., in preparation.'
        global_attributes['VAR-LIST'] = self.str_var_list(var_list)

        if self.grid.grid_type == 'lcc':
            global_attributes['GDTYP'] = np.int32(2)
            global_attributes['NCOLS'] = np.int32(self.grid.nx)
            global_attributes['NROWS'] = np.int32(self.grid.ny)
            global_attributes['P_ALP'] = np.float(self.grid.lat_1)
            global_attributes['P_BET'] = np.float(self.grid.lat_2)
            global_attributes['P_GAM'] = np.float(self.grid.lon_0)
            global_attributes['XCENT'] = np.float(self.grid.lon_0)
            global_attributes['YCENT'] = np.float(self.grid.lat_0)
            global_attributes['XORIG'] = np.float(self.grid.x_0) - np.float(self.grid.inc_x) / 2
            global_attributes['YORIG'] = np.float(self.grid.y_0) - np.float(self.grid.inc_y) / 2
            global_attributes['XCELL'] = np.float(self.grid.inc_x)
            global_attributes['YCELL'] = np.float(self.grid.inc_y)

        return global_attributes

    @staticmethod
    def create_cmaq_netcdf(netcdf_path, center_latitudes, center_longitudes, data_list, levels=None, date=None,
                           hours=None, regular_lat_lon=False, rotated=False, nx=None, ny=None, lat_1=None, lat_2=None,
                           lon_0=None, lat_0=None, x_0=None, y_0=None, inc_x=None, inc_y=None):
        # TODO Documentation
        """

        :param netcdf_path:
        :param center_latitudes:
        :param center_longitudes:
        :param data_list:
        :param levels:
        :param date:
        :param hours:
        :param regular_lat_lon:
        :param rotated:
        :param nx:
        :param ny:
        :param lat_1:
        :param lat_2:
        :param lon_0:
        :param lat_0:
        :param x_0:
        :param y_0:
        :param inc_x:
        :param inc_y:
        :return:
        """

        data_list, var_list = WriterCmaq.change_variable_attributes(data_list)

        if settings.writing_serial:
            WriterCmaq.write_serial_netcdf(
                netcdf_path, center_latitudes, center_longitudes, data_list,
                levels=levels, date=date, hours=hours,
                global_attributes=WriterCmaq.create_global_attributes(date, nx, ny, len(levels), lat_1, lat_2, lon_0,
                                                                      lat_0, x_0, y_0, inc_x, inc_y, var_list),
                regular_lat_lon=regular_lat_lon,
                rotated=rotated, )
        else:
            WriterCmaq.write_parallel_netcdf(
                netcdf_path, center_latitudes, center_longitudes, data_list,
                levels=levels, date=date, hours=hours,
                global_attributes=WriterCmaq.create_global_attributes(date, nx, ny, len(levels), lat_1, lat_2, lon_0,
                                                                      lat_0, x_0, y_0, inc_x, inc_y, var_list),
                regular_lat_lon=regular_lat_lon,
                rotated=rotated, )

    @staticmethod
    def write_netcdf(netcdf_path, center_latitudes, center_longitudes, data_list, levels=None, date=None, hours=None,
                     global_attributes=None, regular_lat_lon=False, rotated=False):
        # TODO Documentation
        """

        :param netcdf_path:
        :param center_latitudes:
        :param center_longitudes:
        :param data_list:
        :param levels:
        :param date:
        :param hours:
        :param global_attributes:
        :param regular_lat_lon:
        :param rotated:
        :return:
        """
        if regular_lat_lon:
            settings.write_log('ERROR: Check the .err file to get more info.')
            if settings.rank == 0:
                raise TypeError('ERROR: Regular Lat Lon grid not implemented for CMAQ')
            sys.exit(1)

        elif rotated:
            settings.write_log('ERROR: Check the .err file to get more info.')
            if settings.rank == 0:
                raise TypeError('ERROR: Rotated grid not implemented for CMAQ')
            sys.exit(1)

        netcdf = Dataset(netcdf_path, mode='w', format="NETCDF4")

        # ===== Dimensions =====
        netcdf.createDimension('TSTEP', len(hours))
        netcdf.createDimension('DATE-TIME', 2)
        netcdf.createDimension('LAY', len(levels))
        netcdf.createDimension('VAR', len(data_list))
        netcdf.createDimension('ROW', center_latitudes.shape[0])
        netcdf.createDimension('COL', center_longitudes.shape[1])

        # ===== Variables =====
        tflag = netcdf.createVariable('TFLAG', 'i', ('TSTEP', 'VAR', 'DATE-TIME',))
        tflag.setncatts({'units': "{:<16}".format('<YYYYDDD,HHMMSS>'), 'long_name': "{:<16}".format('TFLAG'),
                         'var_desc': "{:<80}".format('Timestep-valid flags:  (1) YYYYDDD or (2) HHMMSS')})
        tflag[:] = WriterCmaq.create_tflag(date, hours, len(data_list))

        # Rest of variables
        for variable in data_list:
            var = netcdf.createVariable(variable['name'], 'f', ('TSTEP', 'LAY', 'ROW', 'COL',), zlib=True)
            var.units = variable['units']
            var.long_name = str(variable['long_name'])
            var.var_desc = str(variable['var_desc'])
            var[:] = variable['data']

        # ===== Global attributes =====
        global_attributes, order = global_attributes
        for attribute in order:
            netcdf.setncattr(attribute, global_attributes[attribute])

        netcdf.close()

    def create_parallel_netcdf(self):
        # TODO Documentation
        """
        Create an empty netCDF
        """
        st_time = timeit.default_timer()
        settings.write_log("\tCreating parallel NetCDF file.", level=2)
        # netcdf = Dataset(netcdf_path, mode='w', format="NETCDF4", parallel=True, comm=settings.comm, info=MPI.Info())
        netcdf = Dataset(self.path, mode='w', format="NETCDF4")

        # ===== Dimensions =====
        settings.write_log("\t\tCreating NetCDF dimensions.", level=2)
        netcdf.createDimension('TSTEP', len(self.hours))
        # netcdf.createDimension('TSTEP', None)
        settings.write_log("\t\t\t'TSTEP' dimension: {0}".format('UNLIMITED ({0})'.format(len(self.hours))), level=3)

        netcdf.createDimension('DATE-TIME', 2)
        settings.write_log("\t\t\t'DATE-TIME' dimension: {0}".format(2), level=3)

        netcdf.createDimension('LAY', len(self.levels))
        settings.write_log("\t\t\t'LAY' dimension: {0}".format(len(self.levels)), level=3)

        netcdf.createDimension('VAR', len(self.variables_attributes))
        settings.write_log("\t\t\t'VAR' dimension: {0}".format(len(self.variables_attributes)), level=3)

        netcdf.createDimension('ROW', self.grid.center_latitudes.shape[0])
        settings.write_log("\t\t\t'ROW' dimension: {0}".format(self.grid.center_latitudes.shape[0]), level=3)

        netcdf.createDimension('COL', self.grid.center_longitudes.shape[1])
        settings.write_log("\t\t\t'COL' dimension: {0}".format(self.grid.center_longitudes.shape[1]), level=3)

        # ===== Variables =====
        settings.write_log("\t\tCreating NetCDF variables.", level=2)
        tflag = netcdf.createVariable('TFLAG', 'i', ('TSTEP', 'VAR', 'DATE-TIME',))
        tflag.setncatts({'units': "{:<16}".format('<YYYYDDD,HHMMSS>'), 'long_name': "{:<16}".format('TFLAG'),
                         'var_desc': "{:<80}".format('Timestep-valid flags:  (1) YYYYDDD or (2) HHMMSS')})
        tflag[:] = self.create_tflag(self.date, self.hours, len(self.variables_attributes))
        settings.write_log("\t\t\t'TFLAG' variable created with size: {0}".format(tflag[:].shape), level=3)

        index = 0
        # data_list, var_list = self.change_variable_attributes(self.variables_attributes)
        for var_name in self.variables_attributes.iterkeys():
            index += 1
            var = netcdf.createVariable(var_name, 'f', ('TSTEP', 'LAY', 'ROW', 'COL',), zlib=self.compress)
            var.setncatts(self.variables_attributes[var_name])
            settings.write_log("\t\t\t'{0}' variable created with size: {1}".format(var_name, var[:].shape) +
                               "\n\t\t\t\t'{0}' variable will be filled later.".format(var_name), level=3)

        # ===== Global attributes =====
        settings.write_log("\t\tCreating NetCDF metadata.", level=2)
        global_attributes = self.create_global_attributes(self.variables_attributes.keys())
        for attribute in self.global_attributes_order:
            netcdf.setncattr(attribute, global_attributes[attribute])

        netcdf.close()

        settings.write_time('WriterCmaq', 'create_parallel_netcdf', timeit.default_timer() - st_time, level=3)

        return True

    def write_serial_netcdf(self, emission_list):
        """
        Write the netCDF in serial mode.

        :param emission_list: List of the processed emissions for the different emission inventories
        :type emission_list: list

        :return: True when it finish well.
        :rtype: bool
        """
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
            netcdf = Dataset(self.path, mode='w', format="NETCDF4")

            # ===== Dimensions =====
            settings.write_log("\tCreating NetCDF file.", level=2)
            settings.write_log("\t\tCreating NetCDF dimensions.", level=2)
            netcdf.createDimension('TSTEP', len(self.hours))
            settings.write_log("\t\t\t'TSTEP' dimension: {0}".format(len(self.hours)), level=3)
            netcdf.createDimension('DATE-TIME', 2)
            settings.write_log("\t\t\t'DATE-TIME' dimension: {0}".format(2), level=3)
            netcdf.createDimension('LAY', len(self.levels))
            settings.write_log("\t\t\t'LAY' dimension: {0}".format(len(self.levels)), level=3)
            netcdf.createDimension('VAR', len(self.variables_attributes))
            settings.write_log("\t\t\t'VAR' dimension: {0}".format(len(self.variables_attributes)), level=3)
            netcdf.createDimension('ROW', self.grid.center_latitudes.shape[0])
            settings.write_log("\t\t\t'ROW' dimension: {0}".format(self.grid.center_latitudes.shape[0]), level=3)
            netcdf.createDimension('COL', self.grid.center_longitudes.shape[1])
            settings.write_log("\t\t\t'COL' dimension: {0}".format(self.grid.center_longitudes.shape[1]), level=3)

            # ===== Variables =====
            settings.write_log("\t\tCreating NetCDF variables.", level=2)
            tflag = netcdf.createVariable('TFLAG', 'i', ('TSTEP', 'VAR', 'DATE-TIME',))
            tflag.setncatts({'units': "{:<16}".format('<YYYYDDD,HHMMSS>'), 'long_name': "{:<16}".format('TFLAG'),
                             'var_desc': "{:<80}".format('Timestep-valid flags:  (1) YYYYDDD or (2) HHMMSS')})
            tflag[:] = self.create_tflag(self.date, self.hours, len(self.variables_attributes))
            settings.write_log("\t\t\t'TFLAG' variable created with size: {0}".format(tflag[:].shape), level=3)

        full_shape = None
        index = 0
        # data_list, var_list = self.change_variable_attributes(self.variables_attributes)
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
                st_time = timeit.default_timer()
                index += 1

                var = netcdf.createVariable(var_name, 'f', ('TSTEP', 'LAY', 'ROW', 'COL',), zlib=self.compress)
                var.setncatts(self.variables_attributes[var_name])
                # var.units = variable['units']
                # var.long_name = str(variable['long_name'])
                # var.var_desc = str(variable['var_desc'])
                # var[:] = variable['data']

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
                            # print 'Resizeing {0}'.format(i)
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
            # ===== Global attributes =====
            global_attributes = self.create_global_attributes(self.variables_attributes.keys())
            for attribute in self.global_attributes_order:
                netcdf.setncattr(attribute, global_attributes[attribute])

            netcdf.close()
        settings.write_time('WriterCmaq', 'write_serial_netcdf', timeit.default_timer() - st_time, level=3)
        return True
