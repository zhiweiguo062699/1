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
from hermesv3_gr.config import settings
from hermesv3_gr.modules.writing.writer import Writer


class WriterWrfChem(Writer):
    """
   Class to Write the output file for the WRF-CHEM Chemical Transport Model.

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
        super(WriterWrfChem, self).__init__(path, grid, levels, date, hours, global_attributes_path, compress, parallel)

        self.global_attributes_order = [
            'TITLE', 'START_DATE', 'WEST-EAST_GRID_DIMENSION', 'SOUTH-NORTH_GRID_DIMENSION',
            'BOTTOM-TOP_GRID_DIMENSION', 'DX', 'DY', 'GRIDTYPE', 'DIFF_OPT', 'KM_OPT', 'DAMP_OPT', 'DAMPCOEF', 'KHDIF',
            'KVDIF', 'MP_PHYSICS', 'RA_LW_PHYSICS', 'RA_SW_PHYSICS', 'SF_SFCLAY_PHYSICS', 'SF_SURFACE_PHYSICS',
            'BL_PBL_PHYSICS', 'CU_PHYSICS', 'SF_LAKE_PHYSICS', 'SURFACE_INPUT_SOURCE', 'SST_UPDATE', 'GRID_FDDA',
            'GFDDA_INTERVAL_M', 'GFDDA_END_H', 'GRID_SFDDA', 'SGFDDA_INTERVAL_M', 'SGFDDA_END_H',
            'WEST-EAST_PATCH_START_UNSTAG', 'WEST-EAST_PATCH_END_UNSTAG', 'WEST-EAST_PATCH_START_STAG',
            'WEST-EAST_PATCH_END_STAG', 'SOUTH-NORTH_PATCH_START_UNSTAG', 'SOUTH-NORTH_PATCH_END_UNSTAG',
            'SOUTH-NORTH_PATCH_START_STAG', 'SOUTH-NORTH_PATCH_END_STAG', 'BOTTOM-TOP_PATCH_START_UNSTAG',
            'BOTTOM-TOP_PATCH_END_UNSTAG', 'BOTTOM-TOP_PATCH_START_STAG', 'BOTTOM-TOP_PATCH_END_STAG', 'GRID_ID',
            'PARENT_ID', 'I_PARENT_START', 'J_PARENT_START', 'PARENT_GRID_RATIO', 'DT', 'CEN_LAT', 'CEN_LON',
            'TRUELAT1', 'TRUELAT2', 'MOAD_CEN_LAT', 'STAND_LON', 'POLE_LAT', 'POLE_LON', 'GMT', 'JULYR', 'JULDAY',
            'MAP_PROJ', 'MMINLU', 'NUM_LAND_CAT', 'ISWATER', 'ISLAKE', 'ISICE', 'ISURBAN', 'ISOILWATER']

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

            if Unit(units).symbol == Unit('mol.h-1.km-2').symbol:
                # 10e6 -> from m2 to km2
                # 10e3 -> from kmol to mol
                # 3600n -> from s to h
                data = data * 10e6 * 10e3 * 3600
            elif Unit(units).symbol == Unit('ug.s-1.m-2').symbol:
                # 10e9 -> from kg to ug
                data = data * 10e9
            else:
                settings.write_log('ERROR: Check the .err file to get more info.')
                if settings.rank == 0:
                    raise TypeError("The unit '{0}' of specie {1} is not defined correctly.".format(units, variable) +
                                    " Should be 'mol.h-1.km-2' or 'ug.s-1.m-2'")
                sys.exit(1)
        return data

    def change_variable_attributes(self):
        # TODO Documentation
        """

        :return:
        """
        from cf_units import Unit

        new_variable_dict = {}
        for variable in self.variables_attributes:
            if Unit(variable['units']).symbol == Unit('mol.h-1.km-2').symbol:
                new_variable_dict[variable['name']] = {
                    'FieldType': np.int32(104),
                    'MemoryOrder': "XYZ",
                    'description': "EMISSIONS",
                    'units': "mol km^-2 hr^-1",
                    'stagger': "",
                    'coordinates': "XLONG XLAT"
                }
            elif Unit(variable['units']).symbol == Unit('ug.s-1.m-2').symbol:
                new_variable_dict[variable['name']] = {
                    'FieldType': np.int32(104),
                    'MemoryOrder': "XYZ",
                    'description': "EMISSIONS",
                    'units': "ug/m3 m/s",
                    'stagger': "",
                    'coordinates': "XLONG XLAT"
                }
            else:
                settings.write_log('ERROR: Check the .err file to get more info.')
                if settings.rank == 0:
                    raise TypeError("The unit '{0}' of specie {1} is not ".format(variable['units'], variable['name']) +
                                    "defined correctly. Should be 'mol.h-1.km-2' or 'ug.s-1.m-2'")
                sys.exit(1)

        self.variables_attributes = new_variable_dict

    def read_global_attributes(self):
        # TODO Documentation
        """

        :return:
        """
        import pandas as pd
        from warnings import warn as warning

        float_atts = ['DAMPCOEF', 'KHDIF', 'KVDIF', 'CEN_LAT', 'CEN_LON', 'DT']
        int_atts = ['BOTTOM-TOP_GRID_DIMENSION', 'DIFF_OPT', 'KM_OPT', 'DAMP_OPT',
                    'MP_PHYSICS', 'RA_LW_PHYSICS', 'RA_SW_PHYSICS', 'SF_SFCLAY_PHYSICS', 'SF_SURFACE_PHYSICS',
                    'BL_PBL_PHYSICS', 'CU_PHYSICS', 'SF_LAKE_PHYSICS', 'SURFACE_INPUT_SOURCE', 'SST_UPDATE',
                    'GRID_FDDA', 'GFDDA_INTERVAL_M', 'GFDDA_END_H', 'GRID_SFDDA', 'SGFDDA_INTERVAL_M', 'SGFDDA_END_H',
                    'BOTTOM-TOP_PATCH_START_UNSTAG', 'BOTTOM-TOP_PATCH_END_UNSTAG', 'BOTTOM-TOP_PATCH_START_STAG',
                    'BOTTOM-TOP_PATCH_END_STAG', 'GRID_ID', 'PARENT_ID', 'I_PARENT_START', 'J_PARENT_START',
                    'PARENT_GRID_RATIO', 'NUM_LAND_CAT', 'ISWATER', 'ISLAKE', 'ISICE', 'ISURBAN', 'ISOILWATER',
                    'HISTORY']
        str_atts = ['GRIDTYPE', 'MMINLU']
        if self.grid.grid_type == 'lcc':
            lat_ts = np.float32(self.grid.lat_0)
        elif self.grid.grid_type == 'mercator':
            lat_ts = np.float32(self.grid.lat_ts)

        atts_dict = {
            'BOTTOM-TOP_GRID_DIMENSION': np.int32(45),
            'GRIDTYPE': 'C',
            'DIFF_OPT': np.int32(1),
            'KM_OPT': np.int32(4),
            'DAMP_OPT': np.int32(3),
            'DAMPCOEF': np.float32(0.2),
            'KHDIF': np.float32(0.),
            'KVDIF': np.float32(0.),
            'MP_PHYSICS': np.int32(6),
            'RA_LW_PHYSICS': np.int32(4),
            'RA_SW_PHYSICS': np.int32(4),
            'SF_SFCLAY_PHYSICS': np.int32(2),
            'SF_SURFACE_PHYSICS': np.int32(2),
            'BL_PBL_PHYSICS': np.int32(8),
            'CU_PHYSICS': np.int32(0),
            'SF_LAKE_PHYSICS': np.int32(0),
            'SURFACE_INPUT_SOURCE': np.int32(1),
            'SST_UPDATE': np.int32(0),
            'GRID_FDDA': np.int32(0),
            'GFDDA_INTERVAL_M': np.int32(0),
            'GFDDA_END_H': np.int32(0),
            'GRID_SFDDA': np.int32(0),
            'SGFDDA_INTERVAL_M': np.int32(0),
            'SGFDDA_END_H': np.int32(0),
            'BOTTOM-TOP_PATCH_START_UNSTAG': np.int32(1),
            'BOTTOM-TOP_PATCH_END_UNSTAG': np.int32(44),
            'BOTTOM-TOP_PATCH_START_STAG': np.int32(1),
            'BOTTOM-TOP_PATCH_END_STAG': np.int32(45),
            'GRID_ID': np.int32(1),
            'PARENT_ID': np.int32(0),
            'I_PARENT_START': np.int32(1),
            'J_PARENT_START': np.int32(1),
            'PARENT_GRID_RATIO': np.int32(1),
            'DT': np.float32(18.),
            'MMINLU': 'MODIFIED_IGBP_MODIS_NOAH',
            'NUM_LAND_CAT': np.int32(41),
            'ISWATER': np.int32(17),
            'ISLAKE': np.int32(-1),
            'ISICE': np.int32(15),
            'ISURBAN': np.int32(13),
            'ISOILWATER': np.int32(14),
            'CEN_LAT': lat_ts,
            'CEN_LON': np.float32(self.grid.lon_0)
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
                except ValueError:
                    print 'A warning has occurred. Check the .err file to get more information.'
                    if settings.rank == 0:
                        warning('The global attribute {0} is not defined; Using default value {1}'.format(
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

    def create_global_attributes(self):
        # TODO Documentation
        """
        Create the global attributes that have to be filled.
        """

        global_attributes = self.read_global_attributes()

        global_attributes['TITLE'] = 'Emissions generated by HERMESv3_GR.'
        global_attributes['START_DATE'] = self.date.strftime("%Y-%m-%d_%H:%M:%S")
        global_attributes['JULYR'] = np.int32(self.date.year)
        global_attributes['JULDAY'] = np.int32(self.date.strftime("%j"))
        global_attributes['GMT'] = np.float32(self.date.hour)
        global_attributes['HISTORY'] = \
            'Code developed by Barcelona Supercomputing Center (BSC, https://www.bsc.es/). ' + \
            'Developer: Carles Tena Medina (carles.tena@bsc.es). ' + \
            'Reference: Guevara et al., 2018, GMD., in preparation.'

        if self.grid.grid_type == 'lcc' or self.grid.grid_type == 'mercator':
            global_attributes['WEST-EAST_GRID_DIMENSION'] = np.int32(self.grid.nx + 1)
            global_attributes['SOUTH-NORTH_GRID_DIMENSION'] = np.int32(self.grid.ny + 1)
            global_attributes['DX'] = np.float32(self.grid.inc_x)
            global_attributes['DY'] = np.float32(self.grid.inc_y)
            global_attributes['SURFACE_INPUT_SOURCE'] = np.int32(1)
            global_attributes['WEST-EAST_PATCH_START_UNSTAG'] = np.int32(1)
            global_attributes['WEST-EAST_PATCH_END_UNSTAG'] = np.int32(self.grid.nx)
            global_attributes['WEST-EAST_PATCH_START_STAG'] = np.int32(1)
            global_attributes['WEST-EAST_PATCH_END_STAG'] = np.int32(self.grid.nx + 1)
            global_attributes['SOUTH-NORTH_PATCH_START_UNSTAG'] = np.int32(1)
            global_attributes['SOUTH-NORTH_PATCH_END_UNSTAG'] = np.int32(self.grid.ny)
            global_attributes['SOUTH-NORTH_PATCH_START_STAG'] = np.int32(1)
            global_attributes['SOUTH-NORTH_PATCH_END_STAG'] = np.int32(self.grid.ny + 1)

            global_attributes['POLE_LAT'] = np.float32(90)
            global_attributes['POLE_LON'] = np.float32(0)

            if self.grid.grid_type == 'lcc':
                global_attributes['MAP_PROJ'] = np.int32(1)
                global_attributes['TRUELAT1'] = np.float32(self.grid.lat_1)
                global_attributes['TRUELAT2'] = np.float32(self.grid.lat_2)
                global_attributes['MOAD_CEN_LAT'] = np.float32(self.grid.lat_0)
                global_attributes['STAND_LON'] = np.float32(self.grid.lon_0)
            elif self.grid.grid_type == 'mercator':
                global_attributes['MAP_PROJ'] = np.int32(3)
                global_attributes['TRUELAT1'] = np.float32(self.grid.lat_ts)
                global_attributes['TRUELAT2'] = np.float32(0)
                global_attributes['MOAD_CEN_LAT'] = np.float32(self.grid.lat_ts)
                global_attributes['STAND_LON'] = np.float32(self.grid.lon_0)

        return global_attributes

    def create_times_var(self):
        # TODO Documentation
        """

        :return:
        """
        from datetime import timedelta
        import netCDF4

        aux_times_list = []

        for hour in self.hours:
            aux_date = self.date + timedelta(hours=hour)
            aux_times_list.append(aux_date.strftime("%Y-%m-%d_%H:%M:%S"))

        str_out = netCDF4.stringtochar(np.array(aux_times_list))
        return str_out

    def create_parallel_netcdf(self):
        # TODO Documentation
        """

        :return:
        """
        st_time = timeit.default_timer()
        settings.write_log("\tCreating parallel NetCDF file.", level=2)
        netcdf = Dataset(self.path, mode='w', format="NETCDF4")

        # ===== Dimensions =====
        settings.write_log("\t\tCreating NetCDF dimensions.", level=2)
        netcdf.createDimension('Time', len(self.hours))
        # netcdf.createDimension('Time', None)
        settings.write_log("\t\t\t'Time' dimension: {0}".format('UNLIMITED ({0})'.format(len(self.hours))),
                           level=3)
        netcdf.createDimension('DateStrLen', 19)
        settings.write_log("\t\t\t'DateStrLen' dimension: 19", level=3)
        netcdf.createDimension('west_east', self.grid.center_longitudes.shape[1])
        settings.write_log("\t\t\t'west_east' dimension: {0}".format(len(self.hours)), level=3)
        netcdf.createDimension('south_north', self.grid.center_latitudes.shape[0])
        settings.write_log("\t\t\t'south_north' dimension: {0}".format(self.grid.center_latitudes.shape[0]),
                           level=3)
        netcdf.createDimension('emissions_zdim', len(self.levels))
        settings.write_log("\t\t\t'emissions_zdim' dimension: {0}".format(len(self.levels)), level=3)

        # ===== Variables =====
        settings.write_log("\t\tCreating NetCDF variables.", level=2)
        times = netcdf.createVariable('Times', 'S1', ('Time', 'DateStrLen', ))
        times[:] = self.create_times_var()
        settings.write_log("\t\t\t'Times' variable created with size: {0}".format(times[:].shape), level=3)

        index = 0
        for var_name in self.variables_attributes.iterkeys():
            index += 1
            var = netcdf.createVariable(var_name, 'f', ('Time', 'emissions_zdim', 'south_north', 'west_east',),
                                        zlib=self.compress)
            var.setncatts(self.variables_attributes[var_name])
            settings.write_log("\t\t\t'{0}' variable created with size: {1}".format(var_name, var[:].shape) +
                               "\n\t\t\t\t'{0}' variable will be filled later.".format(var_name), level=3)

        # ===== Global attributes =====
        settings.write_log("\t\tCreating NetCDF metadata.", level=2)
        global_attributes = self.create_global_attributes()
        for attribute in self.global_attributes_order:
            netcdf.setncattr(attribute, global_attributes[attribute])

        netcdf.close()

        settings.write_time('WriterCmaq', 'create_parallel_netcdf', timeit.default_timer() - st_time, level=3)

        return True

    def write_serial_netcdf(self, emission_list):
        # TODO Documentation
        """

        :param emission_list:
        :return:
        """
        st_time = timeit.default_timer()

        # Gathering the index
        rank_position = np.array(
            [self.grid.x_lower_bound, self.grid.x_upper_bound, self.grid.y_lower_bound, self.grid.y_upper_bound],
            dtype='i')
        full_position = None
        if settings.rank == 0:
            full_position = np.empty([settings.size, 4], dtype='i')
        settings.comm.Gather(rank_position, full_position, root=0)

        if settings.rank == 0:
            settings.write_log("\tCreating NetCDF file.", level=2)
            netcdf = Dataset(self.path, mode='w', format="NETCDF4")

            # ===== Dimensions =====
            settings.write_log("\t\tCreating NetCDF dimensions.", level=2)
            netcdf.createDimension('Time', None)
            settings.write_log("\t\t\t'Time' dimension: UNLIMITED", level=3)
            netcdf.createDimension('DateStrLen', 19)
            settings.write_log("\t\t\t'DateStrLen' dimension: 19", level=3)
            netcdf.createDimension('west_east', self.grid.center_longitudes.shape[1])
            settings.write_log("\t\t\t'west_east' dimension: {0}".format(len(self.hours)), level=3)
            netcdf.createDimension('south_north', self.grid.center_latitudes.shape[0])
            settings.write_log("\t\t\t'south_north' dimension: {0}".format(self.grid.center_latitudes.shape[0]),
                               level=3)
            netcdf.createDimension('emissions_zdim', len(self.levels))
            settings.write_log("\t\t\t'emissions_zdim' dimension: {0}".format(len(self.levels)), level=3)

            # ===== Variables =====
            settings.write_log("\t\tCreating NetCDF variables.", level=2)
            times = netcdf.createVariable('Times', 'S1', ('Time', 'DateStrLen', ))
            times[:] = self.create_times_var()
            settings.write_log("\t\t\t'Times' variable created with size: {0}".format(times[:].shape), level=3)

        full_shape = None
        index = 0

        # self.change_variable_attributes()

        for var_name in self.variables_attributes.iterkeys():
            if settings.size != 1:
                settings.write_log("\t\t\tGathering {0} data.".format(var_name), level=3)
            rank_data = self.calculate_data_by_var(var_name, emission_list, self.grid.shape)
            if rank_data is not None:
                # root_shape = settings.comm.bcast(rank_data.shape, root=0)
                if full_shape is None:
                    full_shape = settings.comm.allgather(rank_data.shape)

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

            if settings.rank == 0:
                if settings.size != 1:
                    try:
                        data = np.concatenate(data, axis=3)
                    except (UnboundLocalError, TypeError, IndexError):
                        data = 0
                st_time = timeit.default_timer()
                index += 1

                var = netcdf.createVariable(var_name, 'f', ('Time', 'emissions_zdim', 'south_north', 'west_east',),
                                            zlib=self.compress)
                var.setncatts(self.variables_attributes[var_name])

                var_time = timeit.default_timer()

                # data_list = []#np.empty(shape, dtype=np.float64)

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
                settings.write_log("\t\t\t'{0}' variable created with size: {1}".format(var_name, var[:].shape),
                                   level=3)
        settings.write_log("\t\tCreating NetCDF metadata.", level=2)
        if settings.rank == 0:
            # ===== Global attributes =====
            global_attributes = self.create_global_attributes()
            for attribute in self.global_attributes_order:
                netcdf.setncattr(attribute, global_attributes[attribute])

            netcdf.close()
        settings.write_time('WriterWrfChem', 'write_serial_netcdf', timeit.default_timer() - st_time, level=3)
        return True
