#!/usr/bin/env python

import numpy as np
import pandas as pd
from warnings import warn
import sys
from netCDF4 import Dataset, date2num
from hermesv3_bu.writer.writer import Writer
from mpi4py import MPI
import timeit
from hermesv3_bu.logger.log import Log


class CmaqWriter(Writer):
    def __init__(self, comm_world, comm_write, logger, netcdf_path, grid, date_array, pollutant_info,
                 rank_distribution, global_attributes_path, emission_summary=False):
        """
        Initialise the CMAQ writer that will write a NetCDF in the CMAQ input format (IOAPIv3.2).

        :param comm_world: Global communicator for all the calculation process
        :type comm_world: MPI.COMM

        :param comm_write: Sector communicator.
        :type comm_write: MPI.Intracomm

        :param logger: Logger
        :type logger: Log

        :param netcdf_path: Path to the output NetCDF file-
        :type netcdf_path: str

        :param grid: Output grid definition.
        :type grid: hermesv3_bu.grids.grid.Grid

        :param date_array: Array with each time step to be calculated.
        :type date_array: list of datetime.datetime

        :param pollutant_info: Information related with the output pollutants, short description, units...
        :type pollutant_info: DataFrame

        :param rank_distribution: Information of the writing process. That argument is a dictionary with the writing
            process rank as key and another dictionary as value. That other dictionary contains:
            - shape: Shape to write
            - x_min: X minimum position to write on the full array.
            - x_max: X maximum position to write on the full array.
            - y_min: Y minimum position to write on the full array.
            - y_max: Y maximum position to write on the full array.
            - fid_min: Minimum cell ID of a flatten X Y domain.
            - fid_max: Maximum cell ID of a flatten X Y domain.

            e.g. 24 time steps. 48 vertical levels, 10 x 10
            {0: {'fid_min': 0, 'y_min': 0, 'y_max': 5, 'fid_max': 50, 'shape': (24, 48, 5, 10), 'x_max': 10,
                'x_min': 0},
            1: {'fid_min': 50, 'y_min': 5, 'y_max': 10, 'fid_max': 100, 'shape': (24, 48, 5, 10), 'x_max': 10,
                'x_min': 0}}
        :type rank_distribution: dict

        :param global_attributes_path: Path to the file that contains the static global attributes.
        :type global_attributes_path: str

        :param emission_summary: Indicates if you want to create the emission summary files.
        :type emission_summary: bool
        """
        spent_time = timeit.default_timer()
        logger.write_log('CMAQ writer selected.')

        super(CmaqWriter, self).__init__(comm_world, comm_write, logger, netcdf_path, grid, date_array, pollutant_info,
                                         rank_distribution, emission_summary)
        if self.grid.grid_type not in ['Lambert Conformal Conic']:
            raise TypeError("ERROR: Only Lambert Conformal Conic grid is implemented for CMAQ. " +
                            "The current grid type is '{0}'".format(self.grid.grid_type))

        self.global_attributes_order = [
            'IOAPI_VERSION', 'EXEC_ID', 'FTYPE', 'CDATE', 'CTIME', 'WDATE', 'WTIME', 'SDATE', 'STIME', 'TSTEP', 'NTHIK',
            'NCOLS', 'NROWS', 'NLAYS', 'NVARS', 'GDTYP', 'P_ALP', 'P_BET', 'P_GAM', 'XCENT', 'YCENT', 'XORIG', 'YORIG',
            'XCELL', 'YCELL', 'VGTYP', 'VGTOP', 'VGLVLS', 'GDNAM', 'UPNAM', 'FILEDESC', 'HISTORY', 'VAR-LIST']

        self.global_attributes = self.create_global_attributes(global_attributes_path)
        self.pollutant_info = self.change_pollutant_attributes()

        self.logger.write_time_log('CmaqWriter', '__init__', timeit.default_timer() - spent_time)

    def unit_change(self, emissions):
        """
        Change the units from mol/h or g/h to mol/s or g/s.

        :param emissions: Emissions on dataframe.
        :type emissions: DataFrame

        :return: Same emissions as input
        :rtype: DataFrame
        """
        spent_time = timeit.default_timer()

        # From mol/h g/h to mol/s g/s
        emissions = emissions / 3600.0

        self.logger.write_time_log('CmaqWriter', 'unit_change', timeit.default_timer() - spent_time)
        return emissions

    def change_pollutant_attributes(self):
        """
        Modify the emission list to be consistent to use the output as input for CMAQ model.

        :return: Emission list ready for CMAQ
        :rtype: DataFrame
        """
        spent_time = timeit.default_timer()

        new_pollutant_info = pd.DataFrame(columns=['pollutant', 'units', 'var_desc', 'long_name'])

        for i, (pollutant, variable) in enumerate(self.pollutant_info.iterrows()):
            if variable.get('units') not in ['mol.s-1', 'g.s-1', 'mole/s', 'g/s']:
                raise ValueError("'{0}' unit is not supported for CMAQ emission ".format(variable.get('units')) +
                                 "input file. Set mol.s-1 or g.s-1 in the speciation_map file.")
            new_pollutant_info.loc[i, 'pollutant'] = pollutant
            if variable.get('units') in ['mol.s-1', 'mole/s']:
                new_pollutant_info.loc[i, 'units'] = "{:<16}".format('mole/s')
            else:
                new_pollutant_info.loc[i, 'units'] = "{:<16}".format('g/s')
            new_pollutant_info.loc[i, 'var_desc'] = "{:<80}".format(variable.get('description'))
            new_pollutant_info.loc[i, 'long_name'] = "{:<16}".format(pollutant)

        new_pollutant_info.set_index('pollutant', inplace=True)
        self.logger.write_time_log('CmaqWriter', 'change_pollutant_attributes', timeit.default_timer() - spent_time)
        return new_pollutant_info

    def create_tflag(self):
        """
        Create the content of the CMAQ variable TFLAG

        :return: Array with the content of TFLAG
        :rtype: numpy.array
        """
        spent_time = timeit.default_timer()

        a = np.array([[[]]])

        for date in self.date_array:
            b = np.array([[int(date.strftime('%Y%j'))], [int(date.strftime('%H%M%S'))]] * len(self.pollutant_info))
            a = np.append(a, b)

        a.shape = (len(self.date_array), 2, len(self.pollutant_info))
        self.logger.write_time_log('CmaqWriter', 'create_tflag', timeit.default_timer() - spent_time)
        return a

    def str_var_list(self):
        """
        Transform a list to a string with the elements with 16 white spaces.

        :return: List transformed on string.
        :rtype: str
        """
        spent_time = timeit.default_timer()
        str_var_list = ""

        for var in list(self.pollutant_info.index):
            str_var_list += "{:<16}".format(var)

        self.logger.write_time_log('CmaqWriter', 'str_var_list', timeit.default_timer() - spent_time)
        return str_var_list

    def read_global_attributes(self, global_attributes_path):
        spent_time = timeit.default_timer()

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

        df = pd.read_csv(global_attributes_path)

        for att in atts_dict.iterkeys():
            try:
                if att in int_atts:
                    atts_dict[att] = np.int32(df.loc[df['attribute'] == att, 'value'].item())
                elif att in float_atts:
                    atts_dict[att] = np.float32(df.loc[df['attribute'] == att, 'value'].item())
                elif att in str_atts:
                    atts_dict[att] = str(df.loc[df['attribute'] == att, 'value'].item())
                    if att == 'EXEC_ID':
                        atts_dict[att] = '{:<80}'.format(atts_dict[att])
                    elif att == 'GDNAM':
                        atts_dict[att] = '{:<16}'.format(atts_dict[att])
                elif att in list_float_atts:
                    atts_dict[att] = np.array(df.loc[df['attribute'] == att, 'value'].item().split(),
                                              dtype=np.float32)
            except ValueError:
                self.logger.write_log("WARNING: The global attribute {0} is not defined;".format(att) +
                                      " Using default value '{0}'".format(atts_dict[att]))
                if self.comm_write.Get_rank() == 0:
                    warn('WARNING: The global attribute {0} is not defined; Using default value {1}'.format(
                        att, atts_dict[att]))

        self.logger.write_time_log('CmaqWriter', 'read_global_attributes', timeit.default_timer() - spent_time)
        return atts_dict

    def create_global_attributes(self, global_attributes_path):
        """
        Create the global attributes and the order that they have to be filled.

        :return: Dict of global attributes and a list with the keys ordered.
        :rtype: tuple
        """
        from datetime import datetime
        spent_time = timeit.default_timer()

        global_attributes = self.read_global_attributes(global_attributes_path)

        tstep = 1 * 10000

        now = datetime.now()
        global_attributes['IOAPI_VERSION'] = 'None: made only with NetCDF libraries'
        global_attributes['CDATE'] = np.int32(now.strftime('%Y%j'))
        global_attributes['CTIME'] = np.int32(now.strftime('%H%M%S'))
        global_attributes['WDATE'] = np.int32(now.strftime('%Y%j'))
        global_attributes['WTIME'] = np.int32(now.strftime('%H%M%S'))
        global_attributes['SDATE'] = np.int32(self.date_array[0].strftime('%Y%j'))
        global_attributes['STIME'] = np.int32(self.date_array[0].strftime('%H%M%S'))
        global_attributes['TSTEP'] = np.int32(tstep)
        global_attributes['NLAYS'] = np.int32(len(self.grid.vertical_desctiption))
        global_attributes['NVARS'] = np.int32(len(self.pollutant_info))
        global_attributes['UPNAM'] = "{:<16}".format('HERMESv3')
        global_attributes['FILEDESC'] = 'Emissions generated by HERMESv3_BU.'
        global_attributes['HISTORY'] = \
            'Code developed by Barcelona Supercomputing Center (BSC, https://www.bsc.es/).' + \
            'Developer: Carles Tena Medina (carles.tena@bsc.es), Marc Guevara Vilardell. (marc.guevara@bsc.es) '
        global_attributes['VAR-LIST'] = self.str_var_list()

        if self.grid.grid_type == 'Lambert Conformal Conic':
            global_attributes['GDTYP'] = np.int32(2)
            global_attributes['NCOLS'] = np.int32(self.grid.attributes['nx'])
            global_attributes['NROWS'] = np.int32(self.grid.attributes['ny'])
            global_attributes['P_ALP'] = np.float(self.grid.attributes['lat_1'])
            global_attributes['P_BET'] = np.float(self.grid.attributes['lat_2'])
            global_attributes['P_GAM'] = np.float(self.grid.attributes['lon_0'])
            global_attributes['XCENT'] = np.float(self.grid.attributes['lon_0'])
            global_attributes['YCENT'] = np.float(self.grid.attributes['lat_0'])
            global_attributes['XORIG'] = np.float(self.grid.attributes['x_0']) - np.float(
                self.grid.attributes['inc_x']) / 2
            global_attributes['YORIG'] = np.float(self.grid.attributes['y_0']) - np.float(
                self.grid.attributes['inc_y']) / 2
            global_attributes['XCELL'] = np.float(self.grid.attributes['inc_x'])
            global_attributes['YCELL'] = np.float(self.grid.attributes['inc_y'])

        self.logger.write_time_log('CmaqWriter', 'create_global_attributes', timeit.default_timer() - spent_time)
        return global_attributes

    def write_netcdf(self, emissions):
        """
        Create a NetCDF following the IOAPIv3.2 (CMAQ) conventions

        :param emissions: Emissions to write in the NetCDF with 'FID, level & time step as index and pollutant as
            columns.
        :type emissions: DataFrame
        """
        spent_time = timeit.default_timer()

        netcdf = Dataset(self.netcdf_path, mode='w', parallel=True, comm=self.comm_write, info=MPI.Info())

        # ===== DIMENSIONS =====
        self.logger.write_log('\tCreating NetCDF dimensions', message_level=2)
        netcdf.createDimension('TSTEP', len(self.date_array))
        netcdf.createDimension('DATE-TIME', 2)
        netcdf.createDimension('LAY', len(self.grid.vertical_desctiption))
        netcdf.createDimension('VAR', len(self.pollutant_info))
        netcdf.createDimension('ROW', self.grid.center_latitudes.shape[0])
        netcdf.createDimension('COL', self.grid.center_longitudes.shape[1])

        # ========== VARIABLES ==========
        self.logger.write_log('\tCreating NetCDF variables', message_level=2)
        tflag = netcdf.createVariable('TFLAG', 'i', ('TSTEP', 'VAR', 'DATE-TIME',))
        tflag.setncatts({'units': "{:<16}".format('<YYYYDDD,HHMMSS>'), 'long_name': "{:<16}".format('TFLAG'),
                         'var_desc': "{:<80}".format('Timestep-valid flags:  (1) YYYYDDD or (2) HHMMSS')})
        tflag[:] = self.create_tflag()

        # ========== POLLUTANTS ==========
        for var_name in emissions.columns.values:
            self.logger.write_log('\t\tCreating {0} variable'.format(var_name), message_level=3)

            var_data = self.dataframe_to_array(emissions.loc[:, [var_name]])
            var = netcdf.createVariable(var_name, np.float64, ('TSTEP', 'LAY', 'ROW', 'COL',))
            var[:, :,
                self.rank_distribution[self.comm_write.Get_rank()]['y_min']:
                self.rank_distribution[self.comm_write.Get_rank()]['y_max'],
                self.rank_distribution[self.comm_write.Get_rank()]['x_min']:
                self.rank_distribution[self.comm_write.Get_rank()]['x_max']] = var_data

            var.long_name = self.pollutant_info.loc[var_name, 'long_name']
            var.units = self.pollutant_info.loc[var_name, 'units']
            var.var_desc = self.pollutant_info.loc[var_name, 'var_desc']

        # ========== METADATA ==========
        self.logger.write_log('\tCreating NetCDF metadata', message_level=2)

        for attribute in self.global_attributes_order:
            netcdf.setncattr(attribute, self.global_attributes[attribute])

        netcdf.close()
        self.logger.write_log('NetCDF write at {0}'.format(self.netcdf_path))
        self.logger.write_time_log('CmaqWriter', 'write_netcdf', timeit.default_timer() - spent_time)

        return True
