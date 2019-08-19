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


class WrfChemWriter(Writer):
    def __init__(self, comm_world, comm_write, logger, netcdf_path, grid, date_array, pollutant_info,
                 rank_distribution, global_attributes_path, emission_summary=False):
        """
        Initialise the WRF-Chem writer that will write a NetCDF in the CMAQ input format (IOAPIv3.2).

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
        logger.write_log('WRF-Chem writer selected.')

        super(WrfChemWriter, self).__init__(comm_world, comm_write, logger, netcdf_path, grid, date_array,
                                            pollutant_info, rank_distribution, emission_summary)
        if self.grid.grid_type not in ['Lambert Conformal Conic', 'Mercator']:
            raise TypeError("ERROR: Only Lambert Conformal Conic or Mercator grid is implemented for WRF-Chem. " +
                            "The current grid type is '{0}'".format(self.grid.grid_type))

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

        self.global_attributes = self.create_global_attributes(global_attributes_path)
        self.pollutant_info = self.change_pollutant_attributes()

        self.logger.write_time_log('WrfChemWriter', '__init__', timeit.default_timer() - spent_time)

    def unit_change(self, emissions):
        """
        Change the units from mol/h or g/h to mol/s or g/s.

        :param emissions: Emissions on dataframe.
        :type emissions: DataFrame

        :return: Same emissions as input
        :rtype: DataFrame
        """
        spent_time = timeit.default_timer()

        if self.comm_write.Get_rank() == 0:
            self.grid.add_cell_area()

            cell_area = self.grid.shapefile[['FID', 'cell_area']]
            cell_area.set_index('FID', inplace=True)
        else:
            cell_area = None
        cell_area = self.comm_write.bcast(cell_area, root=0)

        # From mol/h or g/h to mol/m2.h or g/m2.h
        emissions = emissions.divide(cell_area['cell_area'], axis=0, level='FID')

        for pollutant, info in self.pollutant_info.iterrows():
            if info.get('units') == "ug/m3 m/s":
                # From g/m2.h to ug/m2.s
                emissions[[pollutant]] = emissions[[pollutant]].mul(10**6 / 3600)
            elif info.get('units') == "mol km^-2 hr^-1":
                # From mol/m2.h to mol/km2.h
                emissions[[pollutant]] = emissions[[pollutant]].mul(10**6)

        self.logger.write_time_log('WrfChemWriter', 'unit_change', timeit.default_timer() - spent_time)
        return emissions

    def change_pollutant_attributes(self):
        """
        Modify the emission list to be consistent to use the output as input for CMAQ model.

        :return: Emission list ready for CMAQ
        :rtype: DataFrame
        """
        spent_time = timeit.default_timer()

        new_pollutant_info = pd.DataFrame(columns=['pollutant', 'units', 'FieldType', 'MemoryOrder', 'description',
                                                   'stagger', 'coordinates'])

        for i, (pollutant, variable) in enumerate(self.pollutant_info.iterrows()):
            if variable.get('units') not in ['mol.h-1.km-2', "mol km^-2 hr^-1", 'ug.s-1.m-2', "ug/m3 m/s"]:
                raise ValueError("'{0}' unit is not supported for WRF-Chem emission ".format(variable.get('units')) +
                                 "input file. Set '{0}' in the speciation_map file.".format(
                                     ['mol.h-1.km-2', "mol km^-2 hr^-1", 'ug.s-1.m-2', "ug/m3 m/s"]))

            new_pollutant_info.loc[i, 'pollutant'] = pollutant
            if variable.get('units') in ['mol.h-1.km-2', "mol km^-2 hr^-1"]:
                new_pollutant_info.loc[i, 'units'] = "mol km^-2 hr^-1"
            else:
                new_pollutant_info.loc[i, 'units'] = "ug/m3 m/s"

            new_pollutant_info.loc[i, 'FieldType'] = np.int32(104)
            new_pollutant_info.loc[i, 'MemoryOrder'] = "XYZ"
            new_pollutant_info.loc[i, 'description'] = "EMISSIONS"
            new_pollutant_info.loc[i, 'stagger'] = ""
            new_pollutant_info.loc[i, 'coordinates'] = "XLONG XLAT"

        new_pollutant_info.set_index('pollutant', inplace=True)
        self.logger.write_time_log('WrfChemWriter', 'change_pollutant_attributes', timeit.default_timer() - spent_time)
        return new_pollutant_info

    def read_global_attributes(self, global_attributes_path):
        spent_time = timeit.default_timer()

        float_atts = ['DAMPCOEF', 'KHDIF', 'KVDIF', 'CEN_LAT', 'CEN_LON', 'DT']
        int_atts = [
            'BOTTOM-TOP_GRID_DIMENSION', 'DIFF_OPT', 'KM_OPT', 'DAMP_OPT', 'MP_PHYSICS', 'RA_LW_PHYSICS',
            'RA_SW_PHYSICS', 'SF_SFCLAY_PHYSICS', 'SF_SURFACE_PHYSICS', 'BL_PBL_PHYSICS', 'CU_PHYSICS',
            'SF_LAKE_PHYSICS', 'SURFACE_INPUT_SOURCE', 'SST_UPDATE', 'GRID_FDDA', 'GFDDA_INTERVAL_M', 'GFDDA_END_H',
            'GRID_SFDDA', 'SGFDDA_INTERVAL_M', 'SGFDDA_END_H', 'BOTTOM-TOP_PATCH_START_UNSTAG',
            'BOTTOM-TOP_PATCH_END_UNSTAG', 'BOTTOM-TOP_PATCH_START_STAG', 'BOTTOM-TOP_PATCH_END_STAG', 'GRID_ID',
            'PARENT_ID', 'I_PARENT_START', 'J_PARENT_START', 'PARENT_GRID_RATIO', 'NUM_LAND_CAT', 'ISWATER', 'ISLAKE',
            'ISICE', 'ISURBAN', 'ISOILWATER', 'HISTORY']
        str_atts = ['GRIDTYPE', 'MMINLU']

        if self.grid.grid_type == 'Lambert Conformal Conic':
            lat_ts = np.float32(self.grid.attributes['lat_0'])
        elif self.grid.grid_type == 'Mercator':
            lat_ts = np.float32(self.grid.attributes['lat_ts'])
        else:
            raise TypeError("ERROR: Only Lambert Conformal Conic or Mercator grid is implemented for WRF-Chem. " +
                            "The current grid type is '{0}'".format(self.grid.grid_type))

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
            'CEN_LON': np.float32(self.grid.attributes['lon_0'])
        }

        df = pd.read_csv(global_attributes_path)

        for att in atts_dict.keys():
            try:
                if att in int_atts:
                    atts_dict[att] = np.int32(df.loc[df['attribute'] == att, 'value'].item())
                elif att in float_atts:
                    atts_dict[att] = np.float32(df.loc[df['attribute'] == att, 'value'].item())
                elif att in str_atts:
                    atts_dict[att] = str(df.loc[df['attribute'] == att, 'value'].item())

            except ValueError:
                self.logger.write_log("WARNING: The global attribute {0} is not defined;".format(att) +
                                      " Using default value '{0}'".format(atts_dict[att]))
                if self.comm_write.Get_rank() == 0:
                    warn('WARNING: The global attribute {0} is not defined; Using default value {1}'.format(
                        att, atts_dict[att]))

        self.logger.write_time_log('WrfChemWriter', 'read_global_attributes', timeit.default_timer() - spent_time)
        return atts_dict

    def create_global_attributes(self, global_attributes_path):
        """
        Create the global attributes and the order that they have to be filled.

        :return: Dict of global attributes and a list with the keys ordered.
        :rtype: tuple
        """
        spent_time = timeit.default_timer()

        global_attributes = self.read_global_attributes(global_attributes_path)

        global_attributes['TITLE'] = 'Emissions generated by HERMESv3_BU.'
        global_attributes['START_DATE'] = self.date_array[0].strftime("%Y-%m-%d_%H:%M:%S")
        global_attributes['JULYR'] = np.int32(self.date_array[0].year)
        global_attributes['JULDAY'] = np.int32(self.date_array[0].strftime("%j"))
        global_attributes['GMT'] = np.float32(self.date_array[0].hour)
        global_attributes['HISTORY'] = \
            'Code developed by Barcelona Supercomputing Center (BSC, https://www.bsc.es/). ' + \
            'Developer: Carles Tena Medina (carles.tena@bsc.es), Marc Guevara Vilardell. (marc.guevara@bsc.es)'

        if self.grid.grid_type in ['Lambert Conformal Conic', 'Mercator']:
            global_attributes['WEST-EAST_GRID_DIMENSION'] = np.int32(self.grid.attributes['nx'] + 1)
            global_attributes['SOUTH-NORTH_GRID_DIMENSION'] = np.int32(self.grid.attributes['ny'] + 1)
            global_attributes['DX'] = np.float32(self.grid.attributes['inc_x'])
            global_attributes['DY'] = np.float32(self.grid.attributes['inc_y'])
            global_attributes['SURFACE_INPUT_SOURCE'] = np.int32(1)
            global_attributes['WEST-EAST_PATCH_START_UNSTAG'] = np.int32(1)
            global_attributes['WEST-EAST_PATCH_END_UNSTAG'] = np.int32(self.grid.attributes['nx'])
            global_attributes['WEST-EAST_PATCH_START_STAG'] = np.int32(1)
            global_attributes['WEST-EAST_PATCH_END_STAG'] = np.int32(self.grid.attributes['nx'] + 1)
            global_attributes['SOUTH-NORTH_PATCH_START_UNSTAG'] = np.int32(1)
            global_attributes['SOUTH-NORTH_PATCH_END_UNSTAG'] = np.int32(self.grid.attributes['ny'])
            global_attributes['SOUTH-NORTH_PATCH_START_STAG'] = np.int32(1)
            global_attributes['SOUTH-NORTH_PATCH_END_STAG'] = np.int32(self.grid.attributes['ny'] + 1)

            global_attributes['POLE_LAT'] = np.float32(90)
            global_attributes['POLE_LON'] = np.float32(0)

            if self.grid.grid_type == 'Lambert Conformal Conic':
                global_attributes['MAP_PROJ'] = np.int32(1)
                global_attributes['TRUELAT1'] = np.float32(self.grid.attributes['lat_1'])
                global_attributes['TRUELAT2'] = np.float32(self.grid.attributes['lat_2'])
                global_attributes['MOAD_CEN_LAT'] = np.float32(self.grid.attributes['lat_0'])
                global_attributes['STAND_LON'] = np.float32(self.grid.attributes['lon_0'])
            elif self.grid.grid_type == 'Mercator':
                global_attributes['MAP_PROJ'] = np.int32(3)
                global_attributes['TRUELAT1'] = np.float32(self.grid.attributes['lat_ts'])
                global_attributes['TRUELAT2'] = np.float32(0)
                global_attributes['MOAD_CEN_LAT'] = np.float32(self.grid.attributes['lat_ts'])
                global_attributes['STAND_LON'] = np.float32(self.grid.attributes['lon_0'])

        self.logger.write_time_log('WrfChemWriter', 'create_global_attributes', timeit.default_timer() - spent_time)
        return global_attributes

    def create_times_var(self):
        # TODO Documentation
        """

        :return:
        """
        import netCDF4

        aux_times_list = []

        for date in self.date_array:
            aux_times_list.append(date.strftime("%Y-%m-%d_%H:%M:%S"))

        str_out = netCDF4.stringtochar(np.array(aux_times_list))
        return str_out

    def write_netcdf(self, emissions):
        """
        Create a NetCDF following the WRF-Chem conventions

        :param emissions: Emissions to write in the NetCDF with 'FID, level & time step as index and pollutant as
            columns.
        :type emissions: DataFrame
        """
        spent_time = timeit.default_timer()

        if self.comm_write.Get_size() > 1:
            netcdf = Dataset(self.netcdf_path, format="NETCDF4", mode='w', parallel=True, comm=self.comm_write,
                             info=MPI.Info())
        else:
            netcdf = Dataset(self.netcdf_path, format="NETCDF4", mode='w')

        # ===== DIMENSIONS =====
        self.logger.write_log('\tCreating NetCDF dimensions', message_level=2)
        netcdf.createDimension('Time', len(self.date_array))

        netcdf.createDimension('DateStrLen', 19)
        netcdf.createDimension('west_east', self.grid.center_longitudes.shape[1])
        netcdf.createDimension('south_north', self.grid.center_latitudes.shape[0])
        netcdf.createDimension('emissions_zdim', len(self.grid.vertical_desctiption))

        # ========== VARIABLES ==========
        self.logger.write_log('\tCreating NetCDF variables', message_level=2)
        times = netcdf.createVariable('Times', 'S1', ('Time', 'DateStrLen',))
        times[:] = self.create_times_var()

        # ========== POLLUTANTS ==========
        for var_name in emissions.columns.values:
            self.logger.write_log('\t\tCreating {0} variable'.format(var_name), message_level=3)

            var = netcdf.createVariable(var_name, np.float64, ('Time', 'emissions_zdim', 'south_north', 'west_east',))

            if self.comm_write.Get_size() > 1:
                var.set_collective(True)

            var_data = self.dataframe_to_array(emissions.loc[:, [var_name]])

            var[:, :,
                self.rank_distribution[self.comm_write.Get_rank()]['y_min']:
                self.rank_distribution[self.comm_write.Get_rank()]['y_max'],
                self.rank_distribution[self.comm_write.Get_rank()]['x_min']:
                self.rank_distribution[self.comm_write.Get_rank()]['x_max']] = var_data

            var.FieldType = self.pollutant_info.loc[var_name, 'FieldType']
            var.MemoryOrder = self.pollutant_info.loc[var_name, 'MemoryOrder']
            var.description = self.pollutant_info.loc[var_name, 'description']
            var.units = self.pollutant_info.loc[var_name, 'units']
            var.stagger = self.pollutant_info.loc[var_name, 'stagger']
            var.coordinates = self.pollutant_info.loc[var_name, 'coordinates']

        # ========== METADATA ==========
        self.logger.write_log('\tCreating NetCDF metadata', message_level=2)

        for attribute in self.global_attributes_order:
            netcdf.setncattr(attribute, self.global_attributes[attribute])

        netcdf.close()
        self.logger.write_log('NetCDF write at {0}'.format(self.netcdf_path))
        self.logger.write_time_log('WrfChemWriter', 'write_netcdf', timeit.default_timer() - spent_time)

        return True
