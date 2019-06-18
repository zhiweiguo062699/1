#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
from mpi4py import MPI
from warnings import warn
import timeit
from hermesv3_bu.logger.log import Log

CHUNKING = True


def select_writer(logger, arguments, grid, date_array):
    """
    Select the writer depending on the arguments passed to HERMESv3_BU

    :param logger: Logger
    :type logger: Log

    :param arguments: Arguments passed to HERMESv3_BU
    :type arguments: Namespace

    :param grid: Output grid definition.
    :type grid: hermesv3_bu.grids.grid.Grid

    :param date_array: Array with each time step to be calculated.
    :type date_array: list of datetime.datetime

    :return: Selected writer.
    :rtype: Writer
    """
    spent_time = timeit.default_timer()
    comm_world = MPI.COMM_WORLD

    if grid.shape[2] % 2 == 0:
        max_procs = grid.shape[2] // 2
    else:
        max_procs = (grid.shape[2] // 2) + 1

    if arguments.writing_processors > min((comm_world.Get_size(), max_procs)):
        warn('Exceeded maximum of writing processors. Setting it to {0}'.format(
            min((comm_world.Get_size(), max_procs))))

        arguments.writing_processors = min((comm_world.Get_size(), max_procs))

    rank_distribution = get_distribution(logger, arguments.writing_processors, grid.shape)

    if comm_world.Get_rank() < arguments.writing_processors:
        color = 99
    else:
        color = 0

    comm_write = comm_world.Split(color, comm_world.Get_rank())

    pollutant_info = pd.read_csv(arguments.speciation_map, usecols=['dst', 'description', 'units'], index_col='dst')
    pollutant_info = pollutant_info.loc[~pollutant_info.index.duplicated(keep='first')]

    if arguments.output_model == 'DEFAULT':
        from hermesv3_bu.writer.default_writer import DefaultWriter
        writer = DefaultWriter(comm_world, comm_write, logger, arguments.output_name, grid, date_array, pollutant_info,
                               rank_distribution, arguments.emission_summary)
    logger.write_time_log('Writer', 'select_writer', timeit.default_timer() - spent_time)

    return writer


def get_distribution(logger, processors, shape):
    """
    Calculate the process distribution for writing.

    :param logger: Logger
    :type logger: Log

    :param processors: Number of writing processors.
    :type processors: int

    :param shape: Complete shape of the destiny domain.
    :type shape: tuple

    :return: Information of the writing process. That argument is a dictionary with the writing
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
    :rtype rank_distribution: dict
    """
    spent_time = timeit.default_timer()
    fid_dist = {}
    total_rows = shape[2]

    aux_rows = total_rows // processors
    if total_rows % processors > 0:
        aux_rows += 1

    rows_sum = 0
    for proc in xrange(processors):
        total_rows -= aux_rows
        if total_rows < 0:
            rows = total_rows + aux_rows
        else:
            rows = aux_rows

        min_fid = proc * aux_rows * shape[3]
        max_fid = (proc + 1) * aux_rows * shape[3]

        fid_dist[proc] = {
            'y_min': rows_sum,
            'y_max': rows_sum + rows,
            'x_min': 0,
            'x_max': shape[3],
            'fid_min': min_fid,
            'fid_max': max_fid,
            'shape': (shape[0], shape[1], rows, shape[3]),
        }

        rows_sum += rows
    logger.write_time_log('Writer', 'get_distribution', timeit.default_timer() - spent_time)
    return fid_dist


class Writer(object):
    def __init__(self, comm_world, comm_write, logger, netcdf_path, grid, date_array, pollutant_info,
                 rank_distribution, emission_summary=False):
        """
        Initialise the Writer class.

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
        :type pollutant_info: pandas.DataFrame

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
        """
        spent_time = timeit.default_timer()

        self.comm_world = comm_world
        self.comm_write = comm_write
        self.logger = logger
        self.netcdf_path = netcdf_path
        self.grid = grid
        self.date_array = date_array
        self.pollutant_info = pollutant_info
        self.rank_distribution = rank_distribution
        self.emission_summary = emission_summary

        if self.emission_summary and self.comm_write.Get_rank() == 0:
            self.emission_summary_paths = {
                'hourly_layer_summary_path': self.netcdf_path.replace('.nc', '_summary_hourly_layer.csv'),
                'hourly_summary_path': self.netcdf_path.replace('.nc', '_summary_hourly.csv'),
                'total_summary_path': self.netcdf_path.replace('.nc', '_summary.csv')
            }
        else:
            self.emission_summary_paths = None

        self.logger.write_time_log('Writer', '__init__', timeit.default_timer() - spent_time)

    def gather_emissions(self, emissions):
        """
        Each writing process recives the emissions for a concrete region of the domain.

        Each calculation process sends a part of the emissions to each writing processor.

        :param emissions: Emissions to be split and sent to the writing processors.
        :type emissions: pandas.DataFrame

        :return: The writing processors will return the emissions to write but the non writer processors will return
            None.
        :rtype: pandas.DataFrame
        """
        spent_time = timeit.default_timer()
        # Sending
        requests = []
        for w_rank, info in self.rank_distribution.iteritems():
            partial_emis = emissions.loc[(emissions.index.get_level_values(0) >= info['fid_min']) &
                                         (emissions.index.get_level_values(0) < info['fid_max'])]
            requests.append(self.comm_world.isend(partial_emis, dest=w_rank))

        # Receiving
        if self.comm_world.Get_rank() in self.rank_distribution.iterkeys():
            data_list = [None] * self.comm_world.Get_size()

            for i_rank in xrange(self.comm_world.Get_size()):
                data_list[i_rank] = self.comm_world.recv(source=i_rank)

            new_emissions = pd.concat(data_list).reset_index().groupby(['FID', 'layer', 'tstep']).sum()
        else:
            new_emissions = None

        self.comm_world.Barrier()

        if self.emission_summary and self.comm_world.Get_rank() in self.rank_distribution.iterkeys():
            self.make_summary(new_emissions)

        self.logger.write_time_log('Writer', 'gather_emissions', timeit.default_timer() - spent_time)

        return new_emissions

    def dataframe_to_array(self, dataframe):
        """
        Set the dataframe emissions to a 4D numpy array in the way taht have to be written.

        :param dataframe: Dataframe with the FID, level and time step as index and pollutant as columns.
        :type dataframe: pandas.DataFrame

        :return: 4D array with the emissions to be written.
        :rtype: numpy.array
        """
        spent_time = timeit.default_timer()
        var_name = dataframe.columns.values[0]
        shape = self.rank_distribution[self.comm_write.Get_rank()]['shape']
        dataframe.reset_index(inplace=True)
        dataframe['FID'] = dataframe['FID'] - self.rank_distribution[self.comm_write.Get_rank()]['fid_min']
        data = np.zeros((shape[0], shape[1], shape[2] * shape[3]))

        for (layer, tstep), aux_df in dataframe.groupby(['layer', 'tstep']):
            data[tstep, layer, aux_df['FID']] = aux_df[var_name]
        self.logger.write_time_log('Writer', 'dataframe_to_array', timeit.default_timer() - spent_time)

        return data.reshape(shape)

    def write(self, emissions):
        """
        Do all the process to write the emissions.

        :param emissions: Emissions to be written.
        :type emissions: pandas.DataFrame

        :return: True if everything finish OK.
        :rtype: bool
        """
        spent_time = timeit.default_timer()
        emissions = self.unit_change(emissions)
        emissions = self.gather_emissions(emissions)
        if self.comm_world.Get_rank() in self.rank_distribution.iterkeys():
            self.write_netcdf(emissions)

        self.comm_world.Barrier()
        self.logger.write_time_log('Writer', 'write', timeit.default_timer() - spent_time)

        return True

    def unit_change(self, emissions):
        """
        Implemented on the inner classes

        :rtype: pandas.DataFrame
        """
        pass

    def write_netcdf(self, emissions):
        """
        Implemented on the inner classes
        """
        pass

    def make_summary(self, emissions):
        """
        Create the files with the summary of the emissions.

        It will create 3 files:
        - Total emissions per pollutant
        - Total emissions per pollutant and hour
        - Total emissions per pollutant, hour and layer

        :param emissions: Emissions
        :type emissions: pandas.DataFrame

        :return: True if everything goes OK
        :rtype: bool
        """
        spent_time = timeit.default_timer()

        summary = emissions.groupby(['tstep', 'layer']).sum().reset_index()

        summary = self.comm_write.gather(summary, root=0)

        if self.comm_write.Get_rank() == 0:
            summary = pd.concat(summary)
            summary = summary.groupby(['tstep', 'layer']).sum()
            summary.to_csv(self.emission_summary_paths['hourly_layer_summary_path'])
            summary.reset_index(inplace=True)
            summary.drop(columns=['layer'], inplace=True)
            summary.groupby('tstep').sum().to_csv(self.emission_summary_paths['hourly_summary_path'])
            summary.drop(columns=['tstep'], inplace=True)
            summary.sum().to_csv(self.emission_summary_paths['total_summary_path'])
        self.logger.write_time_log('Writer', 'make_summary', timeit.default_timer() - spent_time)
