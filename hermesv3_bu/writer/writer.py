#!/usr/bin/env python

import os
import sys
import numpy as np
import pandas as pd
from mpi4py import MPI
from warnings import warn
import timeit
from hermesv3_bu.logger.log import Log
from hermesv3_bu.tools.checker import error_exit

CHUNKING = True
BALANCED = False
MPI_TAG_CONSTANT = 10**6


def select_writer(logger, comm_world, arguments, grid, date_array):
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

    if grid.shape[2] % 2 == 0:
        max_procs = grid.shape[2] // 2
    else:
        max_procs = (grid.shape[2] // 2) + 1

    if arguments.writing_processors > min((comm_world.Get_size(), max_procs)):
        warn('Exceeded maximum of writing processors. Setting it to {0}'.format(
            min((comm_world.Get_size(), max_procs))))

        arguments.writing_processors = min((comm_world.Get_size(), max_procs))

    if BALANCED:
        rank_distribution = get_balanced_distribution(logger, arguments.writing_processors, grid.shape)
    else:
        rank_distribution = get_distribution(logger, arguments.writing_processors, grid.shape)

    logger.write_log('Rank distribution: {0}'.format(rank_distribution), message_level=3)
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
    elif arguments.output_model == 'MONARCH':
        from hermesv3_bu.writer.monarch_writer import MonarchWriter
        writer = MonarchWriter(comm_world, comm_write, logger, arguments.output_name, grid, date_array, pollutant_info,
                               rank_distribution, arguments.emission_summary)
    elif arguments.output_model == 'CMAQ':
        from hermesv3_bu.writer.cmaq_writer import CmaqWriter
        writer = CmaqWriter(comm_world, comm_write, logger, arguments.output_name, grid, date_array, pollutant_info,
                            rank_distribution, arguments.output_attributes, arguments.emission_summary)
    elif arguments.output_model == 'WRF_CHEM':
        from hermesv3_bu.writer.wrfchem_writer import WrfChemWriter
        writer = WrfChemWriter(comm_world, comm_write, logger, arguments.output_name, grid, date_array, pollutant_info,
                               rank_distribution, arguments.output_attributes, arguments.emission_summary)
    else:
        error_exit("Unknown output model '{0}'. ".format(arguments.output_model) +
                   "Only MONARCH, CMAQ, WRF_CHEM or DEFAULT writers are available")

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

    if aux_rows * (processors - 1) >= total_rows:
        aux_rows -= 1

    rows_sum = 0
    for proc in range(processors):
        total_rows -= aux_rows
        if total_rows < 0 or proc == processors - 1:
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


def get_balanced_distribution(logger, processors, shape):
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

    procs_rows = total_rows // processors
    procs_rows_extended = total_rows-(procs_rows*processors)

    rows_sum = 0
    for proc in range(processors):
        if proc < procs_rows_extended:
            aux_rows = procs_rows + 1
        else:
            aux_rows = procs_rows

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
        :type emissions: DataFrame

        :return: The writing processors will return the emissions to write but the non writer processors will return
            None.
        :rtype: DataFrame
        """
        spent_time = timeit.default_timer()
        # Sending
        self.logger.write_log('Sending emissions to the writing processors.', message_level=2)
        requests = []
        for w_rank, info in self.rank_distribution.items():
            partial_emis = emissions.loc[(emissions.index.get_level_values(0) >= info['fid_min']) &
                                         (emissions.index.get_level_values(0) < info['fid_max'])]

            self.logger.write_log('\tFrom {0} sending {1} to {2}'.format(
                self.comm_world.Get_rank(),  sys.getsizeof(partial_emis), w_rank), message_level=3)
            # requests.append(self.comm_world.isend(sys.getsizeof(partial_emis), dest=w_rank,
            #                                       tag=self.comm_world.Get_rank() + MPI_TAG_CONSTANT))
            requests.append(self.comm_world.isend(partial_emis, dest=w_rank, tag=self.comm_world.Get_rank()))

        # Receiving
        self.logger.write_log('Receiving emissions in the writing processors.', message_level=2)
        if self.comm_world.Get_rank() in self.rank_distribution.keys():
            self.logger.write_log("I'm a writing processor.", message_level=3)
            data_list = []

            self.logger.write_log("Prepared to receive", message_level=3)
            for i_rank in range(self.comm_world.Get_size()):
                self.logger.write_log(
                    '\tFrom {0} to {1}'.format(i_rank, self.comm_world.Get_rank()), message_level=3)
                req = self.comm_world.irecv(2**27, source=i_rank, tag=i_rank)
                dataframe = req.wait()
                data_list.append(dataframe.reset_index())

            new_emissions = pd.concat(data_list)
            new_emissions[['FID', 'layer', 'tstep']] = new_emissions[['FID', 'layer', 'tstep']].astype(np.int32)

            new_emissions = new_emissions.groupby(['FID', 'layer', 'tstep']).sum()

        else:
            new_emissions = None
        self.comm_world.Barrier()
        self.logger.write_log('All emissions received.', message_level=2)

        if self.emission_summary and self.comm_world.Get_rank() in self.rank_distribution.keys():
            self.make_summary(new_emissions)

        self.logger.write_time_log('Writer', 'gather_emissions', timeit.default_timer() - spent_time)

        return new_emissions

    def dataframe_to_array(self, dataframe):
        """
        Set the dataframe emissions to a 4D numpy array in the way taht have to be written.

        :param dataframe: Dataframe with the FID, level and time step as index and pollutant as columns.
        :type dataframe: DataFrame

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
        :type emissions: DataFrame

        :return: True if everything finish OK.
        :rtype: bool
        """
        spent_time = timeit.default_timer()
        emissions = self.unit_change(emissions)
        emissions = self.gather_emissions(emissions)
        if self.comm_world.Get_rank() in self.rank_distribution.keys():
            self.write_netcdf(emissions)

        self.comm_world.Barrier()
        self.logger.write_time_log('Writer', 'write', timeit.default_timer() - spent_time)

        return True

    def unit_change(self, emissions):
        """
        Implemented on the inner classes

        :rtype: DataFrame
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
        :type emissions: DataFrame

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
            pd.DataFrame(summary.sum()).to_csv(self.emission_summary_paths['total_summary_path'])
        self.logger.write_time_log('Writer', 'make_summary', timeit.default_timer() - spent_time)
