#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
from mpi4py import MPI
from warnings import warn

CHUNKING = True


def select_writer(arguments, grid, date_array):
    """
    Select the writer depending on the arguments passed to HERMESv3_BU

    :param arguments: Arguments passed to HERMESv3_BU
    :type arguments: Namespace

    :param grid: Output grid definition.
    :type grid: hermesv3_bu.grids.grid.Grid

    :param date_array: Array with each time step to be calculated.
    :type date_array: list of datetime.datetime

    :return: Selected writer.
    :rtype: Writer
    """

    comm_world = MPI.COMM_WORLD

    if grid.shape[2] % 2 == 0:
        max_procs = grid.shape[2] // 2
    else:
        max_procs = (grid.shape[2] // 2) + 1

    if arguments.writing_processors > min((comm_world.Get_size(), max_procs)):
        warn('Exceeded maximum of writing processors. Setting it to {0}'.format(
            min((comm_world.Get_size(), max_procs))))

        arguments.writing_processors = min((comm_world.Get_size(), max_procs))

    rank_distribution = get_distribution(arguments.writing_processors, grid.shape)

    if comm_world.Get_rank() < arguments.writing_processors:
        color = 99
    else:
        color = 0

    comm_write = comm_world.Split(color, comm_world.Get_rank())

    pollutant_info = pd.read_csv(arguments.speciation_map, usecols=['dst', 'description', 'units'], index_col='dst')

    if arguments.output_model == 'DEFAULT':
        from hermesv3_bu.writer.default_writer import DefaultWriter
        writer = DefaultWriter(comm_world, comm_write, arguments.output_name, grid, date_array, pollutant_info,
                               rank_distribution)

    return writer


def get_distribution(processors, shape):
    """

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

    return fid_dist


class Writer(object):
    def __init__(self, comm_world, comm_write, netcdf_path, grid, date_array, pollutant_info, rank_distribution):
        """
        Initialise the Writer class.

        :param comm_wolrd: Global communicator for all the calculation process
        :type comm_wolrd: MPI.COMM

        :param comm_write: Sector communicator.
        :type comm_write: MPI.Intracomm

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

        self.comm_world = comm_world
        self.comm_write = comm_write
        self.netcdf_path = netcdf_path
        self.grid = grid
        self.date_array = date_array
        self.pollutant_info = pollutant_info
        self.rank_distribution = rank_distribution

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

        return new_emissions

    def dataframe_to_array(self, dataframe):
        """
        Set the dataframe emissions to a 4D numpy array in the way taht have to be written.

        :param dataframe: Dataframe with the FID, level and time step as index and pollutant as columns.
        :type dataframe: pandas.DataFrame

        :return: 4D array with the emissions to be written.
        :rtype: numpy.array
        """
        var_name = dataframe.columns.values[0]
        shape = self.rank_distribution[self.comm_write.Get_rank()]['shape']
        dataframe.reset_index(inplace=True)
        dataframe['FID'] = dataframe['FID'] - self.rank_distribution[self.comm_write.Get_rank()]['fid_min']
        data = np.zeros((shape[0], shape[1], shape[2] * shape[3]))

        for (layer, tstep), aux_df in dataframe.groupby(['layer', 'tstep']):
            data[tstep, layer, aux_df['FID']] = aux_df[var_name]
        return data.reshape(shape)

    def write(self, emissions):
        """
        Do all the process to write the emissions.

        :param emissions: Emissions to be written.
        :type emissions: pandas.DataFrame

        :return: True if everything finish OK.
        :rtype: bool
        """
        emissions = self.unit_change(emissions)
        emissions = self.gather_emissions(emissions)
        if self.comm_world.Get_rank() in self.rank_distribution.iterkeys():
            self.write_netcdf(emissions)

        self.comm_world.Barrier()
        return True
