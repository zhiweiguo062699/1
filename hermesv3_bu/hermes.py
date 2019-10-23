#!/usr/bin/env python

import sys
import os
import timeit
from mpi4py import MPI
from datetime import timedelta

from hermesv3_bu.config.config import Config
from hermesv3_bu.grids.grid import select_grid, Grid
from hermesv3_bu.clipping.clip import select_clip
from hermesv3_bu.writer.writer import select_writer
from hermesv3_bu.sectors.sector_manager import SectorManager
from hermesv3_bu.logger.log import Log


class Hermes(object):
    """
    Interface class for HERMESv3.
    """
    def __init__(self, config, comm=None):
        """

        :param config: Configuration file object
        :type config: Config

        :param comm: Communicator
        :type comm: MPI.Comm
        """
        self.__initial_time = timeit.default_timer()
        if comm is None:
            comm = MPI.COMM_WORLD
        self.__comm = comm

        self.arguments = config.arguments
        self.__logger = Log(self.arguments)
        self.__logger.write_log('====== Starting HERMESv3_BU simulation =====')
        self.grid = select_grid(self.__comm, self.__logger, self.arguments)
        self.clip = select_clip(self.__comm, self.__logger, self.arguments.auxiliary_files_path, self.arguments.clipping,
                                self.grid)
        self.date_array = [self.arguments.start_date + timedelta(hours=hour) for hour in
                           range(self.arguments.output_timestep_num)]

        self.__logger.write_log('Dates to simulate:', message_level=3)
        for aux_date in self.date_array:
            self.__logger.write_log('\t{0}'.format(aux_date.strftime("%Y/%m/%d, %H:%M:%S")), message_level=3)

        self.sector_manager = SectorManager(
            self.__comm, self.__logger, self.grid, self.clip, self.date_array, self.arguments)

        self.writer = select_writer(self.__logger, self.arguments, self.grid, self.date_array)

        self.__logger.write_time_log('Hermes', '__init__', timeit.default_timer() - self.__initial_time)

    def main(self):
        """
        Main functionality of the model.
        """
        from datetime import timedelta

        if self.arguments.fist_time:
            self.__logger.write_log('***** HERMESv3_BU First Time finished successfully *****')
        else:
            emis = self.sector_manager.run()
            waiting_time = timeit.default_timer()
            self.__comm.Barrier()
            self.__logger.write_log('All emissions calculated!')
            self.__logger.write_time_log('Hermes', 'Waiting_to_write', timeit.default_timer() - waiting_time)

            self.writer.write(emis)
            self.__comm.Barrier()

            self.__logger.write_log('***** HERMESv3_BU simulation finished successfully *****')
        self.__logger.write_time_log('Hermes', 'TOTAL', timeit.default_timer() - self.__initial_time)
        self.__logger.finish_logs()

        if self.arguments.start_date < self.arguments.end_date:
            return self.arguments.start_date + timedelta(days=1)

        return None


def run(comm=None):
    date = Hermes(Config(comm), comm).main()
    while date is not None:
        date = Hermes(Config(new_date=date)).main()
    sys.exit(0)


if __name__ == '__main__':
    run()
