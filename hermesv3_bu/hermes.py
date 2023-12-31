#!/usr/bin/env python

import sys
import timeit
from mpi4py import MPI
from datetime import timedelta

from hermesv3_bu.config.config import Config
from hermesv3_bu.grids.grid import select_grid
from hermesv3_bu.clipping.clip import select_clip
from hermesv3_bu.writer.writer import select_writer
from hermesv3_bu.sectors.sector_manager import SectorManager
from hermesv3_bu.logger.log import Log


class HermesBu(object):
    """
    Interface class for HERMESv3_BU.
    """
    def __init__(self, config, comm=None):
        """

        :param config: Configuration file object
        :type config: Config

        :param comm: Communicator
        :type comm: MPI.Comm
        """
        self.initial_time = timeit.default_timer()
        if comm is None:
            comm = MPI.COMM_WORLD
        self.comm = comm

        self.arguments = config.arguments
        self.logger = Log(self.arguments)
        self.logger.write_log('====== Starting HERMESv3_BU simulation =====')
        self.grid = select_grid(self.comm, self.logger, self.arguments)
        self.clip = select_clip(self.comm, self.logger, self.arguments.auxiliary_files_path, self.arguments.clipping,
                                self.grid)
        self.date_array = [self.arguments.start_date + timedelta(hours=hour) for hour in
                           range(self.arguments.output_timestep_num)]

        self.logger.write_log('Dates to simulate:', message_level=3)
        for aux_date in self.date_array:
            self.logger.write_log('\t{0}'.format(aux_date.strftime("%Y/%m/%d, %H:%M:%S")), message_level=3)

        self.sector_manager = SectorManager(
            self.comm, self.logger, self.grid, self.clip, self.date_array, self.arguments)

        self.writer = select_writer(self.logger, self.comm, self.arguments, self.grid, self.date_array)

        self.logger.write_time_log('HERMES', '__init__', timeit.default_timer() - self.initial_time)

    def main(self, return_emis=False):
        """
        Main functionality of the model.
        """
        from datetime import timedelta

        if self.arguments.first_time:
            self.logger.write_log('***** HERMESv3_BU First Time finished successfully *****')
        else:
            emis = self.sector_manager.run()
            if return_emis:
                return emis
            waiting_time = timeit.default_timer()
            self.comm.Barrier()
            self.logger.write_log('All emissions calculated!')
            self.logger.write_time_log('HERMES', 'Waiting_to_write', timeit.default_timer() - waiting_time)

            self.writer.write(emis)
            self.comm.Barrier()
            if self.comm.Get_rank() == 0:
                print('***** HERMESv3_BU simulation finished successfully *****')
                sys.stdout.flush()
            self.logger.write_log('***** HERMESv3_BU simulation finished successfully *****')
        self.logger.write_time_log('HERMES', 'TOTAL', timeit.default_timer() - self.initial_time)
        if self.logger.log_level >= 3:
            self.logger.finish_logs()

        if self.arguments.start_date < self.arguments.end_date:
            return self.arguments.start_date + timedelta(days=1)

        return None


def run():
    date = HermesBu(Config()).main()
    while date is not None:
        date = HermesBu(Config(new_date=date)).main()
    sys.exit(0)


if __name__ == '__main__':
    run()
