#!/usr/bin/env python

import sys
import os
from mpi4py import MPI

from timeit import default_timer as gettime

from hermesv3_bu.config.config import Config
from hermesv3_bu.grids.grid import select_grid
from hermesv3_bu.clipping.clip import select_clip
from hermesv3_bu.sectors.sector_manager import SectorManager


class Hermes(object):
    """
    Interface class for HERMESv3.
    """
    def __init__(self, config, new_date=None):
        from shutil import rmtree

        comm_world = MPI.COMM_WORLD

        self.arguments = config.arguments

        # updating starting date
        if new_date is not None:
            self.arguments.start_date = new_date

        self.grid = select_grid(comm_world, self.arguments)

        self.clip = select_clip(comm_world, self.arguments.auxiliar_files_path, self.arguments.clipping,
                                self.grid.shapefile)

        self.sector_manager = SectorManager(comm_world, self.grid, self.clip, self.arguments)

        sys.exit(1)

    # @profile
    def main(self):
        """
        Main functionality of the model.
        """
        from datetime import timedelta

        if self.arguments.start_date < self.options.end_date:
            return self.arguments.start_date + timedelta(days=1)

        return None


def run():
    date = Hermes(Config()).main()
    while date is not None:
        date = Hermes(Config(), new_date=date).main()
    sys.exit(0)


if __name__ == '__main__':
    run()
