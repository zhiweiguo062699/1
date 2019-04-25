#!/usr/bin/env python

"""
Copyright 2018 Earth Sciences Department, BSC-CNS

 This file is part of HERMESv3.

 HERMESv3 is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 HERMESv3 is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with HERMESv3. If not, see <http://www.gnu.org/licenses/>.
"""

__author__ = "Carles Tena"
__copyright__ = "Copyright 2018"
__email__ = "carles.tena@bsc.es"
__license__ = "GNU General Public License"
__maintainer__ = "Carles Tena"
__version__ = "3.3.1"

import sys
import os
from mpi4py import MPI

parentPath = os.path.abspath(os.path.join('..', '..'))
if parentPath not in sys.path:
    sys.path.insert(0, parentPath)

from timeit import default_timer as gettime

from hermesv3_bu.config.config import Config


class Hermes(object):
    """
    Interface class for HERMESv3.
    """
    def __init__(self, config, new_date=None):

        self.arguments = config.arguments

        # updating starting date
        if new_date is not None:
            self.arguments.start_date = new_date

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
