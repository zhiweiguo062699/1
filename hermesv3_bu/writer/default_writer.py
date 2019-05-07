#!/usr/bin/env python

import os
import numpy as np
from hermesv3_bu.writer.writer import Writer


class DefaultWriter(Writer):
    def __init__(self, comm_wolrd, comm_write, netcdf_path, rank_distribution):
        super(DefaultWriter, self).__init__(comm_wolrd, comm_write, netcdf_path, rank_distribution)

        print netcdf_path
