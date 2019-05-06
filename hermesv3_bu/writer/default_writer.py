#!/usr/bin/env python

import os
import numpy as np
from hermesv3_bu.writer.writer import Writer


class DefaultWriter(Writer):
    def __init__(self, comm, netcdf_path, grid):
        super(DefaultWriter, self).__init__(comm, netcdf_path, grid)

        print netcdf_path
