#!/usr/bin/env python

import sys
import os
import timeit
import geopandas as gpd
import pandas as pd
import numpy as np
from hermesv3_bu.sectors.sector import Sector
from hermesv3_bu.io_server.io_shapefile import IoShapefile
from hermesv3_bu.io_server.io_raster import IoRaster
from hermesv3_bu.tools.checker import check_files, error_exit


class SolventSector(Sector):
    def __init__(self, comm, logger, auxiliary_dir, grid, clip, date_array, source_pollutants, vertical_levels,
                 speciation_map_path, molecular_weights_path, speciation_profiles_path, monthly_profile_path,
                 weekly_profile_path, hourly_profile_path, proxies_path, yearly_emissions_by_nut2_path,
                 point_sources_shapefile_path, population_raster_path, land_uses_raster_path):

        spent_time = timeit.default_timer()
        logger.write_log('===== SOLVENTS SECTOR =====')
        check_files([speciation_map_path, molecular_weights_path, speciation_profiles_path, monthly_profile_path,
                     weekly_profile_path, hourly_profile_path, proxies_path, yearly_emissions_by_nut2_path,
                     point_sources_shapefile_path, population_raster_path, land_uses_raster_path])

        super(SolventSector, self).__init__(
            comm, logger, auxiliary_dir, grid, clip, date_array, source_pollutants, vertical_levels,
            monthly_profile_path, weekly_profile_path, hourly_profile_path, speciation_map_path,
            speciation_profiles_path, molecular_weights_path)

        exit()