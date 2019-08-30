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
                 point_sources_shapefile_path, population_raster_path, population_nuts2_path, land_uses_raster_path,
                 land_uses_nuts2_path):

        spent_time = timeit.default_timer()
        logger.write_log('===== SOLVENTS SECTOR =====')
        check_files([speciation_map_path, molecular_weights_path, speciation_profiles_path, monthly_profile_path,
                     weekly_profile_path, hourly_profile_path, proxies_path, yearly_emissions_by_nut2_path,
                     point_sources_shapefile_path, population_raster_path,
                     # population_nuts2_path,
                     land_uses_raster_path, land_uses_nuts2_path])

        super(SolventSector, self).__init__(
            comm, logger, auxiliary_dir, grid, clip, date_array, source_pollutants, vertical_levels,
            monthly_profile_path, weekly_profile_path, hourly_profile_path, speciation_map_path,
            speciation_profiles_path, molecular_weights_path)

        self.proxies = self.read_proxies(proxies_path)
        self.check_profiles()
        self.yearly_emissions = self.read_yearly_emissions(yearly_emissions_by_nut2_path)

        # print(self.speciation_profile)
        # print(self.monthly_profiles)
        # print(self.weekly_profiles)
        # print(self.hourly_profiles)
        print(self.proxies.head())
        print(self.proxies.columns.values)
        exit()

    def read_proxies(self, path):
        proxies_df = pd.read_csv(path, dtype=str)

        proxies_df.set_index('snap', inplace=True)
        proxies_df = proxies_df.loc[proxies_df['CONS'] == '1']
        proxies_df.drop(columns=['activity', 'CONS', 'gnfr'], inplace=True)

        return proxies_df
    
    def check_profiles(self):
        # Checking monthly profiles IDs
        links_month = set(np.unique(self.proxies['P_month'].dropna().values))
        month = set(self.monthly_profiles.index.values)
        month_res = links_month - month
        if len(month_res) > 0:
            error_exit("The following monthly profile IDs reported in the solvent proxies CSV file do not appear " +
                       "in the monthly profiles file. {0}".format(month_res))
        # Checking weekly profiles IDs
        links_week = set(np.unique(self.proxies['P_week'].dropna().values))
        week = set(self.weekly_profiles.index.values)
        week_res = links_week - week
        if len(week_res) > 0:
            error_exit("The following weekly profile IDs reported in the solvent proxies CSV file do not appear " +
                       "in the weekly profiles file. {0}".format(week_res))
        # Checking hourly profiles IDs
        links_hour = set(np.unique(self.proxies['P_hour'].dropna().values))
        hour = set(self.hourly_profiles.index.values)
        hour_res = links_hour - hour
        if len(hour_res) > 0:
            error_exit("The following hourly profile IDs reported in the solvent proxies CSV file do not appear " +
                       "in the hourly profiles file. {0}".format(hour_res))
        # Checking speciation profiles IDs
        links_spec = set(np.unique(self.proxies['P_spec'].dropna().values))
        spec = set(self.speciation_profile.index.values)
        spec_res = links_spec - spec
        if len(spec_res) > 0:
            error_exit("The following speciation profile IDs reported in the solvent proxies CSV file do not appear " +
                       "in the speciation profiles file. {0}".format(spec_res))

    def read_yearly_emissions(self, path):
        year_emis = pd.read_csv(path, dtype={'nuts2_id': str, 'snap': str, 'nmvoc': np.float64})
        year_emis.set_index(['nuts2_id', 'snap'], inplace=True)
        year_emis.drop(columns=['gnfr_description', 'gnfr', 'snap_description', 'nuts2_na'], inplace=True)
        return year_emis

