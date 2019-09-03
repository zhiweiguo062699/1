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

PROXY_NAMES = {'boat_building': 'boat',
               'automobile_manufacturing': 'automobile',
               'car_repairing': 'car_repair',
               'dry_cleaning': 'gry_clean',
               'rubber_processing': 'rubber',
               'paints_manufacturing': 'paints',
               'inks_manufacturing': 'ink',
               'glues_manufacturing': 'glues',
               'pharmaceutical_products_manufacturing': 'pharma',
               'leather_taning': 'leather',
               'printing': 'printing',
               'automobile_manufacturing': 'automobile'
               }


class SolventsSector(Sector):
    def __init__(self, comm, logger, auxiliary_dir, grid, clip, date_array, source_pollutants, vertical_levels,
                 speciation_map_path, molecular_weights_path, speciation_profiles_path, monthly_profile_path,
                 weekly_profile_path, hourly_profile_path, proxies_map_path, yearly_emissions_by_nut2_path,
                 point_sources_shapefile_path, population_raster_path, population_nuts2_path, land_uses_raster_path,
                 land_uses_nuts2_path, nut2_shapefile_path):

        spent_time = timeit.default_timer()
        logger.write_log('===== SOLVENTS SECTOR =====')
        check_files([speciation_map_path, molecular_weights_path, speciation_profiles_path, monthly_profile_path,
                     weekly_profile_path, hourly_profile_path, proxies_map_path, yearly_emissions_by_nut2_path,
                     point_sources_shapefile_path, population_raster_path,
                     # population_nuts2_path,
                     land_uses_raster_path, land_uses_nuts2_path, nut2_shapefile_path])

        super(SolventsSector, self).__init__(
            comm, logger, auxiliary_dir, grid, clip, date_array, source_pollutants, vertical_levels,
            monthly_profile_path, weekly_profile_path, hourly_profile_path, speciation_map_path,
            speciation_profiles_path, molecular_weights_path)

        self.proxies_map = self.read_proxies(proxies_map_path)
        self.check_profiles()

        self.proxy = self.get_proxy_shapefile(population_raster_path, population_nuts2_path, nut2_shapefile_path)

        self.yearly_emissions = self.read_yearly_emissions(yearly_emissions_by_nut2_path)
        exit()
        self.logger.write_time_log('SolventsSector', '__init__', timeit.default_timer() - spent_time)

    def read_proxies(self, path):
        spent_time = timeit.default_timer()
        proxies_df = pd.read_csv(path, dtype=str)

        proxies_df.set_index('snap', inplace=True)
        proxies_df = proxies_df.loc[proxies_df['CONS'] == '1']
        proxies_df.drop(columns=['activity', 'CONS', 'gnfr'], inplace=True)

        proxies_df.loc[proxies_df['spatial_proxy'] == 'population', 'proxy_name'] = 'population'
        proxies_df.loc[proxies_df['spatial_proxy'] == 'land_use', 'proxy_name'] = \
            'lu_' + proxies_df['land_use_code'].replace(' ', '_', regex=True)
        proxies_df.loc[proxies_df['spatial_proxy'] == 'shapefile', 'proxy_name'] = \
            proxies_df['industry_code'].map(PROXY_NAMES)

        self.logger.write_time_log('SolventsSector', 'read_proxies', timeit.default_timer() - spent_time)
        return proxies_df
    
    def check_profiles(self):
        spent_time = timeit.default_timer()
        # Checking monthly profiles IDs
        links_month = set(np.unique(self.proxies_map['P_month'].dropna().values))
        month = set(self.monthly_profiles.index.values)
        month_res = links_month - month
        if len(month_res) > 0:
            error_exit("The following monthly profile IDs reported in the solvent proxies CSV file do not appear " +
                       "in the monthly profiles file. {0}".format(month_res))
        # Checking weekly profiles IDs
        links_week = set(np.unique(self.proxies_map['P_week'].dropna().values))
        week = set(self.weekly_profiles.index.values)
        week_res = links_week - week
        if len(week_res) > 0:
            error_exit("The following weekly profile IDs reported in the solvent proxies CSV file do not appear " +
                       "in the weekly profiles file. {0}".format(week_res))
        # Checking hourly profiles IDs
        links_hour = set(np.unique(self.proxies_map['P_hour'].dropna().values))
        hour = set(self.hourly_profiles.index.values)
        hour_res = links_hour - hour
        if len(hour_res) > 0:
            error_exit("The following hourly profile IDs reported in the solvent proxies CSV file do not appear " +
                       "in the hourly profiles file. {0}".format(hour_res))
        # Checking speciation profiles IDs
        links_spec = set(np.unique(self.proxies_map['P_spec'].dropna().values))
        spec = set(self.speciation_profile.index.values)
        spec_res = links_spec - spec
        if len(spec_res) > 0:
            error_exit("The following speciation profile IDs reported in the solvent proxies CSV file do not appear " +
                       "in the speciation profiles file. {0}".format(spec_res))

        self.logger.write_time_log('SolventsSector', 'check_profiles', timeit.default_timer() - spent_time)
        return True

    def read_yearly_emissions(self, path):
        spent_time = timeit.default_timer()

        year_emis = pd.read_csv(path, dtype={'nuts2_id': str, 'snap': str, 'nmvoc': np.float64})
        year_emis.set_index(['nuts2_id', 'snap'], inplace=True)
        year_emis.drop(columns=['gnfr_description', 'gnfr', 'snap_description', 'nuts2_na'], inplace=True)

        self.logger.write_time_log('SolventsSector', 'read_yearly_emissions', timeit.default_timer() - spent_time)
        return year_emis

    def get_population_proxie(self, pop_raster_path, pop_by_nut2_path, nut2_shapefile_path):

        if self.comm.Get_rank() == 0:
            pop_raster_path = IoRaster(self.comm).clip_raster_with_shapefile_poly(
                pop_raster_path, self.clip.shapefile, os.path.join(self.auxiliary_dir, 'solvents', 'pop.tif'))
        pop_shp = IoRaster(self.comm).to_shapefile_parallel(pop_raster_path, gather=True, bcast=False,
                                                            crs={'init': 'epsg:4326'})
        if self.comm.Get_rank() == 0:
            pop_shp.rename(columns={'data': 'population'}, inplace=True)
            pop_shp = self.add_nut_code(pop_shp, nut2_shapefile_path, nut_value='nuts2_id')
        pop_shp = IoShapefile(self.comm).split_shapefile(pop_shp)

        print(pop_shp)
        # if self.comm.Get_rank() == 0:
        #     pop_shp.to_file('~/temp/pop{0}.shp'.format(self.comm.Get_size()))
        #     print(pop_shp)
        exit()

    def get_proxy_shapefile(self, population_raster_path, population_nuts2_path, nut2_shapefile_path):

        proxies_list = np.unique(self.proxies_map['proxy_name'])
        proxies_list = np.unique(['population'])
        proxy_shp_name = os.path.join(self.auxiliary_dir, 'solvents', 'proxy_distributions.shp')
        if not os.path.exists(proxy_shp_name):
            for proxy in proxies_list:
                proxy_shp = self.grid.shapefile.copy()
                if proxy == 'population':
                    proxy_shp['population'] = self.get_population_proxie(population_raster_path, population_nuts2_path,
                                                                         nut2_shapefile_path)
        else:
            pass
        print(proxy_shp)
        print(proxies_list)



