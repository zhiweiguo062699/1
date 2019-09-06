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
                     point_sources_shapefile_path, population_raster_path,population_nuts2_path,
                     land_uses_raster_path, land_uses_nuts2_path, nut2_shapefile_path])

        super(SolventsSector, self).__init__(
            comm, logger, auxiliary_dir, grid, clip, date_array, source_pollutants, vertical_levels,
            monthly_profile_path, weekly_profile_path, hourly_profile_path, speciation_map_path,
            speciation_profiles_path, molecular_weights_path)

        # self.calculate_land_use_by_nut(land_uses_raster_path, nut2_shapefile_path, land_uses_nuts2_path)
        # exit()

        self.proxies_map = self.read_proxies(proxies_map_path)
        self.check_profiles()

        self.proxy = self.get_proxy_shapefile(population_raster_path, population_nuts2_path, land_uses_raster_path,
                                              land_uses_nuts2_path, nut2_shapefile_path)

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

    def get_population_by_nut2(self, path):
        spent_time = timeit.default_timer()

        pop_by_nut2 = pd.read_csv(path)
        pop_by_nut2.set_index('nuts2_id', inplace=True)
        pop_by_nut2 = pop_by_nut2.to_dict()['pop']

        self.logger.write_time_log('SolventsSector', 'get_pop_by_nut2', timeit.default_timer() - spent_time)
        return pop_by_nut2

    def get_land_use_by_nut2(self, path, land_uses, nut_codes):
        spent_time = timeit.default_timer()

        land_use_by_nut2 = pd.read_csv(path)
        land_use_by_nut2 = land_use_by_nut2[land_use_by_nut2['nuts2_id'].isin(nut_codes)]
        land_use_by_nut2 = land_use_by_nut2[land_use_by_nut2['land_use'].isin(land_uses)]
        land_use_by_nut2.set_index(['nuts2_id', 'land_use'], inplace=True)

        self.logger.write_time_log('SolventsSector', 'get_land_use_by_nut2', timeit.default_timer() - spent_time)
        return land_use_by_nut2

    def get_population_proxy(self, pop_raster_path, pop_by_nut2_path, nut2_shapefile_path):
        spent_time = timeit.default_timer()

        # 1st Clip the raster
        self.logger.write_log("\t\tCreating clipped population raster", message_level=3)
        if self.comm.Get_rank() == 0:
            pop_raster_path = IoRaster(self.comm).clip_raster_with_shapefile_poly(
                pop_raster_path, self.clip.shapefile, os.path.join(self.auxiliary_dir, 'solvents', 'pop.tif'))

        # 2nd Raster to shapefile
        self.logger.write_log("\t\tRaster to shapefile", message_level=3)
        pop_shp = IoRaster(self.comm).to_shapefile_parallel(
            pop_raster_path, gather=False, bcast=False, crs={'init': 'epsg:4326'})

        # 3rd Add NUT code
        self.logger.write_log("\t\tAdding nut codes to the shapefile", message_level=3)
        # if self.comm.Get_rank() == 0:
        pop_shp.drop(columns='CELL_ID', inplace=True)
        pop_shp.rename(columns={'data': 'population'}, inplace=True)
        pop_shp = self.add_nut_code(pop_shp, nut2_shapefile_path, nut_value='nuts2_id')
        pop_shp = pop_shp[pop_shp['nut_code'] != -999]
        pop_shp = IoShapefile(self.comm).balance(pop_shp)
        # pop_shp = IoShapefile(self.comm).split_shapefile(pop_shp)

        # 4th Calculate population percent
        self.logger.write_log("\t\tCalculating population percentage on source resolution", message_level=3)
        pop_by_nut2 = self.get_population_by_nut2(pop_by_nut2_path)
        pop_shp['tot_pop'] = pop_shp['nut_code'].map(pop_by_nut2)
        pop_shp['pop_percent'] = pop_shp['population'] / pop_shp['tot_pop']
        pop_shp.drop(columns=['tot_pop', 'population'], inplace=True)

        # 5th Calculate percent by dest_cell
        self.logger.write_log("\t\tCalculating population percentage on destiny resolution", message_level=3)
        pop_shp.to_crs(self.grid.shapefile.crs, inplace=True)
        pop_shp['src_inter_fraction'] = pop_shp.geometry.area
        pop_shp = self.spatial_overlays(pop_shp.reset_index(), self.grid.shapefile.reset_index())
        pop_shp.drop(columns=['idx1', 'idx2', 'index'], inplace=True)
        pop_shp['src_inter_fraction'] = pop_shp.geometry.area / pop_shp['src_inter_fraction']
        pop_shp['pop_percent'] = pop_shp['pop_percent'] * pop_shp['src_inter_fraction']
        pop_shp.drop(columns=['src_inter_fraction'], inplace=True)

        popu_dist = pop_shp.groupby(['FID', 'nut_code']).sum()
        popu_dist.rename(columns={'pop_percent': 'population'}, inplace=True)

        self.logger.write_time_log('SolventsSector', 'get_population_proxie', timeit.default_timer() - spent_time)
        return popu_dist

    def get_land_use_proxy(self, land_use_raster, land_use_by_nut2_path, land_uses, nut2_shapefile_path):
        spent_time = timeit.default_timer()
        # 1st Clip the raster
        self.logger.write_log("\t\tCreating clipped land use raster", message_level=3)
        lu_raster_path = os.path.join(self.auxiliary_dir, 'solvents', 'lu_{0}.tif'.format(
            '_'.join([str(x) for x in land_uses])))

        if self.comm.Get_rank() == 0:
            if not os.path.exists(lu_raster_path):
                lu_raster_path = IoRaster(self.comm).clip_raster_with_shapefile_poly(
                    land_use_raster, self.clip.shapefile, lu_raster_path, values=land_uses)

        # 2nd Raster to shapefile
        self.logger.write_log("\t\tRaster to shapefile", message_level=3)
        land_use_shp = IoRaster(self.comm).to_shapefile_parallel(lu_raster_path, gather=False, bcast=False)

        # 3rd Add NUT code
        self.logger.write_log("\t\tAdding nut codes to the shapefile", message_level=3)
        # if self.comm.Get_rank() == 0:
        land_use_shp.drop(columns='CELL_ID', inplace=True)
        land_use_shp.rename(columns={'data': 'land_use'}, inplace=True)
        land_use_shp = self.add_nut_code(land_use_shp, nut2_shapefile_path, nut_value='nuts2_id')
        land_use_shp = land_use_shp[land_use_shp['nut_code'] != -999]
        land_use_shp = IoShapefile(self.comm).balance(land_use_shp)
        # land_use_shp = IoShapefile(self.comm).split_shapefile(land_use_shp)

        # 4th Calculate land_use percent
        self.logger.write_log("\t\tCalculating land use percentage on source resolution", message_level=3)

        land_use_shp['area'] = land_use_shp.geometry.area
        land_use_by_nut2 = self.get_land_use_by_nut2(
            land_use_by_nut2_path, land_uses, np.unique(land_use_shp['nut_code']))
        land_use_shp.drop(columns=['land_use'], inplace=True)

        land_use_shp['fraction'] = land_use_shp.apply(
            lambda row: row['area'] / land_use_by_nut2.xs(row['nut_code'], level='nuts2_id').sum(), axis=1)
        land_use_shp.drop(columns='area', inplace=True)
        
        # 5th Calculate percent by dest_cell
        self.logger.write_log("\t\tCalculating land use percentage on destiny resolution", message_level=3)

        land_use_shp.to_crs(self.grid.shapefile.crs, inplace=True)
        land_use_shp['src_inter_fraction'] = land_use_shp.geometry.area
        land_use_shp = self.spatial_overlays(land_use_shp.reset_index(), self.grid.shapefile.reset_index())
        land_use_shp.drop(columns=['idx1', 'idx2', 'index'], inplace=True)
        land_use_shp['src_inter_fraction'] = land_use_shp.geometry.area / land_use_shp['src_inter_fraction']
        land_use_shp['fraction'] = land_use_shp['fraction'] * land_use_shp['src_inter_fraction']
        land_use_shp.drop(columns=['src_inter_fraction'], inplace=True)

        land_use_dist = land_use_shp.groupby(['FID', 'nut_code']).sum()
        land_use_dist.rename(columns={'fraction': 'lu_{0}'.format('_'.join([str(x) for x in land_uses]))}, inplace=True)

        self.logger.write_time_log('SolventsSector', 'get_land_use_proxy', timeit.default_timer() - spent_time)
        return land_use_dist

    def get_proxy_shapefile(self, population_raster_path, population_nuts2_path, land_uses_raster_path,
                            land_uses_nuts2_path, nut2_shapefile_path):
        spent_time = timeit.default_timer()
        self.logger.write_log("Getting proxies shapefile")
        # proxy_names_list = np.unique(self.proxies_map['proxy_name'])
        proxy_names_list = np.unique(['lu_3_8'])
        proxy_shp_name = os.path.join(self.auxiliary_dir, 'solvents', 'proxy_distributions.shp')
        if not os.path.exists(proxy_shp_name):
            for proxy_name in proxy_names_list:
                self.logger.write_log("\tGetting proxy for {0}".format(proxy_name), message_level=2)
                proxies_list = []
                if proxy_name == 'population':
                    pop_proxy = self.get_population_proxy(population_raster_path, population_nuts2_path,
                                                          nut2_shapefile_path)
                    proxies_list.append(pop_proxy)
                if proxy_name[:3] == 'lu_':
                    land_uses = [int(x) for x in proxy_name[3:].split('_')]

                    land_use_proxy = self.get_land_use_proxy(land_uses_raster_path, land_uses_nuts2_path, land_uses,
                                                             nut2_shapefile_path)
                    print(land_use_proxy)
        else:
            pass

        self.logger.write_time_log('SolventsSector', 'get_proxy_shapefile', timeit.default_timer() - spent_time)
        return True
