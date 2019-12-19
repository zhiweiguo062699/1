#!/usr/bin/env python

import sys
import os
import timeit
import geopandas as gpd
import pandas as pd
import numpy as np
from warnings import warn
from hermesv3_bu.sectors.sector import Sector
from hermesv3_bu.io_server.io_shapefile import IoShapefile
from hermesv3_bu.io_server.io_raster import IoRaster
from hermesv3_bu.tools.checker import check_files, error_exit
from pandas import DataFrame
from geopandas import GeoDataFrame
from hermesv3_bu.grids.grid import Grid
from hermesv3_bu.logger.log import Log
from hermesv3_bu.clipping.clip import Clip

PROXY_NAMES = {'boat_building': 'boat',
               'automobile_manufacturing': 'automobile',
               'car_repairing': 'car_repair',
               'dry_cleaning': 'dry_clean',
               'rubber_processing': 'rubber',
               'paints_manufacturing': 'paints',
               'inks_manufacturing': 'ink',
               'glues_manufacturing': 'glues',
               'pharmaceutical_products_manufacturing': 'pharma',
               'leather_taning': 'leather',
               'printing': 'printing',
               }


class SolventsSector(Sector):
    """
    Solvents sector allows to calculate the solvents emissions.

    It first calculates the horizontal distribution for the different sources and store them in an auxiliary file
    during the initialization part.

    Once the initialization is finished it distribute the emissions of the different sub sectors ont he grid to start
    the temporal disaggregation.
    """
    def __init__(self, comm, logger, auxiliary_dir, grid, clip, date_array, source_pollutants, vertical_levels,
                 speciation_map_path, molecular_weights_path, speciation_profiles_path, monthly_profile_path,
                 weekly_profile_path, hourly_profile_path, proxies_map_path, yearly_emissions_by_nut2_path,
                 point_sources_shapefile_path, point_sources_weight_by_nut2_path, population_raster_path,
                 population_nuts2_path, land_uses_raster_path, land_uses_nuts2_path, nut2_shapefile_path):
        """
        :param comm: Communicator for the sector calculation.
        :type comm: MPI.COMM

        :param logger: Logger
        :type logger: Log

        :param auxiliary_dir: Path to the directory where the necessary auxiliary files will be created if them are not
            created yet.
        :type auxiliary_dir: str

        :param grid: Grid object.
        :type grid: Grid

        :param clip: Clip object
        :type clip: Clip

        :param date_array: List of datetimes.
        :type date_array: list(datetime.datetime, ...)

        :param source_pollutants: List of input pollutants to take into account.
        :type source_pollutants: list

        :param vertical_levels: List of top level of each vertical layer.
        :type vertical_levels: list

        :param speciation_map_path: Path to the CSV file that contains the speciation map. The CSV file must contain
            the following columns [dst, src, description]
            The 'dst' column will be used as output pollutant list and the 'src' column as their onw input pollutant
            to be used as a fraction in the speciation profiles.
        :type speciation_map_path: str

        :param molecular_weights_path: Path to the CSV file that contains all the molecular weights needed. The CSV
            file must contain the 'Specie' and 'MW' columns.
        :type molecular_weights_path: str

        :param speciation_profiles_path: Path to the file that contains all the speciation profiles. The CSV file
            must contain the "Code" column with the value of each animal of the animal_list. The rest of columns
            have to be the sames as the column 'dst' of the 'speciation_map_path' file.
        :type speciation_profiles_path: str

        :param hourly_profile_path: Path to the CSV file that contains all the monthly profiles. The CSV file must
            contain the following columns [P_month, January, February, ..., November, December]
            The P_month code have to match with the proxies_map_path file.
        :type hourly_profile_path: str

        :param weekly_profile_path: Path to the CSV file that contains all the weekly profiles. The CSV file must
            contain the following columns [P_week, Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday]
            The P_week code have to match with the proxies_map_path file.
        :type weekly_profile_path: str

        :param hourly_profile_path: Path to the CSV file that contains all the hourly profiles. The CSV file must
            contain the following columns [P_hour, 0, 1, 2, 3, ..., 22, 23]
            The P_week code have to match with the proxies_map_path file.
        :type hourly_profile_path: str

        :param proxies_map_path: Path to the CSV file that contains the proxies map.
        :type proxies_map_path: str

        :param yearly_emissions_by_nut2_path: Path to the CSV file that contains the yearly emissions by subsecotr and
            nuts2 level.
        :type yearly_emissions_by_nut2_path: str

        :param point_sources_shapefile_path: Path to the shapefile that contains the point sources for solvents.
        :type point_sources_shapefile_path: str

        :param point_sources_weight_by_nut2_path: Path to the CSV file that contains the weight for each proxy and nut2.
        :type point_sources_weight_by_nut2_path: str

        :param population_raster_path: Path to the population raster.
        :type population_raster_path: str

        :param population_nuts2_path: Path to the CSV file that contains the amount of population for each nut2.
        :type population_nuts2_path: str

        :param land_uses_raster_path: Path to the land use raster.
        :type land_uses_raster_path: str

        :param land_uses_nuts2_path: Path to the CSV file that contains the amount of land use for each nut2.
        :type land_uses_nuts2_path: str

        :param nut2_shapefile_path: Path to the shapefile that contains the nut2.
        :type nut2_shapefile_path: str
        """
        spent_time = timeit.default_timer()
        logger.write_log('===== SOLVENTS SECTOR =====')

        check_files([speciation_map_path, molecular_weights_path, speciation_profiles_path, monthly_profile_path,
                     weekly_profile_path, hourly_profile_path, proxies_map_path, yearly_emissions_by_nut2_path,
                     point_sources_shapefile_path, point_sources_weight_by_nut2_path, population_raster_path,
                     population_nuts2_path, land_uses_raster_path, land_uses_nuts2_path, nut2_shapefile_path])

        super(SolventsSector, self).__init__(
            comm, logger, auxiliary_dir, grid, clip, date_array, source_pollutants, vertical_levels,
            monthly_profile_path, weekly_profile_path, hourly_profile_path, speciation_map_path,
            speciation_profiles_path, molecular_weights_path)

        self.proxies_map = self.read_proxies(proxies_map_path)
        self.check_profiles()

        self.proxy = self.get_proxy_shapefile(
            population_raster_path, population_nuts2_path, land_uses_raster_path, land_uses_nuts2_path,
            nut2_shapefile_path, point_sources_shapefile_path, point_sources_weight_by_nut2_path)

        self.yearly_emissions_path = yearly_emissions_by_nut2_path
        self.logger.write_time_log('SolventsSector', '__init__', timeit.default_timer() - spent_time)

    def read_proxies(self, path):
        """
        Read the proxy map.

        It will filter the CONS == '1' snaps and add the 'spatial_proxy' column that the content will match with some
        column of the proxy shapefile.

        :param path: path to the CSV file that have the proxy map.
        :type path: str

        :return: Proxy map.
        :rtype: DataFrame
        """
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
        """
        Check that the profiles appear on the profile files.

        It will check the content of the proxies map.
        Check that the 'P_month' content appears on the monthly profiles.
        Check that the 'P_week' content appears on the weekly profiles.
        Check that the 'P_hour' content appears on the hourly profiles.
        Check that the 'P_spec' content appears on the speciation profiles.

        It will stop teh execution if the requirements are not satisfied.

        :return: True when everything is OK.
        :rtype: bool
        """
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

    def read_yearly_emissions(self, path, nut_list):
        """
        Read the yearly emission by snap and nuts2.

        Select only the nuts2 IDs that appear in the selected domain.

        Emissions are provided in T/year -> g/year

        :param path: Path to the CSV file that contains the yearly emissions by snap and nuts2.
        :type path: str

        :param nut_list: List of nut codes
        :type nut_list: list

        :return: Dataframe with thew amount of NMVOC for each snap and nut2
        :rtype: DataFrame
        """
        spent_time = timeit.default_timer()

        year_emis = pd.read_csv(path, dtype={'nuts2_id': int, 'snap': str, 'nmvoc': np.float64})
        # T/year -> g/year
        year_emis['nmvoc'] = year_emis['nmvoc'] * 1000000
        year_emis = year_emis[year_emis['nuts2_id'].isin(nut_list)]
        year_emis.set_index(['nuts2_id', 'snap'], inplace=True)
        year_emis.drop(columns=['gnfr_description', 'gnfr', 'snap_description', 'nuts2_na'], inplace=True)

        self.logger.write_time_log('SolventsSector', 'read_yearly_emissions', timeit.default_timer() - spent_time)
        return year_emis

    def get_population_by_nut2(self, path):
        """
        Read the CSV file that contains the amount of population by nut2.

        :param path: Path to the CSV file that contains the amount of population by nut2.
        :type path: str

        :return: Dataframe with the amount of population by nut2.
        :rtype: DataFrame
        """
        spent_time = timeit.default_timer()

        pop_by_nut2 = pd.read_csv(path)
        pop_by_nut2.set_index('nuts2_id', inplace=True)
        pop_by_nut2 = pop_by_nut2.to_dict()['pop']

        self.logger.write_time_log('SolventsSector', 'get_pop_by_nut2', timeit.default_timer() - spent_time)
        return pop_by_nut2

    def get_point_sources_weights_by_nut2(self, path, proxy_name):
        """
        Read the CSV file that contains the amount of weight by industry and nut2.

        :param path: Path to the CSV file that contains the amount of weight by industry and nut2.
        :type path: str

        :param proxy_name: Proxy to calculate.
        :type proxy_name: str

        :return: DataFrame with the amount of weight by industry and nut2.
        :rtype: DataFrame
        """
        spent_time = timeit.default_timer()

        weights_by_nut2 = pd.read_csv(path)
        weights_by_nut2['nuts2_id'] = weights_by_nut2['nuts2_id'].astype(int)
        weights_by_nut2 = weights_by_nut2[weights_by_nut2['industry_c'] == proxy_name]
        weights_by_nut2.drop(columns=['industry_c'], inplace=True)
        weights_by_nut2.set_index("nuts2_id", inplace=True)
        weights_by_nut2 = weights_by_nut2.to_dict()['weight']

        self.logger.write_time_log('SolventsSector', 'get_point_sources_weights_by_nut2',
                                   timeit.default_timer() - spent_time)
        return weights_by_nut2

    def get_land_use_by_nut2(self, path, land_uses, nut_codes):
        """
        Read the CSV file that contains the amount of land use by nut2.

        :param path: Path to the CSV file that contains the amount of land use by nut2.
        :type path: str

        :param land_uses: List of land uses to take into account.
        :type land_uses: list

        :param nut_codes: List of nut2 codes to take into account.
        :type nut_codes: list

        :return: DataFrame with the amount of land use by nut2.
        :rtype: DataFrame
        """
        spent_time = timeit.default_timer()

        land_use_by_nut2 = pd.read_csv(path)
        land_use_by_nut2 = land_use_by_nut2[land_use_by_nut2['nuts2_id'].isin(nut_codes)]
        land_use_by_nut2 = land_use_by_nut2[land_use_by_nut2['land_use'].isin(land_uses)]
        land_use_by_nut2.set_index(['nuts2_id', 'land_use'], inplace=True)

        self.logger.write_time_log('SolventsSector', 'get_land_use_by_nut2', timeit.default_timer() - spent_time)
        return land_use_by_nut2

    def get_population_proxy(self, pop_raster_path, pop_by_nut2_path, nut2_shapefile_path):
        """
        Calculate the distribution based on the amount of population.

        :param pop_raster_path: Path to the raster file that contains the population information.
        :type pop_raster_path: str

        :param pop_by_nut2_path: Path to the CSV file that contains the amount of population by nut2.
        :type pop_by_nut2_path: str

        :param nut2_shapefile_path: Path to the shapefile that contains the nut2.
        :type nut2_shapefile_path: str

        :return: GeoDataFrame with the population distribution by destiny cell.
        :rtype: GeoDataFrame
        """
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

        # 5th Calculate percent by destiny cell
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
        """
        Calculate the distribution based on the amount of land use.

        :param land_use_raster: Path to the raster file that contains the land use information.
        :type land_use_raster: str

        :param land_use_by_nut2_path: Path to the CSV file that contains the amount of land use by nut2.
        :type land_use_by_nut2_path: str

        :param land_uses: List of land uses to take into account on the distribution.
        :type land_uses: list

        :param nut2_shapefile_path: Path to the shapefile that contains the nut2.
        :type nut2_shapefile_path: str

        :return: GeoDataFrame with the land use distribution for the selected land uses by destiny cell.
        :rtype: GeoDataFrame
        """
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

    def get_point_shapefile_proxy(self, proxy_name, point_shapefile_path, point_sources_weight_by_nut2_path,
                                  nut2_shapefile_path):
        """
        Calculate the distribution for the solvent sub sector in the destiny grid cell.

        :param proxy_name: Name of the proxy to be calculated.
        :type proxy_name: str

        :param point_shapefile_path: Path to the shapefile that contains all the point sources ant their weights.
        :type point_shapefile_path: str

        :param point_sources_weight_by_nut2_path: Path to the CSV file that contains the amount of weight by industry
        and nut2.
        :type point_sources_weight_by_nut2_path: str

        :param nut2_shapefile_path: Path to the shapefile that contains the nut2.
        :type nut2_shapefile_path: str

        :return: GeoDataFrame with the distribution of the selected proxy on the destiny grid cells.
        :rtype: GeoDataFrame
        """
        spent_time = timeit.default_timer()

        point_shapefile = IoShapefile(self.comm).read_shapefile_parallel(point_shapefile_path)
        point_shapefile.drop(columns=['Empresa', 'Empleados', 'Ingresos', 'Consumos', 'LON', 'LAT'], inplace=True)
        point_shapefile = point_shapefile[point_shapefile['industry_c'] ==
                                          [key for key, value in PROXY_NAMES.items() if value == proxy_name][0]]
        point_shapefile = IoShapefile(self.comm).balance(point_shapefile)
        point_shapefile.drop(columns=['industry_c'], inplace=True)
        point_shapefile = self.add_nut_code(point_shapefile, nut2_shapefile_path, nut_value='nuts2_id')
        point_shapefile = point_shapefile[point_shapefile['nut_code'] != -999]

        point_shapefile = IoShapefile(self.comm).gather_shapefile(point_shapefile, rank=0)
        if self.comm.Get_rank() == 0:
            weight_by_nut2 = self.get_point_sources_weights_by_nut2(
                point_sources_weight_by_nut2_path,
                [key for key, value in PROXY_NAMES.items() if value == proxy_name][0])
            point_shapefile[proxy_name] = point_shapefile.apply(
                lambda row: row['weight'] / weight_by_nut2[row['nut_code']], axis=1)
            point_shapefile.drop(columns=['weight'], inplace=True)
            # print(point_shapefile.groupby('nut_code')['weight'].sum())

        point_shapefile = IoShapefile(self.comm).split_shapefile(point_shapefile)
        point_shapefile = gpd.sjoin(point_shapefile.to_crs(self.grid.shapefile.crs), self.grid.shapefile.reset_index())
        point_shapefile.drop(columns=['geometry', 'index_right'], inplace=True)
        point_shapefile = point_shapefile.groupby(['FID', 'nut_code']).sum()

        self.logger.write_time_log('SolventsSector', 'get_point_shapefile_proxy', timeit.default_timer() - spent_time)
        return point_shapefile

    def get_proxy_shapefile(self, population_raster_path, population_nuts2_path, land_uses_raster_path,
                            land_uses_nuts2_path, nut2_shapefile_path, point_sources_shapefile_path,
                            point_sources_weight_by_nut2_path):
        """
        Calcualte (or read) the proxy shapefile.

        It will split the entire shapoefile into as many processors as selected to split the calculation part.

        :param population_raster_path: Path to the raster file that contains the population information.
        :type population_raster_path: str

        :param population_nuts2_path: Path to the CSV file that contains the amount of population by nut2.
        :type population_nuts2_path: str

        :param land_uses_raster_path: Path to the raster file that contains the land use information.
        :type land_uses_raster_path: str

        :param land_uses_nuts2_path: Path to the CSV file that contains the amount of land use by nut2.
        :type land_uses_nuts2_path: str

        :param nut2_shapefile_path: Path to the shapefile that contains the nut2.
        :type nut2_shapefile_path: str

        :param point_sources_shapefile_path: Path to the shapefile that contains all the point sources ant their
        weights.
        :type point_sources_shapefile_path: str

        :param point_sources_weight_by_nut2_path: Path to the CSV file that contains the amount of weight by industry
        and nut2.
        :type point_sources_weight_by_nut2_path: str

        :return: GeoDataFrame with all the proxies
        :rtype: GeoDataFrame
        """
        spent_time = timeit.default_timer()

        self.logger.write_log("Getting proxies shapefile", message_level=1)
        proxy_names_list = np.unique(self.proxies_map['proxy_name'])
        proxy_path = os.path.join(self.auxiliary_dir, 'solvents', 'proxy_distributions.shp')
        if not os.path.exists(proxy_path):
            proxy_list = []
            for proxy_name in proxy_names_list:
                self.logger.write_log("\tGetting proxy for {0}".format(proxy_name), message_level=2)
                if proxy_name == 'population':
                    proxy = self.get_population_proxy(population_raster_path, population_nuts2_path,
                                                      nut2_shapefile_path)
                elif proxy_name[:3] == 'lu_':
                    land_uses = [int(x) for x in proxy_name[3:].split('_')]

                    proxy = self.get_land_use_proxy(land_uses_raster_path, land_uses_nuts2_path, land_uses,
                                                    nut2_shapefile_path)
                else:
                    proxy = self.get_point_shapefile_proxy(proxy_name, point_sources_shapefile_path,
                                                           point_sources_weight_by_nut2_path, nut2_shapefile_path)
                proxy = IoShapefile(self.comm).gather_shapefile(proxy.reset_index())
                if self.comm.Get_rank() == 0:
                    proxy_list.append(proxy)
            if self.comm.Get_rank() == 0:
                proxies = pd.concat(proxy_list, sort=False)
                proxies['FID'] = proxies['FID'].astype(int)
                proxies['nut_code'] = proxies['nut_code'].astype(int)
                proxies = proxies.groupby(['FID', 'nut_code']).sum()
                proxies = GeoDataFrame(proxies)
                # print(self.grid.shapefile.loc[proxies.index.get_level_values('FID'), 'geometry'].values)
                # exit()
                proxies = GeoDataFrame(
                    proxies, geometry=self.grid.shapefile.loc[proxies.index.get_level_values('FID'), 'geometry'].values,
                    crs=self.grid.shapefile.crs)
                # IoShapefile(self.comm).write_shapefile_serial(proxies.reset_index(), proxy_path)
            else:
                proxies = None
            proxies = IoShapefile(self.comm).split_shapefile(proxies)
            proxies = self.add_timezone(proxies)
            IoShapefile(self.comm).write_shapefile_parallel(proxies.reset_index(), proxy_path)

        else:
            proxies = IoShapefile(self.comm).read_shapefile_parallel(proxy_path)

        proxies.set_index(['FID', 'nut_code'], inplace=True)

        self.logger.write_time_log('SolventsSector', 'get_proxy_shapefile', timeit.default_timer() - spent_time)
        return proxies

    def add_dates(self, dataframe, drop_utc=True):
        """
        Add the 'date' and 'tstep' column to the dataframe.

        The dataframe will be replicated as many times as time steps to calculate.

        :param dataframe: Geodataframe to be extended with the dates.
        :type dataframe: GeoDataFrame

        :return: Geodataframe with the dates. The length of the new dataframe is the length of the input dataframe
            multiplied by the number of time steps.
        :rtype: GeoDataFrame
        """
        spent_time = timeit.default_timer()
        # dataframe = self.add_timezone(dataframe)
        df_list = []

        for tstep, date in enumerate(self.date_array):
            df_aux = dataframe.copy()
            df_aux['date'] = pd.to_datetime(date, utc=True)
            df_aux['date_utc'] = pd.to_datetime(date, utc=True)
            df_aux['tstep'] = tstep
            # df_aux = self.to_timezone(df_aux)
            df_list.append(df_aux)
        dataframe = pd.concat(df_list, ignore_index=True)
        dataframe = self.to_timezone(dataframe)
        if drop_utc:
            dataframe.drop('date_utc', axis=1, inplace=True)
        self.logger.write_time_log('Sector', 'add_dates', timeit.default_timer() - spent_time)

        return dataframe

    def calculate_hourly_emissions(self, yearly_emissions):
        """
        Disaggrate to hourly level the yearly emissions.

        :param yearly_emissions: GeoDataFrame with the yearly emissions by destiny cell ID and snap code.
        :type yearly_emissions: GeoDataFrame

        :return: GeoDataFrame with the hourly distribution by FID, snap code and time step.
        :rtype: GeoDataFrame
        """
        def get_mf(df):
            month_factor = self.monthly_profiles.loc[df.name[1], df.name[0]]

            df['MF'] = month_factor
            return df.loc[:, ['MF']]

        def get_wf(df):
            weekly_profile = self.calculate_rebalanced_weekly_profile(self.weekly_profiles.loc[df.name[1], :].to_dict(),
                                                                      df.name[0])
            df['WF'] = weekly_profile[df.name[0].weekday()]
            return df.loc[:, ['WF']]

        def get_hf(df):
            hourly_profile = self.hourly_profiles.loc[df.name[1], :].to_dict()
            hour_factor = hourly_profile[df.name[0]]

            df['HF'] = hour_factor
            return df.loc[:, ['HF']]

        spent_time = timeit.default_timer()

        self.logger.write_log('\tHourly disaggregation', message_level=2)
        emissions = self.add_dates(yearly_emissions.reset_index(), drop_utc=True)

        emissions['month'] = emissions['date'].dt.month
        emissions['weekday'] = emissions['date'].dt.weekday
        emissions['hour'] = emissions['date'].dt.hour
        emissions['date_as_date'] = emissions['date'].dt.date

        emissions['MF'] = emissions.groupby(['month', 'P_month']).apply(get_mf)
        emissions['WF'] = emissions.groupby(['date_as_date', 'P_week']).apply(get_wf)
        emissions['HF'] = emissions.groupby(['hour', 'P_hour']).apply(get_hf)

        emissions['temp_factor'] = emissions['MF'] * emissions['WF'] * emissions['HF']
        emissions.drop(columns=['MF', 'P_month', 'month', 'WF', 'P_week', 'weekday', 'HF', 'P_hour', 'hour', 'date',
                                'date_as_date'], inplace=True)
        emissions['nmvoc'] = emissions['nmvoc'] * emissions['temp_factor']
        emissions.drop(columns=['temp_factor'], inplace=True)
        emissions.set_index(['FID', 'snap', 'tstep'], inplace=True)

        self.logger.write_time_log('SolventsSector', 'calculate_hourly_emissions', timeit.default_timer() - spent_time)
        return emissions

    def distribute_yearly_emissions(self):
        """
        Calcualte the yearly emission by destiny grid cell and snap code.

        :return: GeoDataFrame with the yearly emissions by snap code.
        :rtype: GeoDataFrame
        """
        spent_time = timeit.default_timer()
        self.logger.write_log('\t\tYearly distribution', message_level=2)

        yearly_emis = self.read_yearly_emissions(
            self.yearly_emissions_path, np.unique(self.proxy.index.get_level_values('nut_code')))
        year_nuts = np.unique(yearly_emis.index.get_level_values('nuts2_id'))
        proxy_nuts = np.unique(self.proxy.index.get_level_values('nut_code'))
        unknow_nuts = list(set(proxy_nuts) - set(year_nuts))
        if len(unknow_nuts) > 0:
            warn("*WARNING* The {0} nuts2_id have no emissions in the solvents_yearly_emissions_by_nut2_path.".format(
                str(unknow_nuts)))
            self.proxy.drop(unknow_nuts, level='nut_code', inplace=True)
        emis_list = []
        for snap, snap_df in self.proxies_map.iterrows():
            emis = self.proxy.reset_index()
            emis['snap'] = snap
            emis['P_month'] = snap_df['P_month']
            emis['P_week'] = snap_df['P_week']
            emis['P_hour'] = snap_df['P_hour']
            emis['P_spec'] = snap_df['P_spec']

            emis['nmvoc'] = emis.apply(lambda row: yearly_emis.loc[(row['nut_code'], snap), 'nmvoc'] * row[
                self.proxies_map.loc[snap, 'proxy_name']], axis=1)

            emis.set_index(['FID', 'snap'], inplace=True)
            emis_list.append(emis[['P_month', 'P_week', 'P_hour', 'P_spec', 'nmvoc', 'geometry', 'timezone']])
        emis = pd.concat(emis_list).sort_index()
        emis = emis[emis['nmvoc'] > 0]

        self.logger.write_time_log('SolventsSector', 'distribute_yearly_emissions', timeit.default_timer() - spent_time)
        return emis

    def speciate(self, dataframe, code='default'):
        """
        Spectiate the NMVOC pollutant into as many pollutants as the speciation map indicates.

        :param dataframe: Emissions to be speciated.
        :type dataframe: DataFrame

        :param code: NOt used.

        :return: Speciated emissions.
        :rtype: DataFrame
        """

        def calculate_new_pollutant(x, out_p):
            sys.stdout.flush()
            profile = self.speciation_profile.loc[x.name, ['VOCtoTOG', out_p]]
            x[out_p] = x['nmvoc'] * (profile['VOCtoTOG'] * profile[out_p])
            return x[[out_p]]

        spent_time = timeit.default_timer()
        self.logger.write_log('\tSpeciation emissions', message_level=2)

        new_dataframe = gpd.GeoDataFrame(index=dataframe.index, data=None, crs=dataframe.crs,
                                         geometry=dataframe.geometry)
        for out_pollutant in self.output_pollutants:
            self.logger.write_log('\t\tSpeciating {0}'.format(out_pollutant), message_level=3)
            new_dataframe[out_pollutant] = dataframe.groupby('P_spec').apply(
                lambda x: calculate_new_pollutant(x, out_pollutant))
        new_dataframe.reset_index(inplace=True)

        new_dataframe.drop(columns=['snap', 'geometry'], inplace=True)
        new_dataframe.set_index(['FID', 'tstep'], inplace=True)

        self.logger.write_time_log('SolventsSector', 'speciate', timeit.default_timer() - spent_time)
        return new_dataframe

    def calculate_emissions(self):
        """
        Main function to calculate the emissions.

        :return: Solvent emissions.
        :rtype: DataFrame
        """
        spent_time = timeit.default_timer()
        self.logger.write_log('\tCalculating emissions')

        emissions = self.distribute_yearly_emissions()
        emissions = self.calculate_hourly_emissions(emissions)
        emissions = self.speciate(emissions)

        emissions.reset_index(inplace=True)
        emissions['layer'] = 0
        emissions = emissions.groupby(['FID', 'layer', 'tstep']).sum()

        self.logger.write_time_log('SolventsSector', 'calculate_emissions', timeit.default_timer() - spent_time)
        return emissions
