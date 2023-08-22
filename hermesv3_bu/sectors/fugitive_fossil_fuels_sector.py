#!/usr/bin/env python

import sys
import os
import timeit
import geopandas as gpd
import pandas as pd
import numpy as np
import gc
from warnings import warn
from hermesv3_bu.sectors.sector import Sector
from hermesv3_bu.io_server.io_shapefile import IoShapefile
from hermesv3_bu.io_server.io_raster import IoRaster
from hermesv3_bu.tools.checker import check_files, error_exit
from pandas import DataFrame, MultiIndex
from geopandas import GeoDataFrame
from hermesv3_bu.grids.grid import Grid
from hermesv3_bu.logger.log import Log
from hermesv3_bu.clipping.clip import Clip
from datetime import datetime

PROXY_CODES = {'maritime_terminal':'mar_term',
               'refinery':'refinery',
               'storage_tank':'store_tank',
               'petrol_station':'petrol_st',
               'pipeline':'pipeline',
               'population':'population',
               }

class FugitiveFossilFuelsSector(Sector):
    """
    Fugitive fossil fuels sector allows to calculate the fugitive fossil fuels emissions.

    It first calculates the horizontal distribution for the different sources and store them in an auxiliary file
    during the initialization part.

    Once the initialization is finished it distribute the emissions of the different sub sectors on the grid to start
    the temporal disaggregation.
    """
    def __init__(self, comm, logger, auxiliary_dir, grid, clip, date_array, source_pollutants, vertical_levels,
                 speciation_map_path, molecular_weights_path, speciation_profiles_path, monthly_profile_path,
                 weekly_profile_path, hourly_profile_path, proxies_map_path, yearly_emissions_by_nut2_path, 
                 shapefile_dir, population_raster_path, population_nuts2_path, nut2_shapefile_path):
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

        :param monthly_profile_path: Path to the CSV file that contains all the monthly profiles. The CSV file must
            contain the following columns [P_month, January, February, ..., November, December]
            The P_month code have to match with the proxies_map_path file.
        :type monthly_profile_path: str

        :param weekly_profile_path: Path to the CSV file that contains all the weekly profiles. The CSV file must
            contain the following columns [P_week, Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday]
            The P_week code have to match with the proxies_map_path file.
        :type weekly_profile_path: str

        :param hourly_profile_path: Path to the CSV file that contains all the hourly profiles. The CSV file must
            contain the following columns [P_hour, 0, 1, 2, 3, ..., 22, 23]
            The P_hour code have to match with the proxies_map_path file.
        :type hourly_profile_path: str

        :param proxies_map_path: Path to the CSV file that contains the proxies map.
        :type proxies_map_path: str

        :param yearly_emissions_by_nut2_path: Path to the CSV file that contains the yearly emissions by subsecotr and
            nuts2 level.
        :type yearly_emissions_by_nut2_path: str

        :param shapefile_dir: Dir where to find shapefiles.
        :type shapefile_dir: str

        :param population_raster_path: Path to the population raster.
        :type population_raster_path: str

        :param population_nuts2_path: Path to the CSV file that contains the amount of population for each nut2.
        :type population_nuts2_path: str

        :param nut2_shapefile_path: Path to the shapefile that contains the nut2.
        :type nut2_shapefile_path: str
        """
        spent_time = timeit.default_timer()
        logger.write_log('===== FUGITIVE FOSSIL FUELS SECTOR =====')
        self.active = True

        check_files([speciation_map_path, molecular_weights_path, speciation_profiles_path, monthly_profile_path,
                     weekly_profile_path, hourly_profile_path, proxies_map_path, yearly_emissions_by_nut2_path, 
                     shapefile_dir, population_raster_path, population_nuts2_path, nut2_shapefile_path])

        super(FugitiveFossilFuelsSector, self).__init__(
            comm, logger, auxiliary_dir, grid, clip, date_array, source_pollutants, vertical_levels,
            monthly_profile_path, weekly_profile_path, hourly_profile_path, speciation_map_path,
            speciation_profiles_path, molecular_weights_path)

        self.proxies_map = self.read_proxies(proxies_map_path)
        self.check_profiles()

        try:
            self.proxy = self.get_proxy_shapefile(population_raster_path, population_nuts2_path, 
                                                    nut2_shapefile_path, shapefile_dir)
            if self.proxy.empty:
                self.active = False
        except ValueError as e:
            if "Cannot write empty DataFrame to file." in str(e):
                print("Fugitive Fossil Fuels sector doesn't have data for current domain.")
                self.active = False
            else:
                raise e

        self.yearly_emissions_path = yearly_emissions_by_nut2_path
        self.logger.write_log("FugitiveFossilFuelsSector initialization finished.", message_level=1)
        self.logger.write_time_log('FugitiveFossilFuelsSector', '__init__', timeit.default_timer() - spent_time)

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

        self.logger.write_time_log('FugitiveFossilFuelsSector', 'read_proxies', timeit.default_timer() - spent_time)
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

        self.logger.write_time_log('FugitiveFossilFuelsSector', 'check_profiles', timeit.default_timer() - spent_time)
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
        year_emis = year_emis[year_emis['nuts2_id'].isin(nut_list)]
        # T/year -> g/year
        year_emis['nmvoc'] = year_emis['nmvoc'] * 1000000
        year_emis.set_index(['nuts2_id', 'snap'], inplace=True)
        year_emis.drop(columns=['gnfr_description', 'gnfr', 'snap_description', 'nuts2_na'], inplace=True)

        self.logger.write_time_log('FugitiveFossilFuelsSector', 'read_yearly_emissions', 
                                        timeit.default_timer() - spent_time)
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

        self.logger.write_time_log('FugitiveFossilFuelsSector', 'get_pop_by_nut2', timeit.default_timer() - spent_time)
        return pop_by_nut2

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
                pop_raster_path, self.clip.shapefile, 
                os.path.join(self.auxiliary_dir, 'fugitive_fossil_fuels', 'pop.tif'))

        # 2nd Raster to shapefile
        self.logger.write_log("\t\tRaster to shapefile", message_level=3)
        pop_shp = IoRaster(self.comm).to_shapefile_parallel(
            pop_raster_path, gather=False, bcast=False, crs={'init': 'epsg:4326'})

        # 3rd Add NUT code
        self.logger.write_log("\t\tAdding nut codes to the shapefile", message_level=3)
        # if self.comm.Get_rank() == 0:
        pop_shp.drop(columns='CELL_ID', inplace=True)
        gc.collect()
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
        pop_shp['population'] = pop_shp.geometry.area.astype(np.float32)
        pop_shp = self.spatial_overlays(pop_shp.reset_index(), self.grid.shapefile.reset_index())
        pop_shp.drop(columns=['idx1', 'idx2', 'index'], inplace=True)
        pop_shp['population'] = pop_shp.geometry.area / pop_shp['population']
        pop_shp['population'] = pop_shp['pop_percent'] * pop_shp['population']
        pop_shp.drop(columns=['pop_percent'], inplace=True)

        pop_shp = pop_shp.groupby(['FID', 'nut_code']).sum()

        self.logger.write_time_log('FugitiveFossilFuelsSector', 'get_population_proxy', 
                                    timeit.default_timer() - spent_time)
        self.logger.write_log("\t\tCalculation of population proxy done", message_level=3)
        return pop_shp

    def get_shapefile_for_proxy_code(self, proxy_code, spatial_proxy, shapefile_dir, nut2_shapefile_path):
        """
        Calculate the distribution for the fugitive fossil fuel sub sector in the destination grid cell.

        :param proxy_code: Name of the proxy to be calculated.
        :type proxy_code: str

        :param spatial_proxy: The name of the shapefile for the type of proxy (polygon, point, line)
        :type spatial_proxy: str

        :param shapefile_dir: The directory where to find the shapefile for the spatial_proxy
        :type shapefile_dir: str

        :param nut2_shapefile_path: Path to the shapefile that contains the nut2.
        :type nut2_shapefile_path: str

        :return: GeoDataFrame with the distribution of the selected proxy on the destination grid cells.
        :rtype: GeoDataFrame
        """
        spent_time = timeit.default_timer()

        shapefile_path = os.path.join(shapefile_dir, spatial_proxy + '.shp')

        shapefile = IoShapefile(self.comm).read_shapefile_parallel(shapefile_path)
        shapefile = shapefile[shapefile['proxy_code'].isin([proxy_code])]
        shapefile = IoShapefile(self.comm).balance(shapefile)

        shapefile = shapefile.to_crs(self.grid.shapefile.crs)
        shapefile = gpd.sjoin(shapefile,
                                self.clip.shapefile.to_crs(self.grid.shapefile.crs).reset_index(), how='inner',
                                op='intersects')
        shapefile.drop(columns=['index', 'index_right', 'proxy_code'], inplace=True)
        if 'FID' in shapefile.columns:
            shapefile.drop(columns=['FID'], inplace=True)
        shapefile = IoShapefile(self.comm).balance(shapefile)

        # Translate the actual proxy_code name to a new one with less than 10 characters
        proxy_code = [value for key, value in PROXY_CODES.items() if key == proxy_code][0]

        if "polygon" in spatial_proxy.lower():
            # Each polygon belongs to one single nut code
            shapefile = self.add_nut_code(shapefile, nut2_shapefile_path, nut_value='nuts2_id')
            shapefile = shapefile.loc[shapefile['nut_code'] != -999]
            shapefile = IoShapefile(self.comm).balance(shapefile)
            shapefile[proxy_code] = shapefile.geometry.area
            try:
                # NOTE: avoid communications here
                shapefile = self.spatial_overlays(shapefile.reset_index(), self.grid.shapefile.reset_index())
                shapefile.drop(columns=['idx1', 'idx2', 'index'], inplace=True)
                if 'FID_1' in shapefile.columns and 'FID_2' in shapefile.columns:
                    shapefile.rename(columns={'FID_1': 'FID'}, inplace=True)
                    shapefile.drop(columns=['FID_2'], inplace=True)
                shapefile[proxy_code] = shapefile.geometry.area / shapefile[proxy_code]
                shapefile[proxy_code] = shapefile['weight'] * shapefile[proxy_code]
                shapefile.drop(columns=['Name', 'Location', 'weight'], inplace=True)
            except ValueError:
                # empty shapefile
                shapefile = DataFrame(columns=[proxy_code, 'FID', 'nut_code', 'geometry'])
                #.set_index(['FID', 'nut_code'])
        elif "line" in spatial_proxy.lower():
            nut2_shapefile = gpd.read_file(nut2_shapefile_path)
            nut2_shapefile.rename(columns={'nuts2_id': 'nut_code'}, inplace=True)
            nut2_shapefile = nut2_shapefile.to_crs(self.grid.shapefile.crs)
            shapefile.drop(columns=['FID_Gasodu', 'ORDER01', 'NAME', 'Shape_Leng'], inplace=True)
            # Intersect the lines with the nuts adding the nut code
            shapefile = Sector.line_intersect(shapefile, nut2_shapefile)
            shapefile = shapefile.loc[~shapefile['nut_code'].isna()]
            shapefile.drop(columns=['nuts2_na'], inplace=True)
            shapefile = IoShapefile(self.comm).balance(shapefile)
            shapefile[proxy_code] = shapefile.geometry.length
            # Intersect the lines with the grid cells
            shapefile = Sector.line_intersect(shapefile, self.grid.shapefile.reset_index())
            shapefile[proxy_code] = shapefile.geometry.length / shapefile[proxy_code]
            shapefile[proxy_code] = shapefile['weight'] * shapefile[proxy_code]
            shapefile.drop(columns=['weight'], inplace=True)
        elif "point" in spatial_proxy.lower():
            shapefile.drop(columns=['NAME', 'ORDER01'], inplace=True)
            shapefile = self.add_nut_code(shapefile, nut2_shapefile_path, nut_value='nuts2_id')
            shapefile = shapefile.loc[shapefile['nut_code'] != -999]
            shapefile = IoShapefile(self.comm).balance(shapefile)
            shapefile.rename(columns={'weight': proxy_code}, inplace=True)
            shapefile = gpd.sjoin(shapefile, self.grid.shapefile.reset_index())
            shapefile.drop(columns=['index_right'], inplace=True)
        else:
            error_exit(
                "Wrong spatial_proxy value in fugitive_fossil_fuels_proxies_path file: {0}".format(spatial_proxy))
        
        shapefile.drop(columns=['geometry'], inplace=True)
        
        shapefile = IoShapefile(self.comm).gather_shapefile(shapefile, rank=0)
        if self.comm.Get_rank() == 0:
            shapefile_groupby = shapefile.groupby(['FID', 'nut_code'], as_index=False)
            shapefile = shapefile_groupby.sum()
            shapefile = (shapefile_groupby.count() if shapefile.empty else shapefile).set_index(['FID', 'nut_code'])
            shapefile[proxy_code] = shapefile.groupby(level='nut_code')[proxy_code].transform(lambda x: x / x.sum())
        else:
            shapefile = None
        shapefile = IoShapefile(self.comm).split_shapefile(shapefile)

        self.logger.write_time_log('FugitiveFossilFuelsSector', 'get_shapefile_for_proxy_name', 
                                    timeit.default_timer() - spent_time)
        return shapefile

    def get_proxy_shapefile(self, population_raster_path, population_nuts2_path, nut2_shapefile_path, shapefile_dir):
        """
        Calcualte (or read) the proxy shapefile.

        It will split the entire shapefile into as many processors as selected to split the calculation part.

        :param population_raster_path: Path to the raster file that contains the population information.
        :type population_raster_path: str

        :param population_nuts2_path: Path to the CSV file that contains the amount of population by nut2.
        :type population_nuts2_path: str

        :param nut2_shapefile_path: Path to the shapefile that contains the nut2.
        :type nut2_shapefile_path: str

        :param shapefile_dir: Directory where to find the shapefiles.
        :type shapefile_dir: str

        :return: GeoDataFrame with all the proxies
        :rtype: GeoDataFrame
        """
        spent_time = timeit.default_timer()

        self.logger.write_log("Getting proxies shapefile", message_level=1)
        proxy_path = os.path.join(self.auxiliary_dir, 'fugitive_fossil_fuels', 'proxy_distributions.shp')
        if not os.path.exists(proxy_path):
            proxy_list = []
            for i, row in self.proxies_map.iterrows():
                spatial_proxy = row['spatial_proxy']
                proxy_code = row['proxy_code']
                self.logger.write_log("\tGetting proxy for {0}".format(proxy_code), message_level=2)
                if spatial_proxy == 'population':
                    proxy = self.get_population_proxy(population_raster_path, population_nuts2_path, 
                                                        nut2_shapefile_path)
                else:
                    proxy = self.get_shapefile_for_proxy_code(proxy_code, spatial_proxy, shapefile_dir, 
                                                                nut2_shapefile_path)
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

        self.logger.write_time_log('FugitiveFossilFuelsSector', 'get_proxy_shapefile', 
                                    timeit.default_timer() - spent_time)
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

        self.logger.write_time_log('FugitiveFossilFuelsSector', 'calculate_hourly_emissions', 
                                    timeit.default_timer() - spent_time)
        return emissions

    def distribute_yearly_emissions(self):
        """
        Calculate the yearly emission by destiny grid cell and snap code.

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
                [value for key, value in PROXY_CODES.items() if key == self.proxies_map.loc[snap, 'proxy_code']][0]], axis=1)

            emis.set_index(['FID', 'snap'], inplace=True)
            emis_list.append(emis[['P_month', 'P_week', 'P_hour', 'P_spec', 'nmvoc', 'geometry', 'timezone']])
        emis = pd.concat(emis_list).sort_index()
        emis = emis[emis['nmvoc'] > 0]

        self.logger.write_time_log('FugitiveFossilFuelsSector', 'distribute_yearly_emissions', 
                                    timeit.default_timer() - spent_time)
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

        self.logger.write_time_log('FugitiveFossilFuelsSector', 'speciate', timeit.default_timer() - spent_time)
        return new_dataframe

    def calculate_emissions(self):
        """
        Main function to calculate the emissions.

        :return: Solvent emissions.
        :rtype: DataFrame
        """
        if not self.active:
            msg = "Fugitive Fossil Fuels sector doesn't have data for current domain. "
            msg += "Please, disable it from the config file."
            error_exit(msg)

        spent_time = timeit.default_timer()
        self.logger.write_log('\tCalculating emissions')

        emissions = self.distribute_yearly_emissions()
        emissions = self.calculate_hourly_emissions(emissions)
        emissions = self.speciate(emissions)

        emissions.reset_index(inplace=True)
        emissions['layer'] = 0
        emissions = emissions.groupby(['FID', 'layer', 'tstep']).sum()

        self.logger.write_log('\Fugitive fossil fuels emissions calculated', message_level=2)
        self.logger.write_time_log('FugitiveFossilFuelsSector', 'calculate_emissions', 
                                    timeit.default_timer() - spent_time)
        return emissions
