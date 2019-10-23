#!/usr/bin/env python
import sys
import os
import timeit

import pandas as pd
from pandas import DataFrame
import geopandas as gpd
from geopandas import GeoDataFrame
import numpy as np
from datetime import timedelta
import warnings
from hermesv3_bu.logger.log import Log
from hermesv3_bu.sectors.sector import Sector
from hermesv3_bu.io_server.io_netcdf import IoNetcdf
from hermesv3_bu.tools.checker import check_files, error_exit

from ctypes import cdll, CDLL
cdll.LoadLibrary("libc.so.6")
libc = CDLL("libc.so.6")
libc.malloc_trim(0)

MIN_RAIN = 0.254  # After USEPA (2011)
RECOVERY_RATIO = 0.0872  # After Amato et al. (2012)
FINAL_PROJ = {'init': 'epsg:3035'}  # https://epsg.io/3035 ETRS89 / LAEA Europe


aerosols = ['oc', 'ec', 'pno3', 'pso4', 'pmfine', 'pmc', 'poa', 'poc', 'pec', 'pcl', 'pnh4', 'pna', 'pmg', 'pk', 'pca',
            'pncom', 'pfe', 'pal', 'psi', 'pti', 'pmn', 'ph2o', 'pmothr']
pmc_list = ['pmc', 'PMC']
rline_shp = False


class TrafficSector(Sector):
    # TODO MARC -> to revise these descriptions
    """
    The traffic class does have all the necessary functions to calculate the traffic emission in bottom-up mode.

    Part of the traffic emissions are calculated by roadlink (hot, cold, road wear, tyre wear, brake wear and
        resuspension) differentiating by vehicle type.
        The other emissions (other cities and evaporative) are calculated by cell instead of by road link.

    To calculate the traffic emissions some input files are needed as the shapefile that contains the information and
        geolocalization of each road link, the temporal proxies, the emission factors files and also the information
        relative to the timesteps.
    """

    def __init__(self, comm, logger, auxiliary_dir, grid, clip, date_array, source_pollutants, vertical_levels,
                 road_link_path, fleet_compo_path, speed_hourly_path, monthly_profiles_path, weekly_profiles_path,
                 hourly_mean_profiles_path, hourly_weekday_profiles_path, hourly_saturday_profiles_path,
                 hourly_sunday_profiles_path, ef_common_path, vehicle_list=None, load=0.5, speciation_map_path=None,
                 hot_cold_speciation=None, tyre_speciation=None, road_speciation=None, brake_speciation=None,
                 resuspension_speciation=None, temp_common_path=None, output_dir=None, molecular_weights_path=None,
                 resuspension_correction=True, precipitation_path=None, do_hot=True, do_cold=True, do_tyre_wear=True,
                 do_brake_wear=True, do_road_wear=True, do_resuspension=True, write_rline=False):

        spent_time = timeit.default_timer()
        logger.write_log('===== TRAFFIC SECTOR =====')
        if do_hot:
            check_files(
                [road_link_path, fleet_compo_path, speed_hourly_path, monthly_profiles_path, weekly_profiles_path,
                 hourly_mean_profiles_path, hourly_weekday_profiles_path, hourly_saturday_profiles_path,
                 hourly_sunday_profiles_path, speciation_map_path, molecular_weights_path, hot_cold_speciation] +
                [os.path.join(ef_common_path, "hot_{0}.csv".format(pol)) for pol in source_pollutants])
        if do_cold:
            check_files(
                [road_link_path, fleet_compo_path, speed_hourly_path, monthly_profiles_path, weekly_profiles_path,
                 hourly_mean_profiles_path, hourly_weekday_profiles_path, hourly_saturday_profiles_path,
                 hourly_sunday_profiles_path, speciation_map_path, molecular_weights_path, hot_cold_speciation,
                 temp_common_path] +
                [os.path.join(ef_common_path, "cold_{0}.csv".format(pol)) for pol in source_pollutants])
        if do_tyre_wear:
            check_files(
                [road_link_path, fleet_compo_path, speed_hourly_path, monthly_profiles_path, weekly_profiles_path,
                 hourly_mean_profiles_path, hourly_weekday_profiles_path, hourly_saturday_profiles_path,
                 hourly_sunday_profiles_path, speciation_map_path, molecular_weights_path, tyre_speciation] +
                [os.path.join(ef_common_path, "tyre_{0}.csv".format(pol)) for pol in ['pm']])
        if do_road_wear:
            check_files(
                [road_link_path, fleet_compo_path, speed_hourly_path, monthly_profiles_path, weekly_profiles_path,
                 hourly_mean_profiles_path, hourly_weekday_profiles_path, hourly_saturday_profiles_path,
                 hourly_sunday_profiles_path, speciation_map_path, molecular_weights_path, road_speciation] +
                [os.path.join(ef_common_path, "road_{0}.csv".format(pol)) for pol in ['pm']])
        if do_brake_wear:
            check_files(
                [road_link_path, fleet_compo_path, speed_hourly_path, monthly_profiles_path, weekly_profiles_path,
                 hourly_mean_profiles_path, hourly_weekday_profiles_path, hourly_saturday_profiles_path,
                 hourly_sunday_profiles_path, speciation_map_path, molecular_weights_path, brake_speciation] +
                [os.path.join(ef_common_path, "brake_{0}.csv".format(pol)) for pol in ['pm']])
        if do_resuspension:
            check_files(
                [road_link_path, fleet_compo_path, speed_hourly_path, monthly_profiles_path, weekly_profiles_path,
                 hourly_mean_profiles_path, hourly_weekday_profiles_path, hourly_saturday_profiles_path,
                 hourly_sunday_profiles_path, speciation_map_path, molecular_weights_path, resuspension_speciation] +
                [os.path.join(ef_common_path, "resuspension_{0}.csv".format(pol)) for pol in ['pm']])
            if resuspension_correction:
                check_files(precipitation_path)
        super(TrafficSector, self).__init__(
            comm, logger, auxiliary_dir, grid, clip, date_array, source_pollutants, vertical_levels,
            monthly_profiles_path, weekly_profiles_path, None, speciation_map_path, None, molecular_weights_path)

        self.resuspension_correction = resuspension_correction
        self.precipitation_path = precipitation_path

        self.output_dir = output_dir

        self.link_to_grid_csv = os.path.join(auxiliary_dir, 'traffic', 'link_grid.csv')
        if self.__comm.Get_rank() == 0:
            if not os.path.exists(os.path.dirname(self.link_to_grid_csv)):
                os.makedirs(os.path.dirname(self.link_to_grid_csv))
        self.__comm.Barrier()
        self.crs = None   # crs is the projection of the road links and it is set on the read_road_links function.
        self.write_rline = write_rline
        self.road_links = self.read_road_links(road_link_path)

        self.load = load
        self.ef_common_path = ef_common_path
        self.temp_common_path = temp_common_path

        self.add_local_date(self.date_array[0])

        self.hot_cold_speciation = hot_cold_speciation
        self.tyre_speciation = tyre_speciation
        self.road_speciation = road_speciation
        self.brake_speciation = brake_speciation
        self.resuspension_speciation = resuspension_speciation

        self.fleet_compo = self.read_fleet_compo(fleet_compo_path, vehicle_list)
        self.speed_hourly = self.read_speed_hourly(speed_hourly_path)

        self.hourly_profiles = self.read_all_hourly_profiles(hourly_mean_profiles_path, hourly_weekday_profiles_path,
                                                             hourly_saturday_profiles_path, hourly_sunday_profiles_path)
        self.check_profiles()
        self.expanded = self.expand_road_links()

        del self.fleet_compo, self.speed_hourly, self.monthly_profiles, self.weekly_profiles, self.hourly_profiles

        self.do_hot = do_hot
        self.do_cold = do_cold
        self.do_tyre_wear = do_tyre_wear
        self.do_brake_wear = do_brake_wear
        self.do_road_wear = do_road_wear
        self.do_resuspension = do_resuspension

        self.__logger.write_time_log('TrafficSector', '__init__', timeit.default_timer() - spent_time)

    def check_profiles(self):
        spent_time = timeit.default_timer()
        # Checking speed profiles IDs
        links_speed = set(np.unique(np.concatenate([
            np.unique(self.road_links['sp_hour_su'].dropna().values),
            np.unique(self.road_links['sp_hour_mo'].dropna().values),
            np.unique(self.road_links['sp_hour_tu'].dropna().values),
            np.unique(self.road_links['sp_hour_we'].dropna().values),
            np.unique(self.road_links['sp_hour_th'].dropna().values),
            np.unique(self.road_links['sp_hour_fr'].dropna().values),
            np.unique(self.road_links['sp_hour_sa'].dropna().values),
        ])))
        # The '0' speed profile means that we don't know the speed profile and it will be replaced by a flat profile
        speed = set(np.unique(np.concatenate([self.speed_hourly.index.values, [0]])))

        speed_res = links_speed - speed
        if len(speed_res) > 0:
            error_exit("The following speed profile IDs reported in the road links shapefile do not appear " +
                       "in the hourly speed profiles file. {0}".format(speed_res))

        # Checking monthly profiles IDs
        links_month = set(np.unique(self.road_links['aadt_m_mn'].dropna().values))
        month = set(self.monthly_profiles.index.values)
        month_res = links_month - month
        if len(month_res) > 0:
            error_exit("The following monthly profile IDs reported in the road links shapefile do not appear " +
                       "in the monthly profiles file. {0}".format(month_res))

        # Checking weekly profiles IDs
        links_week = set(np.unique(self.road_links['aadt_week'].dropna().values))
        week = set(self.weekly_profiles.index.values)
        week_res = links_week - week
        if len(week_res) > 0:
            error_exit("The following weekly profile IDs reported in the road links shapefile do not appear " +
                       "in the weekly profiles file. {0}".format(week_res))

        # Checking hourly profiles IDs
        links_hour = set(np.unique(np.concatenate([
            np.unique(self.road_links['aadt_h_mn'].dropna().values),
            np.unique(self.road_links['aadt_h_wd'].dropna().values),
            np.unique(self.road_links['aadt_h_sat'].dropna().values),
            np.unique(self.road_links['aadt_h_sun'].dropna().values),
        ])))
        hour = set(self.hourly_profiles.index.values)
        hour_res = links_hour - hour
        if len(hour_res) > 0:
            error_exit("The following hourly profile IDs reported in the road links shapefile do not appear " +
                       "in the hourly profiles file. {0}".format(hour_res))

        self.__logger.write_time_log('TrafficSector', 'check_profiles', timeit.default_timer() - spent_time)

    def read_all_hourly_profiles(self, hourly_mean_profiles_path, hourly_weekday_profiles_path,
                                 hourly_saturday_profiles_path, hourly_sunday_profiles_path):
        hourly_profiles = pd.concat([self.read_hourly_profiles(hourly_mean_profiles_path),
                                    self.read_hourly_profiles(hourly_weekday_profiles_path),
                                    self.read_hourly_profiles(hourly_saturday_profiles_path),
                                    self.read_hourly_profiles(hourly_sunday_profiles_path)])
        hourly_profiles.index = hourly_profiles.index.astype(str)
        return hourly_profiles

    def read_speciation_map(self, path):
        """
        Read the speciation map.

        The speciation map is the CSV file that contains the relation from the output pollutant and the correspondent
        input pollutant associated. That file also contains a short description of the output pollutant and the units to
        be stored.

        e.g.:
        dst,src,description,units
        NOx,nox_no2,desc_no,mol.s-1
        SOx,so2,desc_so2,mol.s-1
        CO,co,desc_co,mol.s-1
        CO2,co2,desc_co2,mol.s-1
        NMVOC,nmvoc,desc_nmvoc,g.s-1
        PM10,pm10,desc_pm10,g.s-1
        PM25,pm25,desc_pm25,g.s-1
        PMC,,desc_pmc,g.s-1

        :param path: Path to the speciation map file.
        :type path: str

        :return: Dictionary with the output pollutant as key and the input pollutant as value.
        :rtype: dict
        """
        spent_time = timeit.default_timer()
        speciation_map = pd.read_csv(path)
        dataframe = pd.read_csv(path)
        # input_pollutants = list(self.source_pollutants)
        input_pollutants = ['nmvoc' if x == 'voc' else x for x in list(self.source_pollutants)]
        if 'PMC' in dataframe['dst'].values and all(element in input_pollutants for element in ['pm']):
            dataframe_aux = dataframe.loc[dataframe['src'].isin(input_pollutants), :]
            dataframe = pd.concat([dataframe_aux, dataframe.loc[dataframe['dst'] == 'PMC', :]])
        else:
            dataframe = dataframe.loc[dataframe['src'].isin(input_pollutants), :]

        dataframe = dict(zip(dataframe['dst'], dataframe['src']))
        if 'pm' in self.source_pollutants:
            for out_p, in_p in zip(speciation_map[['dst']].values, speciation_map[['src']].values):
                if in_p in ['pm10', 'pm25']:
                    dataframe[out_p[0]] = in_p[0]
        # if 'pm' in self.source_pollutants and 'PM10' in speciation_map[['dst']].values:
        #     dataframe['PM10'] = 'pm10'
        # if 'pm' in self.source_pollutants and 'PM25' in speciation_map[['dst']].values:
        #     dataframe['PM25'] = 'pm25'
        self.__logger.write_time_log('TrafficSector', 'read_speciation_map', timeit.default_timer() - spent_time)

        return dataframe

    def add_local_date(self, utc_date):
        """
        Adds to the road links the starting date in local time.
        This new column is called 'start_date'.

        :param utc_date: Starting date in UTC.
        """
        import pytz
        spent_time = timeit.default_timer()

        self.add_timezones()
        self.road_links.loc[:, 'utc'] = utc_date
        self.road_links['start_date'] = self.road_links.groupby('timezone')['utc'].apply(
            lambda x: pd.to_datetime(x).dt.tz_localize(pytz.utc).dt.tz_convert(x.name).dt.tz_localize(None))

        self.road_links.drop(columns=['utc', 'timezone'], inplace=True)
        libc.malloc_trim(0)

        self.__logger.write_time_log('TrafficSector', 'add_local_date', timeit.default_timer() - spent_time)
        return True

    def add_timezones(self):
        """
        Finds and sets the timezone for each road link.
        """
        spent_time = timeit.default_timer()
        # TODO calculate timezone from the centroid of each roadlink.

        self.road_links['timezone'] = 'Europe/Madrid'

        self.__logger.write_time_log('TrafficSector', 'add_timezones', timeit.default_timer() - spent_time)
        return True

    def read_speed_hourly(self, path):
        # TODO complete description
        """
        Reads the speed hourly file.

        :param path: Path to the speed hourly file.
        :type path: str:

        :return: ...
        :rtype: DataFrame
        """
        spent_time = timeit.default_timer()

        df = pd.read_csv(path, sep=',', dtype=np.float32)
        df['P_speed'] = df['P_speed'].astype(int)
        df.set_index('P_speed', inplace=True)

        self.__logger.write_time_log('TrafficSector', 'read_speed_hourly', timeit.default_timer() - spent_time)
        return df

    def read_fleet_compo(self, path, vehicle_list):
        spent_time = timeit.default_timer()
        df = pd.read_csv(path, sep=',')
        if vehicle_list is not None:
            df = df.loc[df['Code'].isin(vehicle_list), :]
        self.__logger.write_time_log('TrafficSector', 'read_fleet_compo', timeit.default_timer() - spent_time)
        return df

    def read_road_links(self, path):
        def chunk_road_links(df, nprocs):
            def index_marks(nrows, nprocs):
                max_len = int(nrows // nprocs) + 1
                min_len = max_len - 1
                max_num = nrows % nprocs
                min_num = nprocs - max_num
                index_list = []
                prev = 0
                for i in range(max_num):
                    prev += max_len
                    index_list.append(prev)
                if min_num > 0:
                    for i in range(min_num - 1):
                        prev += min_len
                        index_list.append(prev)

                return index_list

            def split(dfm, nprocs):
                indices = index_marks(dfm.shape[0], nprocs)
                return np.split(dfm, indices)

            chunks_aux = split(df, nprocs)
            return chunks_aux

        spent_time = timeit.default_timer()

        if self.__comm.Get_rank() == 0:
            df = gpd.read_file(path)
            try:
                df.drop(columns=['Adminis', 'CCAA', 'NETWORK_ID', 'Province', 'Road_name', 'aadt_m_sat', 'aadt_m_sun',
                                 'aadt_m_wd', 'Source'], inplace=True)
            except KeyError as e:
                error_exit(str(e).replace('axis', 'the road links shapefile'))
            libc.malloc_trim(0)
            # df.to_file('~/temp/road_links.shp')
            df = gpd.sjoin(df, self.clip.shapefile.to_crs(df.crs), how="inner", op='intersects')
            # df.to_file('~/temp/road_links_selected.shp')
            df.drop(columns=['index_right'], inplace=True)
            libc.malloc_trim(0)

            # Filtering road links to CONSiderate.
            df['CONS'] = df['CONS'].astype(np.int16)
            df = df[df['CONS'] != 0]
            df = df[df['aadt'] > 0]

            try:
                df.drop(columns=['CONS'], inplace=True)
            except KeyError as e:
                error_exit(str(e).replace('axis', 'the road links shapefile'))

            df = df.loc[df['aadt_m_mn'] != 'NULL', :]
            libc.malloc_trim(0)

            # Adding identificator of road link
            df['Link_ID'] = range(len(df))
            df.set_index('Link_ID', inplace=True)
            libc.malloc_trim(0)

            chunks = chunk_road_links(df, self.__comm.Get_size())
        else:
            chunks = None
        self.__comm.Barrier()

        df = self.__comm.scatter(chunks, root=0)

        del chunks
        libc.malloc_trim(0)
        df = df.to_crs({'init': 'epsg:4326'})

        self.crs = df.crs

        # Correcting percentages
        df['PcMoto'] = df['PcMoto'] / 100
        df['PcHeavy'] = df['PcHeavy'] / 100
        df['PcMoped'] = df['PcMoped'] / 100
        df['PcLight'] = 1 - (df['PcMoto'] + df['PcHeavy'] + df['PcMoped'])

        # Road_type int to string
        df['Road_type'] = df['Road_type'].astype(str)
        df.loc[df['Road_type'] == '0', 'Road_type'] = 'Highway'
        df.loc[df['Road_type'] == '1', 'Road_type'] = 'Rural'
        df.loc[df['Road_type'] == '2', 'Road_type'] = 'Urban Off Peak'
        df.loc[df['Road_type'] == '3', 'Road_type'] = 'Urban Peak'

        df['road_grad'] = df['road_grad'].astype(np.float32)

        # Check if percents are ok
        if len(df[df['PcLight'] < 0]) is not 0:
            error_exit('PcLight < 0')

        if self.write_rline:
            self.write_rline_roadlinks(df)

        self.__logger.write_time_log('TrafficSector', 'read_road_links', timeit.default_timer() - spent_time)
        libc.malloc_trim(0)

        return df

    def read_ef(self, emission_type, pollutant_name):
        """
        Reads the file that contains the necessary emission factor for the current pollutant and emission type.

        Depending on the emission tyme the file contain different columns.

        :param emission_type: Type of the emission. It can be hot, cold, tyre, road, brake or resuspension.
        :type emission_type: str

        :param pollutant_name: Name of the pollutant to read their emission factor.
        :type pollutant_name:str

        :return: Returns the readed emission factor in DataFrame mode.
        :rtype: Pandas.DataFrame
        """
        spent_time = timeit.default_timer()

        ef_path = os.path.join(self.ef_common_path, '{0}_{1}.csv'.format(emission_type, pollutant_name))
        df = self.read_profiles(ef_path)

        # Pollutants different to NH3
        if pollutant_name != 'nh3':
            try:
                df.drop(columns=['Copert_V_name'], inplace=True)
            except KeyError as e:
                error_exit(str(e).replace('axis', 'the {0} file'.format(ef_path)))

            # For hot emission factors
            if emission_type == 'hot':
                df = df[(df['Load'] == self.load) | (df['Load'].isnull())]

                df.loc[df['Technology'].isnull(), 'Technology'] = ''
                df = df[df['Technology'] != 'EGR']
                try:
                    df.drop(columns=['Technology', 'Load'], inplace=True)
                except KeyError as e:
                    error_exit(str(e).replace('axis', 'the {0} file'.format(ef_path)))

                # Split the EF file into small DataFrames divided by column Road.Slope and Mode restrictions.
                df_code_slope_road = df[df['Road.Slope'].notnull() & df['Mode'].notnull()]
                df_code_slope = df[df['Road.Slope'].notnull() & (df['Mode'].isnull())]
                df_code_road = df[df['Road.Slope'].isnull() & (df['Mode'].notnull())]
                df_code = df[df['Road.Slope'].isnull() & (df['Mode'].isnull())]

                # Checks that the splited DataFrames contain the full DataFrame
                if (len(df_code_slope_road) + len(df_code_slope) + len(df_code_road) + len(df_code)) != len(df):
                    # TODO check that error
                    error_exit('ERROR in blablavbla')

                return df_code_slope_road, df_code_slope, df_code_road, df_code
            elif emission_type == 'cold' or emission_type == 'tyre' or emission_type == 'road' or \
                    emission_type == 'brake' or emission_type == 'resuspension':
                return df
        # NH3 pollutant
        else:
            try:
                df.drop(columns=['Copert_V_name'], inplace=True)
            except KeyError as e:
                error_exit(str(e).replace('axis', 'the {0} file'.format(ef_path)))
            # Specific case for cold NH3 emission factors that needs the hot emission factors and the cold ones.
            if emission_type == 'cold':
                df_hot = self.read_ef('hot', pollutant_name)
                df_hot.columns = [x + '_hot' for x in df_hot.columns.values]

                df = df.merge(df_hot, left_on=['Code', 'Mode'], right_on=['Code_hot', 'Mode_hot'],
                              how='left')
                try:
                    df.drop(columns=['Cmileage_hot', 'Mode_hot', 'Code_hot'], inplace=True)
                except KeyError as e:
                    error_exit(str(e).replace('axis', 'the {0} file'.format(ef_path)))

            return df

        self.__logger.write_time_log('TrafficSector', 'read_ef', timeit.default_timer() - spent_time)
        return None

    def read_mcorr_file(self, pollutant_name):
        spent_time = timeit.default_timer()
        try:
            df_path = os.path.join(self.ef_common_path, 'mcorr_{0}.csv'.format(pollutant_name))

            df = pd.read_csv(df_path, sep=',')
            if 'Copert_V_name' in list(df.columns.values):
                df.drop(columns=['Copert_V_name'], inplace=True)
        except IOError:
            self.__logger.write_log('WARNING! No mileage correction applied to {0}'.format(pollutant_name))
            warnings.warn('No mileage correction applied to {0}'.format(pollutant_name))
            df = None

        self.__logger.write_time_log('TrafficSector', 'read_ef', timeit.default_timer() - spent_time)
        return df

    def calculate_precipitation_factor(self, lon_min, lon_max, lat_min, lat_max, precipitation_dir):
        spent_time = timeit.default_timer()

        dates_to_extract = [self.date_array[0] + timedelta(hours=x - 47) for x in range(47)] + self.date_array

        precipitation = IoNetcdf(self.__comm).get_hourly_data_from_netcdf(
            lon_min, lon_max, lat_min, lat_max, precipitation_dir, 'prlr', dates_to_extract)

        precipitation.set_index('REC', inplace=True, drop=True)

        prlr = precipitation.drop(columns='geometry').values.T

        # From m/s to mm/h
        prlr = prlr * (3600 * 1000)
        prlr = prlr <= MIN_RAIN
        dst = np.empty(prlr.shape)
        last = np.zeros((prlr.shape[-1]))
        for time in range(prlr.shape[0]):
            dst[time, :] = (last + prlr[time, :]) * prlr[time, :]
            last = dst[time, :]

        dst = dst[47:, :]
        dst = 1 - np.exp(- RECOVERY_RATIO * dst)
        # It is assumed that after 48 h without rain the potential emission is equal to one
        dst[dst >= (1 - np.exp(- RECOVERY_RATIO * 48))] = 1.
        # Creates the GeoDataFrame
        df = gpd.GeoDataFrame(dst.T, geometry=precipitation['geometry'].values)
        df.columns = ['PR_{0}'.format(x) for x in df.columns.values[:-1]] + ['geometry']

        df.loc[:, 'REC'] = df.index

        self.__logger.write_time_log('TrafficSector', 'calculate_precipitation_factor',
                                     timeit.default_timer() - spent_time)
        return df

    def update_fleet_value(self, df):
        spent_time = timeit.default_timer()

        def update_by_class(x):
            if x.name == 'light_veh':
                x['value'] = x['PcLight'].mul(x['Fleet_value'] * x['aadt'], axis='index')
            elif x.name == 'heavy_veh':
                x['value'] = x['PcHeavy'].mul(x['Fleet_value'] * x['aadt'], axis='index')
            elif x.name == 'motos':
                x['value'] = x['PcMoto'].mul(x['Fleet_value'] * x['aadt'], axis='index')
            elif x.name == 'mopeds':
                x['value'] = x['PcMoped'].mul(x['Fleet_value'] * x['aadt'], axis='index')
            else:
                x['value'] = np.nan
            return x[['value']]

        df['Fleet_value'] = df.groupby('Fleet_Class').apply(update_by_class)
        for link_id, aux_df in df.groupby('Link_ID'):
            aadt = round(aux_df['aadt'].min(), 1)
            fleet_value = round(aux_df['Fleet_value'].sum(), 1)
            if aadt != fleet_value:
                self.__logger.write_log('link_ID: {0} aadt: {1} sum_fleet: {2}'.format(link_id, aadt, fleet_value),
                                        message_level=2)

        # Drop 0 values
        df = df[df['Fleet_value'] > 0]

        # Deleting unused columns
        try:
            df.drop(columns=['aadt', 'PcLight', 'PcHeavy', 'PcMoto', 'PcMoped', 'Fleet_Class'], inplace=True)
        except KeyError as e:
            error_exit(str(e).replace('axis', 'the road links shapefile'))
        libc.malloc_trim(0)

        self.__logger.write_time_log('TrafficSector', 'update_fleet_value', timeit.default_timer() - spent_time)
        return df

    def calculate_time_dependent_values(self, df):
        spent_time = timeit.default_timer()

        def get_weekday_speed_profile(x):
            # Spead mean
            if x.name <= 4:
                x['speed_mean'] = df['sp_wd']
            else:
                x['speed_mean'] = df['sp_we']

            # Profile_ID
            if x.name == 0:
                x['P_speed'] = x['sp_hour_mo']
            elif x.name == 1:
                x['P_speed'] = x['sp_hour_tu']
            elif x.name == 2:
                x['P_speed'] = x['sp_hour_we']
            elif x.name == 3:
                x['P_speed'] = x['sp_hour_th']
            elif x.name == 4:
                x['P_speed'] = x['sp_hour_fr']
            elif x.name == 5:
                x['P_speed'] = x['sp_hour_sa']
            elif x.name == 6:
                x['P_speed'] = x['sp_hour_su']
            else:
                x['P_speed'] = 1  # Flat profile

            # Flat profile
            x['P_speed'].replace([0, np.nan], 1, inplace=True)
            x['P_speed'] = x['P_speed'].astype(int)

            return x[['speed_mean', 'P_speed']]

        def get_velocity(x):
            speed = self.speed_hourly.loc[np.unique(x['P_speed'].values), :]

            x = pd.merge(x, speed, left_on='P_speed', right_index=True, how='left')
            x['speed'] = x.groupby('hour').apply(lambda y: x[[str(y.name)]])

            return x['speed'] * x['speed_mean']

        def get_temporal_factor(x):
            def get_hourly_id_from_weekday(weekday):
                if weekday <= 4:
                    return 'aadt_h_wd'
                elif weekday == 5:
                    return 'aadt_h_sat'
                elif weekday == 6:
                    return 'aadt_h_sun'
                else:
                    error_exit('Weekday not found')

            # Monthly factor
            x = pd.merge(x, self.monthly_profiles, left_on='aadt_m_mn', right_index=True, how='left')
            x['MF'] = x.groupby('month').apply(lambda y: x[[y.name]])
            x.drop(columns=range(1, 12 + 1), inplace=True)

            # Daily factor
            x = pd.merge(x, self.weekly_profiles, left_on='aadt_week', right_index=True, how='left')
            x['WF'] = x.groupby('weekday').apply(lambda y: x[[y.name]])
            x.drop(columns=range(0, 7), inplace=True)

            # Hourly factor
            x.fillna(value=pd.np.nan, inplace=True)
            x['hourly_profile'] = x.groupby('weekday').apply(lambda y: x[[get_hourly_id_from_weekday(y.name)]])
            x['hourly_profile'].fillna(x['aadt_h_mn'], inplace=True)

            x = pd.merge(x, self.hourly_profiles, left_on='hourly_profile', right_index=True, how='left')
            x['HF'] = x.groupby('hour').apply(lambda y: x[[y.name]])
            x.drop(columns=range(0, 24), inplace=True)
            x['factor'] = x['MF'] * x['WF'] * x['HF']

            return x[['factor']]

        for i_t, tstep in enumerate(self.date_array):
            df['aux_date'] = df['start_date'] + (tstep - self.date_array[0])
            df['month'] = df['aux_date'].dt.month
            df['weekday'] = df['aux_date'].dt.weekday
            df['hour'] = df['aux_date'].dt.hour

            df[['speed_mean', 'P_speed']] = df.groupby('weekday').apply(get_weekday_speed_profile)

            df['v_{0}'.format(i_t)] = get_velocity(df[['hour', 'speed_mean', 'P_speed']])
            df['f_{0}'.format(i_t)] = get_temporal_factor(
                df[['month', 'weekday', 'hour', 'aadt_m_mn', 'aadt_week', 'aadt_h_mn', 'aadt_h_wd', 'aadt_h_sat',
                    'aadt_h_sun']])

        try:
            df.drop(columns=['month', 'weekday', 'hour', 'P_speed', 'speed_mean', 'sp_wd', 'sp_we', 'sp_hour_mo',
                             'sp_hour_tu', 'sp_hour_we', 'sp_hour_th', 'sp_hour_fr', 'sp_hour_sa', 'sp_hour_su',
                             'aux_date', 'aadt_m_mn', 'aadt_h_mn', 'aadt_h_wd', 'aadt_h_sat', 'aadt_h_sun', 'aadt_week',
                             'start_date'], inplace=True)
        except KeyError as e:
            error_exit(str(e).replace('axis', 'the road links shapefile'))
        libc.malloc_trim(0)

        self.__logger.write_time_log('TrafficSector', 'calculate_time_dependent_values',
                                     timeit.default_timer() - spent_time)
        return df

    def expand_road_links(self):
        spent_time = timeit.default_timer()

        # Expands each road link by any vehicle type that the selected road link has.
        df_list = []
        road_link_aux = self.road_links.copy().reset_index()

        road_link_aux.drop(columns='geometry', inplace=True)
        libc.malloc_trim(0)

        for zone, compo_df in road_link_aux.groupby('fleet_comp'):
            fleet = self.find_fleet(zone)
            df_aux = pd.merge(compo_df, fleet, how='left', on='fleet_comp')
            df_aux.drop(columns='fleet_comp', inplace=True)
            df_list.append(df_aux)

        df = pd.concat(df_list, ignore_index=True)

        df.set_index(['Link_ID', 'Fleet_Code'], inplace=True)
        libc.malloc_trim(0)

        df = self.update_fleet_value(df)
        df = self.calculate_time_dependent_values(df)

        self.__logger.write_time_log('TrafficSector', 'expand_road_links', timeit.default_timer() - spent_time)

        return df

    def find_fleet(self, zone):
        spent_time = timeit.default_timer()

        try:
            fleet = self.fleet_compo[['Code', 'Class', zone]]
        except KeyError as e:
            error_exit(e.message + ' of the fleet_compo file')
        fleet.columns = ['Fleet_Code', 'Fleet_Class', 'Fleet_value']

        fleet = fleet[fleet['Fleet_value'] > 0]

        fleet['fleet_comp'] = zone

        self.__logger.write_time_log('TrafficSector', 'find_fleet', timeit.default_timer() - spent_time)

        return fleet

    def calculate_hot(self):
        spent_time = timeit.default_timer()

        expanded_aux = self.expanded.copy().reset_index()

        for pollutant in self.source_pollutants:
            if pollutant != 'nh3':

                ef_code_slope_road, ef_code_slope, ef_code_road, ef_code = self.read_ef('hot', pollutant)

                df_code_slope_road = expanded_aux.merge(
                    ef_code_slope_road, left_on=['Fleet_Code', 'road_grad', 'Road_type'],
                    right_on=['Code', 'Road.Slope', 'Mode'], how='inner')
                df_code_slope = expanded_aux.merge(ef_code_slope, left_on=['Fleet_Code', 'road_grad'],
                                                   right_on=['Code', 'Road.Slope'], how='inner')
                df_code_road = expanded_aux.merge(ef_code_road, left_on=['Fleet_Code', 'Road_type'],
                                                  right_on=['Code', 'Mode'], how='inner')
                df_code = expanded_aux.merge(ef_code, left_on=['Fleet_Code'], right_on=['Code'], how='inner')

                del ef_code_slope_road, ef_code_slope, ef_code_road, ef_code

                expanded_aux = pd.concat([df_code_slope_road, df_code_slope, df_code_road, df_code])

                expanded_aux.drop(columns=['Code', 'Road.Slope', 'Mode'], inplace=True)
            else:
                ef_code_road = self.read_ef('hot', pollutant)
                expanded_aux = expanded_aux.merge(ef_code_road, left_on=['Fleet_Code', 'Road_type'],
                                                  right_on=['Code', 'Mode'], how='inner')

                expanded_aux.drop(columns=['Code', 'Mode'], inplace=True)

            # Warnings and Errors
            original_ef_profile = np.unique(self.expanded.index.get_level_values('Fleet_Code'))
            calculated_ef_profiles = np.unique(expanded_aux['Fleet_Code'])
            resta_1 = [item for item in original_ef_profile if item not in calculated_ef_profiles]  # Warining
            resta_2 = [item for item in calculated_ef_profiles if item not in original_ef_profile]  # Error

            if len(resta_1) > 0:
                self.__logger.write_log('WARNING! Exists some fleet codes that not appear on the EF file: {0}'.format(
                    resta_1))
                warnings.warn('Exists some fleet codes that not appear on the EF file: {0}'.format(resta_1), Warning)
            if len(resta_2) > 0:
                error_exit('Exists some fleet codes duplicated on the EF file: {0}'.format(resta_2))

            m_corr = self.read_mcorr_file(pollutant)
            if m_corr is not None:
                expanded_aux = expanded_aux.merge(m_corr, left_on='Fleet_Code', right_on='Code', how='left')
                expanded_aux.drop(columns=['Code'], inplace=True)

            for tstep in range(len(self.date_array)):
                ef_name = 'ef_{0}_{1}'.format(pollutant, tstep)
                p_column = '{0}_{1}'.format(pollutant, tstep)
                if pollutant != 'nh3':
                    expanded_aux['v_aux'] = expanded_aux['v_{0}'.format(tstep)]
                    expanded_aux.loc[expanded_aux['v_aux'] < expanded_aux['Min.Speed'], 'v_aux'] = expanded_aux.loc[
                        expanded_aux['v_aux'] < expanded_aux['Min.Speed'], 'Min.Speed']
                    expanded_aux.loc[expanded_aux['v_aux'] > expanded_aux['Max.Speed'], 'v_aux'] = expanded_aux.loc[
                        expanded_aux['v_aux'] > expanded_aux['Max.Speed'], 'Max.Speed']

                    # EF
                    expanded_aux[ef_name] = \
                        ((expanded_aux.Alpha * expanded_aux.v_aux**2 + expanded_aux.Beta * expanded_aux.v_aux +
                          expanded_aux.Gamma + (expanded_aux.Delta / expanded_aux.v_aux)) /
                         (expanded_aux.Epsilon * expanded_aux.v_aux**2 + expanded_aux.Zita * expanded_aux.v_aux +
                          expanded_aux.Hta)) * (1 - expanded_aux.RF) * \
                        (expanded_aux.PF * expanded_aux['T'] / expanded_aux.Q)

                    # COPERT V equation can give nan for CH4
                    expanded_aux[ef_name].fillna(0, inplace=True)
                else:
                    expanded_aux[ef_name] = \
                        ((expanded_aux['a'] * expanded_aux['Cmileage'] + expanded_aux['b']) *
                         (expanded_aux['EFbase'] * expanded_aux['TF'])) / 1000

                # Mcorr
                if m_corr is not None:
                    expanded_aux.loc[expanded_aux['v_aux'] <= 19., 'Mcorr'] = \
                        expanded_aux.A_urban * expanded_aux['M'] + expanded_aux.B_urban
                    expanded_aux.loc[expanded_aux['v_aux'] >= 63., 'Mcorr'] = \
                        expanded_aux.A_road * expanded_aux['M'] + expanded_aux.B_road
                    expanded_aux.loc[(expanded_aux['v_aux'] > 19.) & (expanded_aux['v_aux'] < 63.), 'Mcorr'] = \
                        (expanded_aux.A_urban * expanded_aux['M'] + expanded_aux.B_urban) + \
                        ((expanded_aux.v_aux - 19) *
                         ((expanded_aux.A_road * expanded_aux['M'] + expanded_aux.B_road) -
                          (expanded_aux.A_urban * expanded_aux['M'] + expanded_aux.B_urban))) / 44.
                    expanded_aux.loc[expanded_aux['Mcorr'].isnull(), 'Mcorr'] = 1
                else:
                    expanded_aux.loc[:, 'Mcorr'] = 1

                # Full formula
                expanded_aux.loc[:, p_column] = \
                    expanded_aux['Fleet_value'] * expanded_aux[ef_name] * expanded_aux['Mcorr'] * \
                    expanded_aux['f_{0}'.format(tstep)]
                expanded_aux.drop(columns=[ef_name, 'Mcorr'], inplace=True)

            if pollutant != 'nh3':
                expanded_aux.drop(columns=['v_aux', 'Min.Speed', 'Max.Speed', 'Alpha', 'Beta', 'Gamma', 'Delta',
                                           'Epsilon', 'Zita', 'Hta', 'RF', 'Q', 'PF', 'T'], inplace=True)
            else:
                expanded_aux.drop(columns=['a', 'Cmileage', 'b', 'EFbase', 'TF'], inplace=True)

            if m_corr is not None:
                expanded_aux.drop(columns=['A_urban', 'B_urban', 'A_road', 'B_road', 'M'], inplace=True)
        expanded_aux.drop(columns=['road_grad'], inplace=True)
        expanded_aux.drop(columns=['f_{0}'.format(x) for x in range(len(self.date_array))], inplace=True)

        libc.malloc_trim(0)

        self.__logger.write_time_log('TrafficSector', 'calculate_hot', timeit.default_timer() - spent_time)

        return expanded_aux

    def calculate_cold(self, hot_expanded):
        spent_time = timeit.default_timer()

        cold_links = self.road_links.copy().reset_index()
        cold_links.drop(columns=['aadt', 'PcHeavy', 'PcMoto', 'PcMoped', 'sp_wd', 'sp_we', 'sp_hour_su', 'sp_hour_mo',
                                 'sp_hour_tu', 'sp_hour_we', 'sp_hour_th', 'sp_hour_fr', 'sp_hour_sa', 'Road_type',
                                 'aadt_m_mn', 'aadt_h_mn', 'aadt_h_wd', 'aadt_h_sat', 'aadt_h_sun', 'aadt_week',
                                 'fleet_comp', 'road_grad', 'PcLight', 'start_date'], inplace=True)
        libc.malloc_trim(0)

        cold_links['centroid'] = cold_links['geometry'].centroid
        link_lons = cold_links['geometry'].centroid.x
        link_lats = cold_links['geometry'].centroid.y

        temperature = IoNetcdf(self.__comm).get_hourly_data_from_netcdf(
            link_lons.min(), link_lons.max(), link_lats.min(), link_lats.max(), self.temp_common_path, 'tas',
            self.date_array)
        temperature.rename(columns={x: 't_{0}'.format(x) for x in range(len(self.date_array))}, inplace=True)
        # From Kelvin to Celsius degrees
        temperature[['t_{0}'.format(x) for x in range(len(self.date_array))]] = \
            temperature[['t_{0}'.format(x) for x in range(len(self.date_array))]] - 273.15

        unary_union = temperature.unary_union
        cold_links['REC'] = cold_links.apply(self.nearest, geom_union=unary_union, df1=cold_links, df2=temperature,
                                             geom1_col='centroid', src_column='REC', axis=1)

        cold_links.drop(columns=['geometry', 'centroid', 'geometry'], inplace=True)
        libc.malloc_trim(0)

        cold_links = cold_links.merge(temperature, left_on='REC', right_on='REC', how='left')
        cold_links.drop(columns=['REC'], inplace=True)
        libc.malloc_trim(0)

        c_expanded = hot_expanded.merge(cold_links, left_on='Link_ID', right_on='Link_ID', how='left')

        df_list = []
        for pollutant in self.source_pollutants:

            ef_cold = self.read_ef('cold', pollutant)

            if pollutant != 'nh3':
                ef_cold.loc[ef_cold['Tmin'].isnull(), 'Tmin'] = -999
                ef_cold.loc[ef_cold['Tmax'].isnull(), 'Tmax'] = 999
                ef_cold.loc[ef_cold['Min.Speed'].isnull(), 'Min.Speed'] = -999
                ef_cold.loc[ef_cold['Max.Speed'].isnull(), 'Max.Speed'] = 999

            c_expanded_p = c_expanded.merge(ef_cold, left_on=['Fleet_Code', 'Road_type'],
                                            right_on=['Code', 'Mode'], how='inner')
            cold_exp_p_aux = c_expanded_p.copy()

            cold_exp_p_aux.drop(columns=['Road_type', 'Fleet_value', 'Code'], inplace=True)
            libc.malloc_trim(0)

            for tstep in range(len(self.date_array)):
                v_column = 'v_{0}'.format(tstep)
                p_column = '{0}_{1}'.format(pollutant, tstep)
                t_column = 't_{0}'.format(tstep)
                if pollutant != 'nh3':
                    cold_exp_p_aux = cold_exp_p_aux.loc[cold_exp_p_aux[t_column] >= cold_exp_p_aux['Tmin'], :]
                    cold_exp_p_aux = cold_exp_p_aux.loc[cold_exp_p_aux[t_column] < cold_exp_p_aux['Tmax'], :]
                    cold_exp_p_aux = cold_exp_p_aux.loc[cold_exp_p_aux[v_column] >= cold_exp_p_aux['Min.Speed'], :]
                    cold_exp_p_aux = cold_exp_p_aux.loc[cold_exp_p_aux[v_column] < cold_exp_p_aux['Max.Speed'], :]

                # Beta
                cold_exp_p_aux['Beta'] = \
                    (0.6474 - (0.02545 * cold_exp_p_aux['ltrip']) - (0.00974 - (0.000385 * cold_exp_p_aux['ltrip'])) *
                     cold_exp_p_aux[t_column]) * cold_exp_p_aux['bc']
                if pollutant != 'nh3':
                    cold_exp_p_aux['cold_hot'] = \
                        cold_exp_p_aux['A'] * cold_exp_p_aux[v_column] + cold_exp_p_aux['B'] * \
                        cold_exp_p_aux[t_column] + cold_exp_p_aux['C']

                else:
                    cold_exp_p_aux['cold_hot'] = \
                        ((cold_exp_p_aux['a'] * cold_exp_p_aux['Cmileage'] + cold_exp_p_aux['b']) *
                         cold_exp_p_aux['EFbase'] * cold_exp_p_aux['TF']) / \
                        ((cold_exp_p_aux['a_hot'] * cold_exp_p_aux['Cmileage'] + cold_exp_p_aux['b_hot']) *
                         cold_exp_p_aux['EFbase_hot'] * cold_exp_p_aux['TF_hot'])
                cold_exp_p_aux.loc[cold_exp_p_aux['cold_hot'] < 1, 'cold_hot'] = 1

                # Formula Cold emissions
                cold_exp_p_aux[p_column] = \
                    cold_exp_p_aux[p_column] * cold_exp_p_aux['Beta'] * (cold_exp_p_aux['cold_hot'] - 1)
                df_list.append((cold_exp_p_aux[['Link_ID', 'Fleet_Code', p_column]]).set_index(
                    ['Link_ID', 'Fleet_Code']))

        try:
            cold_df = pd.concat(df_list, axis=1, ).reset_index()
        except Exception:
            error_fleet_code = []
            for df in df_list:
                orig = list(df.index.values)
                uni = list(np.unique(df.index.values))

                for o in orig:
                    try:
                        uni.remove(o)
                    except Exception:
                        error_fleet_code.append(o)
            error_exit('There are duplicated values for {0} codes in the cold EF files.'.format(error_fleet_code))

        for tstep in range(len(self.date_array)):
            if 'pm' in self.source_pollutants:
                cold_df['pm10_{0}'.format(tstep)] = cold_df['pm_{0}'.format(tstep)]
                cold_df['pm25_{0}'.format(tstep)] = cold_df['pm_{0}'.format(tstep)]
                cold_df.drop(columns=['pm_{0}'.format(tstep)], inplace=True)
                libc.malloc_trim(0)
            if 'voc' in self.source_pollutants and 'ch4' in self.source_pollutants:
                cold_df['nmvoc_{0}'.format(tstep)] = \
                    cold_df['voc_{0}'.format(tstep)] - cold_df['ch4_{0}'.format(tstep)]
                cold_df.drop(columns=['voc_{0}'.format(tstep)], inplace=True)
                libc.malloc_trim(0)
            else:
                self.__logger.write_log("WARNING! nmvoc emissions cannot be estimated because voc or ch4 are not " +
                                      "selected in the pollutant list.")
                warnings.warn("nmvoc emissions cannot be estimated because voc or ch4 are not selected in the " +
                              "pollutant list.")

        cold_df = self.speciate_traffic(cold_df, self.hot_cold_speciation)
        libc.malloc_trim(0)
        self.__logger.write_time_log('TrafficSector', 'calculate_cold', timeit.default_timer() - spent_time)
        return cold_df

    def compact_hot_expanded(self, expanded):
        spent_time = timeit.default_timer()

        columns_to_delete = ['Road_type', 'Fleet_value'] + ['v_{0}'.format(x) for x in range(len(self.date_array))]
        expanded.drop(columns=columns_to_delete, inplace=True)

        for tstep in range(len(self.date_array)):
            if 'pm' in self.source_pollutants:
                expanded['pm10_{0}'.format(tstep)] = expanded['pm_{0}'.format(tstep)]
                expanded['pm25_{0}'.format(tstep)] = expanded['pm_{0}'.format(tstep)]
                expanded.drop(columns=['pm_{0}'.format(tstep)], inplace=True)

            if 'voc' in self.source_pollutants and 'ch4' in self.source_pollutants:
                expanded['nmvoc_{0}'.format(tstep)] = expanded['voc_{0}'.format(tstep)] - \
                                                      expanded['ch4_{0}'.format(tstep)]
                # For certain vehicles (mostly diesel) and speeds, in urban road CH4 > than VOC according to COPERT V
                expanded.loc[expanded['nmvoc_{0}'.format(tstep)] < 0, 'nmvoc_{0}'.format(tstep)] = 0
                expanded.drop(columns=['voc_{0}'.format(tstep)], inplace=True)
            else:
                self.__logger.write_log("nmvoc emissions cannot be estimated because voc or ch4 are not selected in " +
                                      "the pollutant list.")
                warnings.warn(
                    "nmvoc emissions cannot be estimated because voc or ch4 are not selected in the pollutant list.")

        compacted = self.speciate_traffic(expanded, self.hot_cold_speciation)

        self.__logger.write_time_log('TrafficSector', 'compact_hot_expanded', timeit.default_timer() - spent_time)
        return compacted

    def calculate_tyre_wear(self):
        spent_time = timeit.default_timer()

        pollutants = ['pm']
        for pollutant in pollutants:
            ef_tyre = self.read_ef('tyre', pollutant)
            df = pd.merge(self.expanded.reset_index(), ef_tyre, left_on='Fleet_Code', right_on='Code', how='inner')
            df.drop(columns=['road_grad', 'Road_type', 'Code'], inplace=True)

            for tstep in range(len(self.date_array)):
                p_column = '{0}_{1}'.format(pollutant, tstep)
                f_column = 'f_{0}'.format(tstep)
                v_column = 'v_{0}'.format(tstep)
                df.loc[df[v_column] < 40, p_column] = df['Fleet_value'] * df['EFbase'] * df[f_column] * 1.39
                df.loc[(df[v_column] >= 40) & (df[v_column] <= 90), p_column] = \
                    df['Fleet_value'] * df['EFbase'] * df[f_column] * (-0.00974 * df[v_column] + 1.78)
                df.loc[df[v_column] > 90, p_column] = df['Fleet_value'] * df['EFbase'] * df[f_column] * 0.902

                # from PM to PM10 & PM2.5
                if pollutant == 'pm':
                    df['pm10_{0}'.format(tstep)] = df[p_column] * 0.6
                    df['pm25_{0}'.format(tstep)] = df[p_column] * 0.42
                    df.drop(columns=[p_column], inplace=True)

        # Cleaning df
        columns_to_delete = ['f_{0}'.format(x) for x in range(len(self.date_array))] + \
                            ['v_{0}'.format(x) for x in range(len(self.date_array))] + \
                            ['Fleet_value', 'EFbase']
        df.drop(columns=columns_to_delete, inplace=True)
        df = self.speciate_traffic(df, self.tyre_speciation)

        self.__logger.write_time_log('TrafficSector', 'calculate_tyre_wear', timeit.default_timer() - spent_time)
        return df

    def calculate_brake_wear(self):
        spent_time = timeit.default_timer()

        pollutants = ['pm']
        for pollutant in pollutants:
            ef_tyre = self.read_ef('brake', pollutant)
            df = pd.merge(self.expanded.reset_index(), ef_tyre, left_on='Fleet_Code', right_on='Code', how='inner')
            df.drop(columns=['road_grad', 'Road_type', 'Code'], inplace=True)
            for tstep in range(len(self.date_array)):
                p_column = '{0}_{1}'.format(pollutant, tstep)
                f_column = 'f_{0}'.format(tstep)
                v_column = 'v_{0}'.format(tstep)
                df.loc[df[v_column] < 40, p_column] = df['Fleet_value'] * df['EFbase'] * df[f_column] * 1.67
                df.loc[(df[v_column] >= 40) & (df[v_column] <= 95), p_column] = \
                    df['Fleet_value'] * df['EFbase'] * df[f_column] * (-0.027 * df[v_column] + 2.75)
                df.loc[df[v_column] > 95, p_column] = df['Fleet_value'] * df['EFbase'] * df[f_column] * 0.185

                # from PM to PM10 & PM2.5
                if pollutant == 'pm':
                    df.loc[:, 'pm10_{0}'.format(tstep)] = df[p_column] * 0.98
                    df.loc[:, 'pm25_{0}'.format(tstep)] = df[p_column] * 0.39
                    df.drop(columns=[p_column], inplace=True)

        # Cleaning df
        columns_to_delete = ['f_{0}'.format(x) for x in range(len(self.date_array))] + \
                            ['v_{0}'.format(x) for x in range(len(self.date_array))] + \
                            ['Fleet_value', 'EFbase']
        df.drop(columns=columns_to_delete, inplace=True)
        libc.malloc_trim(0)

        df = self.speciate_traffic(df, self.brake_speciation)

        self.__logger.write_time_log('TrafficSector', 'calculate_brake_wear', timeit.default_timer() - spent_time)
        return df

    def calculate_road_wear(self):
        spent_time = timeit.default_timer()

        pollutants = ['pm']
        for pollutant in pollutants:
            ef_tyre = self.read_ef('road', pollutant)
            df = pd.merge(self.expanded.reset_index(), ef_tyre, left_on='Fleet_Code', right_on='Code', how='inner')
            df.drop(columns=['road_grad', 'Road_type', 'Code'], inplace=True)
            for tstep in range(len(self.date_array)):
                p_column = '{0}_{1}'.format(pollutant, tstep)
                f_column = 'f_{0}'.format(tstep)
                df.loc[:, p_column] = df['Fleet_value'] * df['EFbase'] * df[f_column]

                # from PM to PM10 & PM2.5
                if pollutant == 'pm':
                    df.loc[:, 'pm10_{0}'.format(tstep)] = df[p_column] * 0.5
                    df.loc[:, 'pm25_{0}'.format(tstep)] = df[p_column] * 0.27
                    df.drop(columns=[p_column], inplace=True)

        # Cleaning df
        columns_to_delete = ['f_{0}'.format(x) for x in range(len(self.date_array))] + \
                            ['v_{0}'.format(x) for x in range(len(self.date_array))] + \
                            ['Fleet_value', 'EFbase']
        df.drop(columns=columns_to_delete, inplace=True)

        df = self.speciate_traffic(df, self.road_speciation)

        self.__logger.write_time_log('TrafficSector', 'calculate_road_wear', timeit.default_timer() - spent_time)
        return df

    def calculate_resuspension(self):
        spent_time = timeit.default_timer()

        if self.resuspension_correction:
            road_link_aux = self.road_links[['geometry']].copy().reset_index()

            road_link_aux['centroid'] = road_link_aux['geometry'].centroid
            link_lons = road_link_aux['geometry'].centroid.x
            link_lats = road_link_aux['geometry'].centroid.y

            p_factor = self.calculate_precipitation_factor(link_lons.min(), link_lons.max(), link_lats.min(),
                                                           link_lats.max(), self.precipitation_path)
            unary_union = p_factor.unary_union

            road_link_aux['REC'] = road_link_aux.apply(self.nearest, geom_union=unary_union, df1=road_link_aux,
                                                       df2=p_factor, geom1_col='centroid', src_column='REC', axis=1)
            road_link_aux.drop(columns=['centroid'], inplace=True)
            p_factor.drop(columns=['geometry'], inplace=True)

            road_link_aux = road_link_aux.merge(p_factor, left_on='REC', right_on='REC', how='left')

            road_link_aux.drop(columns=['REC'], inplace=True)

        pollutants = ['pm']
        for pollutant in pollutants:
            ef_tyre = self.read_ef('resuspension', pollutant)
            df = pd.merge(self.expanded.reset_index(), ef_tyre, left_on='Fleet_Code', right_on='Code', how='inner')
            if self.resuspension_correction:
                df = df.merge(road_link_aux, left_on='Link_ID', right_on='Link_ID', how='left')

            df.drop(columns=['road_grad', 'Road_type', 'Code'], inplace=True)
            for tstep in range(len(self.date_array)):
                p_column = '{0}_{1}'.format(pollutant, tstep)
                f_column = 'f_{0}'.format(tstep)
                if self.resuspension_correction:
                    pr_column = 'PR_{0}'.format(tstep)
                    df[p_column] = df['Fleet_value'] * df['EFbase'] * df[pr_column] * df[f_column]
                else:
                    df[p_column] = df['Fleet_value'] * df['EFbase'] * df[f_column]

                # from PM to PM10 & PM2.5
                if pollutant == 'pm':
                    df['pm10_{0}'.format(tstep)] = df[p_column]
                    # TODO Check fraction of pm2.5
                    df['pm25_{0}'.format(tstep)] = df[p_column] * 0.5
                    df.drop(columns=[p_column], inplace=True)

        # Cleaning df
        columns_to_delete = ['f_{0}'.format(x) for x in range(len(self.date_array))] + \
                            ['v_{0}'.format(x) for x in range(len(self.date_array))] + \
                            ['Fleet_value', 'EFbase']
        df.drop(columns=columns_to_delete, inplace=True)

        df = self.speciate_traffic(df, self.resuspension_speciation)

        self.__logger.write_time_log('TrafficSector', 'calculate_resuspension', timeit.default_timer() - spent_time)
        return df

    def transform_df(self, df):
        spent_time = timeit.default_timer()

        df_list = []
        for tstep in range(len(self.date_array)):
            pollutants_to_rename = [p for p in list(df.columns.values) if p.endswith('_{0}'.format(tstep))]
            pollutants_renamed = []
            for p_name in pollutants_to_rename:
                p_name_new = p_name.replace('_{0}'.format(tstep), '')
                df.rename(columns={p_name: p_name_new}, inplace=True)
                pollutants_renamed.append(p_name_new)

            df_aux = df[['Link_ID', 'Fleet_Code'] + pollutants_renamed].copy()
            df_aux['tstep'] = tstep

            df_list.append(df_aux)
            df.drop(columns=pollutants_renamed, inplace=True)

        df = pd.concat(df_list, ignore_index=True)
        self.__logger.write_time_log('TrafficSector', 'transform_df', timeit.default_timer() - spent_time)
        return df

    def speciate_traffic(self, df, speciation):
        spent_time = timeit.default_timer()

        # Reads speciation profile
        speciation = self.read_profiles(speciation)

        speciation.drop(columns=['Copert_V_name'], inplace=True)
        # Transform dataset into timestep rows instead of timestep columns
        df = self.transform_df(df)

        in_list = list(df.columns.values)

        in_columns = ['Link_ID', 'Fleet_Code', 'tstep']
        for in_col in in_columns:
            in_list.remove(in_col)

        df_out_list = []

        # PMC
        if not set(speciation.columns.values).isdisjoint(pmc_list):
            out_p = set(speciation.columns.values).intersection(pmc_list).pop()
            speciation_by_in_p = speciation[[out_p] + ['Code']].copy()

            speciation_by_in_p.rename(columns={out_p: 'f_{0}'.format(out_p)}, inplace=True)
            df_aux = df[['pm10', 'pm25', 'Fleet_Code', 'tstep', 'Link_ID']]
            df_aux = df_aux.merge(speciation_by_in_p, left_on='Fleet_Code', right_on='Code', how='left')
            df_aux.drop(columns=['Code'], inplace=True)

            df_aux[out_p] = df_aux['pm10'] - df_aux['pm25']

            df_out_list.append(df_aux[[out_p] + ['tstep', 'Link_ID']].groupby(['tstep', 'Link_ID']).sum())

        for in_p in in_list:
            involved_out_pollutants = [key for key, value in self.speciation_map.items() if value == in_p]

            # Selecting only necessary speciation profiles
            speciation_by_in_p = speciation[involved_out_pollutants + ['Code']].copy()

            # Adding "f_" in the formula column names
            for p in involved_out_pollutants:
                speciation_by_in_p.rename(columns={p: 'f_{0}'.format(p)}, inplace=True)
            # Getting a slice of the full dataset to be merged
            df_aux = df[[in_p] + ['Fleet_Code', 'tstep', 'Link_ID']]
            df_aux = df_aux.merge(speciation_by_in_p, left_on='Fleet_Code', right_on='Code', how='left')
            df_aux.drop(columns=['Code'], inplace=True)

            # Renaming pollutant columns by adding "old_" to the beginning.
            df_aux.rename(columns={in_p: 'old_{0}'.format(in_p)}, inplace=True)
            for p in involved_out_pollutants:
                if in_p is not np.nan:
                    if in_p != 0:
                        df_aux[p] = df_aux['old_{0}'.format(in_p)].multiply(df_aux['f_{0}'.format(p)])
                        try:
                            if in_p == 'nmvoc':
                                mol_w = 1.0
                            else:
                                mol_w = self.molecular_weights[in_p]
                        except KeyError:
                            error_exit('{0} not found in the molecular weights file.'.format(in_p))
                        # from g/km.h to mol/km.h or g/km.h (aerosols)
                        df_aux[p] = df_aux.loc[:, p] / mol_w

                    else:
                        df_aux.loc[:, p] = 0

                df_out_list.append(df_aux[[p] + ['tstep', 'Link_ID']].groupby(['tstep', 'Link_ID']).sum())
            del df_aux
            df.drop(columns=[in_p], inplace=True)

        df_out = pd.concat(df_out_list, axis=1)

        self.__logger.write_time_log('TrafficSector', 'speciate_traffic', timeit.default_timer() - spent_time)
        return df_out

    def calculate_emissions(self):
        spent_time = timeit.default_timer()
        version = 1
        self.__logger.write_log('\tCalculating Road traffic emissions', message_level=1)
        df_accum = pd.DataFrame()

        if version == 2:
            if self.do_hot:
                self.__logger.write_log('\t\tCalculating Hot emissions.', message_level=2)
                df_accum = pd.concat([df_accum, self.compact_hot_expanded(self.calculate_hot())]).groupby(
                    ['tstep', 'Link_ID']).sum()
            if self.do_cold:
                self.__logger.write_log('\t\tCalculating Cold emissions.', message_level=2)
                df_accum = pd.concat([df_accum, self.calculate_cold(self.calculate_hot())]).groupby(
                    ['tstep', 'Link_ID']).sum()
        else:
            if self.do_hot or self.do_cold:
                self.__logger.write_log('\t\tCalculating Hot emissions.', message_level=2)
                hot_emis = self.calculate_hot()

            if self.do_hot:
                self.__logger.write_log('\t\tCompacting Hot emissions.', message_level=2)
                df_accum = pd.concat([df_accum, self.compact_hot_expanded(hot_emis.copy())]).groupby(
                    ['tstep', 'Link_ID']).sum()
                libc.malloc_trim(0)
            if self.do_cold:
                self.__logger.write_log('\t\tCalculating Cold emissions.', message_level=2)
                df_accum = pd.concat([df_accum, self.calculate_cold(hot_emis)]).groupby(
                    ['tstep', 'Link_ID']).sum()
                libc.malloc_trim(0)
            if self.do_hot or self.do_cold:
                del hot_emis
                libc.malloc_trim(0)

        if self.do_tyre_wear:
            self.__logger.write_log('\t\tCalculating Tyre wear emissions.', message_level=2)
            df_accum = pd.concat([df_accum, self.calculate_tyre_wear()], sort=False).groupby(['tstep', 'Link_ID']).sum()
            libc.malloc_trim(0)
        if self.do_brake_wear:
            self.__logger.write_log('\t\tCalculating Brake wear emissions.', message_level=2)
            df_accum = pd.concat([df_accum, self.calculate_brake_wear()], sort=False).groupby(
                ['tstep', 'Link_ID']).sum()
            libc.malloc_trim(0)
        if self.do_road_wear:
            self.__logger.write_log('\t\tCalculating Road wear emissions.', message_level=2)
            df_accum = pd.concat([df_accum, self.calculate_road_wear()], sort=False).groupby(['tstep', 'Link_ID']).sum()
            libc.malloc_trim(0)
        if self.do_resuspension:
            self.__logger.write_log('\t\tCalculating Resuspension emissions.', message_level=2)
            df_accum = pd.concat([df_accum, self.calculate_resuspension()], sort=False).groupby(
                ['tstep', 'Link_ID']).sum()
            libc.malloc_trim(0)
        df_accum = df_accum.reset_index().merge(self.road_links.reset_index().loc[:, ['Link_ID', 'geometry']],
                                                on='Link_ID', how='left')
        df_accum = gpd.GeoDataFrame(df_accum, crs=self.crs)
        libc.malloc_trim(0)
        df_accum.set_index(['Link_ID', 'tstep'], inplace=True)

        if self.write_rline:
            self.write_rline_output(df_accum.copy())
        self.__logger.write_log('\t\tRoad link emissions to grid.', message_level=2)
        df_accum = self.links_to_grid(df_accum)
        libc.malloc_trim(0)

        self.__logger.write_log('\tRoad traffic emissions calculated', message_level=2)
        self.__logger.write_time_log('TrafficSector', 'calculate_emissions', timeit.default_timer() - spent_time)
        return df_accum

    def links_to_grid(self, link_emissions):
        spent_time = timeit.default_timer()

        link_emissions.reset_index(inplace=True)
        if not os.path.exists(self.link_to_grid_csv):
            link_emissions_aux = link_emissions.loc[link_emissions['tstep'] == 0, :]

            if self.grid.grid_type in ['Lambert Conformal Conic', 'Mercator']:
                grid_aux = self.grid.shapefile
            else:
                # For REGULAR and ROTATED grids, shapefile projection is transformed to a metric projected coordinate
                # system to derive the length in km.
                grid_aux = self.grid.shapefile.to_crs(FINAL_PROJ)

            link_emissions_aux = link_emissions_aux.to_crs(grid_aux.crs)

            link_emissions_aux = gpd.sjoin(link_emissions_aux, grid_aux.reset_index(), how="inner", op='intersects')

            link_emissions_aux = link_emissions_aux.loc[:, ['Link_ID', 'geometry', 'FID']]

            link_emissions_aux = link_emissions_aux.merge(grid_aux.reset_index().loc[:, ['FID', 'geometry']],
                                                          on='FID', how='left')

            length_list = []
            link_id_list = []
            fid_list = []
            count = 1
            for i, line in link_emissions_aux.iterrows():
                count += 1
                aux = line.get('geometry_x').intersection(line.get('geometry_y'))
                if not aux.is_empty:
                    link_id_list.append(line.get('Link_ID'))
                    fid_list.append(line.get('FID'))
                    # Length of road links from m to km
                    length_list.append(aux.length / 1000)

            link_grid = pd.DataFrame({'Link_ID': link_id_list, 'FID': fid_list, 'length': length_list})

            # Writing link to grid file
            data = self.__comm.gather(link_grid, root=0)
            if self.__comm.Get_rank() == 0:
                if not os.path.exists(os.path.dirname(self.link_to_grid_csv)):
                    os.makedirs(os.path.dirname(self.link_to_grid_csv))
                data = pd.concat(data)
                data.to_csv(self.link_to_grid_csv)

            self.__comm.Barrier()

        else:
            link_grid = pd.read_csv(self.link_to_grid_csv)
            link_grid = link_grid[link_grid['Link_ID'].isin(link_emissions['Link_ID'].values)]

        link_emissions.drop(columns=['geometry'], inplace=True)
        link_grid = link_grid.merge(link_emissions, left_on='Link_ID', right_on='Link_ID')
        if 'Unnamed: 0' in link_grid.columns.values:
            link_grid.drop(columns=['Unnamed: 0'], inplace=True)

        cols_to_update = list(link_grid.columns.values)
        cols_to_update.remove('length')
        cols_to_update.remove('tstep')
        cols_to_update.remove('FID')
        for col in cols_to_update:
            link_grid.loc[:, col] = link_grid[col] * link_grid['length']
        link_grid.drop(columns=['length', 'Link_ID'], inplace=True)
        link_grid['layer'] = 0
        link_grid = link_grid.groupby(['FID', 'layer', 'tstep']).sum()

        self.__logger.write_time_log('TrafficSector', 'links_to_grid', timeit.default_timer() - spent_time)

        return link_grid

    def write_rline_output(self, emissions):
        from datetime import timedelta
        spent_time = timeit.default_timer()

        emissions.drop(columns=['geometry'], inplace=True)
        for poll in emissions.columns.values:
            mol_w = self.molecular_weights[self.speciation_map[poll]]
            # From g/km.h to g/m.s
            emissions.loc[:, poll] = emissions.loc[:, poll] * mol_w / (1000 * 3600)

        emissions.reset_index(inplace=True)

        emissions_list = self.__comm.gather(emissions, root=0)
        if self.__comm.Get_rank() == 0:
            emissions = pd.concat(emissions_list)
            p_list = list(emissions.columns.values)
            p_list.remove('tstep')
            p_list.remove('Link_ID')
            for p in p_list:
                link_list = ['L_{0}'.format(x) for x in list(pd.unique(emissions['Link_ID']))]
                out_df = pd.DataFrame(columns=["Year", "Mon", "Day", "JDay", "Hr"] + link_list)
                for tstep, aux in emissions.loc[:, ['tstep', 'Link_ID', p]].groupby('tstep'):
                    aux_date = self.date_array[0] + timedelta(hours=tstep)
                    out_df.loc[tstep, 'Year'] = aux_date.strftime('%y')
                    out_df.loc[tstep, 'Mon'] = aux_date.month
                    out_df.loc[tstep, 'Day'] = aux_date.day
                    out_df.loc[tstep, 'JDay'] = aux_date.strftime('%j')
                    out_df.loc[tstep, 'Hr'] = aux_date.hour
                    out_df.loc[tstep, link_list] = aux.loc[:, [p]].transpose().values

                out_df.to_csv(os.path.join(self.output_dir, 'rline_{1}_{0}.csv'.format(
                    p, self.date_array[0].strftime('%Y%m%d'))), index=False)

        self.__comm.Barrier()

        self.__logger.write_time_log('TrafficSector', 'write_rline_output', timeit.default_timer() - spent_time)
        return True

    def write_rline_roadlinks(self, df_in):
        spent_time = timeit.default_timer()

        df_in_list = self.__comm.gather(df_in, root=0)
        if self.__comm.Get_rank() == 0:
            df_in = pd.concat(df_in_list)

            df_out = pd.DataFrame(
                columns=['Group', 'X_b', 'Y_b', 'Z_b', 'X_e', 'Y_e', 'Z_e', 'dCL', 'sigmaz0', '#lanes',
                         'lanewidth', 'Emis', 'Hw1', 'dw1', 'Hw2', 'dw2', 'Depth', 'Wtop', 'Wbottom',
                         'l_bh2sw', 'l_avgbh', 'l_avgbdensity', 'l_bhdev', 'X0_af', 'X45_af',
                         'X90_af', 'X135_af', 'X180_af', 'X225_af', 'X270_af', 'X315_af', 'l_maxbh', 'Link_ID'])
            df_err_list = []

            df_in = df_in.to_crs({u'units': u'm', u'no_defs': True, u'ellps': u'intl', u'proj': u'utm', u'zone': 31})
            if rline_shp:
                df_in.to_file(os.path.join(self.output_dir, 'roads.shp'))

            count = 0
            for i, line in df_in.iterrows():
                try:
                    df_out.loc[count] = pd.Series({
                        'Group': 'G1',
                        'X_b': round(line.get('geometry').coords[0][0], 3),
                        'Y_b': round(line.get('geometry').coords[0][1], 3),
                        'Z_b': 1,
                        'X_e': round(line.get('geometry').coords[-1][0], 3),
                        'Y_e': round(line.get('geometry').coords[-1][1], 3),
                        'Z_e': 1,
                        'dCL': 0,
                        'sigmaz0': 2,
                        '#lanes': 3,
                        'lanewidth': 2.5,
                        'Emis': 1,
                        'Hw1': 0,
                        'dw1': 0,
                        'Hw2': 0,
                        'dw2': 0,
                        'Depth': 0,
                        'Wtop': 0,
                        'Wbottom': 0,
                        'l_bh2sw': round(line.get('bh_2_sw'), 3),
                        'l_avgbh': round(line.get('mean_heigh'), 3),
                        'l_avgbdensity': round(line.get('area_densi'), 3),
                        'l_bhdev': round(line.get('sd_height'), 3),
                        'X0_af': round(line.get('af_0'), 3),
                        'X45_af': round(line.get('af_45'), 3),
                        'X90_af': round(line.get('af_90'), 3),
                        'X135_af': round(line.get('af_135'), 3),
                        'X180_af': round(line.get('af_180'), 3),
                        'X225_af': round(line.get('af_225'), 3),
                        'X270_af': round(line.get('af_270'), 3),
                        'X315_af': round(line.get('af_315'), 3),
                        'l_maxbh': round(line.get('max_height'), 3),
                        'Link_ID': line.get('Link_ID'),
                    })
                    count += 1
                except Exception:
                    # df_err_list.append(line)
                    pass

            df_out.set_index('Link_ID', inplace=True)
            df_out.sort_index(inplace=True)
            df_out.to_csv(os.path.join(self.output_dir, 'roads.txt'), index=False, sep=' ')
        self.__comm.Barrier()
        self.__logger.write_log('\t\tTraffic emissions calculated', message_level=2)
        self.__logger.write_time_log('TrafficSector', 'write_rline_roadlinks', timeit.default_timer() - spent_time)
        return True
