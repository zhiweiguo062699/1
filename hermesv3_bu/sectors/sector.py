#!/usr/bin/env python

import sys
import os
import timeit
import numpy as np
import pandas as pd
import geopandas as gpd
from mpi4py import MPI

from geopandas import GeoDataFrame
from hermesv3_bu.logger.log import Log
from hermesv3_bu.grids.grid import Grid


class Sector(object):

    def __init__(self, comm, logger, auxiliary_dir, grid, clip, date_array, source_pollutants, vertical_levels,
                 monthly_profiles_path, weekly_profiles_path, hourly_profiles_path, speciation_map_path,
                 speciation_profiles_path, molecular_weights_path):
        """
        Initialize the main sector class with the common arguments and methods.

        :param comm: Communicator for the sector calculation.
        :type comm: MPI.Comm

        :param logger: Logger
        :type logger: Log

        :param auxiliary_dir: Path to the directory where the necessary auxiliary files will be created if them are not
            created yet.
        :type auxiliary_dir: str

        :param grid: Grid object
        :type grid: Grid

        :param date_array: List of datetimes.
        :type date_array: list(datetime.datetime, ...)

        :param source_pollutants: List of input pollutants to take into account.
        :type source_pollutants: list

        :param vertical_levels: List of top level of each vertical layer.
        :type vertical_levels: list

        :param monthly_profiles_path: Path to the CSV file that contains all the monthly profiles. The CSV file must
            contain the following columns [P_month, January, February, March, April, May, June, July, August, September,
            October, November, December]
        :type monthly_profiles_path: str

        :param weekly_profiles_path: Path to the CSV file that contains all the weekly profiles. The CSV file must
            contain the following columns [P_week, Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday]
        :type weekly_profiles_path: str

        :param hourly_profiles_path: Path to the CSV file that contains all the hourly profiles. The CSV file must
            contain the following columns [P_hour, 0, 1, 2, 3, ..., 22, 23]
        :type hourly_profiles_path: str

        :param speciation_map_path: Path to the CSV file that contains the speciation map. The CSV file must contain
            the following columns [dst, src, description]
            The 'dst' column will be used as output pollutant list and the 'src' column as their onw input pollutant
            to be used as a fraction in the speciation profiles.
        :type speciation_map_path: str

        :param speciation_profiles_path: Path to the file that contains all the speciation profiles. The CSV file
            must contain the "Code" column with the value of each animal of the animal_list. The rest of columns
            have to be the sames as the column 'dst' of the 'speciation_map_path' file.
        :type speciation_profiles_path: str

        :param molecular_weights_path: Path to the CSV file that contains all the molecular weights needed. The CSV
            file must contain the 'Specie' and 'MW' columns.
        :type molecular_weights_path: str

        """
        spent_time = timeit.default_timer()
        self.comm = comm
        self.logger = logger
        self.auxiliary_dir = auxiliary_dir
        self.grid = grid
        self.clip = clip
        self.date_array = date_array
        self.source_pollutants = source_pollutants

        self.vertical_levels = vertical_levels

        # Reading temporal profiles
        self.monthly_profiles = self.read_monthly_profiles(monthly_profiles_path)
        self.weekly_profiles = self.read_weekly_profiles(weekly_profiles_path)
        self.hourly_profiles = self.read_hourly_profiles(hourly_profiles_path)

        # Reading speciation files
        self.speciation_map = self.read_speciation_map(speciation_map_path)
        self.speciation_profile = self.read_speciation_profiles(speciation_profiles_path)
        self.molecular_weights = self.read_molecular_weights(molecular_weights_path)

        self.output_pollutants = list(self.speciation_map.keys())

        self.logger.write_time_log('Sector', '__init__', timeit.default_timer() - spent_time)

    def read_speciation_profiles(self, path):
        """
        Read all the speciation profiles.

        The CSV must contain the column ID with the identification of that profile. The rest of columns are the output
        pollutant and the value is the fraction of input pollutant that goes to that output pollutant.

        e.g.:
        ID,NOx,SOx,CO,NMVOC,PM10,PM25,PMC,CO2
        default,1,1,1,1,1,1,1,1

        :param path: Path to the CSV that contains the speciation profiles.
        :type path: str

        :return: Dataframe with the speciation profile and the ID as index.
        :rtype: DataFrame
        """
        spent_time = timeit.default_timer()
        if path is None:
            dataframe = None
        else:
            dataframe = pd.read_csv(path)
            dataframe.set_index('ID', inplace=True)

        self.logger.write_time_log('Sector', 'read_speciation_profiles', timeit.default_timer() - spent_time)
        return dataframe

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
        dataframe = pd.read_csv(path)
        if 'PMC' in dataframe['dst'].values and all(element in self.source_pollutants for element in ['pm10', 'pm25']):
            dataframe_aux = dataframe.loc[dataframe['src'].isin(self.source_pollutants), :]
            dataframe = pd.concat([dataframe_aux, dataframe.loc[dataframe['dst'] == 'PMC', :]])
        else:
            dataframe = dataframe.loc[dataframe['src'].isin(self.source_pollutants), :]

        dataframe = dict(zip(dataframe['dst'], dataframe['src']))
        self.logger.write_time_log('Sector', 'read_speciation_map', timeit.default_timer() - spent_time)

        return dataframe

    def read_molecular_weights(self, path):
        """
        Read the CSV file that contains the molecular weights

        e.g.:
        Specie,MW
        nox_no,30.01
        nox_no2,46.01
        co,28.01
        co2,44.01
        so2,64.06
        nh3,17.03

        :param path: Path to the CSV file.
        :type path: str

        :return: Dictionary with the specie as key and the molecular weight as value.
        :rtype: dict
        """
        spent_time = timeit.default_timer()
        dataframe = pd.read_csv(path)
        # dataframe = dataframe.loc[dataframe['Specie'].isin(self.source_pollutants)]

        mol_wei = dict(zip(dataframe['Specie'], dataframe['MW']))
        self.logger.write_time_log('Sector', 'read_molecular_weights', timeit.default_timer() - spent_time)

        return mol_wei

    def read_profiles(self, path, sep=','):
        """
        Read the CSV profile.

        :param path: Path to the CSV file that contains the profiles
        :type path: str

        :param sep: Separator of the values. [default -> ',']
        :type sep: str

        :return: DataFrame with the profiles.
        :rtype: DataFrame
        """
        spent_time = timeit.default_timer()
        dataframe = pd.read_csv(path, sep=sep)
        self.logger.write_time_log('Sector', 'read_profiles', timeit.default_timer() - spent_time)

        return dataframe

    def read_monthly_profiles(self, path):
        """
        Read the DataFrame of the monthly profiles with the month number as columns.

        :param path: Path to the file that contains the monthly profiles.
        :type path: str

        :return: DataFrame of the monthly profiles.
        :rtype: DataFrame
        """
        spent_time = timeit.default_timer()
        if path is None:
            profiles = None
        else:
            profiles = pd.read_csv(path)

            profiles.rename(
                columns={'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7,
                         'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12},
                inplace=True)
            profiles.set_index('P_month', inplace=True)

        self.logger.write_time_log('Sector', 'read_monthly_profiles', timeit.default_timer() - spent_time)
        return profiles

    def read_weekly_profiles(self, path):
        """
        Read the Dataset of the weekly profiles with the weekdays as numbers (Monday: 0 - Sunday:6) as columns.


        :param path: Path to the file that contains the weekly profiles.
        :type path: str

        :return: Dataset od the weekly profiles.
        :rtype: DataFrame
        """
        spent_time = timeit.default_timer()
        if path is None:
            profiles = None
        else:
            profiles = pd.read_csv(path)

            profiles.rename(
                columns={'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5,
                         'Sunday': 6, }, inplace=True)
            profiles.set_index('P_week', inplace=True)
        self.logger.write_time_log('Sector', 'read_weekly_profiles', timeit.default_timer() - spent_time)
        return profiles

    def read_hourly_profiles(self, path):
        """
        Read the Dataset of the hourly profiles with the hours (int) as columns.

        :param path: Path to the file that contains the monthly profiles.
        :type path: str

        :return: Dataset od the monthly profiles.
        :rtype: DataFrame
        """
        spent_time = timeit.default_timer()
        if path is None:
            profiles = None
        else:
            profiles = pd.read_csv(path)
            profiles.rename(
                columns={'P_hour': -1, '00': 0, '01': 1, '02': 2, '03': 3, '04': 4, '05': 5, '06': 6, '07': 7,
                         '08': 8, '09': 9, '10': 10, '11': 11, '12': 12, '13': 13, '14': 14, '15': 15, '16': 16,
                         '17': 17, '18': 18, '19': 19, '20': 20, '21': 21, '22': 22, '23': 23}, inplace=True)
            profiles.columns = profiles.columns.astype(int)
            profiles.rename(columns={-1: 'P_hour'}, inplace=True)
            profiles.set_index('P_hour', inplace=True)
        self.logger.write_time_log('Sector', 'read_hourly_profiles', timeit.default_timer() - spent_time)
        return profiles

    def calculate_rebalanced_weekly_profile(self, profile, date):
        """
        Correct the weekly profile depending on the date selected.

        If we sum the weekday factor of each day of the full month it mus sum 1.

        :param profile: Profile to be corrected.
            {0: 1.0414, 1: 1.0310, 2: 1.0237, 3: 1.0268, 4: 1.0477, 5: 0.9235, 6: 0.9058}
        :type profile: dict

        :param date: Date to select the month to evaluate.
        :type date: datetime.datetime

        :return: Profile already rebalanced.
        :rtype: dict
        """
        spent_time = timeit.default_timer()
        weekdays = self.calculate_weekdays(date)

        rebalanced_profile = self.calculate_weekday_factor_full_month(profile, weekdays)
        self.logger.write_time_log('Sector', 'calculate_rebalanced_weekly_profile', timeit.default_timer() - spent_time)

        return rebalanced_profile

    def calculate_weekday_factor_full_month(self, profile, weekdays):
        """
        Operate with all the days of the month to get the sum of daily factors of the full month.

        :param profile: input profile
        :type profile: dict

        :param weekdays: Dictionary with the number of days of each day type (Monday, Tuesday, ...)
        :type weekdays: dict

        :return: Dictionary with the corrected profile.
        :rtype: dict
        """
        spent_time = timeit.default_timer()
        weekdays_factors = 0
        num_days = 0
        for day in range(7):
            weekdays_factors += profile[day] * weekdays[day]
            num_days += weekdays[day]
        increment = float(num_days - weekdays_factors) / num_days
        for day in range(7):
            profile[day] = (increment + profile[day]) / num_days
        self.logger.write_time_log('Sector', 'calculate_weekday_factor_full_month', timeit.default_timer() - spent_time)

        return profile

    def calculate_weekdays(self, date):
        """
        Calculate the number of days of each day type for the given month of the year.

        :param date: Date to select the month to evaluate.
        :type date: datetime.datetime

        :return: Dictionary with the number of days of each day type (Monday, Tuesday, ...)
        :rtype: dict
        """
        from calendar import monthrange, weekday, MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY
        spent_time = timeit.default_timer()
        weekdays = [MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY]
        days = [weekday(date.year, date.month, d + 1) for d in range(monthrange(date.year, date.month)[1])]

        weekdays_dict = {}
        for i, day in enumerate(weekdays):
            weekdays_dict[i] = days.count(day)
        self.logger.write_time_log('Sector', 'calculate_weekdays', timeit.default_timer() - spent_time)
        return weekdays_dict

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
        dataframe = self.add_timezone(dataframe)
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

    def add_timezone(self, dataframe):
        """
        Add the timezone os the centroid of each geometry of the input geodataframe.

        :param dataframe: Geodataframe where add the timezone.
        :type dataframe: geopandas.GeoDataframe

        :return: Geodataframe with the timezone column.
        :rtype: geopandas.GeoDataframe        """
        from timezonefinder import TimezoneFinder
        spent_time = timeit.default_timer()
        dataframe = dataframe.to_crs({'init': 'epsg:4326'})
        tzfinder = TimezoneFinder()
        dataframe['timezone'] = dataframe.centroid.apply(lambda x: tzfinder.timezone_at(lng=x.x, lat=x.y))
        dataframe.reset_index(inplace=True)
        self.logger.write_time_log('Sector', 'add_timezone', timeit.default_timer() - spent_time)
        return dataframe

    def to_timezone(self, dataframe):
        """
        Set the local date with the correspondent timezone substituting the UTC date.

        :param dataframe: DataFrame with the UTC date column.
        :type dataframe: DataFrame

        :return: Catalog with the local date column.
        :rtype: DataFrame
        """
        spent_time = timeit.default_timer()
        dataframe['date'] = dataframe.groupby('timezone')['date'].apply(
            lambda x: x.dt.tz_convert(x.name).dt.tz_localize(None))

        dataframe.drop('timezone', axis=1, inplace=True)
        self.logger.write_time_log('Sector', 'to_timezone', timeit.default_timer() - spent_time)

        return dataframe

    def add_nut_code(self, shapefile, nut_shapefile_path, nut_value='nuts2_id'):
        """
        Add 'nut_code' column into the shapefile based on the 'nut_value' column of the 'nut_shapefile_path' shapefile.

        The elements that are not into any NUT will be dropped.
        If an element belongs to two NUTs will be set the fist one that appear in the 'nut_shapefile_path' shapefile.

        :param shapefile: Shapefile where add the NUT code.
        :type shapefile: geopandas.GeoDataframe

        :param nut_shapefile_path: Path to the shapefile with the polygons that contains the NUT code into the
            'nut_value' column.
        :type nut_shapefile_path: str

        :param nut_value: Column name of the NUT codes.
        :type nut_value: str

        :return: Shapefile with the 'nut_code' column set.
        :rtype: geopandas.GeoDataframe
        """
        spent_time = timeit.default_timer()
        nut_shapefile = gpd.read_file(nut_shapefile_path).to_crs(shapefile.crs)
        shapefile = gpd.sjoin(shapefile, nut_shapefile.loc[:, [nut_value, 'geometry']], how='left', op='intersects')
        del nut_shapefile
        # shapefile = shapefile[~shapefile.index.duplicated(keep='first')]
        shapefile.drop('index_right', axis=1, inplace=True)

        shapefile.rename(columns={nut_value: 'nut_code'}, inplace=True)
        shapefile.loc[shapefile['nut_code'].isna(), 'nut_code'] = -999
        shapefile['nut_code'] = shapefile['nut_code'].astype(np.int16)
        self.logger.write_time_log('Sector', 'add_nut_code', timeit.default_timer() - spent_time)

        return shapefile

    def spatial_overlays(self, df1, df2, how='intersection'):
        """
        Compute overlay intersection of two GeoPandasDataFrames df1 and df2

        https://github.com/geopandas/geopandas/issues/400

        :param df1: GeoDataFrame
        :param df2: GeoDataFrame
        :param how: Operation to do
        :return: GeoDataFrame
        """
        from functools import reduce

        spent_time = timeit.default_timer()
        df1 = df1.copy()
        df2 = df2.copy()
        df1['geometry'] = df1.geometry.buffer(0)
        df2['geometry'] = df2.geometry.buffer(0)
        if how == 'intersection':
            # Spatial Index to create intersections
            spatial_index = df2.sindex
            df1['bbox'] = df1.geometry.apply(lambda x: x.bounds)
            df1['histreg'] = df1.bbox.apply(lambda x: list(spatial_index.intersection(x)))
            pairs = df1['histreg'].to_dict()
            nei = []
            for i, j in pairs.items():
                for k in j:
                    nei.append([i, k])

            pairs = gpd.GeoDataFrame(nei, columns=['idx1', 'idx2'], crs=df1.crs)
            pairs = pairs.merge(df1, left_on='idx1', right_index=True)
            pairs = pairs.merge(df2, left_on='idx2', right_index=True, suffixes=['_1', '_2'])
            pairs['Intersection'] = pairs.apply(lambda x: (x['geometry_1'].intersection(x['geometry_2'])).buffer(0),
                                                axis=1)
            pairs = gpd.GeoDataFrame(pairs, columns=pairs.columns, crs=df1.crs)
            cols = pairs.columns.tolist()
            cols.remove('geometry_1')
            cols.remove('geometry_2')
            cols.remove('histreg')
            cols.remove('bbox')
            cols.remove('Intersection')
            dfinter = pairs[cols + ['Intersection']].copy()
            dfinter.rename(columns={'Intersection': 'geometry'}, inplace=True)
            dfinter = gpd.GeoDataFrame(dfinter, columns=dfinter.columns, crs=pairs.crs)
            dfinter = dfinter.loc[~dfinter.geometry.is_empty]
            return_value = dfinter
        elif how == 'difference':
            spatial_index = df2.sindex
            df1['bbox'] = df1.geometry.apply(lambda x: x.bounds)
            df1['histreg'] = df1.bbox.apply(lambda x: list(spatial_index.intersection(x)))
            df1['new_g'] = df1.apply(lambda x: reduce(lambda x, y: x.difference(y).buffer(0),
                                                      [x.geometry] + list(df2.iloc[x.histreg].geometry)), axis=1)
            df1.geometry = df1.new_g
            df1 = df1.loc[~df1.geometry.is_empty].copy()
            df1.drop(['bbox', 'histreg', 'new_g'], axis=1, inplace=True)
            return_value = df1
        self.logger.write_time_log('Sector', 'spatial_overlays', timeit.default_timer() - spent_time)

        return return_value

    def nearest(self, row, geom_union, df1, df2, geom1_col='geometry', geom2_col='geometry', src_column=None):
        """Finds the nearest point and return the corresponding value from specified column.
        https://automating-gis-processes.github.io/2017/lessons/L3/nearest-neighbour.html#nearest-points-using-geopandas
        """
        from shapely.ops import nearest_points
        spent_time = timeit.default_timer()
        # Find the geometry that is closest
        nearest = df2[geom2_col] == nearest_points(row[geom1_col], geom_union)[1]
        # Get the corresponding value from df2 (matching is based on the geometry)
        value = df2[nearest][src_column].get_values()[0]
        self.logger.write_time_log('Sector', 'nearest', timeit.default_timer() - spent_time)

        return value

    def speciate(self, dataframe, code='default'):
        spent_time = timeit.default_timer()
        self.logger.write_log('\t\tSpeciating {0} emissions'.format(code), message_level=2)

        new_dataframe = pd.DataFrame(index=dataframe.index, data=None)
        for out_pollutant in self.output_pollutants:
            if out_pollutant != 'PMC':
                self.logger.write_log("\t\t\t{0} = ({1}/{2})*{3}".format(
                    out_pollutant, self.speciation_map[out_pollutant],
                    self.molecular_weights[self.speciation_map[out_pollutant]],
                    self.speciation_profile.loc[code, out_pollutant]), message_level=3)
                if self.speciation_map[out_pollutant] in dataframe.columns.values:
                    new_dataframe[out_pollutant] = (dataframe[self.speciation_map[out_pollutant]] /
                                                    self.molecular_weights[self.speciation_map[out_pollutant]]) * \
                                                   self.speciation_profile.loc[code, out_pollutant]
            else:
                self.logger.write_log("\t\t\t{0} = ({1}/{2} - {4}/{5})*{3}".format(
                    out_pollutant, 'pm10', self.molecular_weights['pm10'],
                    self.speciation_profile.loc[code, out_pollutant], 'pm25', self.molecular_weights['pm25']),
                    message_level=3)

                new_dataframe[out_pollutant] = \
                    ((dataframe['pm10'] / self.molecular_weights['pm10']) -
                     (dataframe['pm25'] / self.molecular_weights['pm25'])) * \
                    self.speciation_profile.loc[code, out_pollutant]
        self.logger.write_time_log('Sector', 'speciate', timeit.default_timer() - spent_time)
        return new_dataframe

    def get_output_pollutants(self, input_pollutant):
        spent_time = timeit.default_timer()
        return_value = [outs for outs, ints in self.speciation_map.items() if ints == input_pollutant]
        self.logger.write_time_log('Sector', 'get_output_pollutants', timeit.default_timer() - spent_time)
        return return_value
