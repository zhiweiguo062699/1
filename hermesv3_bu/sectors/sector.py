#!/usr/bin/env python

import sys
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from mpi4py import MPI


class Sector(object):

    def __init__(self, comm, auxiliary_dir, grid_shp, clip, date_array, source_pollutants, vertical_levels,
                 weekly_profiles_path, hourly_profiles_path, speciation_map_path, speciation_profiles_path,
                 molecular_weights_path):
        """
        Initialize the main sector class with the common arguments and methods.

        :param comm: Communicator for the sector calculation.
        :type comm: MPI.Comm

        :param auxiliary_dir: Path to the directory where the necessary auxiliary files will be created if them are not
            created yet.
        :type auxiliary_dir: str

        :param grid_shp: Shapefile with the grid horizontal distribution.
        :type grid_shp: geopandas.GeoDataFrame

        :param date_array: List of datetimes.
        :type date_array: list(datetime.datetime, ...)

        :param source_pollutants: List of input pollutants to take into account.
        :type source_pollutants: list

        :param weekly_profiles_path: Path to the CSV file that contains all the weekly profiles. The CSV file must
            contain the following columns [P_week, Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday]
            The P_week code have to be the input pollutant.
        :type weekly_profiles_path: str

        :param hourly_profiles_path: Path to the CSV file that contains all the hourly profiles. The CSV file must
            contain the following columns [P_hour, 0, 1, 2, 3, ..., 22, 23]
            The P_week code have to be the input pollutant.
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
        self.comm = comm
        self.auxiliray_dir = auxiliary_dir
        self.grid_shp = grid_shp
        self.clip = clip
        self.date_array = date_array
        self.source_pollutants = source_pollutants

        self.vertical_levels = vertical_levels

        # Reading temporal profiles
        self.weekly_profiles = self.read_weekly_profiles(weekly_profiles_path)
        self.hourly_profiles = self.read_hourly_profiles(hourly_profiles_path)

        # Reading speciation files
        self.speciation_map = self.read_speciation_map(speciation_map_path)
        self.speciation_profile = self.read_speciation_profiles(speciation_profiles_path)
        self.molecular_weights = self.read_molecular_weights(molecular_weights_path)

        self.output_pollutants = self.speciation_map.keys()

    def read_speciation_profiles(self, path):
        dataframe = pd.read_csv(path)
        dataframe.set_index('ID', inplace=True)
        return dataframe

    def read_speciation_map(self, path):
        dataframe = pd.read_csv(path)
        if 'PMC' in dataframe['dst'].values and all(element in self.source_pollutants for element in ['pm10', 'pm25']):
            dataframe_aux = dataframe.loc[dataframe['src'].isin(self.source_pollutants), :]
            dataframe = pd.concat([dataframe_aux, dataframe.loc[dataframe['dst'] == 'PMC', :]])
        else:
            dataframe = dataframe.loc[dataframe['src'].isin(self.source_pollutants), :]

        dataframe = dict(zip(dataframe['dst'], dataframe['src']))
        return dataframe

    def read_molecular_weights(self, path):
        dataframe = pd.read_csv(path)
        dataframe = dataframe.loc[dataframe['Specie'].isin(self.source_pollutants)]

        mol_wei = dict(zip(dataframe['Specie'], dataframe['MW']))
        return mol_wei

    @staticmethod
    def read_profiles(path, sep=','):
        """
        Read the CSV profile.

        :param path: Path to the CSV file that contains the profiles
        :type path: str

        :param sep: Separator of the values. [default -> ',']
        :type sep: str

        :return: Dataframe with the profiles.
        :rtype: pandas.Dataframe
        """
        dataframe = pd.read_csv(path, sep=sep)
        return dataframe

    @staticmethod
    def read_monthly_profiles(path):
        """
        Read the Dataset of the monthly profiles with the month number as columns.

        :param path: Path to the file that contains the monthly profiles.
        :type path: str

        :return: Dataset od the monthly profiles.
        :rtype: pandas.DataFrame
        """
        if path is None:
            return None
        profiles = pd.read_csv(path)

        profiles.rename(
            columns={'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7,
                     'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12},
            inplace=True)

        return profiles

    @staticmethod
    def read_weekly_profiles(path):
        """
        Read the Dataset of the weekly profiles with the weekdays as numbers (Monday: 0 - Sunday:6) as columns.


        :param path: Path to the file that contains the weekly profiles.
        :type path: str

        :return: Dataset od the weekly profiles.
        :rtype: pandas.DataFrame
        """
        if path is None:
            return None
        profiles = pd.read_csv(path)

        profiles.rename(
            columns={'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5,
                     'Sunday': 6, }, inplace=True)
        profiles.set_index('P_week', inplace=True)
        return profiles

    @staticmethod
    def read_hourly_profiles(path):
        """
        Read the Dataset of the hourly profiles with the hours (int) as columns.

        :param path: Path to the file that contains the monthly profiles.
        :type path: str

        :return: Dataset od the monthly profiles.
        :rtype: pandas.DataFrame
        """
        if path is None:
            return None
        profiles = pd.read_csv(path)
        profiles.rename(
            columns={'P_hour': -1, '00': 0, '01': 1, '02': 2, '03': 3, '04': 4, '05': 5, '06': 6, '07': 7,
                     '08': 8, '09': 9, '10': 10, '11': 11, '12': 12, '13': 13, '14': 14, '15': 15, '16': 16,
                     '17': 17, '18': 18, '19': 19, '20': 20, '21': 21, '22': 22, '23': 23}, inplace=True)
        profiles.columns = profiles.columns.astype(int)
        profiles.rename(columns={-1: 'P_hour'}, inplace=True)

        return profiles

    def calculate_rebalance_factor(self, profile_array, date):
        profile = {}
        for day, factor in enumerate(profile_array):
            profile[day] = factor

        # print [hour.date() for hour in date]
        weekdays = self.calculate_weekdays(date)

        rebalanced_profile = self.calculate_weekday_factor_full_month(profile, weekdays)

        return rebalanced_profile

    @staticmethod
    def calculate_weekday_factor_full_month(profile, weekdays):
        """
        Operate with all the days of the month to get the sum of daily factors of the full month.

        :param profile: input profile
        :type profile: dict

        :param weekdays: Dictionary with the number of days of each day type (Monday, Tuesday, ...)
        :type weekdays: dict

        :return: Dictionary with the corrected profile.
        :rtype: dict
        """
        weekdays_factors = 0
        num_days = 0
        for day in xrange(7):
            weekdays_factors += profile[day] * weekdays[day]
            num_days += weekdays[day]
        increment = float(num_days - weekdays_factors) / num_days
        for day in xrange(7):
            profile[day] = (increment + profile[day]) / num_days
        return profile

    @staticmethod
    def calculate_weekdays(date):
        """
        Calculate the number of days of each day type for the given month of the year.

        :param date: Date to select the month to evaluate.
        :type date: datetime.datetime

        :return: Dictionary with the number of days of each day type (Monday, Tuesday, ...)
        :rtype: dict
        """
        from calendar import monthrange, weekday, MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY
        weekdays = [MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY]
        days = [weekday(date.year, date.month, d + 1) for d in xrange(monthrange(date.year, date.month)[1])]

        weekdays_dict = {}
        for i, day in enumerate(weekdays):
            weekdays_dict[i] = days.count(day)

        return weekdays_dict

    def add_dates(self, dataframe):
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
        dataframe.drop('date_utc', axis=1, inplace=True)
        return dataframe

    @staticmethod
    def add_timezone(dataframe):
        from timezonefinder import TimezoneFinder

        dataframe = dataframe.to_crs({'init': 'epsg:4326'})
        tzfinder = TimezoneFinder()
        dataframe['timezone'] = dataframe.centroid.apply(lambda x: tzfinder.timezone_at(lng=x.x, lat=x.y))
        dataframe.reset_index(inplace=True)

        return dataframe

    @staticmethod
    def to_timezone(dataframe):
        """
        Set the local date with the correspondent timezone substituting the UTC date.

        :param dataframe: DataFrame with the UTC date column.
        :type dataframe: pandas.DataFrame

        :return: Catalog with the local date column.
        :rtype: pandas.DataFrame
        """
        dataframe['date'] = dataframe.groupby('timezone')['date'].apply(
            lambda x: x.dt.tz_convert(x.name).dt.tz_localize(None))

        dataframe.drop('timezone', axis=1, inplace=True)

        return dataframe

    @staticmethod
    def spatial_overlays(df1, df2, how='intersection'):
        '''Compute overlay intersection of two
            GeoPandasDataFrames df1 and df2
            https://github.com/geopandas/geopandas/issues/400
        '''
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
            dfinter = dfinter.loc[dfinter.geometry.is_empty == False]
            return dfinter
        elif how == 'difference':
            spatial_index = df2.sindex
            df1['bbox'] = df1.geometry.apply(lambda x: x.bounds)
            df1['histreg'] = df1.bbox.apply(lambda x: list(spatial_index.intersection(x)))
            df1['new_g'] = df1.apply(lambda x: reduce(lambda x, y: x.difference(y).buffer(0),
                                                      [x.geometry] + list(df2.iloc[x.histreg].geometry)), axis=1)
            df1.geometry = df1.new_g
            df1 = df1.loc[df1.geometry.is_empty == False].copy()
            df1.drop(['bbox', 'histreg', 'new_g'], axis=1, inplace=True)
            return df1

    def speciate(self, dataframe, code):
        print('Speciation')
        new_dataframe = pd.DataFrame(index=dataframe.index, data=None)
        for out_pollutant in self.output_pollutants:
            if out_pollutant != 'PMC':
                print "{0} = ({1}/{2})*{3}".format(out_pollutant,
                                                   self.speciation_map[out_pollutant],
                                                   self.molecular_weights[self.speciation_map[out_pollutant]],
                                                   self.speciation_profile.loc[code, out_pollutant],)
                if self.speciation_map[out_pollutant] in dataframe.columns.values:
                    new_dataframe[out_pollutant] = (dataframe[self.speciation_map[out_pollutant]] /
                                                    self.molecular_weights[self.speciation_map[out_pollutant]]) * \
                                                   self.speciation_profile.loc[code, out_pollutant]
            else:
                print "{0} = ({1}/{2} - {4}/{5})*{3}".format(out_pollutant, 'pm10', self.molecular_weights['pm10'],
                                                             self.speciation_profile.loc[code, out_pollutant],
                                                             'pm25', self.molecular_weights['pm25'],)
                new_dataframe[out_pollutant] = \
                    ((dataframe['pm10'] / self.molecular_weights['pm10']) -
                     (dataframe['pm25'] / self.molecular_weights['pm25'])) * \
                    self.speciation_profile.loc[code, out_pollutant]
        return new_dataframe
