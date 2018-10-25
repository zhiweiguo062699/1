#!/usr/bin/env python

import sys
import os
from timeit import default_timer as gettime
# import pandas as pd
# import geopandas as gpd
# import numpy as np
# from shapely.ops import nearest_points
# import warnings
import IN.src.config.settings as settings

# TODO some pollutants


class PointSource(object):
    """
    Class to calculate the Point Source emissions

    :param grid: Grid of the destination domain
    :type grid: Grid

    :param catalog_path: Path to the fine that contains all the information for each point source.
    :type catalog_path: str

    :param monthly_profiles_path: Path to the file that contains the monthly profiles.
    :type monthly_profiles_path: str

    :param daily_profiles_path: Path to the file that contains the daily profiles.
    :type daily_profiles_path: str

    :param hourly_profiles_path: Path to the file that contains the hourly profile.
    :type hourly_profiles_path: str

    :param speciation_map_path: Path to the file that contains the speciation map.
    :type speciation_map_path: str

    :param speciation_profiles_path: Path to the file that contains the speciation profiles.
    :type speciation_profiles_path: str

    :param sector_list: List os sectors (SNAPS) to take into account. 01, 03, 04, 09
    :type sector_list: list
    """
    def __init__(self, grid, catalog_path, monthly_profiles_path, daily_profiles_path, hourly_profiles_path,
                 speciation_map_path, speciation_profiles_path, sector_list, effective_stack_height, pollutant_list,
                 measured_emission_path, molecular_weights_path=None):
        import pandas as pd

        self.pollutant_list = pollutant_list

        self.catalog = self.read_catalog(catalog_path, sector_list)
        self.catalog_measured = self.read_catalog_for_measured_emissions(catalog_path, sector_list)
        self.measured_path = measured_emission_path

        self.grid = grid

        self.monthly_profiles = self.read_monthly_profiles(monthly_profiles_path)
        self.daily_profiles = self.read_daily_profiles(daily_profiles_path)
        self.hourly_profiles = self.read_hourly_profiles(hourly_profiles_path)

        self.speciation_map = self.read_speciation_map(speciation_map_path)
        self.speciation_profiles = self.read_speciation_profiles(speciation_profiles_path)
        self.effective_stack_height = effective_stack_height

        self.molecular_weigths = pd.read_csv(molecular_weights_path, sep=';')

    @staticmethod
    def read_speciation_map(path):
        """
        Read the Dataset of the speciation map.

        :param path: Path to the file that contains the speciation map.
        :type path: str

        :return: Dataset od the speciation map.
        :rtype: pandas.DataFrame
        """
        import pandas as pd

        map = pd.read_csv(path, sep=';')

        return map

    @staticmethod
    def read_speciation_profiles(path):
        """
        Read the Dataset of the speciation profiles.

        :param path: Path to the file that contains the speciation profiles.
        :type path: str

        :return: Dataset od the speciation profiles.
        :rtype: pandas.DataFrame
        """
        import pandas as pd

        profiles = pd.read_csv(path, sep=',')

        return profiles

    @staticmethod
    def read_monthly_profiles(path):
        """
        Read the Dataset of the monthly profiles with the month number as columns.

        :param path: Path to the file that contains the monthly profiles.
        :type path: str

        :return: Dataset od the monthly profiles.
        :rtype: pandas.DataFrame
        """
        import pandas as pd

        profiles = pd.read_csv(path)

        profiles.rename(columns={'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7,
                                 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12},
                        inplace=True)
        return profiles

    @staticmethod
    def read_daily_profiles(path):
        """
        Read the Dataset of the daily profiles with the days as numbers (Monday: 0 - Sunday:6) as columns.


        :param path: Path to the file that contains the daily profiles.
        :type path: str

        :return: Dataset od the daily profiles.
        :rtype: pandas.DataFrame
        """
        import pandas as pd

        profiles = pd.read_csv(path)

        profiles.rename(columns={'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5,
                                 'Sunday': 6, }, inplace=True)
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
        import pandas as pd

        profiles = pd.read_csv(path)
        profiles.rename(columns={'P_hour': -1, '00': 0, '01': 1, '02': 2, '03': 3, '04': 4, '05': 5, '06': 6, '07': 7,
                                 '08': 8, '09': 9, '10': 10, '11': 11, '12': 12, '13': 13, '14': 14, '15': 15, '16': 16,
                                 '17': 17, '18': 18, '19': 19, '20': 20, '21': 21, '22': 22, '23': 23}, inplace=True)
        profiles.columns = profiles.columns.astype(int)
        profiles.rename(columns={-1: 'P_hour'}, inplace=True)
        return profiles

    def read_catalog(self, catalog_path, sector_list):
        """
        Read the catalog

        :param catalog_path: path to the catalog
        :type catalog_path: str

        :param sector_list: List of sectors to take into account
        :type sector_list: list

        :return: catalog
        :rtype: pandas.DataFrame
        """
        import pandas as pd
        import numpy as np

        columns = {"Code": np.str, "Cons": np.bool, "SNAP": np.str, "Lon": settings.precision,
                   "Lat": settings.precision, "Height": settings.precision, "AF": settings.precision,
                   "P_month": np.str, "P_week": np.str, "P_hour": np.str, "P_spec": np.str}
        for pollutant in self.pollutant_list:
            columns['EF_{0}'.format(pollutant)] = settings.precision

        catalog_df = pd.read_csv(catalog_path, usecols=columns.keys(), dtype=columns)

        # Filtering
        catalog_df = catalog_df.loc[catalog_df['Cons'] == 1, :]
        catalog_df.drop('Cons', axis=1, inplace=True)

        # Filtering
        catalog_df = catalog_df.loc[catalog_df['AF'] != -1, :]

        if sector_list is not None:
            catalog_df = catalog_df.loc[catalog_df['SNAP'].str[:2].isin(sector_list)]
        catalog_df.drop('SNAP', axis=1, inplace=True)

        # TODO Select only involved point sources in the working domain

        return catalog_df

    def read_catalog_for_measured_emissions(self, catalog_path, sector_list):
        """
        Read the catalog

        :param catalog_path: path to the catalog
        :type catalog_path: str

        :param sector_list: List of sectors to take into account
        :type sector_list: list

        :return: catalog
        :rtype: pandas.DataFrame
        """
        import pandas as pd
        import numpy as np

        columns = {"Code": np.str, "Cons": np.bool, "SNAP": np.str, "Lon": settings.precision,
                   "Lat": settings.precision, "Height": settings.precision, "AF": settings.precision, "P_spec": np.str}
        # for pollutant in self.pollutant_list:
        #     columns['EF_{0}'.format(pollutant)] = settings.precision

        catalog_df = pd.read_csv(catalog_path, usecols=columns.keys(), dtype=columns)

        # Filtering
        catalog_df = catalog_df.loc[catalog_df['Cons'] == 1, :]
        catalog_df.drop('Cons', axis=1, inplace=True)

        # Filtering
        catalog_df = catalog_df.loc[catalog_df['AF'] == -1, :]
        catalog_df.drop('AF', axis=1, inplace=True)

        if sector_list is not None:
            catalog_df = catalog_df.loc[catalog_df['SNAP'].str[:2].isin(sector_list)]
        catalog_df.drop('SNAP', axis=1, inplace=True)

        # TODO Select only involved point sources in the working domain

        return catalog_df

    @staticmethod
    def to_geodataframe(catalog):
        """
        Convert a simple DataFrame with Lat, Lon columns into a GeoDataFrame as a shape

        :param catalog: DataFrame with all the information of each point source.
        :type catalog: pandas.DataFrame

        :return: GeoDataFrame with all the information of each point source.
        :rtype: geopandas.GeoDataFrame
        """
        import geopandas as gpd
        from shapely.geometry import Point

        geometry = [Point(xy) for xy in zip(catalog.Lon, catalog.Lat)]
        catalog.drop(['Lon', 'Lat'], axis=1, inplace=True)
        crs = {'init': 'epsg:4326'}
        catalog = gpd.GeoDataFrame(catalog, crs=crs, geometry=geometry)

        # catalog.to_file('/home/Earth/ctena/Models/HERMESv3/OUT/test/point_source.shp')

        return catalog

    def add_dates(self, catalog, st_date_utc, delta_hours):
        """
        Add to the catalog the 'date' column (in local time) and the time step ('tstep').

        :param catalog: Catalog to update
        :type catalog: pandas.DataFrame

        :param st_date_utc: Starting date in UTC.
        :type st_date_utc: datetime.datetime

        :param delta_hours: List of hours that have to sum to the first hour for each time step.
        :type delta_hours: list

        :return: Catalog with the dates
        :rtype: pandas.DataFrame
        """
        from datetime import timedelta
        import pandas as pd

        catalog = self.add_timezone(catalog)

        list_catalogs = []
        for index, hour in enumerate(delta_hours):
            catalog_aux = catalog.copy()
            catalog_aux['date'] = pd.to_datetime(st_date_utc + timedelta(hours=hour), utc=True)
            catalog_aux['tstep'] = index
            list_catalogs.append(catalog_aux)

        catalog = pd.concat(list_catalogs)
        catalog.reset_index(drop=True, inplace=True)

        catalog = self.to_timezone(catalog)

        return catalog

    @staticmethod
    def add_timezone(catalog):
        """
        Add the timezone column with the timezone of the location of each point source.

        :param catalog: Catalog where add the timezone.
        :type catalog: pandas.DataFrame

        :return: Catalog with the added timezone column.
        :rtype: pandas.DataFrame
        """
        from timezonefinder import TimezoneFinder

        tzfinder = TimezoneFinder()

        catalog['timezone'] = catalog['geometry'].apply(lambda x: tzfinder.timezone_at(lng=x.x, lat=x.y))

        return catalog

    @staticmethod
    def to_timezone(catalog):
        """
        Set the local date with the correspondent timezone substituting the UTC date.

        :param catalog: Catalog with the UTC date column.
        :type catalog: pandas.DataFrame

        :return: Catalog with the local date column.
        :rtype: pandas.DataFrame
        """
        import pandas as pd

        catalog['date'] = catalog.groupby('timezone')['date'].apply(lambda x: x.dt.tz_convert(x.name).dt.tz_localize(None))

        catalog.drop('timezone', axis=1, inplace=True)

        return catalog

    def get_yearly_emissions(self, catalog):
        """
        Calculate yearly emissions.

        :param catalog: Catalog with the activity factor (AF) column and all the emission factor column for each
        pollutant.
        :type catalog: pandas.DataFrame

        :return: Catalog with yearly emissions of each point source for all the pollutants (as column names).
        :rtype: pandas.DataFrame
        """
        for pollutant in self.pollutant_list:
            catalog.rename(columns={u'EF_{0}'.format(pollutant): pollutant}, inplace=True)
            catalog[pollutant] = catalog[pollutant] * catalog['AF']

        catalog.drop('AF', axis=1, inplace=True)
        return catalog

    @staticmethod
    def calculate_rebalance_factor(profile, date):
        """
        Calculate the necessary factor to make consistent the full month data. This is needed for the months that if you
        sum the daily factor of each day of the month it doesn't sum as the number of days of the month.

        :param profile: Daily profile.
        :type profile: dict

        :param date: Date of the timestep to simulate.
        :type date: datetime.datetime

        :return: Dataset with the corrected values for the daily profiles.
        :rtype: pandas.DataFrame
        """
        import pandas as pd
        weekdays = PointSource.calculate_weekdays(date)
        rebalanced_profile = PointSource.calculate_weekday_factor_full_month(profile, weekdays)
        rebalanced_profile = pd.DataFrame.from_dict(rebalanced_profile)

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
        increment = (num_days - weekdays_factors) / num_days

        for day in xrange(7):
            profile[day] = [(increment + profile[day]) / num_days]

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

    def get_temporal_factors(self, catalog, st_date):
        """
        Calculates the temporal factor for each point source and each time step.

        :param catalog: Catalog with the activity factor (AF) column and all the emission factor column for each
        pollutant.
        :type catalog: pandas.DataFrame

        :return: Catalog with yearly emissions of each point source for all the pollutants (as column names).
        :rtype: pandas.DataFrame
        """
        import pandas as pd

        def set_monthly_profile(x):
            """
            Extracts the monthly profile for the given Series.

            :param x: Series to modify
            :type x: pandas.Series

            :return: Modified Series.
            :rtype: pandas.Series
            """
            profile = self.monthly_profiles[self.monthly_profiles['P_month'] == x.name]
            for month, aux_df in x.groupby(x.dt.month):
                x.loc[aux_df.index] = profile[month].values[0]
            return x

        def set_daily_profile(x, st_date):
            """
            Extracts the daily profile for the given Series and correct them with the rebalanced factor.

            :param x: Series to modify
            :type x: pandas.Series

            :param st_date: Date to evaluate. Necessary for rebalanced the factor.
            :type st_date: datetime.datetime

            :return: Modified Series.
            :rtype: pandas.Series
            """
            profile = self.daily_profiles[self.daily_profiles['P_week'] == x.name]
            profile = self.calculate_rebalance_factor(profile.to_dict('records')[0], st_date)

            for weekday, aux_df in x.groupby(x.dt.weekday):
                x.loc[aux_df.index] = profile[weekday].values[0]
            return x

        def set_hourly_profile(x):
            """
            Extracts the hourly profile for the given Series.

            :param x: Series to modify
            :type x: pandas.Series

            :return: Modified Series.
            :rtype: pandas.Series
            """
            profile = self.hourly_profiles[self.hourly_profiles['P_hour'] == x.name]
            for hour, aux_df in x.groupby(x.dt.hour):
                x.loc[aux_df.index] = profile[hour].values[0]
            return x

        catalog['P_month'] = catalog.groupby('P_month')['date'].apply(set_monthly_profile)
        catalog['P_week'] = catalog.groupby('P_week')['date'].apply(lambda x: set_daily_profile(x, st_date))
        catalog['P_hour'] = catalog.groupby('P_hour')['date'].apply(set_hourly_profile)

        catalog['temp_factor'] = catalog['P_month'] * catalog['P_week'] * catalog['P_hour']
        catalog.drop(['P_month', 'P_week', 'P_hour'], axis=1, inplace=True)

        for pollutant in self.pollutant_list:
            catalog[pollutant] = catalog[pollutant] * catalog['temp_factor']
        catalog.drop('temp_factor', axis=1, inplace=True)

        return catalog

    def calculate_hourly_emissions(self, catalog, st_date):
        """
        Calculate the hourly emissions

        :param catalog: Catalog to calculate.
        :type catalog: pandas.DataFrame

        :param st_date: Starting date to simulate (UTC).
        :type st_date: dateitme.datetime

        :return: Catalog with the hourly emissions.
        :rtype: pandas.DataFrame
        """

        catalog = self.get_yearly_emissions(catalog)
        catalog = self.get_temporal_factors(catalog, st_date)

        return catalog

    def calculate_vertical_distribution(self, catalog, vertical_levels):
        """
        Add the layer column to indicate at what layer the emission have to go.

        :param catalog: Catalog to calculate.
        :type catalog: pandas.DataFrame

        :param vertical_levels: List with the maximum altitude of each layer in meters.
        :type vertical_levels: list

        :return: Catalog with the level.
        :rtype: pandas.DataFrame
        """
        import numpy as np

        if self.effective_stack_height:
            catalog['Height'] = catalog['Height'] * 1.2

        catalog['layer'] = np.searchsorted(vertical_levels, catalog['Height'], side='left')

        catalog.drop('Height', axis=1, inplace=True)

        return catalog

    def speciate(self, catalog):
        """
        Speciate the catalog for the output pollutants.

        :param catalog: Catalog to speciate.
        :type catalog: pandas.DataFrame

        :return: Speciated catalog.
        :rtype: pandas.DataFrame
        """
        import pandas as pd
        import numpy as np

        def do_speciation(x, input_pollutant, output_pollutant):
            """
            Do the speciation for a specific pollutant.

            :param x: Serie with the pollutant to specieate.
            :type x: pandas.Series

            :param input_pollutant: Name of the input pollutant.
            :type input_pollutant: str

            :param output_pollutant: Name of the output pollutant.
            :type output_pollutant: str

            :return: Speciated Series
            :rtype: pandas.Series
            """
            mol_weight = self.molecular_weigths.loc[self.molecular_weigths['Specie'] == input_pollutant, 'MW'].values[0]

            profile = self.speciation_profiles[self.speciation_profiles['P_spec'] == x.name]
            if output_pollutant == 'PMC':
                x = catalog.loc[x.index, 'pm10'] - catalog.loc[x.index, 'pm25']

            if input_pollutant == 'nmvoc':
                x = x * profile['VOCtoTOG'].values[0] * (profile[output_pollutant].values[0] / mol_weight)
            else:
                x = x * (profile[out_p].values[0] / mol_weight)
            return x

        speciated_catalog = catalog.drop(self.pollutant_list, axis=1)

        for out_p in self.speciation_map['dst'].values:
            in_p = self.speciation_map.loc[self.speciation_map['dst'] == out_p, 'src'].values[0]
            if type(in_p) == float and np.isnan(in_p):
                in_p = 'pm10'
            speciated_catalog[out_p] = catalog.groupby('P_spec')[in_p].apply(lambda x: do_speciation(x, in_p, out_p))

        speciated_catalog.drop('P_spec', axis=1, inplace=True)

        return speciated_catalog

    def add_measured_emissions(self, catalog, st_date, delta_hours):
        def func(x, pollutant):
            import pandas as pd
            from datetime import timedelta
            measured_emissions = self.measured_path.replace('<Code>', x.name)
            measured_emissions = pd.read_csv(measured_emissions, sep=';')
            measured_emissions = measured_emissions.loc[measured_emissions['Code'] == x.name, :]

            measured_emissions['date'] = pd.to_datetime(measured_emissions['date']) + pd.to_timedelta(
                measured_emissions['local_to_UTC'], unit='h')

            measured_emissions.drop('local_to_UTC', axis=1, inplace=True)

            # dates_array = [st_date + timedelta(hours=hour) for hour in delta_hours]
            # measured_emissions = measured_emissions.loc[measured_emissions['date'].isin(dates_array), :]
            code = x.name
            x = pd.DataFrame(x)
            x.rename(columns={code: 'date'}, inplace=True)

            test = pd.merge(left=x, right=measured_emissions.loc[:, ['date', pollutant]], on='date', how='inner')
            test.set_index(x.index, inplace=True)

            return test[pollutant]
        for pollutant in self.pollutant_list:
            catalog[pollutant] = catalog.groupby('Code')['date'].apply(lambda x: func(x, pollutant))

        return catalog

    def calculate_measured_emissions(self, catalog, st_date, delta_hours):
        if len(catalog) == 0:
            return None
        else:
            catalog = self.to_geodataframe(catalog)
            catalog = self.add_dates(catalog, st_date, delta_hours)

            catalog = self.add_measured_emissions(catalog, st_date, delta_hours)

            return catalog

    def merge_catalogs(self, catalog_list):
        import pandas as pd

        catalog = pd.concat(catalog_list)

        catalog.reset_index(inplace=True)

        return catalog

    def calculate_point_source_emissions(self, st_date, delta_hours, vertical_levels):
        """
        Process to calculate the poitn source emissions.

        :param st_date: Starting date to simulate (UTC).
        :type st_date: dateitme.datetime

        :param delta_hours: List of hours that have to sum to the first hour for each time step.
        :type delta_hours: list

        :param vertical_levels: List with the maximum altitude of each layer in meters.
        :type vertical_levels: list

        :return: Catalog with the calculated emissions.
        :rtype: pandas.DataFrame
        """
        self.catalog = self.to_geodataframe(self.catalog)

        self.catalog = self.add_dates(self.catalog, st_date, delta_hours)
        self.catalog = self.calculate_hourly_emissions(self.catalog, st_date)

        self.catalog_measured = self.calculate_measured_emissions(self.catalog_measured, st_date, delta_hours)

        if self.catalog_measured is not None:
            self.catalog = self.merge_catalogs([self.catalog, self.catalog_measured])
        self.catalog = self.calculate_vertical_distribution(self.catalog, vertical_levels)

        self.catalog = self.speciate(self.catalog)

        # self.catalog['date'] = self.catalog['date'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        # self.catalog.to_file('/home/Earth/ctena/Models/HERMESv3/OUT/test/catalog.shp')
        return self.catalog

    def points_to_grid(self, catalog, grid_shape, out_list):
        """
        Add the cell location for each point source and dum the values that goes to the same cell, time step and layer.

        :param catalog: Catalog to find their possition.
        :type catalog: pandas.DataFrame

        :param grid_shape: Shapefile of the oputput grid.
        :type grid_shape: geopandas.GeoDataFrame

        :param out_list: List of output pollutants.
        :type out_list: list

        :return: List of dictionaries with the necessary information to write the netCDF.
        :rtype: list
        """
        import geopandas as gpd

        catalog = catalog.to_crs(grid_shape.crs)
        catalog = gpd.sjoin(catalog, grid_shape, how="inner", op='intersects')
        # Drops duplicates when the point source is on the boundary of the cell
        catalog = catalog[~catalog.index.duplicated(keep='first')]

        try:
            catalog.drop(['Code', 'index_right', 'date', 'geometry'], axis=1, inplace=True)
        except ValueError:
            pass

        catalog = catalog.groupby(['tstep', 'layer', 'FID']).sum()
        catalog.reset_index(inplace=True)

        emission_list = []
        for out_p in out_list:
            aux_data = catalog.loc[:, [out_p, 'tstep', 'layer', 'FID']]
            aux_data = aux_data.loc[aux_data[out_p] > 0, :]
            dict_aux = {
                'name': out_p,
                'units': '',
                'data': aux_data
            }
            # print dict_aux
            emission_list.append(dict_aux)

        return emission_list
