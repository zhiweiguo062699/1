#!/usr/bin/env python

import sys
import os
import timeit
from warnings import warn

import geopandas as gpd
import pandas as pd
import numpy as np

from hermesv3_bu.sectors.agricultural_sector import AgriculturalSector
from hermesv3_bu.io_server.io_shapefile import IoShapefile
from hermesv3_bu.tools.checker import check_files


class AgriculturalMachinerySector(AgriculturalSector):
    def __init__(self, comm_agr, comm, logger, auxiliary_dir, grid, clip, date_array, source_pollutants,
                 vertical_levels, crop_list, nut_shapefile, machinery_list, land_uses_path, ef_files_dir,
                 monthly_profiles_path, weekly_profiles_path, hourly_profiles_path, speciation_map_path,
                 speciation_profiles_path, molecular_weights_path, landuse_by_nut, crop_by_nut, crop_from_landuse_path,
                 machinery_distribution_nut_shapefile_path, deterioration_factor_path, load_factor_path,
                 vehicle_ratio_path, vehicle_units_path, vehicle_workhours_path, vehicle_power_path,
                 crop_machinery_nuts3):
        spent_time = timeit.default_timer()

        logger.write_log('===== AGRICULTURAL MACHINERY SECTOR =====')
        check_files(
            [nut_shapefile, land_uses_path, ef_files_dir, monthly_profiles_path, weekly_profiles_path,
             hourly_profiles_path, speciation_map_path, speciation_profiles_path, molecular_weights_path,
             landuse_by_nut, crop_by_nut, crop_from_landuse_path, machinery_distribution_nut_shapefile_path,
             deterioration_factor_path, load_factor_path, vehicle_ratio_path, vehicle_units_path,
             vehicle_workhours_path, vehicle_power_path, crop_machinery_nuts3])
        super(AgriculturalMachinerySector, self).__init__(
            comm_agr, comm, logger, auxiliary_dir, grid, clip, date_array, nut_shapefile, source_pollutants,
            vertical_levels, crop_list, land_uses_path, landuse_by_nut, crop_by_nut, crop_from_landuse_path,
            ef_files_dir, monthly_profiles_path, weekly_profiles_path, hourly_profiles_path, speciation_map_path,
            speciation_profiles_path, molecular_weights_path)

        self.machinery_list = machinery_list
        self.crop_machinery_nuts3 = self.read_profiles(crop_machinery_nuts3)

        self.crop_distribution = self.get_crop_distribution_by_nut(
            self.crop_distribution, machinery_distribution_nut_shapefile_path, nut_code='nuts3_id')

        self.months = self.get_date_array_by_month()

        self.deterioration_factor = self.read_profiles(deterioration_factor_path)
        self.load_factor = self.read_profiles(load_factor_path)
        self.vehicle_ratio = self.read_profiles(vehicle_ratio_path)
        self.vehicle_units = self.read_profiles(vehicle_units_path)
        self.vehicle_workhours = self.read_profiles(vehicle_workhours_path)
        self.vehicle_power = self.read_profiles(vehicle_power_path)
        self.emission_factors = self.read_profiles(ef_files_dir)

        self.logger.write_time_log('AgriculturalMachinerySector', '__init__', timeit.default_timer() - spent_time)

    def get_crop_distribution_by_nut(self, crop_distribution, nut_shapefile, nut_code=None, write_crop_by_nut=False):
        spent_time = timeit.default_timer()

        def get_fraction(dataframe):
            total_crop_sum = self.crop_machinery_nuts3.loc[self.crop_machinery_nuts3[nut_code] == int(dataframe.name),
                                                           self.crop_list].values.sum()
            dataframe['fraction'] = dataframe[self.crop_list].sum(axis=1) / total_crop_sum

            return dataframe.loc[:, ['fraction']]

        crop_distribution.reset_index(inplace=True)

        crop_distribution_nut_path = os.path.join(self.auxiliary_dir, 'agriculture', 'crops_nuts3')
        if not os.path.exists(crop_distribution_nut_path):
            nut_shapefile = gpd.read_file(nut_shapefile)
            if nut_code is not None:
                nut_shapefile = nut_shapefile.loc[:, [nut_code, 'geometry']]

            nut_shapefile = nut_shapefile.to_crs(crop_distribution.crs)
            crop_distribution['src_inter_fraction'] = crop_distribution.geometry.area

            crop_distribution = self.spatial_overlays(crop_distribution, nut_shapefile, how='intersection')
            crop_distribution['src_inter_fraction'] = \
                crop_distribution.geometry.area / crop_distribution['src_inter_fraction']

            crop_distribution[self.crop_list] = \
                crop_distribution.loc[:, self.crop_list].multiply(crop_distribution["src_inter_fraction"], axis="index")

            crop_distribution.drop(columns=['src_inter_fraction', 'idx1', 'idx2'], inplace=True)

            if write_crop_by_nut:
                crop_distribution.loc[:, self.crop_list + [nut_code]].groupby(nut_code).sum().reset_index().to_csv(
                    self.crop_machinery_nuts3)
            crop_distribution['fraction'] = crop_distribution.groupby(nut_code).apply(get_fraction)
            crop_distribution.drop(columns=self.crop_list, inplace=True)
            crop_distribution.rename(columns={nut_code: 'NUT_code'}, inplace=True)

            IoShapefile(self.comm).write_shapefile_parallel(crop_distribution, crop_distribution_nut_path)
        else:
            crop_distribution = IoShapefile(self.comm).read_shapefile(crop_distribution_nut_path)

        self.logger.write_time_log('AgriculturalMachinerySector', 'get_crop_distribution_by_nut',
                                   timeit.default_timer() - spent_time)

        return crop_distribution

    def get_date_array_by_month(self):
        spent_time = timeit.default_timer()
        month_array = [hour.date().month for hour in self.date_array]
        month_list, num_days = np.unique(month_array, return_counts=True)

        month_dict = {}
        for month in month_list:
            month_dict[month] = np.array(self.date_array)[month_array == month]

        self.logger.write_time_log('AgriculturalMachinerySector', 'get_date_array_by_month',
                                   timeit.default_timer() - spent_time)
        return month_dict

    def calcualte_yearly_emissions_by_nut_vehicle(self):
        spent_time = timeit.default_timer()

        def get_n(df):
            try:
                df['N'] = self.vehicle_units.loc[self.vehicle_units['nuts3_id'] == df.name[0], df.name[1]].values[0]
            except IndexError:
                warn("*WARNING*: NUT3_ID {0} not found in the {1} file".format(
                    df.name[0], 'crop_machinery_vehicle_units_path'))
                df['N'] = 0.0
            return df.loc[:, ['N']]

        def get_s(df):
            try:
                df['S'] = self.vehicle_ratio.loc[
                    (self.vehicle_ratio['nuts3_id'] == df.name[0]) & (self.vehicle_ratio['technology'] == df.name[2]),
                    df.name[1]].values[0]
            except IndexError:
                warn("*WARNING*: NUT3_ID {0} not found in the {1} file".format(
                    df.name[0], 'crop_machinery_vehicle_ratio_path'))
                df['S'] = 0.0
            return df.loc[:, ['S']]

        def get_t(df):

            try:
                df['T'] = self.vehicle_workhours.loc[(self.vehicle_workhours['nuts3_id'] == df.name[0]) &
                                                     (self.vehicle_workhours['technology'] == df.name[2]),
                                                     df.name[1]].values[0]
            except IndexError:
                df['T'] = np.nan
            try:
                df.loc[df['T'].isna(), 'T'] = self.vehicle_workhours.loc[
                    (self.vehicle_workhours['nuts3_id'] == df.name[0]) & (self.vehicle_workhours['technology'] ==
                                                                          'default'), df.name[1]].values[0]
            except IndexError:
                warn("*WARNING*: NUT3_ID {0} not found in the {1} file".format(
                    df.name[0], 'crop_machinery_vehicle_workhours_path'))
                df.loc[df['T'].isna(), 'T'] = 0.0
            return df.loc[:, ['T']]

        def get_p(df):
            try:
                df['P'] = self.vehicle_power.loc[self.vehicle_power['nuts3_id'] == df.name[0], df.name[1]].values[0]
            except IndexError:
                warn("*WARNING*: NUT3_ID {0} not found in the {1} file".format(
                    df.name[0], 'crop_machinery_vehicle_power_path'))
                df['P'] = 0.0
            return df.loc[:, ['P']]

        def get_lf(df):
            df['LF'] = self.load_factor.loc[self.load_factor['vehicle'] == df.name, 'LF'].values[0]
            return df.loc[:, ['LF']]

        def get_df(df):
            try:
                df['DF_{0}'.format(in_p)] = 1 + self.deterioration_factor.loc[
                    (self.deterioration_factor['vehicle'] == df.name[0]) & (
                            self.deterioration_factor['technology'] == df.name[1]), 'DF_{0}'.format(in_p)].values[0]
            except (KeyError, IndexError):
                df['DF_{0}'.format(in_p)] = 1
            return df.loc[:, ['DF_{0}'.format(in_p)]]

        def get_ef(df):
            emission_factors = self.emission_factors.loc[(self.emission_factors['vehicle'] == df.name[0]) &
                                                         (self.emission_factors['technology'] == df.name[1]),
                                                         ['power_min', 'power_max', 'EF_{0}'.format(in_p)]]
            df['EF_{0}'.format(in_p)] = None
            for i, emission_factor in emission_factors.iterrows():
                if np.isnan(emission_factor['power_min']) and not np.isnan(emission_factor['power_max']):
                    df.loc[df['P'] < emission_factor['power_max'], 'EF_{0}'.format(in_p)] = emission_factor[
                        'EF_{0}'.format(in_p)]
                elif not np.isnan(emission_factor['power_min']) and not np.isnan(emission_factor['power_max']):
                    df.loc[(df['P'] >= emission_factor['power_min']) & (df['P'] < emission_factor['power_max']),
                           'EF_{0}'.format(in_p)] = emission_factor['EF_{0}'.format(in_p)]
                elif not np.isnan(emission_factor['power_min']) and np.isnan(emission_factor['power_max']):
                    df.loc[df['P'] >= emission_factor['power_min'], 'EF_{0}'.format(in_p)] = emission_factor[
                        'EF_{0}'.format(in_p)]
                else:
                    df['EF_{0}'.format(in_p)] = emission_factor['EF_{0}'.format(in_p)]

            return df.loc[:, ['EF_{0}'.format(in_p)]]

        nut_codes = np.unique(self.crop_distribution['NUT_code'].values.astype(np.int16))
        tech = np.unique(self.vehicle_ratio['technology'].values)

        database = pd.DataFrame(None, pd.MultiIndex.from_product(
            [nut_codes, self.machinery_list, tech], names=['NUT_code', 'vehicle', 'technology']))
        database['N'] = database.groupby(['NUT_code', 'vehicle']).apply(get_n)
        database['S'] = database.groupby(['NUT_code', 'vehicle', 'technology']).apply(get_s)
        database.dropna(inplace=True)
        database['T'] = database.groupby(['NUT_code', 'vehicle', 'technology']).apply(get_t)
        database['P'] = database.groupby(['NUT_code', 'vehicle']).apply(get_p)
        database['LF'] = database.groupby('vehicle').apply(get_lf)
        for in_p in self.source_pollutants:
            database['DF_{0}'.format(in_p)] = database.groupby(['vehicle', 'technology']).apply(get_df)

            database['EF_{0}'.format(in_p)] = database.groupby(['vehicle', 'technology'])[['P']].apply(get_ef)

            database[in_p] = database['N'] * database['S'] * database['T'] * database['P'] * database['LF'] * \
                database['DF_{0}'.format(in_p)] * database['EF_{0}'.format(in_p)]

            database.drop(columns=['DF_{0}'.format(in_p), 'EF_{0}'.format(in_p)], inplace=True)

        database.drop(columns=['N', 'S', 'T', 'P', 'LF'], inplace=True)

        database = database.groupby(['NUT_code', 'vehicle']).sum()
        self.logger.write_time_log('AgriculturalMachinerySector', 'calcualte_yearly_emissions_by_nut_vehicle',
                                   timeit.default_timer() - spent_time)
        return database

    def calculate_monthly_emissions_by_nut(self, month):
        spent_time = timeit.default_timer()

        def get_mf(df, month_num):
            df['MF'] = self.monthly_profiles.loc[df.name, month_num]
            return df.loc[:, ['MF']]
        # month_distribution = self.crop_distribution.loc[:, ['FID', 'timezone', 'geometry']].copy()
        dataframe = self.calcualte_yearly_emissions_by_nut_vehicle().reset_index()
        dataframe['MF'] = dataframe.groupby('vehicle').apply(
            lambda x: get_mf(x, month)
        )
        dataframe[self.source_pollutants] = dataframe[self.source_pollutants].multiply(dataframe['MF'], axis=0)

        dataframe.drop(columns=['MF'], inplace=True)

        dataframe = dataframe.groupby('NUT_code').sum()

        self.logger.write_time_log('AgriculturalMachinerySector', 'calculate_monthly_emissions_by_nut',
                                   timeit.default_timer() - spent_time)
        return dataframe

    def distribute(self, dataframe):
        spent_time = timeit.default_timer()

        def distribute_by_nut(df, nut_emissions):
            aux = df.apply(lambda row: row * nut_emissions)
            return aux.loc[:, self.source_pollutants]

        self.crop_distribution.reset_index(inplace=True)
        self.crop_distribution[self.source_pollutants] = self.crop_distribution.groupby('NUT_code')['fraction'].apply(
            lambda x: distribute_by_nut(x, dataframe.loc[int(x.name), self.source_pollutants])
        )
        self.crop_distribution.drop(columns=['fraction', 'NUT_code'], inplace=True)
        timezones = self.crop_distribution.groupby('FID')[['timezone']].first()
        self.crop_distribution = self.crop_distribution.reset_index().groupby('FID').sum()

        self.crop_distribution['timezone'] = timezones
        self.crop_distribution.reset_index(inplace=True)
        self.logger.write_time_log('AgriculturalMachinerySector', 'distribute',
                                   timeit.default_timer() - spent_time)
        return self.crop_distribution

    def add_dates(self, df_by_month):
        spent_time = timeit.default_timer()

        df_list = []
        for tstep, date in enumerate(self.date_array):
            df_aux = df_by_month[date.date().month].copy()
            df_aux['date'] = pd.to_datetime(date, utc=True)
            df_aux['date_utc'] = pd.to_datetime(date, utc=True)
            df_aux['tstep'] = tstep
            # df_aux = self.to_timezone(df_aux)
            df_list.append(df_aux)
        dataframe_by_day = pd.concat(df_list, ignore_index=True)

        dataframe_by_day = self.to_timezone(dataframe_by_day)
        self.logger.write_time_log('AgriculturalMachinerySector', 'add_dates', timeit.default_timer() - spent_time)
        return dataframe_by_day

    def calculate_hourly_emissions(self):
        spent_time = timeit.default_timer()

        def get_wf(df):
            """
            Get the Weekly Factor for the given dataframe depending on the date.

            :param df: DataFrame where find the weekly factor. df.name is the date.
            :type df: DataFrame

            :return: DataFrame with only the WF column.
            :rtype: DataFrame
            """
            weekly_profile = self.calculate_rebalanced_weekly_profile(self.weekly_profiles.loc['default', :].to_dict(),
                                                                      df.name)
            df['WF'] = weekly_profile[df.name.weekday()]
            return df.loc[:, ['WF']]

        def get_hf(df):
            """
            Get the Hourly Factor for the given dataframe depending on the hour.

            :param df: DataFrame where find the hourly factor. df.name is the hour.
            :type df: DataFrame

            :return: DataFrame with only the HF column.
            :rtype: DataFrame
            """
            hourly_profile = self.hourly_profiles.loc['default', :].to_dict()
            hour_factor = hourly_profile[df.name]

            df['HF'] = hour_factor
            return df.loc[:, ['HF']]

        self.crop_distribution['date_as_date'] = self.crop_distribution['date'].dt.date
        self.crop_distribution['month'] = self.crop_distribution['date'].dt.weekday
        self.crop_distribution['weekday'] = self.crop_distribution['date'].dt.weekday
        self.crop_distribution['hour'] = self.crop_distribution['date'].dt.hour

        for pollutant in self.source_pollutants:
            self.crop_distribution['WF'] = self.crop_distribution.groupby(['date_as_date']).apply(get_wf)

            self.crop_distribution['HF'] = self.crop_distribution.groupby('hour').apply(get_hf)
            self.crop_distribution[pollutant] = self.crop_distribution[pollutant].multiply(
                self.crop_distribution['HF'] * self.crop_distribution['WF'], axis=0)

        self.crop_distribution.drop(columns=['month', 'weekday', 'hour', 'WF', 'HF', 'date_as_date'], inplace=True)
        self.logger.write_time_log('AgriculturalMachinerySector', 'calculate_hourly_emissions',
                                   timeit.default_timer() - spent_time)
        return self.crop_distribution

    def calculate_emissions(self):
        spent_time = timeit.default_timer()
        self.logger.write_log('\tCalculating emissions')

        distribution_by_month = {}
        for month in self.months.keys():
            distribution_by_month[month] = self.calculate_monthly_emissions_by_nut(month)
            distribution_by_month[month] = self.distribute(distribution_by_month[month])

        self.crop_distribution = self.add_dates(distribution_by_month)
        self.crop_distribution.drop('date_utc', axis=1, inplace=True)
        self.crop_distribution = self.calculate_hourly_emissions()
        self.crop_distribution['layer'] = 0
        self.crop_distribution = self.crop_distribution.groupby(['FID', 'layer', 'tstep']).sum()
        self.crop_distribution = self.speciate(self.crop_distribution)

        self.logger.write_log('\t\tAgricultural machinery emissions calculated', message_level=2)
        self.logger.write_time_log('AgriculturalMachinerySector', 'calculate_emissions',
                                   timeit.default_timer() - spent_time)
        return self.crop_distribution
