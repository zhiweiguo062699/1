#!/usr/bin/env python

import os
import timeit

import numpy as np
import pandas as pd
import geopandas as gpd

from hermesv3_bu.sectors.sector import Sector
from hermesv3_bu.io_server.io_shapefile import IoShapefile
from hermesv3_bu.io_server.io_raster import IoRaster
from hermesv3_bu.logger.log import Log


class RecreationalBoatsSector(Sector):
    def __init__(self, comm, logger, auxiliary_dir, grid_shp, clip, date_array, source_pollutants, vertical_levels,
                 boat_list, density_map_path, boats_data_path, ef_file_path, monthly_profiles_path,
                 weekly_profiles_path, hourly_profiles_path, speciation_map_path, speciation_profiles_path,
                 molecular_weights_path):
        spent_time = timeit.default_timer()

        super(RecreationalBoatsSector, self).__init__(
            comm, logger, auxiliary_dir, grid_shp, clip, date_array, source_pollutants, vertical_levels,
            monthly_profiles_path, weekly_profiles_path, hourly_profiles_path, speciation_map_path,
            speciation_profiles_path, molecular_weights_path)

        self.boat_list = boat_list

        # TODO Change nodata value of the raster
        self.density_map = self.create_density_map(density_map_path)
        self.boats_data_path = boats_data_path
        self.ef_file_path = ef_file_path

        self.logger.write_time_log('RecreationalBoatsSector', '__init__', timeit.default_timer() - spent_time)

    def create_density_map(self, density_map_path):
        spent_time = timeit.default_timer()
        if self.comm.Get_rank() == 0:
            density_map_auxpath = os.path.join(self.auxiliary_dir, 'recreational_boats', 'density_map.shp')
            if not os.path.exists(density_map_auxpath):
                src_density_map = IoRaster(self.comm).to_shapefile_serie(density_map_path, nodata=0)
                src_density_map = src_density_map.loc[src_density_map['data'] > 0]
                src_density_map['data'] = src_density_map['data'] / src_density_map['data'].sum()
                src_density_map.to_crs(self.grid_shp.crs, inplace=True)
                src_density_map['src_inter_fraction'] = src_density_map.area
                src_density_map = self.spatial_overlays(src_density_map, self.grid_shp.reset_index(),
                                                        how='intersection')
                src_density_map['src_inter_fraction'] = src_density_map.area / src_density_map['src_inter_fraction']

                src_density_map['data'] = src_density_map.loc[:, 'data'].multiply(src_density_map["src_inter_fraction"],
                                                                                  axis="index")

                src_density_map = src_density_map.loc[:, ['FID', 'data']].groupby('FID').sum()
                src_density_map = gpd.GeoDataFrame(src_density_map, crs=self.grid_shp.crs,
                                                   geometry=self.grid_shp.loc[src_density_map.index, 'geometry'])
                src_density_map.reset_index(inplace=True)

                IoShapefile(self.comm).write_shapefile_serial(src_density_map, density_map_auxpath)
            else:
                src_density_map = IoShapefile(self.comm).read_shapefile_serial(density_map_auxpath)
        else:
            src_density_map = None
        src_density_map = IoShapefile(self.comm).split_shapefile(src_density_map)
        src_density_map.set_index('FID', inplace=True)

        self.logger.write_time_log('RecreationalBoatsSector', 'create_density_map', timeit.default_timer() - spent_time)
        return src_density_map

    def speciate_dict(self, annual_emissions_dict):
        spent_time = timeit.default_timer()

        speciated_emissions = {}
        for out_pollutant in self.output_pollutants:
            if out_pollutant != 'PMC':
                self.logger.write_log("\t\t\t{0} = ({1}/{2})*{3}".format(
                    out_pollutant, self.speciation_map[out_pollutant],
                    self.molecular_weights[self.speciation_map[out_pollutant]],
                    self.speciation_profile.loc['default', out_pollutant]), message_level=3)

                speciated_emissions[out_pollutant] = (annual_emissions_dict[self.speciation_map[out_pollutant]] /
                                                      self.molecular_weights[self.speciation_map[out_pollutant]]
                                                      ) * self.speciation_profile.loc['default', out_pollutant]
            else:
                self.logger.write_log("\t\t\t{0} = ({1}/{2} - {4}/{5})*{3}".format(
                    out_pollutant, 'pm10', self.molecular_weights['pm10'],
                    self.speciation_profile.loc['default', out_pollutant], 'pm25', self.molecular_weights['pm25']),
                    message_level=3)
                speciated_emissions[out_pollutant] = ((annual_emissions_dict['pm10'] / self.molecular_weights['pm10']) -
                                                      (annual_emissions_dict['pm25'] / self.molecular_weights['pm25'])
                                                      ) * self.speciation_profile.loc['default', out_pollutant]
        self.logger.write_time_log('RecreationalBoatsSector', 'speciate_dict', timeit.default_timer() - spent_time)
        return speciated_emissions

    def get_annual_emissions(self):
        spent_time = timeit.default_timer()

        emissions_dict = {}

        data = pd.read_csv(self.boats_data_path, usecols=['code', 'number', 'nominal_power', 'Ann_hours', 'LF'])
        data['AF'] = data['number'] * data['Ann_hours'] * data['nominal_power'] * data['LF']
        # Emission Factor in g/kWh
        ef_dataframe = pd.read_csv(self.ef_file_path)
        dataframe = pd.merge(data.loc[:, ['code', 'AF']], ef_dataframe, on='code')
        for in_p in self.source_pollutants:
            emissions_dict[in_p] = dataframe['AF'].multiply(dataframe['EF_{0}'.format(in_p)]).sum()

        self.logger.write_time_log('RecreationalBoatsSector', 'get_annual_emissions',
                                   timeit.default_timer() - spent_time)
        return emissions_dict

    def calculate_yearly_emissions(self, annual_emissions):
        spent_time = timeit.default_timer()

        new_dataframe = self.density_map.copy()
        new_dataframe.drop(columns='data', inplace=True)

        for pollutant, annual_value in annual_emissions.items():
            new_dataframe[pollutant] = self.density_map['data'] * annual_value

        self.logger.write_time_log('RecreationalBoatsSector', 'calculate_yearly_emissions',
                                   timeit.default_timer() - spent_time)
        return new_dataframe

    def dates_to_month_weekday_hour(self, dataframe):
        spent_time = timeit.default_timer()
        dataframe['month'] = dataframe['date'].dt.month
        dataframe['weekday'] = dataframe['date'].dt.weekday
        dataframe['hour'] = dataframe['date'].dt.hour

        self.logger.write_time_log('RecreationalBoatsSector', 'dates_to_month_weekday_hour',
                                   timeit.default_timer() - spent_time)
        return dataframe

    def calculate_hourly_emissions(self, annual_distribution):
        spent_time = timeit.default_timer()

        def get_mf(df):
            month_factor = self.monthly_profiles.loc['default', df.name]

            df['MF'] = month_factor
            return df.loc[:, ['MF']]

        def get_wf(df):
            weekly_profile = self.calculate_rebalanced_weekly_profile(self.weekly_profiles.loc['default', :].to_dict(),
                                                                      df.name)
            df['WF'] = weekly_profile[df.name.weekday()]
            return df.loc[:, ['WF']]

        def get_hf(df):
            hourly_profile = self.hourly_profiles.loc['default', :].to_dict()
            hour_factor = hourly_profile[df.name]

            df['HF'] = hour_factor
            return df.loc[:, ['HF']]

        dataframe = self.add_dates(annual_distribution)
        dataframe = self.dates_to_month_weekday_hour(dataframe)

        dataframe['date_as_date'] = dataframe['date'].dt.date

        dataframe['MF'] = dataframe.groupby('month').apply(get_mf)
        dataframe[self.output_pollutants] = dataframe[self.output_pollutants].multiply(dataframe['MF'], axis=0)
        dataframe.drop(columns=['month', 'MF'], inplace=True)

        dataframe['WF'] = dataframe.groupby('date_as_date').apply(get_wf)
        dataframe[self.output_pollutants] = dataframe[self.output_pollutants].multiply(dataframe['WF'], axis=0)
        dataframe.drop(columns=['weekday', 'date', 'date_as_date', 'WF'], inplace=True)

        dataframe['HF'] = dataframe.groupby('hour').apply(get_hf)
        dataframe[self.output_pollutants] = dataframe[self.output_pollutants].multiply(dataframe['HF'], axis=0)
        dataframe.drop(columns=['hour', 'HF'], inplace=True)

        self.logger.write_time_log('RecreationalBoatsSector', 'calculate_hourly_emissions',
                                   timeit.default_timer() - spent_time)
        return dataframe

    def calculate_emissions(self):
        spent_time = timeit.default_timer()
        self.logger.write_log('\tCalculating emissions')

        annual_emissions = self.get_annual_emissions()
        annual_emissions = self.speciate_dict(annual_emissions)

        distribution = self.calculate_yearly_emissions(annual_emissions)
        distribution = self.calculate_hourly_emissions(distribution)
        distribution.drop(columns=['geometry'], inplace=True)
        distribution['layer'] = 0
        distribution.set_index(['FID', 'layer', 'tstep'], inplace=True)
        self.logger.write_log('\t\tRecreational boats emissions calculated', message_level=2)
        self.logger.write_time_log('RecreationalBoatsSector', 'calculate_emissions',
                                   timeit.default_timer() - spent_time)
        return distribution
