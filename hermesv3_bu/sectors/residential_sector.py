#!/usr/bin/env python

import sys
import os
import timeit

import numpy as np
import pandas as pd
import geopandas as gpd

from hermesv3_bu.sectors.sector import Sector
from hermesv3_bu.io_server.io_raster import IoRaster
from hermesv3_bu.io_server.io_shapefile import IoShapefile
from hermesv3_bu.io_server.io_netcdf import IoNetcdf
from hermesv3_bu.logger.log import Log


class ResidentialSector(Sector):
    def __init__(self, comm, logger, auxiliary_dir, grid_shp, clip, date_array, source_pollutants, vertical_levels,
                 fuel_list, prov_shapefile, ccaa_shapefile, population_density_map, population_type_map,
                 population_type_by_ccaa, population_type_by_prov, energy_consumption_by_prov,
                 energy_consumption_by_ccaa, residential_spatial_proxies, residential_ef_files_path,
                 heating_degree_day_path, temperature_path, hourly_profiles_path, speciation_map_path,
                 speciation_profiles_path, molecular_weights_path):
        spent_time = timeit.default_timer()

        super(ResidentialSector, self).__init__(
            comm, logger, auxiliary_dir, grid_shp, clip, date_array, source_pollutants, vertical_levels,
            None, None, hourly_profiles_path, speciation_map_path,
            speciation_profiles_path, molecular_weights_path)

        # Heating Degree Day constants
        self.hdd_base_temperature = 15.5
        self.hdd_base_temperature = 15.5
        self.hdd_f_biomass = 0.0
        self.hdd_f_others = 0.2

        self.fuel_list = fuel_list
        self.day_dict = self.calculate_num_days()

        self.pop_type_by_prov = population_type_by_prov
        self.pop_type_by_ccaa = population_type_by_ccaa

        self.energy_consumption_by_prov = pd.read_csv(energy_consumption_by_prov)
        self.energy_consumption_by_ccaa = pd.read_csv(energy_consumption_by_ccaa)
        self.residential_spatial_proxies = self.read_residential_spatial_proxies(residential_spatial_proxies)
        self.ef_profiles = self.read_ef_file(residential_ef_files_path)

        if self.comm.Get_rank() == 0:
            self.fuel_distribution = self.get_fuel_distribution(prov_shapefile, ccaa_shapefile, population_density_map,
                                                                population_type_map, create_pop_csv=False)
        else:
            self.fuel_distribution = None
        self.fuel_distribution = IoShapefile().split_shapefile(self.fuel_distribution)

        self.heating_degree_day_path = heating_degree_day_path
        self.temperature_path = temperature_path

        self.logger.write_time_log('ResidentialSector', '__init__', timeit.default_timer() - spent_time)

    def read_ef_file(self, path):
        spent_time = timeit.default_timer()

        df_ef = pd.read_csv(path)
        df_ef = df_ef.loc[df_ef['fuel_type'].isin(self.fuel_list), ['fuel_type'] + self.source_pollutants]

        self.logger.write_time_log('ResidentialSector', 'read_ef_file', timeit.default_timer() - spent_time)
        return df_ef

    def calculate_num_days(self):
        spent_time = timeit.default_timer()

        day_array = [hour.date() for hour in self.date_array]
        days, num_days = np.unique(day_array, return_counts=True)

        day_dict = {}
        for key, value in zip(days, num_days):
            day_dict[key] = value

        self.logger.write_time_log('ResidentialSector', 'calculate_num_days', timeit.default_timer() - spent_time)
        return day_dict

    def read_residential_spatial_proxies(self, path):
        spent_time = timeit.default_timer()

        spatial_proxies = pd.read_csv(path)
        spatial_proxies = spatial_proxies.loc[spatial_proxies['fuel_type'].isin(self.fuel_list), :]

        self.logger.write_time_log('ResidentialSector', 'read_residential_spatial_proxies',
                                   timeit.default_timer() - spent_time)
        return spatial_proxies

    def get_spatial_proxy(self, fuel_type):
        spent_time = timeit.default_timer()

        proxy = self.residential_spatial_proxies.loc[self.residential_spatial_proxies['fuel_type'] == fuel_type, ['nuts_level', 'proxy']].values[0]

        if proxy[0] == 3:
            nut_level = 'prov'
        elif proxy[0] == 2:
            nut_level = 'ccaa'
        else:
            nut_level = proxy[0]

        if proxy[1] == 'urban':
            proxy_type = 3
        elif proxy[1] == 'rural':
            proxy_type = 1
        else:
            proxy_type = proxy[1]

        self.logger.write_time_log('ResidentialSector', 'get_spatial_proxy', timeit.default_timer() - spent_time)
        return {'nut_level': nut_level, 'proxy_type': proxy_type}

    def to_dst_resolution(self, src_distribution):
        spent_time = timeit.default_timer()

        src_distribution.to_crs(self.grid_shp.crs, inplace=True)
        src_distribution.to_file(os.path.join(self.auxiliary_dir, 'residential', 'fuel_distribution_src.shp'))
        src_distribution['src_inter_fraction'] = src_distribution.geometry.area
        src_distribution = self.spatial_overlays(src_distribution, self.grid_shp, how='intersection')
        src_distribution.to_file(os.path.join(self.auxiliary_dir, 'residential', 'fuel_distribution_raw.shp'))
        src_distribution['src_inter_fraction'] = src_distribution.geometry.area / src_distribution[
            'src_inter_fraction']

        src_distribution[self.fuel_list] = src_distribution.loc[:, self.fuel_list].multiply(
            src_distribution["src_inter_fraction"], axis="index")

        src_distribution = src_distribution.loc[:, self.fuel_list + ['FID']].groupby('FID').sum()
        src_distribution = gpd.GeoDataFrame(src_distribution, crs=self.grid_shp.crs,
                                            geometry=self.grid_shp.loc[src_distribution.index, 'geometry'])
        src_distribution.reset_index(inplace=True)

        self.logger.write_time_log('ResidentialSector', 'to_dst_resolution', timeit.default_timer() - spent_time)
        return src_distribution

    def get_fuel_distribution(self, prov_shapefile, ccaa_shapefile, population_density_map, population_type_map,
                              create_pop_csv=False):
        spent_time = timeit.default_timer()

        fuel_distribution_path = os.path.join(self.auxiliary_dir, 'residential', 'fuel_distribution.shp')

        if not os.path.exists(fuel_distribution_path):

            population_density = IoRaster().clip_raster_with_shapefile_poly(
                population_density_map, self.clip.shapefile,
                os.path.join(self.auxiliary_dir, 'residential', 'population_density.tif'))
            population_density = IoRaster().to_shapefile(population_density)

            population_density.rename(columns={'data': 'pop'}, inplace=True)

            population_type = IoRaster().clip_raster_with_shapefile_poly(
                population_type_map, self.clip.shapefile,
                os.path.join(self.auxiliary_dir, 'residential', 'population_type.tif'))
            population_type = IoRaster().to_shapefile(population_type)
            population_type.rename(columns={'data': 'type'}, inplace=True)

            population_density['type'] = population_type['type']
            population_density.loc[population_density['type'] == 2, 'type'] = 3

            population_density = self.add_nut_code(population_density, prov_shapefile, nut_value='ORDER07')
            population_density.rename(columns={'nut_code': 'prov'}, inplace=True)

            population_density = population_density.loc[population_density['prov'] != -999, :]
            population_density = self.add_nut_code(population_density, ccaa_shapefile, nut_value='ORDER06')
            population_density.rename(columns={'nut_code': 'ccaa'}, inplace=True)
            population_density = population_density.loc[population_density['ccaa'] != -999, :]

            if create_pop_csv:
                population_density.loc[:, ['prov', 'pop', 'type']].groupby(['prov', 'type']).sum().reset_index().to_csv(
                    self.pop_type_by_prov)
                population_density.loc[:, ['ccaa', 'pop', 'type']].groupby(['ccaa', 'type']).sum().reset_index().to_csv(
                    self.pop_type_by_ccaa)

            self.pop_type_by_ccaa = pd.read_csv(self.pop_type_by_ccaa).set_index(['ccaa', 'type'])
            self.pop_type_by_prov = pd.read_csv(self.pop_type_by_prov).set_index(['prov', 'type'])

            fuel_distribution = population_density.loc[:, ['CELL_ID', 'geometry']].copy()

            for fuel in self.fuel_list:
                fuel_distribution[fuel] = 0

                spatial_proxy = self.get_spatial_proxy(fuel)

                if spatial_proxy['nut_level'] == 'ccaa':
                    for ccaa in np.unique(population_density['ccaa']):
                        if spatial_proxy['proxy_type'] == 'all':
                            total_pop = self.pop_type_by_ccaa.loc[
                                self.pop_type_by_ccaa.index.get_level_values('ccaa') == ccaa, 'pop'].sum()
                            energy_consumption = self.energy_consumption_by_ccaa.loc[
                                self.energy_consumption_by_ccaa['code'] == ccaa, fuel].values[0]

                            fuel_distribution.loc[
                                population_density['ccaa'] == ccaa, fuel] = population_density['pop'].multiply(
                                energy_consumption / total_pop)
                        else:
                            total_pop = self.pop_type_by_ccaa.loc[
                                (self.pop_type_by_ccaa.index.get_level_values('ccaa') == ccaa) &
                                (self.pop_type_by_ccaa.index.get_level_values('type') == spatial_proxy['proxy_type']),
                                'pop'].values[0]
                            energy_consumption = self.energy_consumption_by_ccaa.loc[
                                self.energy_consumption_by_ccaa['code'] == ccaa, fuel].values[0]

                            fuel_distribution.loc[(population_density['ccaa'] == ccaa) &
                                                  (population_density['type'] == spatial_proxy['proxy_type']),
                                                  fuel] = population_density['pop'].multiply(
                                energy_consumption / total_pop)
                if spatial_proxy['nut_level'] == 'prov':
                    for prov in np.unique(population_density['prov']):
                        if spatial_proxy['proxy_type'] == 'all':
                            total_pop = self.pop_type_by_prov.loc[self.pop_type_by_prov.index.get_level_values(
                                'prov') == prov, 'pop'].sum()
                            energy_consumption = self.energy_consumption_by_prov.loc[
                                self.energy_consumption_by_prov['code'] == prov, fuel].values[0]

                            fuel_distribution.loc[population_density['prov'] == prov, fuel] = population_density[
                                'pop'].multiply(energy_consumption / total_pop)
                        else:
                            total_pop = self.pop_type_by_prov.loc[
                                (self.pop_type_by_prov.index.get_level_values('prov') == prov) &
                                (self.pop_type_by_prov.index.get_level_values('type') == spatial_proxy['proxy_type']),
                                'pop'].values[0]
                            energy_consumption = self.energy_consumption_by_prov.loc[
                                self.energy_consumption_by_prov['code'] == prov, fuel].values[0]

                            fuel_distribution.loc[(population_density['prov'] == prov) &
                                                  (population_density['type'] == spatial_proxy['proxy_type']),
                                                  fuel] = population_density['pop'].multiply(
                                energy_consumption / total_pop)
            fuel_distribution = self.to_dst_resolution(fuel_distribution)

            IoShapefile().write_shapefile(fuel_distribution, fuel_distribution_path)
        else:
            fuel_distribution = IoShapefile().read_serial_shapefile(fuel_distribution_path)

        self.logger.write_time_log('ResidentialSector', 'get_fuel_distribution', timeit.default_timer() - spent_time)
        return fuel_distribution

    def calculate_daily_distribution(self, day):
        import calendar
        spent_time = timeit.default_timer()

        if calendar.isleap(day.year):
            num_days = 366
        else:
            num_days = 365

        geometry_shp = self.fuel_distribution.loc[:, ['FID', 'geometry']].to_crs({'init': 'epsg:4326'})
        geometry_shp['c_lat'] = geometry_shp.centroid.y
        geometry_shp['c_lon'] = geometry_shp.centroid.x
        geometry_shp['centroid'] = geometry_shp.centroid
        geometry_shp.drop(columns='geometry', inplace=True)

        meteo = IoNetcdf(self.comm).get_data_from_netcdf(
            os.path.join(self.temperature_path, 'tas_{0}{1}.nc'.format(day.year, str(day.month).zfill(2))),
            'tas', 'daily', day, geometry_shp)
        # From K to Celsius degrees
        meteo['tas'] = meteo['tas'] - 273.15

        # HDD(x,y,d) = max(Tb - Tout(x,y,d), 1)
        meteo['hdd'] = np.maximum(self.hdd_base_temperature - meteo['tas'], 1)
        meteo.drop('tas', axis=1, inplace=True)

        meteo['hdd_mean'] = IoNetcdf(self.comm).get_data_from_netcdf(self.heating_degree_day_path.replace(
            '<year>', str(day.year)), 'HDD', 'yearly', day, geometry_shp).loc[:, 'HDD']

        daily_distribution = self.fuel_distribution.copy()

        daily_distribution = daily_distribution.to_crs({'init': 'epsg:4326'})
        daily_distribution['centroid'] = daily_distribution.centroid

        daily_distribution['REC'] = daily_distribution.apply(
            self.nearest, geom_union=meteo.unary_union, df1=daily_distribution, df2=meteo, geom1_col='centroid',
            src_column='REC', axis=1)
        daily_distribution = pd.merge(daily_distribution, meteo, how='left', on='REC')

        daily_distribution.drop(columns=['centroid', 'REC', 'geometry_y'], axis=1, inplace=True)
        daily_distribution.rename(columns={'geometry_x': 'geometry'}, inplace=True)

        for fuel in self.fuel_list:
            # Selection of factor for HDD as a function of fuel type
            if fuel.startswith('B_'):
                hdd_f = self.hdd_f_biomass
            else:
                hdd_f = self.hdd_f_others

            daily_distribution.loc[:, fuel] = daily_distribution[fuel].multiply(
                (daily_distribution['hdd'] + hdd_f * daily_distribution['hdd_mean']) /
                (num_days*((1 + hdd_f)*daily_distribution['hdd_mean']))
            )

        daily_distribution.drop(['hdd', 'hdd_mean'], axis=1, inplace=True)

        self.logger.write_time_log('ResidentialSector', 'calculate_daily_distribution',
                                   timeit.default_timer() - spent_time)
        return daily_distribution

    def get_fuel_distribution_by_day(self):
        spent_time = timeit.default_timer()

        daily_distribution = {}
        for day in self.day_dict.keys():
            daily_distribution[day] = self.calculate_daily_distribution(day)

        self.logger.write_time_log('ResidentialSector', 'get_fuel_distribution_by_day',
                                   timeit.default_timer() - spent_time)
        return daily_distribution

    def calculate_hourly_distribution(self, fuel_distribution):
        spent_time = timeit.default_timer()

        fuel_distribution['hour'] = fuel_distribution['date'].dt.hour
        for fuel in self.fuel_list:
            if fuel.startswith('B_'):
                fuel_distribution.loc[:, fuel] = fuel_distribution.groupby('hour')[fuel].apply(
                    lambda x: x.multiply(self.hourly_profiles.loc['biomass', x.name])
                )
            else:
                fuel_distribution.loc[:, fuel] = fuel_distribution.groupby('hour')[fuel].apply(
                    lambda x: x.multiply(self.hourly_profiles.loc['others', x.name])
                )
        fuel_distribution.drop('hour', axis=1, inplace=True)

        self.logger.write_time_log('ResidentialSector', 'calculate_hourly_distribution',
                                   timeit.default_timer() - spent_time)
        return fuel_distribution

    def add_dates(self, df_by_day):
        spent_time = timeit.default_timer()

        df_list = []
        for tstep, date in enumerate(self.date_array):
            df_aux = df_by_day[date.date()].copy()
            df_aux['date'] = pd.to_datetime(date, utc=True)
            df_aux['date_utc'] = pd.to_datetime(date, utc=True)
            df_aux['tstep'] = tstep
            # df_aux = self.to_timezone(df_aux)
            df_list.append(df_aux)
        dataframe_by_day = pd.concat(df_list, ignore_index=True)

        dataframe_by_day = self.to_timezone(dataframe_by_day)
        self.logger.write_time_log('ResidentialSector', 'add_dates', timeit.default_timer() - spent_time)

        return dataframe_by_day

    def calculate_fuel_distribution_by_hour(self):
        spent_time = timeit.default_timer()

        self.fuel_distribution = self.add_timezone(self.fuel_distribution)

        fuel_distribution_by_day = self.get_fuel_distribution_by_day()

        fuel_distribution_by_day = self.add_dates(fuel_distribution_by_day)

        fuel_distribution = self.calculate_hourly_distribution(fuel_distribution_by_day)

        self.logger.write_time_log('ResidentialSector', 'calculate_fuel_distribution_by_hour',
                                   timeit.default_timer() - spent_time)
        return fuel_distribution

    def calculate_emissions_from_fuel_distribution(self, fuel_distribution):
        spent_time = timeit.default_timer()

        emissions = fuel_distribution.loc[:, ['date', 'date_utc', 'tstep', 'geometry']].copy()
        for in_p in self.source_pollutants:
            emissions[in_p] = 0
            for i, fuel_type_ef in self.ef_profiles.iterrows():
                emissions[in_p] += fuel_distribution.loc[:, fuel_type_ef['fuel_type']].multiply(fuel_type_ef[in_p])
        self.logger.write_time_log('ResidentialSector', 'calculate_fuel_distribution_by_hour',
                                   timeit.default_timer() - spent_time)

        return emissions

    def calculate_output_emissions_from_fuel_distribution(self, fuel_distribution):
        spent_time = timeit.default_timer()

        emissions = fuel_distribution.loc[:, ['FID', 'date', 'date_utc', 'tstep', 'geometry']].copy()
        for out_p in self.output_pollutants:
            emissions[out_p] = 0
            if out_p == 'PMC':
                pm10_df = None
                for i, fuel_type_ef in self.ef_profiles.iterrows():
                    if fuel_type_ef['fuel_type'].startswith('B_'):
                        speciation_factor = self.speciation_profile.loc['biomass', out_p]
                    else:
                        speciation_factor = self.speciation_profile.loc['others', out_p]

                    if pm10_df is None:
                        pm10_df = fuel_distribution.loc[:, fuel_type_ef['fuel_type']].multiply(
                            fuel_type_ef['pm10'] * speciation_factor)
                    else:
                        pm10_df += fuel_distribution.loc[:, fuel_type_ef['fuel_type']].multiply(
                            fuel_type_ef['pm10'] * speciation_factor)
                pm10_df.divide(self.molecular_weights['pm10'])

                pm25_df = None
                for i, fuel_type_ef in self.ef_profiles.iterrows():
                    if fuel_type_ef['fuel_type'].startswith('B_'):
                        speciation_factor = self.speciation_profile.loc['biomass', out_p]
                    else:
                        speciation_factor = self.speciation_profile.loc['others', out_p]

                    if pm25_df is None:
                        pm25_df = fuel_distribution.loc[:, fuel_type_ef['fuel_type']].multiply(
                            fuel_type_ef['pm25'] * speciation_factor)
                    else:
                        pm25_df += fuel_distribution.loc[:, fuel_type_ef['fuel_type']].multiply(
                            fuel_type_ef['pm25'] * speciation_factor)
                pm25_df.divide(self.molecular_weights['pm25'])

                emissions[out_p] = pm10_df - pm25_df
            else:
                in_p = self.speciation_map[out_p]
                in_df = None
                for i, fuel_type_ef in self.ef_profiles.iterrows():
                    if fuel_type_ef['fuel_type'].startswith('B_'):
                        speciation_factor = self.speciation_profile.loc['biomass', out_p]
                    else:
                        speciation_factor = self.speciation_profile.loc['others', out_p]

                    if in_df is None:
                        in_df = fuel_distribution.loc[:, fuel_type_ef['fuel_type']].multiply(
                            fuel_type_ef[in_p] * speciation_factor)
                    else:
                        in_df += fuel_distribution.loc[:, fuel_type_ef['fuel_type']].multiply(
                            fuel_type_ef[in_p] * speciation_factor)
                emissions[out_p] = in_df.divide(self.molecular_weights[in_p])

        self.logger.write_time_log('ResidentialSector', 'calculate_output_emissions_from_fuel_distribution',
                                   timeit.default_timer() - spent_time)
        return emissions

    def calculate_emissions(self):
        spent_time = timeit.default_timer()
        fuel_distribution_by_hour = self.calculate_fuel_distribution_by_hour()

        emissions = self.calculate_output_emissions_from_fuel_distribution(fuel_distribution_by_hour)

        self.logger.write_time_log('ResidentialSector', 'calculate_emissions', timeit.default_timer() - spent_time)
        emissions.drop(columns=['date', 'date_utc', 'geometry'], inplace=True)
        emissions['layer'] = 0
        emissions.set_index(['FID', 'layer', 'tstep'], inplace=True)

        return emissions
