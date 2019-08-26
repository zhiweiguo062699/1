#!/usr/bin/env python

import sys
import os
import timeit
import geopandas as gpd
import pandas as pd
import numpy as np
from hermesv3_bu.sectors.sector import Sector
from hermesv3_bu.io_server.io_shapefile import IoShapefile
from hermesv3_bu.io_server.io_netcdf import IoNetcdf

pmc_list = ['pmc', 'PMC']


class TrafficAreaSector(Sector):
    def __init__(self, comm, logger, auxiliary_dir, grid_shp, clip, date_array, source_pollutants, vertical_levels,
                 population_tif_path, speciation_map_path, molecular_weights_path,
                 do_evaporative, gasoline_path, total_pop_by_prov, nuts_shapefile, speciation_profiles_evaporative,
                 evaporative_ef_file, temperature_dir,
                 do_small_cities, small_cities_shp, speciation_profiles_small_cities, small_cities_ef_file,
                 small_cities_monthly_profile, small_cities_weekly_profile, small_cities_hourly_profile):
        spent_time = timeit.default_timer()
        logger.write_log('===== TRAFFIC AREA SECTOR =====')

        super(TrafficAreaSector, self).__init__(
            comm, logger, auxiliary_dir, grid_shp, clip, date_array, source_pollutants, vertical_levels,
            None, None, None, speciation_map_path, None, molecular_weights_path)

        self.do_evaporative = do_evaporative
        self.temperature_dir = temperature_dir
        self.speciation_profiles_evaporative = self.read_speciation_profiles(speciation_profiles_evaporative)
        self.evaporative_ef_file = evaporative_ef_file
        if do_evaporative:
            logger.write_log('\tInitialising evaporative emissions.', message_level=2)
            self.evaporative = self.init_evaporative(population_tif_path, nuts_shapefile, gasoline_path,
                                                     total_pop_by_prov)
        else:
            self.evaporative = None

        self.do_small_cities = do_small_cities
        self.speciation_profiles_small_cities = self.read_speciation_profiles(speciation_profiles_small_cities)
        self.small_cities_ef_file = small_cities_ef_file
        self.small_cities_monthly_profile = self.read_monthly_profiles(small_cities_monthly_profile)
        self.small_cities_weekly_profile = self.read_weekly_profiles(small_cities_weekly_profile)
        self.small_cities_hourly_profile = self.read_hourly_profiles(small_cities_hourly_profile)
        if do_small_cities:
            logger.write_log('\tInitialising small cities emissions.', message_level=2)
            self.small_cities = self.init_small_cities(population_tif_path, small_cities_shp)
        else:
            self.small_cities = None

        self.logger.write_time_log('TrafficAreaSector', '__init__', timeit.default_timer() - spent_time)

    def init_evaporative(self, global_path, provinces_shapefile, gasoline_path, total_pop_by_prov):
        spent_time = timeit.default_timer()

        if self.comm.Get_rank() == 0:
            if not os.path.exists(os.path.join(self.auxiliary_dir, 'traffic_area', 'vehicle_by_cell.shp')):
                self.logger.write_log('\t\tCreating population shapefile.', message_level=3)
                pop = self.get_clipped_population(
                    global_path, os.path.join(self.auxiliary_dir, 'traffic_area', 'population.shp'), write_file=False)
                self.logger.write_log('\t\tCreating population shapefile by NUT.', message_level=3)
                pop = self.make_population_by_nuts(
                    pop, provinces_shapefile, os.path.join(self.auxiliary_dir, 'traffic_area', 'pop_NUT.shp'),
                    write_file=False)
                self.logger.write_log('\t\tCreating population shapefile by NUT and cell.', message_level=3)
                pop = self.make_population_by_nuts_cell(
                    pop,  os.path.join(self.auxiliary_dir, 'traffic_area', 'pop_NUT_cell.shp'), write_file=False)
                self.logger.write_log('\t\tCreating vehicle shapefile by cell.', message_level=3)
                veh_cell = self.make_vehicles_by_cell(
                    pop, gasoline_path, pd.read_csv(total_pop_by_prov),
                    os.path.join(self.auxiliary_dir, 'traffic_area', 'vehicle_by_cell.shp'))
            else:
                self.logger.write_log('\t\tReading vehicle shapefile by cell.', message_level=3)
                veh_cell = IoShapefile(self.comm).read_shapefile_serial(
                    os.path.join(self.auxiliary_dir, 'traffic_area', 'vehicle_by_cell.shp'))
        else:
            veh_cell = None

        veh_cell = IoShapefile(self.comm).split_shapefile(veh_cell)

        self.logger.write_time_log('TrafficAreaSector', 'init_evaporative', timeit.default_timer() - spent_time)
        return veh_cell

    def init_small_cities(self, global_path, small_cities_shapefile):
        spent_time = timeit.default_timer()
        if self.comm.Get_rank() == 0:
            if not os.path.exists(os.path.join(self.auxiliary_dir, 'traffic_area', 'pop_SMALL_cell.shp')):
                self.logger.write_log('\t\tCreating population shapefile.', message_level=3)
                pop = self.get_clipped_population(
                    global_path, os.path.join(self.auxiliary_dir, 'traffic_area', 'population.shp'), write_file=False)
                self.logger.write_log('\t\tCreating population small cities shapefile.', message_level=3)
                pop = self.make_population_by_nuts(
                    pop, small_cities_shapefile, os.path.join(self.auxiliary_dir, 'traffic_area', 'pop_SMALL.shp'),
                    write_file=False)
                self.logger.write_log('\t\tCreating population small cities shapefile by cell.', message_level=3)
                pop = self.make_population_by_nuts_cell(
                    pop, os.path.join(self.auxiliary_dir, 'traffic_area', 'pop_SMALL_cell.shp'), write_file=True)
            else:
                self.logger.write_log('\t\tReading population small cities shapefile by cell.', message_level=3)
                pop = IoShapefile(self.comm).read_shapefile_serial(
                    os.path.join(self.auxiliary_dir, 'traffic_area', 'pop_SMALL_cell.shp'))
        else:
            pop = None
        pop = IoShapefile(self.comm).split_shapefile(pop)

        self.logger.write_time_log('TrafficAreaSector', 'init_small_cities', timeit.default_timer() - spent_time)
        return pop

    def get_clipped_population(self, global_path, population_shapefile_path, write_file=True):
        from hermesv3_bu.io_server.io_raster import IoRaster
        spent_time = timeit.default_timer()

        if not os.path.exists(population_shapefile_path):
            population_density = IoRaster(self.comm).clip_raster_with_shapefile_poly(
                global_path, self.clip.shapefile,
                os.path.join(self.auxiliary_dir, 'traffic_area', 'population.tif'))
            population_density = IoRaster(self.comm).to_shapefile_serie_by_cell(
                population_density, out_path=population_shapefile_path, write=write_file)
        else:
            population_density = IoShapefile(self.comm).read_shapefile_serial(population_shapefile_path)

        self.logger.write_time_log('TrafficAreaSector', 'get_clipped_population', timeit.default_timer() - spent_time)

        return population_density

    def make_population_by_nuts(self, population_shape, nut_shp, pop_by_nut_path, write_file=True, csv_path=None,
                                column_id='nuts3_id'):
        spent_time = timeit.default_timer()

        if not os.path.exists(pop_by_nut_path):
            nut_df = IoShapefile(self.comm).read_shapefile_serial(nut_shp)
            population_shape['area_in'] = population_shape.geometry.area
            df = gpd.overlay(population_shape, nut_df.to_crs(population_shape.crs), how='intersection')
            df.crs = population_shape.crs
            df.loc[:, 'data'] = df['data'] * (df.geometry.area / df['area_in'])
            del df['area_in']
            if write_file:
                IoShapefile(self.comm).write_shapefile_serial(df, pop_by_nut_path)
            if csv_path is not None:
                df = df.loc[:, ['data', column_id]].groupby(column_id).sum()
                df.to_csv(csv_path)
        else:
            df = IoShapefile(self.comm).read_shapefile_serial(pop_by_nut_path)

        self.logger.write_time_log('TrafficAreaSector', 'make_population_by_nuts', timeit.default_timer() - spent_time)
        return df

    def make_population_by_nuts_cell(self, pop_by_nut, pop_nut_cell_path, write_file=True):
        spent_time = timeit.default_timer()

        if not os.path.exists(pop_nut_cell_path):

            pop_by_nut = pop_by_nut.to_crs(self.grid_shp.crs)
            # del pop_by_nut['']
            pop_by_nut['area_in'] = pop_by_nut.geometry.area

            # df = gpd.overlay(pop_by_nut, grid_shp, how='intersection')
            df = self.spatial_overlays(pop_by_nut, self.grid_shp.reset_index(), how='intersection')

            df.crs = self.grid_shp.crs
            df.loc[:, 'data'] = df['data'] * (df.geometry.area / df['area_in'])
            del pop_by_nut['area_in']
            if write_file:
                IoShapefile(self.comm).write_shapefile_serial(df, pop_nut_cell_path)
        else:
            df = IoShapefile(self.comm).read_shapefile_serial(pop_nut_cell_path)

        self.logger.write_time_log('TrafficAreaSector', 'make_population_by_nuts_cell',
                                   timeit.default_timer() - spent_time)
        return df

    def make_vehicles_by_cell(self, pop_nut_cell, gasoline_path, total_pop_by_nut, veh_by_cell_path,
                              column_id='nuts3_id'):
        spent_time = timeit.default_timer()

        if not os.path.exists(veh_by_cell_path):

            total_pop_by_nut.loc[:, column_id] = total_pop_by_nut[column_id].astype(np.int16)
            pop_nut_cell.loc[:, column_id] = pop_nut_cell[column_id].astype(np.int16)

            df = pd.merge(pop_nut_cell, total_pop_by_nut, left_on=column_id, right_on=column_id, how='left')

            df['pop_percent'] = df['data'] / df['population']
            df.drop(columns=['data', 'population'], inplace=True)

            gas_df = pd.read_csv(gasoline_path, index_col='COPERT_V_name').transpose()
            vehicle_type_list = list(gas_df.columns.values)
            gas_df.loc[:, column_id] = gas_df.index.astype(np.int16)

            df = df.merge(gas_df, left_on=column_id, right_on=column_id, how='left')
            for vehicle_type in vehicle_type_list:
                df.loc[:, vehicle_type] = df[vehicle_type] * df['pop_percent']

            del df['pop_percent'], df[column_id]

            aux_df = df.loc[:, ['FID'] + vehicle_type_list].groupby('FID').sum()
            aux_df.loc[:, 'FID'] = aux_df.index

            geom = self.grid_shp.loc[aux_df.index, 'geometry']

            df = gpd.GeoDataFrame(aux_df, geometry=geom, crs=pop_nut_cell.crs)
            IoShapefile(self.comm).write_shapefile_serial(df, veh_by_cell_path)
        else:
            df = IoShapefile(self.comm).read_shapefile_serial(veh_by_cell_path)

        self.logger.write_time_log('TrafficAreaSector', 'make_vehicles_by_cell', timeit.default_timer() - spent_time)
        return df

    def get_profiles_from_temperature(self, temperature, default=False):
        spent_time = timeit.default_timer()

        temperature = temperature.copy()
        if default:
            default_profile = np.array(
                [0.025, 0.025, 0.025, 0.025, 0.025, 0.027083, 0.03125, 0.0375, 0.045833, 0.05625, 0.060417, 0.066667,
                 0.06875, 0.072917, 0.070833, 0.064583, 0.05625, 0.045833, 0.0375, 0.03125, 0.027083, 0.025, 0.025,
                 0.025])
            for x in range(24):
                temperature['t_{0}'.format(x)] = default_profile[x]

        else:
            temp_list = ['t_{0}'.format(x) for x in range(24)]
            temperature.loc[:, temp_list] = temperature[temp_list] + 273.15

            temperature.loc[:, temp_list] = temperature[temp_list].subtract(temperature[temp_list].min(axis=1), axis=0)

            temperature.loc[:, temp_list] = temperature[temp_list].div(
                temperature[temp_list].max(axis=1) - temperature[temp_list].min(axis=1), axis=0)

            aux = temperature[temp_list].replace({0: np.nan})
            second_min = aux[temp_list].min(axis=1)

            temperature.loc[:, temp_list] = temperature[temp_list].add(second_min, axis=0)
            temperature.loc[:, temp_list] = temperature[temp_list].div(temperature[temp_list].sum(axis=1), axis=0)

        self.logger.write_time_log('TrafficAreaSector', 'get_profiles_from_temperature',
                                   timeit.default_timer() - spent_time)
        return temperature

    def calculate_evaporative_emissions(self):
        spent_time = timeit.default_timer()

        veh_list = list(self.evaporative.columns.values)
        veh_list.remove('FID')
        veh_list.remove('geometry')
        if 'T_REC' in veh_list:
            veh_list.remove('T_REC')

        crs = self.evaporative.crs
        geom = self.evaporative.geometry

        # get average daily temperature by cell
        aux_df = self.evaporative.loc[:, 'geometry'].to_crs({'init': 'epsg:4326'})
        self.evaporative['c_lat'] = aux_df.centroid.y
        self.evaporative['c_lon'] = aux_df.centroid.x
        self.evaporative['centroid'] = aux_df.centroid

        temperature = IoNetcdf(self.comm).get_hourly_data_from_netcdf(
            self.evaporative['c_lon'].min(), self.evaporative['c_lon'].max(), self.evaporative['c_lat'].min(),
            self.evaporative['c_lat'].max(), self.temperature_dir, 'tas', self.date_array)
        temperature.rename(columns={x: 't_{0}'.format(x) for x in range(len(self.date_array))}, inplace=True)
        # From Kelvin to Celsius degrees
        temperature.loc[:, ['t_{0}'.format(x) for x in range(len(self.date_array))]] = \
            temperature.loc[:, ['t_{0}'.format(x) for x in range(len(self.date_array))]] - 273.15

        temperature_mean = gpd.GeoDataFrame(temperature[['t_{0}'.format(x) for x in
                                                         range(len(self.date_array))]].mean(axis=1),
                                            columns=['temp'], geometry=temperature.geometry)
        temperature_mean['REC'] = temperature['REC']

        if 'T_REC' not in self.evaporative.columns.values:
            self.evaporative['T_REC'] = self.evaporative.apply(self.nearest, geom_union=temperature_mean.unary_union,
                                                               df1=self.evaporative, df2=temperature_mean,
                                                               geom1_col='centroid', src_column='REC', axis=1)
            del self.evaporative['c_lat'], self.evaporative['c_lon'], self.evaporative['centroid']
            IoShapefile(self.comm).write_shapefile_parallel(
                self.evaporative, os.path.join(self.auxiliary_dir, 'traffic_area', 'vehicle_by_cell.shp'))
        else:
            del self.evaporative['c_lat'], self.evaporative['c_lon'], self.evaporative['centroid']

        self.evaporative = self.evaporative.merge(temperature_mean, left_on='T_REC', right_on='REC', how='left')

        ef_df = pd.read_csv(self.evaporative_ef_file, sep=',')
        ef_df.drop(columns=['canister', 'Copert_V_name'], inplace=True)
        ef_df.loc[ef_df['Tmin'].isnull(), 'Tmin'] = -999
        ef_df.loc[ef_df['Tmax'].isnull(), 'Tmax'] = 999

        for vehicle_type in veh_list:

            self.evaporative['EF'] = np.nan
            ef_aux = ef_df.loc[ef_df['Code'] == vehicle_type]
            for i, line in ef_aux.iterrows():
                self.evaporative.loc[(self.evaporative['temp'] < line.get('Tmax')) &
                                     (self.evaporative['temp'] >= line.get('Tmin')), 'EF'] = \
                    line.get('EFbase') * line.get('TF')

            self.evaporative.loc[:, vehicle_type] = self.evaporative[vehicle_type] * self.evaporative['EF']

        self.evaporative.loc[:, 'nmvoc'] = self.evaporative.loc[:, veh_list].sum(axis=1)
        self.evaporative = gpd.GeoDataFrame(self.evaporative.loc[:, ['nmvoc', 'T_REC', 'FID']], geometry=geom, crs=crs)

        self.evaporative = self.speciate_evaporative()

        self.evaporative = self.evaporative_temporal_distribution(self.get_profiles_from_temperature(temperature))

        self.evaporative.set_index(['FID', 'tstep'], inplace=True)

        self.logger.write_time_log('TrafficAreaSector', 'calculate_evaporative_emissions',
                                   timeit.default_timer() - spent_time)
        return self.evaporative

    def evaporative_temporal_distribution(self, temporal_profiles):
        spent_time = timeit.default_timer()

        aux = self.evaporative.merge(temporal_profiles, left_on='T_REC', right_on='REC', how='left')

        temporal_df_list = []
        pollutant_list = [e for e in self.evaporative.columns.values if e not in ('T_REC', 'FID', 'geometry')]

        for tstep, date in enumerate(self.date_array):
            aux_temporal = aux[pollutant_list].multiply(aux['t_{0}'.format(date.hour)], axis=0)
            aux_temporal['FID'] = aux['FID']
            aux_temporal['tstep'] = tstep
            temporal_df_list.append(aux_temporal)
        df = pd.concat(temporal_df_list)

        self.logger.write_time_log('TrafficAreaSector', 'evaporative_temporal_distribution',
                                   timeit.default_timer() - spent_time)
        return df

    def speciate_evaporative(self):
        spent_time = timeit.default_timer()

        speciated_df = self.evaporative.drop(columns=['nmvoc'])
        out_p_list = [out_p for out_p, in_p_aux in self.speciation_map.items() if in_p_aux == 'nmvoc']

        for p in out_p_list:
            # From g/day to mol/day
            speciated_df[p] = self.evaporative['nmvoc'] * self.speciation_profiles_evaporative.loc['default', p]

        self.logger.write_time_log('TrafficAreaSector', 'speciate_evaporative', timeit.default_timer() - spent_time)
        return speciated_df

    def small_cities_emissions_by_population(self, df):
        spent_time = timeit.default_timer()

        df = df.loc[:, ['data', 'FID']].groupby('FID').sum()
        ef_df = pd.read_csv(self.small_cities_ef_file, sep=',')
        ef_df.drop(['Code', 'Copert_V_name'], axis=1, inplace=True)
        for pollutant in ef_df.columns.values:
            df[pollutant] = df['data'] * ef_df[pollutant].iloc[0]
        df.drop('data', axis=1, inplace=True)

        self.logger.write_time_log('TrafficAreaSector', 'small_cities_emissions_by_population',
                                   timeit.default_timer() - spent_time)
        return df

    def add_timezones(self, grid, default=False):
        from timezonefinder import TimezoneFinder
        spent_time = timeit.default_timer()

        if default:
            grid['timezone'] = 'Europe/Madrid'
        else:
            tz = TimezoneFinder()
            aux_grid = grid.to_crs({'init': 'epsg:4326'})
            aux_grid['lats'] = aux_grid.geometry.centroid.y
            aux_grid['lons'] = aux_grid.geometry.centroid.x
            inc = 1

            while len(grid.loc[grid['timezone'] == '', :]) > 0:
                grid.loc[grid['timezone'] == '', 'timezone'] = aux_grid.loc[grid['timezone'] == '', :].apply(
                    lambda x: tz.closest_timezone_at(lng=x['lons'], lat=x['lats'], delta_degree=inc), axis=1)
                inc += 1

        self.logger.write_time_log('TrafficAreaSector', 'add_timezones', timeit.default_timer() - spent_time)
        return grid

    def temporal_distribution_small(self, small_cities):
        import pytz
        spent_time = timeit.default_timer()

        p_names = small_cities.columns.values

        aux_grid = self.grid_shp.loc[small_cities.index.values, :].reset_index().copy()

        aux_grid = self.add_timezone(aux_grid)
        aux_grid.set_index('FID', inplace=True)

        small_cities = small_cities.merge(aux_grid.loc[:, ['timezone']], left_index=True, right_index=True,
                                          how='left')
        small_cities['utc'] = self.date_array[0]
        small_cities['date'] = small_cities.groupby('timezone')['utc'].apply(
            lambda x: pd.to_datetime(x).dt.tz_localize(pytz.utc).dt.tz_convert(x.name).dt.tz_localize(None))
        small_cities.drop(columns=['utc', 'timezone'], inplace=True)
        df_list = []
        for tstep in range(len(self.date_array)):
            small_cities['month'] = small_cities['date'].dt.month
            small_cities['weekday'] = small_cities['date'].dt.dayofweek
            small_cities['hour'] = small_cities['date'].dt.hour
            small_cities.loc[small_cities['weekday'] <= 4, 'day_type'] = 'Weekday'
            small_cities.loc[small_cities['weekday'] == 5, 'day_type'] = 'Saturday'
            small_cities.loc[small_cities['weekday'] == 6, 'day_type'] = 'Sunday'

            for i, aux in small_cities.groupby(['month', 'weekday', 'hour', 'day_type']):
                aux_date = pd.Timestamp(aux['date'].values[0])

                balanced_weekly_profile = self.calculate_rebalanced_weekly_profile(
                    self.small_cities_weekly_profile.loc['default', :].to_dict(), aux_date)
                small_cities.loc[aux.index, 'f'] = self.small_cities_monthly_profile.loc['default', i[0]] * \
                    balanced_weekly_profile[i[1]] * self.small_cities_hourly_profile.loc[i[3], i[2]]

            aux_df = small_cities.loc[:, p_names].multiply(small_cities['f'], axis=0)
            aux_df['tstep'] = tstep
            aux_df.set_index('tstep', append=True, inplace=True)
            df_list.append(aux_df)

            small_cities['date'] = small_cities['date'] + pd.to_timedelta(1, unit='h')
        df = pd.concat(df_list)

        self.logger.write_time_log('TrafficAreaSector', 'temporal_distribution_small',
                                   timeit.default_timer() - spent_time)
        return df

    def calculate_small_cities_emissions(self):
        spent_time = timeit.default_timer()

        # EF
        self.small_cities = self.small_cities_emissions_by_population(self.small_cities)

        # Spectiacion
        self.speciation_profile = self.speciation_profiles_small_cities
        self.small_cities = self.speciate(self.small_cities)
        # From kmol/h or kg/h to mol/h or g/h
        self.small_cities = self.small_cities.mul(1000.0)

        # Temporal
        # grid = self.add_timezones(gpd.read_file(os.path.join(self.auxiliary_dir, 'shapefile', 'grid_shapefile.shp')),
        #                           default=True)
        self.small_cities = self.temporal_distribution_small(self.small_cities)

        self.logger.write_time_log('TrafficAreaSector', 'calculate_small_cities_emissions',
                                   timeit.default_timer() - spent_time)

        return True

    def to_grid(self):
        spent_time = timeit.default_timer()

        if self.do_evaporative and self.do_small_cities:
            dataset = pd.concat([self.evaporative, self.small_cities], sort=False)
        elif self.do_evaporative:
            dataset = self.evaporative
        elif self.do_small_cities:
            dataset = self.small_cities
        else:
            raise ValueError('No traffic area emission selected. do_evaporative and do_small_cities are False')

        dataset['layer'] = 0
        dataset = dataset.groupby(['FID', 'layer', 'tstep']).sum()

        self.logger.write_time_log('TrafficAreaSector', 'to_grid', timeit.default_timer() - spent_time)
        return dataset

    def calculate_emissions(self):
        spent_time = timeit.default_timer()

        if self.do_evaporative:
            self.logger.write_log('\tCalculating evaporative emissions.', message_level=2)
            self.calculate_evaporative_emissions()
        if self.do_small_cities:
            self.logger.write_log('\tCalculating small cities emissions.', message_level=2)
            self.calculate_small_cities_emissions()

        emissions = self.to_grid()

        self.logger.write_log('\t\tTraffic area emissions calculated', message_level=2)
        self.logger.write_time_log('TrafficAreaSector', 'calculate_emissions', timeit.default_timer() - spent_time)
        return emissions
