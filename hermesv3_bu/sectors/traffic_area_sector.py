#!/usr/bin/env python

import sys
import os
import timeit
import geopandas as gpd
import pandas as pd
import numpy as np
from hermesv3_bu.sectors.sector import Sector
from hermesv3_bu.sectors.traffic_sector import TrafficSector

pmc_list = ['pmc', 'PMC']


class TrafficAreaSector(Sector):
    def __init__(self, population_tiff_path, auxiliary_dir,
                 do_evaporative=True, gasoline_path=None, total_pop_by_prov=None, nuts_shapefile=None,
                 do_small_cities=True, small_cities_shp=None):
        spent_time = timeit.default_timer()

        self.auxiliary_dir = auxiliary_dir

        if not os.path.exists(os.path.join(auxiliary_dir, 'population')):
            os.makedirs(os.path.join(auxiliary_dir, 'population'))

        if do_evaporative:
            self.evaporative = self.init_evaporative(population_tiff_path, auxiliary_dir, nuts_shapefile, gasoline_path,
                                                     total_pop_by_prov)

        if do_small_cities:
            self.small_cities = self.init_small_cities(population_tiff_path, auxiliary_dir, small_cities_shp)

        self.logger.write_time_log('TrafficAreaSector', '__init__', timeit.default_timer() - spent_time)

        return None

    def init_evaporative(self, global_path, auxiliary_dir, provinces_shapefile, gasoline_path, total_pop_by_prov):
        spent_time = timeit.default_timer()

        if not os.path.exists(os.path.join(auxiliary_dir, 'vehicle_by_cell.shp')):
            grid_shape_path = os.path.join(auxiliary_dir, 'shapefile', 'grid_shapefile.shp')
            pop = self.get_clipped_population(global_path, os.path.join(auxiliary_dir, 'population', 'population.shp'))
            pop_nut = self.make_population_by_nuts(pop, provinces_shapefile,
                                                   os.path.join(auxiliary_dir, 'population', 'pop_NUT.shp'))
            pop_nut_cell = self.make_population_by_nuts_cell(pop_nut, grid_shape_path,
                                                             os.path.join(auxiliary_dir, 'population',
                                                                          'pop_NUT_cell.shp'))
            veh_cell = self.make_vehicles_by_cell(pop_nut_cell, gasoline_path, pd.read_csv(total_pop_by_prov),
                                                  grid_shape_path, os.path.join(auxiliary_dir, 'vehicle_by_cell.shp'))
        else:
            veh_cell = gpd.read_file(os.path.join(auxiliary_dir, 'vehicle_by_cell.shp'))

        self.logger.write_time_log('TrafficAreaSector', 'init_evaporative', timeit.default_timer() - spent_time)
        return veh_cell

    def init_small_cities(self, global_path, auxiliary_dir, small_cities_shapefile):
        spent_time = timeit.default_timer()

        if not os.path.exists(os.path.join(auxiliary_dir, 'population', 'pop_SMALL_cell.shp')):
            grid_shape_path = os.path.join(auxiliary_dir, 'shapefile', 'grid_shapefile.shp')
            pop = self.get_clipped_population(global_path, os.path.join(auxiliary_dir, 'population', 'population.shp'))
            pop_nut = self.make_population_by_nuts(pop, small_cities_shapefile,
                                                   os.path.join(auxiliary_dir, 'population', 'pop_SMALL.shp'))
            pop_nut_cell = self.make_population_by_nuts_cell(pop_nut, grid_shape_path,
                                                             os.path.join(auxiliary_dir, 'population',
                                                                          'pop_SMALL_cell.shp'))
        else:
            pop_nut_cell = gpd.read_file(os.path.join(auxiliary_dir, 'population', 'pop_SMALL_cell.shp'))

        self.logger.write_time_log('TrafficAreaSector', 'init_small_cities', timeit.default_timer() - spent_time)
        return pop_nut_cell

    def get_clipped_population(self, global_path, population_shapefile_path):
        from hermesv3_bu.io_server.io_raster import IoRaster
        spent_time = timeit.default_timer()

        if not os.path.exists(population_shapefile_path):
            IoRaster(self.comm).clip_raster_with_shapefile_poly(
                global_path, self.clip.shapefile, os.path.join(self.auxiliary_dir, 'traffic', 'population_clip.tiff'))
            df = IoRaster(self.comm).to_shapefile(os.path.join(self.auxiliary_dir, 'traffic', 'population_clip.tiff'),
                                                  population_shapefile_path, write=True)
        else:
            df = gpd.read_file(population_shapefile_path)

        self.logger.write_time_log('TrafficAreaSector', 'get_clipped_population', timeit.default_timer() - spent_time)

        return df

    def make_population_by_nuts(self, population_shape, nut_shp, pop_by_nut_path, write_file=True, csv_path=None,
                                column_id='ORDER07'):
        spent_time = timeit.default_timer()

        if not os.path.exists(pop_by_nut_path):
            nut_df = gpd.read_file(nut_shp)
            population_shape['area_in'] = population_shape.geometry.area
            df = gpd.overlay(population_shape, nut_df.to_crs(population_shape.crs), how='intersection')
            df.crs = population_shape.crs
            df.loc[:, 'data'] = df['data'] * (df.geometry.area / df['area_in'])
            del df['area_in']
            if write_file:
                df.to_file(pop_by_nut_path)
            if csv_path is not None:
                df = df.loc[:, ['data', column_id]].groupby(column_id).sum()
                df.to_csv(csv_path)
        else:
            df = gpd.read_file(pop_by_nut_path)

        self.logger.write_time_log('TrafficAreaSector', 'make_population_by_nuts', timeit.default_timer() - spent_time)
        return df

    def make_population_by_nuts_cell(self, pop_by_nut, grid_shp_path, pop_nut_cell_path, write_file=True):
        spent_time = timeit.default_timer()

        if not os.path.exists(pop_nut_cell_path):

            grid_shp = gpd.read_file(grid_shp_path)

            pop_by_nut = pop_by_nut.to_crs(grid_shp.crs)

            del pop_by_nut['NAME']
            pop_by_nut['area_in'] = pop_by_nut.geometry.area

            # df = gpd.overlay(pop_by_nut, grid_shp, how='intersection')
            df = self.spatial_overlays(pop_by_nut, grid_shp, how='intersection')

            df.crs = grid_shp.crs
            df.loc[:, 'data'] = df['data'] * (df.geometry.area / df['area_in'])
            del pop_by_nut['area_in']
            if write_file:
                df.to_file(pop_nut_cell_path)
        else:
            df = gpd.read_file(pop_nut_cell_path)

        self.logger.write_time_log('TrafficAreaSector', 'make_population_by_nuts_cell',
                                   timeit.default_timer() - spent_time)
        return df

    def make_vehicles_by_cell(self, pop_nut_cell, gasoline_path, total_pop_by_nut, grid_shape_path, veh_by_cell_path,
                              column_id='ORDER07'):
        spent_time = timeit.default_timer()

        if not os.path.exists(veh_by_cell_path):
            total_pop_by_nut.loc[:, column_id] = total_pop_by_nut[column_id].astype(np.int16)
            pop_nut_cell.loc[:, column_id] = pop_nut_cell[column_id].astype(np.int16)

            df = pop_nut_cell.merge(total_pop_by_nut, left_on=column_id, right_on=column_id, how='left')

            df['pop_percent'] = df['data_x'] / df['data_y']
            del df['data_x'], df['data_y'], df['CELL_ID']

            gas_df = pd.read_csv(gasoline_path, index_col='COPERT_V_name').transpose()
            vehicle_type_list = list(gas_df.columns.values)
            gas_df.loc[:, column_id] = gas_df.index.astype(np.int16)

            df = df.merge(gas_df, left_on=column_id, right_on=column_id, how='left')
            for vehicle_type in vehicle_type_list:
                df.loc[:, vehicle_type] = df[vehicle_type] * df['pop_percent']

            del df['pop_percent'], df[column_id]

            aux_df = df.loc[:, ['FID'] + vehicle_type_list].groupby('FID').sum()
            aux_df.loc[:, 'FID'] = aux_df.index
            grid_shape = gpd.read_file(grid_shape_path)
            geom = grid_shape.loc[aux_df.index, 'geometry']

            df = gpd.GeoDataFrame(aux_df, geometry=geom, crs=pop_nut_cell.crs)

            df.to_file(veh_by_cell_path)

        else:
            df = gpd.read_file(veh_by_cell_path)

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
            for x in xrange(24):
                temperature['t_{0}'.format(x)] = default_profile[x]

        else:
            temp_list = ['t_{0}'.format(x) for x in xrange(24)]
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

    def calculate_evaporative_emissions(self, temperature_dir, ef_file, date, tstep_num, tstep_frq, speciation_map_path,
                                        speciation_profile_path):
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

        temperature = TrafficSector.read_temperature(
            self.evaporative['c_lon'].min(), self.evaporative['c_lon'].max(), self.evaporative['c_lat'].min(),
            self.evaporative['c_lat'].max(), temperature_dir, date.replace(hour=0, minute=0, second=0, microsecond=0),
            24, 1)

        temperature_mean = gpd.GeoDataFrame(temperature[['t_{0}'.format(x) for x in xrange(24)]].mean(axis=1),
                                            columns=['temp'], geometry=temperature.geometry)
        temperature_mean['REC'] = temperature['REC']

        if 'T_REC' not in self.evaporative.columns.values:
            self.evaporative['T_REC'] = self.evaporative.apply(self.nearest, geom_union=temperature_mean.unary_union,
                                                               df1=self.evaporative, df2=temperature_mean,
                                                               geom1_col='centroid', src_column='REC', axis=1)
            del self.evaporative['c_lat'], self.evaporative['c_lon'], self.evaporative['centroid']

            self.evaporative.to_file(os.path.join(self.auxiliary_dir, 'vehicle_by_cell.shp'))
        else:
            del self.evaporative['c_lat'], self.evaporative['c_lon'], self.evaporative['centroid']

        self.evaporative = self.evaporative.merge(temperature_mean, left_on='T_REC', right_on='REC', how='left')

        ef_df = pd.read_csv(ef_file, sep=';')
        del ef_df['canister'], ef_df['Copert_V_name']
        ef_df.loc[ef_df['Tmin'].isnull(), 'Tmin'] = -999
        ef_df.loc[ef_df['Tmax'].isnull(), 'Tmax'] = 999

        for vehicle_type in veh_list:

            self.evaporative['EF'] = np.nan
            ef_aux = ef_df.loc[ef_df['CODE_HERMESv3'] == vehicle_type]
            for i, line in ef_aux.iterrows():
                self.evaporative.loc[(self.evaporative['temp'] < line.get('Tmax')) &
                                     (self.evaporative['temp'] >= line.get('Tmin')), 'EF'] = \
                    line.get('EFbase') * line.get('TF')

            self.evaporative.loc[:, vehicle_type] = self.evaporative[vehicle_type] * self.evaporative['EF']

        self.evaporative.loc[:, 'nmvoc'] = self.evaporative.loc[:, veh_list].sum(axis=1)
        self.evaporative = gpd.GeoDataFrame(self.evaporative.loc[:, ['nmvoc', 'T_REC', 'FID']], geometry=geom, crs=crs)

        # TODO change units function 3600 cell area
        self.evaporative = self.speciate_evaporative()

        self.evaporative = self.evaporative_temporal_distribution(
            self.get_profiles_from_temperature(temperature), date, tstep_num, tstep_frq)

        self.evaporative.set_index(['FID', 'tstep'], inplace=True)

        self.logger.write_time_log('TrafficAreaSector', 'calculate_evaporative_emissions',
                                   timeit.default_timer() - spent_time)
        return True

    def evaporative_temporal_distribution(self, temporal_profiles, date, tstep_num, tstep_frq):
        from datetime import timedelta
        spent_time = timeit.default_timer()

        aux = self.evaporative.merge(temporal_profiles, left_on='T_REC', right_on='REC', how='left')

        temporal_df_list = []
        pollutant_list = [e for e in self.evaporative.columns.values if e not in ('T_REC', 'FID', 'geometry')]

        for tstep in xrange(tstep_num):
            aux_temporal = aux[pollutant_list].multiply(aux['t_{0}'.format(date.hour)], axis=0)
            aux_temporal['FID'] = aux['FID']
            aux_temporal['tstep'] = tstep
            temporal_df_list.append(aux_temporal)
            date = date + timedelta(hours=tstep_frq)
        df = pd.concat(temporal_df_list)

        self.logger.write_time_log('TrafficAreaSector', 'evaporative_temporal_distribution',
                                   timeit.default_timer() - spent_time)
        return df

    def speciate_evaporative(self):
        spent_time = timeit.default_timer()

        speciated_df = self.evaporative.drop(columns=['nmvoc'])

        for p in self.output_pollutants:
            # From g/day to mol/day
            speciated_df[p] = self.evaporative['nmvoc'] * self.speciation_profile.get(p)

        self.logger.write_time_log('TrafficAreaSector', 'speciate_evaporative', timeit.default_timer() - spent_time)
        return speciated_df

    def small_cities_emissions_by_population(self, df, ef_file):
        spent_time = timeit.default_timer()

        df = df.loc[:, ['data', 'FID']].groupby('FID').sum()
        # print pop_nut_cell
        ef_df = pd.read_csv(ef_file, sep=';')
        # print ef_df
        ef_df.drop(['CODE_HERMESv3', 'Copert_V_name'], axis=1, inplace=True)
        for pollutant in ef_df.columns.values:
            # print ef_df[pollutant].iloc[0]
            df[pollutant] = df['data'] * ef_df[pollutant].iloc[0]
        df.drop('data', axis=1, inplace=True)

        self.logger.write_time_log('TrafficAreaSector', 'small_cities_emissions_by_population',
                                   timeit.default_timer() - spent_time)
        return df

    def speciate_small_cities(self, small_cities):
        spent_time = timeit.default_timer()

        in_p_list = list(small_cities.columns.values)
        df = pd.DataFrame()
        for in_p in in_p_list:
            for out_p in self.output_pollutants:
                # from kg/year to mol/year (gases) or g/year (aerosols)
                df[out_p] = small_cities[in_p] * (self.speciation_profile[out_p].iloc[0] / 1000 *
                                                  self.molecular_weights[in_p])
        if not set(self.speciation_profile.columns.values).isdisjoint(pmc_list):
            out_p = set(self.speciation_profile.columns.values).intersection(pmc_list).pop()
            try:
                df[out_p] = small_cities['pm10'] - small_cities['pm25']
            except KeyError as e:
                raise KeyError('{0} pollutant do not appear on the evaporative EF.'.format(e))

        self.logger.write_time_log('TrafficAreaSector', 'speciate_small_cities', timeit.default_timer() - spent_time)
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
                print len(grid.loc[grid['timezone'] == '', :])
                grid.loc[grid['timezone'] == '', 'timezone'] = aux_grid.loc[grid['timezone'] == '', :].apply(
                    lambda x: tz.closest_timezone_at(lng=x['lons'], lat=x['lats'], delta_degree=inc), axis=1)
                inc += 1
            grid.to_file('/home/Earth/ctena/Models/HERMESv3/OUT/timezones_2.shp')

        self.logger.write_time_log('TrafficAreaSector', 'add_timezones', timeit.default_timer() - spent_time)
        return grid

    def temporal_distribution_small(self, small_cities, starting_date, tstep_num, tstep_frq):
        import pytz
        spent_time = timeit.default_timer()

        p_names = small_cities.columns.values

        small_cities = small_cities.merge(self.grid_shp.loc[:, ['timezone']], left_index=True, right_index=True,
                                          how='left')

        small_cities.loc[:, 'utc'] = starting_date
        small_cities['date'] = small_cities.groupby('timezone')['utc'].apply(
            lambda x: pd.to_datetime(x).dt.tz_localize(pytz.utc).dt.tz_convert(x.name).dt.tz_localize(None))
        small_cities.drop(['utc', 'timezone'], inplace=True, axis=1)
        # print small_cities

        df_list = []
        for tstep in xrange(tstep_num):
            small_cities['month'] = small_cities['date'].dt.month
            small_cities['weekday'] = small_cities['date'].dt.dayofweek
            small_cities['hour'] = small_cities['date'].dt.hour
            small_cities.loc[small_cities['weekday'] <= 4, 'day_type'] = 'Weekday'
            small_cities.loc[small_cities['weekday'] == 5, 'day_type'] = 'Saturday'
            small_cities.loc[small_cities['weekday'] == 6, 'day_type'] = 'Sunday'

            for i, aux in small_cities.groupby(['month', 'weekday', 'hour', 'day_type']):
                small_cities.loc[aux.index, 'f'] = self.montly_profile.loc[str(i[0]), 1] * \
                                                   self.weekly_profile.loc[str(i[1]), 1] * \
                                                   self.hourly_profile.loc[str(i[2]), i[3]] * \
                                                   1/3600

            aux_df = small_cities.loc[:, p_names].multiply(small_cities['f'], axis=0)
            aux_df['tstep'] = tstep
            aux_df.set_index('tstep', append=True, inplace=True)
            df_list.append(aux_df)

            small_cities['date'] = small_cities['date'] + pd.to_timedelta(tstep_frq, unit='h')
        df = pd.concat(df_list)

        self.logger.write_time_log('TrafficAreaSector', 'temporal_distribution_small',
                                   timeit.default_timer() - spent_time)
        return df

    def calculate_small_cities_emissions(self, ef_file, starting_date, tstep_num, tstep_frq):
        spent_time = timeit.default_timer()

        # EF
        self.small_cities = self.small_cities_emissions_by_population(self.small_cities, ef_file)

        # Spectiacion
        self.small_cities = self.speciate_small_cities(self.small_cities)
        # Temporal
        grid = self.add_timezones(gpd.read_file(os.path.join(self.auxiliary_dir, 'shapefile', 'grid_shapefile.shp')),
                                  default=True)
        self.small_cities = self.temporal_distribution_small(self.small_cities, starting_date, tstep_num, tstep_frq)

        self.logger.write_time_log('TrafficAreaSector', 'calculate_small_cities_emissions',
                                   timeit.default_timer() - spent_time)
