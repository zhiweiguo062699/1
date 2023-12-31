#!/usr/bin/env python

import sys
import os
import timeit
import geopandas as gpd
import pandas as pd
import numpy as np
from hermesv3_bu.sectors.sector import Sector
from hermesv3_bu.io_server.io_raster import IoRaster
from hermesv3_bu.io_server.io_shapefile import IoShapefile
from hermesv3_bu.io_server.io_netcdf import IoNetcdf
from hermesv3_bu.tools.checker import check_files, error_exit

from pandas import DataFrame
from geopandas import GeoDataFrame


pmc_list = ['pmc', 'PMC']


class TrafficAreaSector(Sector):
    def __init__(self, comm, logger, auxiliary_dir, grid, clip, date_array, source_pollutants, vertical_levels,
                 population_tif_path, speciation_map_path, molecular_weights_path,
                 do_evaporative, gasoline_path, population_nuts3, nuts_shapefile, speciation_profiles_evaporative,
                 evaporative_ef_file, temperature_dir,
                 do_small_cities, small_cities_shp, speciation_profiles_small_cities, small_cities_ef_file,
                 small_cities_monthly_profile, small_cities_weekly_profile, small_cities_hourly_profile):
        spent_time = timeit.default_timer()
        logger.write_log('===== TRAFFIC AREA SECTOR =====')

        if do_evaporative:
            check_files([population_tif_path, speciation_map_path, molecular_weights_path,
                         gasoline_path, population_nuts3, nuts_shapefile, speciation_profiles_evaporative,
                         evaporative_ef_file, temperature_dir])
        if do_small_cities:
            check_files([population_tif_path, speciation_map_path, molecular_weights_path,
                         small_cities_shp, speciation_profiles_small_cities, small_cities_ef_file,
                         small_cities_monthly_profile, small_cities_weekly_profile, small_cities_hourly_profile])
        super(TrafficAreaSector, self).__init__(
            comm, logger, auxiliary_dir, grid, clip, date_array, source_pollutants, vertical_levels,
            None, None, None, speciation_map_path, None, molecular_weights_path)

        self.do_evaporative = do_evaporative
        self.temperature_dir = temperature_dir
        self.speciation_profiles_evaporative = self.read_speciation_profiles(speciation_profiles_evaporative)
        self.evaporative_ef_file = evaporative_ef_file

        # self.create_population_by_nut(population_tif_path, nuts_shapefile, population_nuts3, nut_column='nuts3_id')

        if do_evaporative:
            logger.write_log('\tInitialising evaporative emissions.', message_level=2)
            self.population_percent = self.get_population_percent(population_tif_path, population_nuts3, nuts_shapefile)
            self.evaporative = self.init_evaporative(gasoline_path)
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

    def get_population_by_nut2(self, path):
        """
        Read the CSV file that contains the amount of population by nut3.

        :param path: Path to the CSV file that contains the amount of population by nut3.
        :type path: str

        :return: DataFrame with the amount of population by nut3.
        :rtype: DataFrame
        """
        spent_time = timeit.default_timer()

        pop_by_nut3 = pd.read_csv(path)
        pop_by_nut3.set_index('nuts3_id', inplace=True)
        pop_by_nut3 = pop_by_nut3.to_dict()['population']

        self.logger.write_time_log('TrafficAreaSector', 'get_pop_by_nut3', timeit.default_timer() - spent_time)
        return pop_by_nut3

    def get_population_percent(self, pop_raster_path, pop_by_nut_path, nut_shapefile_path):
        """
        Calculate the percentage based on the amount of population.

        :param pop_raster_path: Path to the raster file that contains the population information.
        :type pop_raster_path: str

        :param pop_by_nut_path: Path to the CSV file that contains the amount of population by nut3.
        :type pop_by_nut_path: str

        :param nut_shapefile_path: Path to the shapefile that contains the nut3.
        :type nut_shapefile_path: str

        :return: DataFrame with the population distribution by destiny cell.
        :rtype: DataFrame
        """
        spent_time = timeit.default_timer()

        pop_percent_path = os.path.join(self.auxiliary_dir, 'traffic_area', 'population_percent')
        if not os.path.exists(pop_percent_path):
            # 1st Clip the raster
            self.logger.write_log("\t\tCreating clipped population raster", message_level=3)
            if self.comm.Get_rank() == 0:
                pop_raster_path = IoRaster(self.comm).clip_raster_with_shapefile_poly(
                    pop_raster_path, self.clip.shapefile, os.path.join(self.auxiliary_dir, 'traffic_area', 'pop.tif'))

            # 2nd Raster to shapefile
            self.logger.write_log("\t\tRaster to shapefile", message_level=3)
            pop_shp = IoRaster(self.comm).to_shapefile_parallel(
                pop_raster_path, gather=False, bcast=False, crs={'init': 'epsg:4326'})

            # 3rd Add NUT code
            self.logger.write_log("\t\tAdding nut codes to the shapefile", message_level=3)
            # if self.comm.Get_rank() == 0:
            pop_shp.drop(columns='CELL_ID', inplace=True)
            pop_shp.rename(columns={'data': 'population'}, inplace=True)
            pop_shp = self.add_nut_code(pop_shp, nut_shapefile_path, nut_value='nuts3_id')
            pop_shp = pop_shp[pop_shp['nut_code'] != -999]
            pop_shp = IoShapefile(self.comm).balance(pop_shp)

            # 4th Calculate population percent
            self.logger.write_log("\t\tCalculating population percentage on source resolution", message_level=3)
            pop_by_nut2 = self.get_population_by_nut2(pop_by_nut_path)
            pop_shp['tot_pop'] = pop_shp['nut_code'].map(pop_by_nut2)
            pop_shp['pop_per'] = pop_shp['population'] / pop_shp['tot_pop']
            pop_shp.drop(columns=['tot_pop', 'population'], inplace=True)

            # 5th Calculate percent by destiny cell
            self.logger.write_log("\t\tCalculating population percentage on destiny resolution", message_level=3)
            pop_shp.to_crs(self.grid.shapefile.crs, inplace=True)
            pop_shp['src_inter_fraction'] = pop_shp.geometry.area
            pop_shp = self.spatial_overlays(pop_shp.reset_index(), self.grid.shapefile.reset_index())
            pop_shp.drop(columns=['idx1', 'idx2', 'index'], inplace=True)
            pop_shp['src_inter_fraction'] = pop_shp.geometry.area / pop_shp['src_inter_fraction']
            pop_shp['pop_per'] = pop_shp['pop_per'] * pop_shp['src_inter_fraction']
            pop_shp.drop(columns=['src_inter_fraction'], inplace=True)

            pop_shp = IoShapefile(self.comm).gather_shapefile(pop_shp)
            if self.comm.Get_rank() == 0:
                popu_dist = pop_shp.groupby(['FID', 'nut_code']).sum()
                popu_dist = GeoDataFrame(
                    popu_dist,
                    geometry=self.grid.shapefile.loc[popu_dist.index.get_level_values('FID'), 'geometry'].values,
                    crs=self.grid.shapefile.crs)
                IoShapefile(self.comm).write_shapefile_serial(popu_dist.reset_index(), pop_percent_path)
            else:
                popu_dist = None
            popu_dist = IoShapefile(self.comm).split_shapefile(popu_dist)
        else:
            popu_dist = IoShapefile(self.comm).read_shapefile_parallel(pop_percent_path)
            popu_dist.set_index(['FID', 'nut_code'], inplace=True)

        self.logger.write_time_log('TrafficAreaSector', 'get_population_percent', timeit.default_timer() - spent_time)
        return popu_dist

    def get_population(self, pop_raster_path, nut_shapefile_path):
        """
        Calculate the amount of population.

        :param pop_raster_path: Path to the raster file that contains the population information.
        :type pop_raster_path: str

        :param nut_shapefile_path: Path to the shapefile that contains the small cities.
        :type nut_shapefile_path: str

        :return: DataFrame with the amount of population distribution by small city.
        :rtype: DataFrame
        """
        spent_time = timeit.default_timer()

        pop_path = os.path.join(self.auxiliary_dir, 'traffic_area', 'population_small')
        if not os.path.exists(pop_path):
            # 1st Clip the raster
            self.logger.write_log("\t\tCreating clipped population raster", message_level=3)
            if self.comm.Get_rank() == 0:
                pop_raster_path = IoRaster(self.comm).clip_raster_with_shapefile_poly(
                    pop_raster_path, self.clip.shapefile, os.path.join(self.auxiliary_dir, 'traffic_area', 'pop.tif'))

            # 2nd Raster to shapefile
            self.logger.write_log("\t\tRaster to shapefile", message_level=3)
            pop_shp = IoRaster(self.comm).to_shapefile_parallel(
                pop_raster_path, gather=False, bcast=False, crs={'init': 'epsg:4326'})

            # 3rd Add NUT code
            self.logger.write_log("\t\tAdding nut codes to the shapefile", message_level=3)
            # if self.comm.Get_rank() == 0:
            pop_shp.drop(columns='CELL_ID', inplace=True)
            pop_shp.rename(columns={'data': 'population'}, inplace=True)

            pop_shp = self.add_nut_code(pop_shp, nut_shapefile_path, nut_value='ORDER08')
            pop_shp = pop_shp[pop_shp['nut_code'] != -999]
            pop_shp = IoShapefile(self.comm).balance(pop_shp)

            # 4th Calculate percent by destiny cell
            self.logger.write_log("\t\tCalculating population percentage on destiny resolution", message_level=3)
            pop_shp.to_crs(self.grid.shapefile.crs, inplace=True)
            pop_shp['src_inter_fraction'] = pop_shp.geometry.area
            pop_shp = self.spatial_overlays(pop_shp.reset_index(), self.grid.shapefile.reset_index())
            pop_shp.drop(columns=['idx1', 'idx2', 'index'], inplace=True)
            pop_shp['src_inter_fraction'] = pop_shp.geometry.area / pop_shp['src_inter_fraction']
            pop_shp['population'] = pop_shp['population'] * pop_shp['src_inter_fraction']
            pop_shp.drop(columns=['src_inter_fraction', 'nut_code'], inplace=True)

            pop_shp = IoShapefile(self.comm).gather_shapefile(pop_shp)
            if self.comm.Get_rank() == 0:
                popu_dist = pop_shp.groupby(['FID']).sum()
                popu_dist = GeoDataFrame(
                    popu_dist,
                    geometry=self.grid.shapefile.loc[popu_dist.index.get_level_values('FID'), 'geometry'].values,
                    crs=self.grid.shapefile.crs)
                IoShapefile(self.comm).write_shapefile_serial(popu_dist.reset_index(), pop_path)
            else:
                popu_dist = None
            popu_dist = IoShapefile(self.comm).split_shapefile(popu_dist)
        else:
            popu_dist = IoShapefile(self.comm).read_shapefile_parallel(pop_path)
            popu_dist.set_index(['FID'], inplace=True)

        self.logger.write_time_log('TrafficAreaSector', 'get_population_percent', timeit.default_timer() - spent_time)
        return popu_dist

    def init_evaporative(self, gasoline_path):
        """
        Create the gasoline vehicle by destiny cell.

        :param gasoline_path: Path to the CSV file that contains the amount of vehicles by NUTS3.
        :type gasoline_path: str

        :return: Shapefile with the vehicle distribution.
        :rtype: GeoDataFrame
        """
        spent_time = timeit.default_timer()
        veh_cell_path = os.path.join(self.auxiliary_dir, 'traffic_area', 'vehicle_by_cell')
        if not os.path.exists(veh_cell_path):
            veh_cell = self.make_vehicles_by_cell(gasoline_path)
            IoShapefile(self.comm).write_shapefile_parallel(veh_cell.reset_index(), veh_cell_path)
        else:
            self.logger.write_log('\t\tReading vehicle shapefile by cell.', message_level=3)
            veh_cell = IoShapefile(self.comm).read_shapefile_parallel(veh_cell_path)
            veh_cell.set_index('FID', inplace=True)

        self.logger.write_time_log('TrafficAreaSector', 'init_evaporative', timeit.default_timer() - spent_time)
        return veh_cell

    def init_small_cities(self, global_path, small_cities_shapefile):
        spent_time = timeit.default_timer()
        pop = self.get_population(global_path, small_cities_shapefile)

        self.logger.write_time_log('TrafficAreaSector', 'init_small_cities', timeit.default_timer() - spent_time)
        return pop

    def read_vehicles_by_nut(self, path):
        spent_time = timeit.default_timer()

        vehicles_by_nut = pd.read_csv(path, index_col='COPERT_V_name')
        vehicle_list = vehicles_by_nut.index.values
        nut_list = vehicles_by_nut.columns.values.astype(np.int32)
        vehicles_by_nut = pd.DataFrame(vehicles_by_nut.values.T, index=nut_list, columns=vehicle_list)
        vehicles_by_nut.index.name = 'nuts3_id'

        self.logger.write_time_log('TrafficAreaSector', 'read_vehicles_by_nut', timeit.default_timer() - spent_time)
        return vehicles_by_nut

    def make_vehicles_by_cell(self, gasoline_path):
        spent_time = timeit.default_timer()
        vehicles_by_nut = self.read_vehicles_by_nut(gasoline_path)

        vehicle_list = vehicles_by_nut.columns.values
        vehicle_by_cell = pd.merge(self.population_percent.reset_index(), vehicles_by_nut.reset_index(),
                                   left_on='nut_code', right_on='nuts3_id', how='left')
        vehicle_by_cell.drop(columns=['nut_code', 'nuts3_id'], inplace=True)
        vehicle_by_cell[vehicle_list] = vehicle_by_cell[vehicle_list].multiply(
            vehicle_by_cell['pop_per'], axis='index')
        vehicle_by_cell.drop(columns=['pop_per'], inplace=True)
        vehicle_by_cell = IoShapefile(self.comm).gather_shapefile(vehicle_by_cell, rank=0)
        if self.comm.Get_rank() == 0:
            vehicle_by_cell = vehicle_by_cell.groupby('FID').sum()
        else:
            vehicle_by_cell = None
        vehicle_by_cell = IoShapefile(self.comm).split_shapefile(vehicle_by_cell)

        vehicle_by_cell = GeoDataFrame(
            vehicle_by_cell,
            geometry=self.grid.shapefile.loc[vehicle_by_cell.index.get_level_values('FID'), 'geometry'].values,
            crs=self.grid.shapefile.crs)

        self.logger.write_time_log('TrafficAreaSector', 'make_vehicles_by_cell', timeit.default_timer() - spent_time)
        return vehicle_by_cell

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

        self.evaporative.reset_index(inplace=True)
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
                self.evaporative, os.path.join(self.auxiliary_dir, 'traffic_area', 'vehicle_by_cell'))
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

    def small_cities_emissions_by_population(self, pop_by_cell):
        spent_time = timeit.default_timer()

        ef_df = pd.read_csv(self.small_cities_ef_file, sep=',')
        ef_df.drop(['Code', 'Copert_V_name'], axis=1, inplace=True)
        for pollutant in ef_df.columns.values:
            pop_by_cell[pollutant] = pop_by_cell['population'] * ef_df[pollutant].iloc[0]
        pop_by_cell.drop(columns=['population'], inplace=True)

        self.logger.write_time_log('TrafficAreaSector', 'small_cities_emissions_by_population',
                                   timeit.default_timer() - spent_time)
        return pop_by_cell

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

        aux_grid = self.grid.shapefile.loc[small_cities.index.values, :].reset_index().copy()

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
            error_exit('No traffic area emission selected. do_evaporative and do_small_cities are False')

        dataset['layer'] = 0
        dataset = dataset.groupby(['FID', 'layer', 'tstep']).sum()

        self.logger.write_time_log('TrafficAreaSector', 'to_grid', timeit.default_timer() - spent_time)
        return dataset

    def calculate_emissions(self):
        spent_time = timeit.default_timer()
        self.logger.write_log('\tCalculating traffic area.', message_level=2)

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
