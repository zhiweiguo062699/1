#!/usr/bin/env python

import sys
import os
import timeit
import numpy as np
import pandas as pd
import geopandas as gpd
from warnings import warn
from hermesv3_bu.sectors.sector import Sector
from hermesv3_bu.io_server.io_shapefile import IoShapefile
# from hermesv3_bu.io_server.io_netcdf import IoNetcdf
from hermesv3_bu.logger.log import Log
from hermesv3_bu.tools.checker import check_files, error_exit

INTERPOLATION_TYPE = 'linear'
# GRAVITI m/s-2
GRAVITY = 9.81
# CP J/kg*K
CP = 1005


class PointSourceSector(Sector):
    """
    Class to calculate the Point Source emissions

    :param grid: Grid of the destination domain
    :type grid: Grid

    :param catalog_path: Path to the fine that contains all the information for each point source.
    :type catalog_path: str

    :param monthly_profiles_path: Path to the file that contains the monthly profiles.
    :type monthly_profiles_path: str

    :param weekly_profiles_path: Path to the file that contains the weekly profiles.
    :type weekly_profiles_path: str

    :param hourly_profiles_path: Path to the file that contains the hourly profile.
    :type hourly_profiles_path: str

    :param speciation_map_path: Path to the file that contains the speciation map.
    :type speciation_map_path: str

    :param speciation_profiles_path: Path to the file that contains the speciation profiles.
    :type speciation_profiles_path: str

    :param sector_list: List os sectors (SNAPS) to take into account. 01, 03, 04, 09
    :type sector_list: list
    """
    def __init__(self, comm, logger, auxiliary_dir, grid, clip, date_array, source_pollutants, vertical_levels,
                 catalog_path, monthly_profiles_path, weekly_profiles_path, hourly_profiles_path,
                 speciation_map_path, speciation_profiles_path, sector_list, measured_emission_path,
                 molecular_weights_path, plume_rise=False, plume_rise_pahts=None):
        spent_time = timeit.default_timer()
        logger.write_log('===== POINT SOURCES SECTOR =====')
        check_files(
            [catalog_path, monthly_profiles_path, weekly_profiles_path, hourly_profiles_path, speciation_map_path,
             speciation_profiles_path])
        super(PointSourceSector, self).__init__(
            comm, logger, auxiliary_dir, grid, clip, date_array, source_pollutants, vertical_levels,
            monthly_profiles_path, weekly_profiles_path, hourly_profiles_path, speciation_map_path,
            speciation_profiles_path, molecular_weights_path)

        self.plume_rise = plume_rise
        self.catalog = self.read_catalog_shapefile(catalog_path, sector_list)
        self.check_catalog()
        self.catalog_measured = self.read_catalog_for_measured_emissions(catalog_path, sector_list)
        self.measured_path = measured_emission_path
        self.plume_rise_pahts = plume_rise_pahts

        self.logger.write_time_log('PointSourceSector', '__init__', timeit.default_timer() - spent_time)

    def check_catalog(self):
        # Checking monthly profiles IDs
        links_month = set(np.unique(self.catalog['P_month'].dropna().values))
        month = set(self.monthly_profiles.index.values)
        month_res = links_month - month
        if len(month_res) > 0:
            error_exit("The following monthly profile IDs reported in the point sources shapefile do not appear " +
                       "in the monthly profiles file. {0}".format(month_res))
        # Checking weekly profiles IDs
        links_week = set(np.unique(self.catalog['P_week'].dropna().values))
        week = set(self.weekly_profiles.index.values)
        week_res = links_week - week
        if len(week_res) > 0:
            error_exit("The following weekly profile IDs reported in the point sources shapefile do not appear " +
                       "in the weekly profiles file. {0}".format(week_res))
        # Checking hourly profiles IDs
        links_hour = set(np.unique(self.catalog['P_hour'].dropna().values))
        hour = set(self.hourly_profiles.index.values)
        hour_res = links_hour - hour
        if len(hour_res) > 0:
            error_exit("The following hourly profile IDs reported in the point sources shapefile do not appear " +
                       "in the hourly profiles file. {0}".format(hour_res))
        # Checking specly profiles IDs
        links_spec = set(np.unique(self.catalog['P_spec'].dropna().values))
        spec = set(self.speciation_profile.index.values)
        spec_res = links_spec - spec
        if len(spec_res) > 0:
            error_exit("The following speciation profile IDs reported in the point sources shapefile do not appear " +
                       "in the speciation profiles file. {0}".format(spec_res))

    def read_catalog_csv(self, catalog_path, sector_list):
        """
        Read the catalog

        :param catalog_path: path to the catalog
        :type catalog_path: str

        :param sector_list: List of sectors to take into account
        :type sector_list: list

        :return: catalog
        :rtype: DataFrame
        """
        spent_time = timeit.default_timer()

        if self.comm.Get_rank() == 0:
            if self.plume_rise:
                columns = {"Code": np.str, "Cons": np.bool, "SNAP": np.str, "Lon": np.float64,
                           "Lat": np.float64, "Height": np.float64, "Diameter": np.float64,
                           "Speed": np.float64, "Temp": np.float64, "AF": np.float64,
                           "P_month": np.str, "P_week": np.str, "P_hour": np.str, "P_spec": np.str}
            else:
                columns = {"Code": np.str, "Cons": np.bool, "SNAP": np.str, "Lon": np.float64,
                           "Lat": np.float64, "Height": np.float64, "AF": np.float64,
                           "P_month": np.str, "P_week": np.str, "P_hour": np.str, "P_spec": np.str}
            for pollutant in self.source_pollutants:
                # EF in Kg / Activity factor
                columns['EF_{0}'.format(pollutant)] = np.float64

            catalog_df = pd.read_csv(catalog_path, usecols=columns.keys(), dtype=columns)

            # Filtering
            catalog_df = catalog_df.loc[catalog_df['Cons'] == 1, :]
            catalog_df.drop('Cons', axis=1, inplace=True)

            # Filtering
            catalog_df = catalog_df.loc[catalog_df['AF'] != -1, :]

            if sector_list is not None:
                catalog_df = catalog_df.loc[catalog_df['SNAP'].str[:2].isin(sector_list)]
            catalog_df.drop('SNAP', axis=1, inplace=True)
            catalog_df.sort_values('Lat', inplace=True)

            catalog_df = self.to_geodataframe(catalog_df)

            catalog_df = gpd.sjoin(catalog_df, self.clip.shapefile.to_crs(catalog_df.crs), how='inner')
            catalog_df.drop(columns=['index_right'], inplace=True)

        else:
            catalog_df = None
        self.comm.Barrier()
        catalog_df = IoShapefile(self.comm).split_shapefile(catalog_df)
        self.logger.write_time_log('PointSourceSector', 'read_catalog', timeit.default_timer() - spent_time)
        return catalog_df

    def read_catalog_shapefile(self, catalog_path, sector_list):
        """
        Read the catalog

        :param catalog_path: path to the catalog
        :type catalog_path: str

        :param sector_list: List of sectors to take into account
        :type sector_list: list

        :return: catalog
        :rtype: DataFrame
        """
        spent_time = timeit.default_timer()

        if self.comm.Get_rank() == 0:
            if self.plume_rise:
                columns = {"Code": np.str, "Cons": np.bool, "SNAP": np.str, "Height": np.float64,
                           "Diameter": np.float64, "Speed": np.float64, "Temp": np.float64, "AF": np.float64,
                           "P_month": np.str, "P_week": np.str, "P_hour": np.str, "P_spec": np.str}
            else:
                columns = {"Code": np.str, "Cons": np.bool, "SNAP": np.str, "Height": np.float64, "AF": np.float64,
                           "P_month": np.str, "P_week": np.str, "P_hour": np.str, "P_spec": np.str}
            for pollutant in self.source_pollutants:
                # EF in Kg / Activity factor
                columns['EF_{0}'.format(pollutant)] = np.float64

            catalog_df = gpd.read_file(catalog_path)

            columns_to_drop = list(set(catalog_df.columns.values) - set(list(columns.keys()) + ['geometry']))

            if len(columns_to_drop) > 0:
                catalog_df.drop(columns=columns_to_drop, inplace=True)
            for col, typ in columns.items():
                catalog_df[col] = catalog_df[col].astype(typ)

            # Filtering
            catalog_df = catalog_df.loc[catalog_df['Cons'] == 1, :]
            catalog_df.drop('Cons', axis=1, inplace=True)

            # Filtering
            catalog_df = catalog_df.loc[catalog_df['AF'] != -1, :]

            if sector_list is not None:
                catalog_df = catalog_df.loc[catalog_df['SNAP'].str[:2].isin(sector_list)]
            catalog_df.drop('SNAP', axis=1, inplace=True)

            catalog_df = gpd.sjoin(catalog_df, self.clip.shapefile.to_crs(catalog_df.crs), how='inner')
            catalog_df.drop(columns=['index_right'], inplace=True)

        else:
            catalog_df = None
        self.comm.Barrier()
        catalog_df = IoShapefile(self.comm).split_shapefile(catalog_df)
        self.logger.write_time_log('PointSourceSector', 'read_catalog', timeit.default_timer() - spent_time)
        return catalog_df

    def read_catalog_for_measured_emissions_csv(self, catalog_path, sector_list):
        """
        Read the catalog

        :param catalog_path: path to the catalog
        :type catalog_path: str

        :param sector_list: List of sectors to take into account
        :type sector_list: list

        :return: catalog
        :rtype: DataFrame
        """
        spent_time = timeit.default_timer()

        if self.plume_rise:
            columns = {"Code": np.str, "Cons": np.bool, "SNAP": np.str, "Lon": np.float64, "Lat": np.float64,
                       "Height": np.float64, "Diameter": np.float64, "Speed": np.float64, "Temp": np.float64,
                       "AF": np.float64, "P_spec": np.str}
        else:
            columns = {"Code": np.str, "Cons": np.bool, "SNAP": np.str, "Lon": np.float64, "Lat": np.float64,
                       "Height": np.float64, "AF": np.float64, "P_spec": np.str}
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

        self.logger.write_time_log('PointSourceSector', 'read_catalog_for_measured_emissions',
                                   timeit.default_timer() - spent_time)
        return catalog_df

    def read_catalog_for_measured_emissions(self, catalog_path, sector_list):
        """
        Read the catalog

        :param catalog_path: path to the catalog
        :type catalog_path: str

        :param sector_list: List of sectors to take into account
        :type sector_list: list

        :return: catalog
        :rtype: DataFrame
        """
        spent_time = timeit.default_timer()

        if self.plume_rise:
            columns = {"Code": np.str, "Cons": np.bool, "SNAP": np.str, "Lon": np.float64, "Lat": np.float64,
                       "Height": np.float64, "Diameter": np.float64, "Speed": np.float64, "Temp": np.float64,
                       "AF": np.float64, "P_spec": np.str}
        else:
            columns = {"Code": np.str, "Cons": np.bool, "SNAP": np.str, "Lon": np.float64, "Lat": np.float64,
                       "Height": np.float64, "AF": np.float64, "P_spec": np.str}
        # for pollutant in self.pollutant_list:
        #     columns['EF_{0}'.format(pollutant)] = settings.precision

        catalog_df = gpd.read_file(catalog_path)

        columns_to_drop = list(set(catalog_df.columns.values) - set(list(columns.keys()) + ['geometry']))

        if len(columns_to_drop) > 0:
            catalog_df.drop(columns=columns_to_drop, inplace=True)
        for col, typ in columns.items():
            catalog_df[col] = catalog_df[col].astype(typ)

        # Filtering
        catalog_df = catalog_df.loc[catalog_df['Cons'] == 1, :]
        catalog_df.drop('Cons', axis=1, inplace=True)

        # Filtering
        catalog_df = catalog_df.loc[catalog_df['AF'] == -1, :]
        catalog_df.drop('AF', axis=1, inplace=True)

        if sector_list is not None:
            catalog_df = catalog_df.loc[catalog_df['SNAP'].str[:2].isin(sector_list)]
        catalog_df.drop('SNAP', axis=1, inplace=True)

        self.logger.write_time_log('PointSourceSector', 'read_catalog_for_measured_emissions',
                                   timeit.default_timer() - spent_time)
        return catalog_df

    def to_geodataframe(self, catalog):
        """
        Convert a simple DataFrame with Lat, Lon columns into a GeoDataFrame as a shape

        :param catalog: DataFrame with all the information of each point source.
        :type catalog: DataFrame

        :return: GeoDataFrame with all the information of each point source.
        :rtype: GeoDataFrame
        """
        from shapely.geometry import Point
        spent_time = timeit.default_timer()

        geometry = [Point(xy) for xy in zip(catalog.Lon, catalog.Lat)]
        catalog.drop(['Lon', 'Lat'], axis=1, inplace=True)
        crs = {'init': 'epsg:4326'}
        catalog = gpd.GeoDataFrame(catalog, crs=crs, geometry=geometry)
        self.logger.write_time_log('PointSourceSector', 'to_geodataframe', timeit.default_timer() - spent_time)
        return catalog

    def get_yearly_emissions(self, catalog):
        """
        Calculate yearly emissions.

        :param catalog: Catalog with the activity factor (AF) column and all the emission factor column for each
        pollutant.
        :type catalog: DataFrame

        :return: Catalog with yearly emissions of each point source for all the pollutants (as column names).
        :rtype: DataFrame
        """
        spent_time = timeit.default_timer()
        self.logger.write_log('\tCalculating yearly emissions', message_level=2)
        for pollutant in self.source_pollutants:
            catalog.rename(columns={u'EF_{0}'.format(pollutant): pollutant}, inplace=True)
            catalog[pollutant] = catalog[pollutant] * catalog['AF']

        catalog.drop('AF', axis=1, inplace=True)
        self.logger.write_time_log('PointSourceSector', 'get_yearly_emissions', timeit.default_timer() - spent_time)
        return catalog

    def get_temporal_factors(self, catalog):
        """
        Calculates the temporal factor for each point source and each time step.

        :param catalog: Catalog with the activity factor (AF) column and all the emission factor column for each
        pollutant.
        :type catalog: DataFrame

        :return: Catalog with yearly emissions of each point source for all the pollutants (as column names).
        :rtype: DataFrame
        """
        spent_time = timeit.default_timer()
        self.logger.write_log('\tCalculating hourly emissions', message_level=2)

        def get_mf(df):
            month_factor = self.monthly_profiles.loc[df.name[1], df.name[0]]

            df['MF'] = month_factor
            return df.loc[:, ['MF']]

        def get_wf(df):
            weekly_profile = self.calculate_rebalanced_weekly_profile(self.weekly_profiles.loc[df.name[1], :].to_dict(),
                                                                      df.name[0])
            df['WF'] = weekly_profile[df.name[0].weekday()]
            return df.loc[:, ['WF']]

        def get_hf(df):
            hourly_profile = self.hourly_profiles.loc[df.name[1], :].to_dict()
            hour_factor = hourly_profile[df.name[0]]

            df['HF'] = hour_factor
            return df.loc[:, ['HF']]

        catalog['month'] = catalog['date'].dt.month
        catalog['weekday'] = catalog['date'].dt.weekday
        catalog['hour'] = catalog['date'].dt.hour
        catalog['date_as_date'] = catalog['date'].dt.date

        catalog['MF'] = catalog.groupby(['month', 'P_month']).apply(get_mf)
        catalog['WF'] = catalog.groupby(['date_as_date', 'P_week']).apply(get_wf)
        catalog['HF'] = catalog.groupby(['hour', 'P_hour']).apply(get_hf)

        catalog['temp_factor'] = catalog['MF'] * catalog['WF'] * catalog['HF']
        catalog.drop(['MF', 'WF', 'HF'], axis=1, inplace=True)

        catalog[self.source_pollutants] = catalog[self.source_pollutants].multiply(catalog['temp_factor'], axis=0)

        catalog.drop('temp_factor', axis=1, inplace=True)

        self.logger.write_time_log('PointSourceSector', 'get_temporal_factors', timeit.default_timer() - spent_time)
        return catalog

    def calculate_hourly_emissions(self, catalog):
        """
        Calculate the hourly emissions

        :param catalog: Catalog to calculate.
        :type catalog: DataFrame

        :return: Catalog with the hourly emissions.
        :rtype: DataFrame
        """
        spent_time = timeit.default_timer()

        catalog = self.get_yearly_emissions(catalog)
        catalog = self.get_temporal_factors(catalog)

        catalog.set_index(['Code', 'tstep'], inplace=True)
        self.logger.write_time_log('PointSourceSector', 'calculate_hourly_emissions',
                                   timeit.default_timer() - spent_time)
        return catalog

    def get_meteo_xy(self, dataframe, netcdf_path):
        spent_time = timeit.default_timer()

        def nearest(row, geom_union, df1, df2, geom1_col='geometry', geom2_col='geometry', src_column=None):
            """Finds the nearest point and return the corresponding value from specified column.
            https://automating-gis-processes.github.io/2017/lessons/L3/nearest-neighbour.html
            """
            from shapely.ops import nearest_points

            # Find the geometry that is closest
            nearest = df2[geom2_col] == nearest_points(row[geom1_col], geom_union)[1]
            # Get the corresponding value from df2 (matching is based on the geometry)
            value = df2[nearest][src_column].get_values()[0]
            return value
        from netCDF4 import Dataset
        from shapely.geometry import Point
        import numpy as np
        import pandas as pd
        import geopandas as gpd
        check_files(netcdf_path)
        nc = Dataset(netcdf_path, mode='r')
        try:
            lats = nc.variables['lat'][:]
            lons = nc.variables['lon'][:]
        except KeyError as e:
            error_exit("{0} variable not found in {1} file.".format(str(e), netcdf_path))
        x = np.array([np.arange(lats.shape[1])] * lats.shape[0])
        y = np.array([np.arange(lats.shape[0]).T] * lats.shape[1]).T

        nc_dataframe = pd.DataFrame.from_dict({'X': x.flatten(), 'Y': y.flatten()})
        nc_dataframe = gpd.GeoDataFrame(nc_dataframe,
                                        geometry=[Point(xy) for xy in list(zip(lons.flatten(), lats.flatten()))],
                                        crs={'init': 'epsg:4326'})
        nc_dataframe['index'] = nc_dataframe.index

        union = nc_dataframe.unary_union
        dataframe['meteo_index'] = dataframe.apply(
            nearest, geom_union=union, df1=dataframe, df2=nc_dataframe, src_column='index', axis=1)

        dataframe['X'] = nc_dataframe.loc[dataframe['meteo_index'], 'X'].values
        dataframe['Y'] = nc_dataframe.loc[dataframe['meteo_index'], 'Y'].values

        self.logger.write_time_log('PointSourceSector', 'get_meteo_xy', timeit.default_timer() - spent_time)
        return dataframe[['X', 'Y']]

    def get_plumerise_meteo(self, catalog):
        def get_sfc_value(dataframe, dir_path, var_name):
            from netCDF4 import Dataset, num2date
            nc_path = os.path.join(dir_path,
                                   '{0}_{1}.nc'.format(var_name, dataframe.name.replace(hour=0).strftime("%Y%m%d%H")))
            check_files(nc_path)
            netcdf = Dataset(nc_path, mode='r')
            # time_index
            try:
                time = netcdf.variables['time']
            except KeyError as e:
                error_exit("{0} variable not found in {1} file.".format(str(e), nc_path))
            nc_times = [x.replace(minute=0, second=0, microsecond=0) for x in
                        num2date(time[:], time.units, time.calendar)]
            time_index = nc_times.index(dataframe.name.to_pydatetime().replace(tzinfo=None))

            try:
                var = netcdf.variables[var_name][time_index, 0, :]
            except KeyError as e:
                error_exit("{0} variable not found in {1} file.".format(str(e), nc_path))
            netcdf.close()
            dataframe[var_name] = var[dataframe['Y'], dataframe['X']]

            return dataframe[[var_name]]

        def get_layers(dataframe, dir_path, var_name):
            from netCDF4 import Dataset, num2date
            nc_path = os.path.join(dir_path,
                                   '{0}_{1}.nc'.format(var_name, dataframe.name.replace(hour=0).strftime("%Y%m%d%H")))
            check_files(nc_path)
            netcdf = Dataset(nc_path, mode='r')
            # time_index
            try:
                time = netcdf.variables['time']
            except KeyError as e:
                error_exit("{0} variable not found in {1} file.".format(str(e), nc_path))
            nc_times = [x.replace(minute=0, second=0, microsecond=0) for x in
                        num2date(time[:], time.units, time.calendar)]
            time_index = nc_times.index(dataframe.name.to_pydatetime().replace(tzinfo=None))

            try:
                var = np.flipud(netcdf.variables[var_name][time_index, :, :, :])
            except KeyError as e:
                error_exit("{0} variable not found in {1} file.".format(str(e), nc_path))
            netcdf.close()
            var = var[:, dataframe['Y'], dataframe['X']]

            pre_t_lay = 0
            lay_list = []
            # lay_list = ['l_sfc']
            # dataframe['l_sfc'] = 2
            for i, t_lay in enumerate(var):
                t_lay += pre_t_lay
                dataframe['l_{0}'.format(i)] = t_lay
                pre_t_lay = t_lay
                lay_list.append('l_{0}'.format(i))

            dataframe['layers'] = dataframe[lay_list].values.tolist()

            return dataframe[['layers']]

        def get_temp_top(dataframe, dir_path, var_name):
            from netCDF4 import Dataset, num2date
            from scipy.interpolate import interp1d as interpolate

            nc_path = os.path.join(dir_path,
                                   '{0}_{1}.nc'.format(var_name, dataframe.name.replace(hour=0).strftime("%Y%m%d%H")))
            check_files(nc_path)
            netcdf = Dataset(nc_path, mode='r')
            # time_index
            try:
                time = netcdf.variables['time']
            except KeyError as e:
                error_exit("{0} variable not found in {1} file.".format(str(e), nc_path))
            nc_times = [x.replace(minute=0, second=0, microsecond=0) for x in
                        num2date(time[:], time.units, time.calendar)]
            time_index = nc_times.index(dataframe.name.to_pydatetime().replace(tzinfo=None))

            try:
                var = np.flipud(netcdf.variables[var_name][time_index, :, :, :])
            except KeyError as e:
                error_exit("{0} variable not found in {1} file.".format(str(e), nc_path))
            netcdf.close()
            var = var[:, dataframe['Y'], dataframe['X']]

            lay_list = ['temp_sfc']
            for i, t_lay in enumerate(var):
                dataframe['t_{0}'.format(i)] = t_lay
                lay_list.append('t_{0}'.format(i))

            # Setting minimum height to 2 because is the lowest temperature layer height.
            dataframe.loc[dataframe['Height'] < 2, 'Height'] = 2

            dataframe['temp_list'] = dataframe[lay_list].values.tolist()
            dataframe.drop(columns=lay_list, inplace=True)
            # Interpolation
            for ind, row in dataframe.iterrows():
                f_temp = interpolate([2] + row.get('layers'), row.get('temp_list'), kind=INTERPOLATION_TYPE)
                dataframe.loc[ind, 'temp_top'] = f_temp(row.get('Height'))

            return dataframe[['temp_top']]

        def get_wind_speed_10m(dataframe, u_dir_path, v_dir_path, u_var_name, v_var_name):
            from netCDF4 import Dataset, num2date
            # === u10 ===
            u10_nc_path = os.path.join(
                u_dir_path, '{0}_{1}.nc'.format(u_var_name, dataframe.name.replace(hour=0).strftime("%Y%m%d%H")))
            check_files(u10_nc_path)
            u10_netcdf = Dataset(u10_nc_path, mode='r')
            # time_index
            try:
                time = u10_netcdf.variables['time']
            except KeyError as e:
                error_exit("{0} variable not found in {1} file.".format(str(e), u10_nc_path))
            nc_times = [x.replace(minute=0, second=0, microsecond=0) for x in
                        num2date(time[:], time.units, time.calendar)]
            time_index = nc_times.index(dataframe.name.to_pydatetime().replace(tzinfo=None))

            try:
                var = u10_netcdf.variables[u_var_name][time_index, 0, :]
            except KeyError as e:
                error_exit("{0} variable not found in {1} file.".format(str(e), u10_nc_path))
            u10_netcdf.close()
            dataframe['u10'] = var[dataframe['Y'], dataframe['X']]

            # === v10 ===
            v10_nc_path = os.path.join(
                v_dir_path, '{0}_{1}.nc'.format(v_var_name, dataframe.name.replace(hour=0).strftime("%Y%m%d%H")))
            check_files(v10_nc_path)
            v10_netcdf = Dataset(v10_nc_path, mode='r')

            try:
                var = v10_netcdf.variables[v_var_name][time_index, 0, :]
            except KeyError as e:
                error_exit("{0} variable not found in {1} file.".format(str(e), v10_nc_path))
            v10_netcdf.close()
            dataframe['v10'] = var[dataframe['Y'], dataframe['X']]

            # === wind speed ===
            dataframe['wSpeed_10'] = np.linalg.norm(dataframe[['u10', 'v10']].values, axis=1)

            return dataframe[['wSpeed_10']]

        def get_wind_speed_top(dataframe, u_dir_path, v_dir_path, u_var_name, v_var_name):
            from netCDF4 import Dataset, num2date
            from scipy.interpolate import interp1d as interpolate
            # === u10 ===
            u10_nc_path = os.path.join(
                u_dir_path, '{0}_{1}.nc'.format(u_var_name, dataframe.name.replace(hour=0).strftime("%Y%m%d%H")))
            check_files(u10_nc_path)
            u10_netcdf = Dataset(u10_nc_path, mode='r')
            # time_index
            try:
                time = u10_netcdf.variables['time']
            except KeyError as e:
                error_exit("{0} variable not found in {1} file.".format(str(e), u10_nc_path))
            nc_times = [x.replace(minute=0, second=0, microsecond=0) for x in
                        num2date(time[:], time.units, time.calendar)]
            time_index = nc_times.index(dataframe.name.to_pydatetime().replace(tzinfo=None))

            try:
                var = np.flipud(u10_netcdf.variables[u_var_name][time_index, :, :, :])
            except KeyError as e:
                error_exit("{0} variable not found in {1} file.".format(str(e), u10_nc_path))

            u10_netcdf.close()
            var = var[:, dataframe['Y'], dataframe['X']]

            for i, t_lay in enumerate(var):
                dataframe['u_{0}'.format(i)] = t_lay

            # === v10 ===
            v10_nc_path = os.path.join(
                v_dir_path, '{0}_{1}.nc'.format(v_var_name, dataframe.name.replace(hour=0).strftime("%Y%m%d%H")))
            check_files(v10_nc_path)
            v10_netcdf = Dataset(v10_nc_path, mode='r')

            try:
                var = np.flipud(v10_netcdf.variables[v_var_name][time_index, :, :, :])
            except KeyError as e:
                error_exit("{0} variable not found in {1} file.".format(str(e), v10_nc_path))
            v10_netcdf.close()
            var = var[:, dataframe['Y'], dataframe['X']]

            ws_lay_list = ['wSpeed_10']
            for i, t_lay in enumerate(var):
                dataframe['v_{0}'.format(i)] = t_lay
                ws_lay_list.append('ws_{0}'.format(i))
                dataframe['ws_{0}'.format(i)] = np.linalg.norm(dataframe[['u_{0}'.format(i), 'v_{0}'.format(i)]].values,
                                                               axis=1)
            # Setting minimum height to 10 because is the lowest wind Speed layer.
            dataframe.loc[dataframe['Height'] < 10, 'Height'] = 10
            dataframe['ws_list'] = dataframe[ws_lay_list].values.tolist()

            for ind, row in dataframe.iterrows():
                f_ws = interpolate([10] + row.get('layers'), row.get('ws_list'), kind=INTERPOLATION_TYPE)
                dataframe.loc[ind, 'windSpeed_top'] = f_ws(row.get('Height'))

            return dataframe[['windSpeed_top']]

        # TODO Use IoNetCDF
        spent_time = timeit.default_timer()

        meteo_xy = self.get_meteo_xy(catalog.groupby('Code').first(), os.path.join(
            self.plume_rise_pahts['temperature_sfc_dir'],
            't2_{0}.nc'.format(self.date_array[0].replace(hour=0).strftime("%Y%m%d%H"))))

        catalog = catalog.merge(meteo_xy, left_index=True, right_index=True)

        # ===== 3D Meteo variables =====
        # Adding stc_temp
        self.logger.write_log('\t\tGetting temperature from {0}'.format(self.plume_rise_pahts['temperature_sfc_dir']),
                              message_level=3)
        catalog['temp_sfc'] = catalog.groupby('date_utc')['X', 'Y'].apply(
            lambda x: get_sfc_value(x, self.plume_rise_pahts['temperature_sfc_dir'], 't2'))
        self.logger.write_log('\t\tGetting friction velocity from {0}'.format(
            self.plume_rise_pahts['friction_velocity_dir']),  message_level=3)
        catalog['friction_v'] = catalog.groupby('date_utc')['X', 'Y'].apply(
            lambda x: get_sfc_value(x, self.plume_rise_pahts['friction_velocity_dir'], 'ustar'))
        self.logger.write_log('\t\tGetting PBL height from {0}'.format(
            self.plume_rise_pahts['pblh_dir']), message_level=3)
        catalog['pbl'] = catalog.groupby('date_utc')['X', 'Y'].apply(
            lambda x: get_sfc_value(x, self.plume_rise_pahts['pblh_dir'], 'mixed_layer_height'))
        self.logger.write_log('\t\tGetting obukhov length from {0}'.format(
            self.plume_rise_pahts['obukhov_length_dir']), message_level=3)
        catalog['obukhov_len'] = catalog.groupby('date_utc')['X', 'Y'].apply(
            lambda x: get_sfc_value(x, self.plume_rise_pahts['obukhov_length_dir'], 'rmol'))
        catalog['obukhov_len'] = 1. / catalog['obukhov_len']

        self.logger.write_log('\t\tGetting layer thickness from {0}'.format(
            self.plume_rise_pahts['layer_thickness_dir']), message_level=3)
        catalog['layers'] = catalog.groupby('date_utc')['X', 'Y'].apply(
            lambda x: get_layers(x, self.plume_rise_pahts['layer_thickness_dir'], 'layer_thickness'))
        self.logger.write_log('\t\tGetting temperatue at the top from {0}'.format(
            self.plume_rise_pahts['temperature_4d_dir']), message_level=3)
        catalog['temp_top'] = catalog.groupby('date_utc')['X', 'Y', 'Height', 'layers', 'temp_sfc'].apply(
            lambda x: get_temp_top(x, self.plume_rise_pahts['temperature_4d_dir'], 't'))
        self.logger.write_log('\t\tGetting wind speed at 10 m', message_level=3)
        catalog['wSpeed_10'] = catalog.groupby('date_utc')['X', 'Y'].apply(
            lambda x: get_wind_speed_10m(x, self.plume_rise_pahts['u10_wind_speed_dir'],
                                         self.plume_rise_pahts['v10_wind_speed_dir'], 'u10', 'v10'))
        self.logger.write_log('\t\tGetting wind speed at the top', message_level=3)
        catalog['wSpeed_top'] = catalog.groupby('date_utc')['X', 'Y', 'Height', 'layers', 'wSpeed_10'].apply(
            lambda x: get_wind_speed_top(x, self.plume_rise_pahts['u_wind_speed_4d_dir'],
                                         self.plume_rise_pahts['v_wind_speed_4d_dir'], 'u', 'v'))
        catalog.drop(columns=['wSpeed_10', 'layers', 'X', 'Y'], inplace=True)
        self.logger.write_time_log('PointSourceSector', 'get_plumerise_meteo', timeit.default_timer() - spent_time)
        return catalog

    def get_plume_rise_top_bot(self, catalog):
        spent_time = timeit.default_timer()

        catalog = self.get_plumerise_meteo(catalog).reset_index()

        # Step 1: Bouyancy flux
        catalog.loc[catalog['Temp'] <= catalog['temp_top'], 'Fb'] = 0
        try:
            catalog.loc[catalog['Temp'] > catalog['temp_top'], 'Fb'] = ((catalog['Temp'] - catalog['temp_top']) / catalog[
                'Temp']) * ((catalog['Speed'] * np.square(catalog['Diameter'])) / 4.) * GRAVITY
        except ValueError as e:
            print(catalog)
            sys.stdout.flush()
            error_exit(str(e))

        # Step 2: Stability parameter
        catalog['S'] = np.maximum(
            (GRAVITY / catalog['temp_top']) * (((catalog['temp_top'] - catalog['temp_sfc']) / catalog['Height']) +
                                               (GRAVITY / CP)),
            0.047 / catalog['temp_top'])

        # Step 3: Plume thickness
        # catalog.reset_index(inplace=True)
        neutral_atm = (catalog['obukhov_len'] > 2. * catalog['Height']) | (
                    catalog['obukhov_len'] < -0.25 * catalog['Height'])
        stable_atm = ((catalog['obukhov_len'] > 0) & (catalog['obukhov_len'] < 2 * catalog['Height'])) | (
                    catalog['Height'] > catalog['pbl'])
        unstable_atm = ((catalog['obukhov_len'] > -0.25 * catalog['Height']) & (catalog['obukhov_len'] < 0))

        catalog.loc[neutral_atm, 'Ah'] = np.minimum(
            39 * (np.power(catalog['Fb'], 3. / 5.) / catalog['wSpeed_top']),
            1.2 * np.power(catalog['Fb'] / (catalog['wSpeed_top'] * np.square(catalog['friction_v'])), 3. / 5.) *
            np.power(catalog['Height'] + (1.3 * (catalog['Fb'] / (catalog['wSpeed_top'] * np.square(
                catalog['friction_v'])))), 2. / 5.))
        # catalog.loc[unstable_atm, 'Ah'] = 30. * np.power(catalog['Fb'] / catalog['wSpeed_top'], 3./5.)
        catalog.loc[unstable_atm, 'Ah'] = np.minimum(
            3. * np.power(catalog['Fb'] / catalog['wSpeed_top'], 3. / 5.) * np.power(
                -2.5 * np.power(catalog['friction_v'], 3.) / catalog['obukhov_len'], -2. / 5.),
            30. * np.power(catalog['Fb'] / catalog['wSpeed_top'], 3. / 5.))
        catalog.loc[stable_atm, 'Ah'] = 2.6 * np.power(catalog['Fb'] / (catalog['wSpeed_top'] * catalog['S']), 1. / 3.)

        # Step 4: Plume rise
        catalog['h_top'] = (1.5 * catalog['Ah']) + catalog['Height']
        catalog['h_bot'] = (0.5 * catalog['Ah']) + catalog['Height']

        catalog.drop(columns=['Height', 'Diameter', 'Speed', 'Temp', 'date_utc', 'temp_sfc', 'friction_v', 'pbl',
                              'obukhov_len', 'temp_top', 'wSpeed_top', 'Fb', 'S', 'Ah'], inplace=True)
        self.logger.write_time_log('PointSourceSector', 'get_plume_rise_top_bot', timeit.default_timer() - spent_time)
        return catalog

    def set_layer(self, catalog):
        spent_time = timeit.default_timer()

        # catalog.set_index(['Code', 'tstep'], inplace=True)
        catalog['percent'] = 1.
        catalog_by_layer = []
        last_layer = 0
        for layer, v_lev in enumerate(self.vertical_levels):
            # filtering catalog
            aux_catalog = catalog.loc[(catalog['percent'] > 0) & (catalog['h_bot'] < v_lev), :].copy()

            aux_catalog['aux_percent'] = (((v_lev - aux_catalog['h_bot']) * aux_catalog['percent']) /
                                          (aux_catalog['h_top'] - aux_catalog['h_bot']))
            # inf are the ones that h_top == h_bot
            aux_catalog['aux_percent'].replace(np.inf, 1., inplace=True)
            # percentages higher than 'percent' are due to the ones that are the last layer
            aux_catalog.loc[aux_catalog['aux_percent'] > aux_catalog['percent'], 'aux_percent'] = \
                aux_catalog['percent']

            aux_catalog[self.source_pollutants] = aux_catalog[self.source_pollutants].multiply(
                aux_catalog['aux_percent'], axis=0)
            aux_catalog['layer'] = layer

            catalog.loc[aux_catalog.index, 'percent'] = aux_catalog['percent'] - aux_catalog['aux_percent']

            catalog.loc[aux_catalog.index, 'h_bot'] = v_lev

            aux_catalog.drop(columns=['h_top', 'h_bot', 'percent', 'aux_percent'], inplace=True)
            catalog_by_layer.append(aux_catalog)
            last_layer = layer

        # catalog_by_layer = pd.concat(catalog_by_layer)

        unused = catalog.loc[catalog['percent'] > 0, :]

        # catalog_by_layer.set_index(['Code', 'tstep', 'layer'], inplace=True)
        if len(unused) > 0:
            warn('WARNING: Some point sources have to allocate the emissions higher than the last vertical level:\n' +
                 '{0}'.format(unused.loc[:, ['Code', 'tstep', 'h_top']]))
            unused['layer'] = last_layer
            # unused.set_index(['Code', 'tstep', 'layer'], inplace=True)
            unused[self.source_pollutants] = unused[self.source_pollutants].multiply(unused['percent'], axis=0)
            unused.drop(columns=['h_top', 'h_bot', 'percent'], inplace=True)
            catalog_by_layer.append(unused)

        catalog_by_layer = pd.concat(catalog_by_layer)
        catalog_by_layer.set_index(['Code', 'tstep', 'layer'], inplace=True)

        new_catalog = catalog_by_layer[~catalog_by_layer.index.duplicated(keep='first')]
        new_catalog[self.source_pollutants] = catalog_by_layer.groupby(['Code', 'tstep', 'layer'])[
            self.source_pollutants].sum()
        self.logger.write_time_log('PointSourceSector', 'set_layer', timeit.default_timer() - spent_time)
        return new_catalog

    def calculate_vertical_distribution(self, catalog):
        """
        Add the layer column to indicate at what layer the emission have to go.

        :param catalog: Catalog to calculate.
        :type catalog: DataFrame

        :return: Catalog with the level.
        :rtype: DataFrame
        """
        spent_time = timeit.default_timer()

        if self.plume_rise:

            catalog = self.get_plume_rise_top_bot(catalog)
            catalog = self.set_layer(catalog)

        else:
            catalog['Height'] = catalog['Height'] * 1.2

            catalog['layer'] = np.searchsorted(self.vertical_levels, catalog['Height'], side='left')

            catalog.drop('Height', axis=1, inplace=True)
            catalog.reset_index(inplace=True)
            catalog.set_index(['Code', 'tstep', 'layer'], inplace=True)

        self.logger.write_time_log('PointSourceSector', 'calculate_vertical_distribution',
                                   timeit.default_timer() - spent_time)
        return catalog

    def add_measured_emissions(self, catalog):
        spent_time = timeit.default_timer()

        def func(x, pollutant):
            measured_emissions = self.measured_path.replace('<Code>', x.name)
            measured_emissions = pd.read_csv(measured_emissions, sep=';')
            measured_emissions = measured_emissions.loc[measured_emissions['Code'] == x.name, :]

            measured_emissions['date'] = pd.to_datetime(measured_emissions['date']) + pd.to_timedelta(
                measured_emissions['local_to_UTC'], unit='h')

            measured_emissions.drop('local_to_UTC', axis=1, inplace=True)

            code = x.name
            x = pd.DataFrame(x)
            x.rename(columns={code: 'date'}, inplace=True)

            test = pd.merge(left=x, right=measured_emissions.loc[:, ['date', pollutant]], on='date', how='inner')

            try:
                test.set_index(x.index, inplace=True)
            except ValueError:
                error_exit('No measured emissions for the selected dates: {0}'.format(x.values))

            return test[pollutant]

        for pollutant in self.source_pollutants:
            catalog[pollutant] = catalog.groupby('Code')['date'].apply(lambda x: func(x, pollutant))

        self.logger.write_time_log('PointSourceSector', 'add_measured_emissions', timeit.default_timer() - spent_time)
        return catalog

    def calculate_measured_emissions(self, catalog):
        spent_time = timeit.default_timer()

        if len(catalog) == 0:
            catalog = None
        else:
            catalog = self.to_geodataframe(catalog)
            catalog = self.add_dates(catalog, drop_utc=False)
            catalog = self.add_measured_emissions(catalog)

            catalog.set_index(['Code', 'tstep'], inplace=True)
        self.logger.write_time_log('PointSourceSector', 'calculate_measured_emissions',
                                   timeit.default_timer() - spent_time)
        return catalog

    def merge_catalogs(self, catalog_list):
        spent_time = timeit.default_timer()

        catalog = pd.concat(catalog_list).reset_index()
        catalog.set_index(['Code', 'tstep'], inplace=True)
        self.logger.write_time_log('PointSourceSector', 'merge_catalogs', timeit.default_timer() - spent_time)
        return catalog

    def speciate(self, dataframe, code='default'):
        spent_time = timeit.default_timer()
        self.logger.write_log('\t\tSpeciating {0} emissions'.format(code), message_level=2)

        new_dataframe = gpd.GeoDataFrame(index=dataframe.index, data=None, crs=dataframe.crs,
                                         geometry=dataframe.geometry)
        for out_pollutant in self.output_pollutants:
            input_pollutant = self.speciation_map[out_pollutant]
            if input_pollutant == 'nmvoc' and input_pollutant in dataframe.columns.values:
                self.logger.write_log("\t\t\t{0} = {4}*({1}/{2})*{3}".format(
                    out_pollutant, input_pollutant, self.molecular_weights[input_pollutant],
                    self.speciation_profile.loc[code, out_pollutant],
                    self.speciation_profile.loc[code, 'VOCtoTOG']), message_level=3)
                new_dataframe[out_pollutant] = \
                    self.speciation_profile.loc[code, 'VOCtoTOG'] * (
                            dataframe[input_pollutant] /
                            self.molecular_weights[input_pollutant]) * self.speciation_profile.loc[code, out_pollutant]
            else:
                if out_pollutant != 'PMC':
                    self.logger.write_log("\t\t\t{0} = ({1}/{2})*{3}".format(
                        out_pollutant, input_pollutant,
                        self.molecular_weights[input_pollutant],
                        self.speciation_profile.loc[code, out_pollutant]), message_level=3)
                    if input_pollutant in dataframe.columns.values:
                        new_dataframe[out_pollutant] = (dataframe[input_pollutant] /
                                                        self.molecular_weights[input_pollutant]) * \
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

    def point_source_to_fid(self, catalog):
        catalog.reset_index(inplace=True)
        catalog = catalog.to_crs(self.grid.shapefile.crs)

        catalog = gpd.sjoin(catalog, self.grid.shapefile.reset_index(), how="inner", op='intersects')

        # Drops duplicates when the point source is on the boundary of the cell
        catalog = catalog[~catalog.index.duplicated(keep='first')]

        columns_to_drop = ['Code', 'index_right', 'index']
        for del_col in columns_to_drop:
            if del_col in catalog.columns.values:
                catalog.drop(columns=[del_col], inplace=True)

        catalog = catalog.groupby(['FID', 'layer', 'tstep']).sum()

        return catalog

    def calculate_emissions(self):
        spent_time = timeit.default_timer()
        self.logger.write_log('\tCalculating emissions')

        emissions = self.add_dates(self.catalog, drop_utc=False)
        emissions = self.calculate_hourly_emissions(emissions)

        if self.comm.Get_rank() == 0:
            emissions_measured = self.calculate_measured_emissions(self.catalog_measured)
        else:
            emissions_measured = None

        if emissions_measured is not None:
            emissions = self.merge_catalogs([emissions, emissions_measured])
        emissions = self.calculate_vertical_distribution(emissions)

        emis_list = []
        for spec, spec_df in emissions.groupby('P_spec'):
            emis_list.append(self.speciate(spec_df, spec))
        emissions = pd.concat(emis_list)

        emissions = self.point_source_to_fid(emissions)
        # From kmol/h or kg/h to mol/h or g/h
        emissions = emissions.mul(1000.0)

        self.logger.write_log('\t\tPoint sources emissions calculated', message_level=2)
        self.logger.write_time_log('PointSourceSector', 'calculate_emissions', timeit.default_timer() - spent_time)

        return emissions
