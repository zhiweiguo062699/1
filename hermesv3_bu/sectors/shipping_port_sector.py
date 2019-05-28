#!/usr/bin/env python

from hermesv3_bu.sectors.sector import Sector
import pandas as pd
import geopandas as gpd
import numpy as np


class ShippingPortSector(Sector):
    def __init__(self, comm, auxiliary_dir, grid_shp, clip, date_array, source_pollutants, vertical_levels, vessel_list,
                 port_list, hoteling_shapefile_path, maneuvering_shapefile_path, ef_dir, engine_percent_path,
                 tonnage_path, load_factor_path, power_path, monthly_profiles_path, weekly_profiles_path,
                 hourly_profiles_path, speciation_map_path, speciation_profiles_path, molecular_weights_path):
        """
        :param comm: Communicator for the sector calculation.
        :type comm: MPI.COMM

        :param auxiliary_dir: Path to the directory where the necessary auxiliary files will be created if them are not
            created yet.
        :type auxiliary_dir: str

        :param grid_shp: Shapefile with the grid horizontal distribution.
        :type grid_shp: geopandas.GeoDataFrame

        :param date_array: List of datetimes.
        :type date_array: list(datetime.datetime, ...)

        :param source_pollutants: List of input pollutants to take into account.
        :type source_pollutants: list

        :param vertical_levels: List of top level of each vertical layer.
        :type vertical_levels: list

        :param vessel_list: List of vessels to take into account.
        :type vessel_list: list

        :param port_list: List of ports to take into account.
        :type port_list: list

        :param ef_dir: Path to the CSV that contains all the emission factors.
            Units: g/kWh
        :type ef_dir: str

        :param monthly_profiles_path: Path to the CSV file that contains all the monthly profiles. The CSV file must
            contain the following columns [P_month, January, February, March, April, May, June, July, August, September,
            October, November, December]
        :type monthly_profiles_path: str

        :param weekly_profiles_path: Path to the CSV file that contains all the weekly profiles. The CSV file must
            contain the following columns [P_week, Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday]
            The P_week code have to be the input pollutant.
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
        super(ShippingPortSector, self).__init__(
            comm, auxiliary_dir, grid_shp, clip, date_array, source_pollutants, vertical_levels, monthly_profiles_path,
            weekly_profiles_path, hourly_profiles_path, speciation_map_path, speciation_profiles_path,
            molecular_weights_path)

        self.ef_engine = self.read_profiles(ef_dir)

        self.vessel_list = vessel_list
        if port_list is not None:
            unknown_ports = [x for x in port_list if x not in list(pd.read_csv(tonnage_path).code.values)]
            if len(unknown_ports) > 0:
                raise ValueError("The port(s) {0} are not listed on the {1} file.".format(unknown_ports, tonnage_path))
            self.port_list = port_list
        else:
            self.port_list = list(self.read_profiles(tonnage_path).code.values)

        self.hoteling_shapefile_path = hoteling_shapefile_path
        self.maneuvering_shapefile_path = maneuvering_shapefile_path

        self.engine_percent = self.read_profiles(engine_percent_path)
        self.tonnage = self.read_profiles(tonnage_path)
        self.tonnage.set_index('code', inplace=True)
        self.load_factor = self.read_profiles(load_factor_path)
        self.power_values = self.read_profiles(power_path)

    @staticmethod
    def read_monthly_profiles(path):
        """
        Read the DataFrame of the monthly profiles with the month number as columns.

        :param path: Path to the file that contains the monthly profiles.
        :type path: str

        :return: DataFrame of the monthly profiles.
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

    def add_timezone(self, dataframe, shapefile_path):
        from timezonefinder import TimezoneFinder

        shapefile = gpd.read_file(shapefile_path)
        shapefile = shapefile.loc[:, ['code', 'geometry']]
        shapefile.drop_duplicates('code', keep='first', inplace=True)

        shapefile = shapefile.to_crs({'init': 'epsg:4326'})
        tzfinder = TimezoneFinder()
        shapefile['timezone'] = shapefile.centroid.apply(lambda x: tzfinder.timezone_at(lng=x.x, lat=x.y))
        dataframe.reset_index(inplace=True)
        dataframe = pd.merge(dataframe, shapefile.loc[:, ['code', 'timezone']], on='code')
        dataframe.set_index(['code', 'vessel'], inplace=True)

        return dataframe

    def add_dates(self, dataframe):
        dataframe.reset_index(inplace=True)
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
        dataframe.set_index(['code', 'vessel', 'tstep'], inplace=True)
        # del dataframe['date_utc']

        return dataframe

    def get_constants(self):
        def get_n(df):
            aux = self.tonnage.loc[:, ['N_{0}'.format(df.name)]].reset_index()
            aux['vessel'] = df.name
            aux.set_index(['code', 'vessel'], inplace=True)
            df['N'] = aux['N_{0}'.format(df.name)]
            return df.loc[:, ['N']]

        def get_p(df):
            aux = self.tonnage.loc[:, ['GT_{0}'.format(df.name)]].reset_index()
            aux.rename(columns={'GT_{0}'.format(df.name): 'GT'}, inplace=True)
            aux['vessel'] = df.name
            aux.set_index(['code', 'vessel'], inplace=True)
            aux['P'] = np.power(aux['GT'], self.power_values.loc[self.power_values['Type_vessel'] == df.name,
                                                                 'GT_exp'].values[0])
            df['P'] = aux['P'].multiply(self.power_values.loc[self.power_values['Type_vessel'] == df.name,
                                                              'Value'].values[0])
            return df.loc[:, ['P']]

        def get_rae(df):
            df['Rae'] = self.power_values.loc[self.power_values['Type_vessel'] == df.name, 'Ratio_AE'].values[0]
            return df.loc[:, ['Rae']]

        def get_t(df, phase):
            df['T'] = self.load_factor.loc[(self.load_factor['Type_vessel'] == df.name) &
                                           (self.load_factor['Phase'] == phase), 'time'].values[0]
            return df.loc[:, ['T']]

        def get_lf(df, phase, engine):
            if engine == 'main':
                col_name = 'LF_ME'
            else:
                col_name = 'LF_AE'
            df['LF'] = self.load_factor.loc[(self.load_factor['Type_vessel'] == df.name) &
                                            (self.load_factor['Phase'] == phase), col_name].values[0]
            return df.loc[:, ['LF']]

        def get_ef(df, engine, pollutant):
            if engine == 'main':
                engine = 'ME'
            else:
                engine = 'AE'
            aux1 = self.engine_percent.loc[(self.engine_percent['Type_vessel'] == df.name) &
                                           (self.engine_percent['Engine'] == engine), ['Engine_fuel', 'Factor']]
            aux2 = self.ef_engine.loc[(self.ef_engine['Engine'] == engine) &
                                      (self.ef_engine['Engine_fuel'].isin(aux1['Engine_fuel'].values)),
                                      ['Engine_fuel', 'EF_{0}'.format(pollutant)]]

            aux = pd.merge(aux1, aux2, on='Engine_fuel')
            aux['value'] = aux['Factor'] * aux['EF_{0}'.format(pollutant)]
            df['EF'] = aux['value'].sum()
            return df.loc[:, ['EF']]

        dataframe = pd.DataFrame(index=pd.MultiIndex.from_product([self.port_list, self.vessel_list],
                                                                  names=['code', 'vessel']))
        dataframe['N'] = dataframe.groupby('vessel').apply(get_n)
        dataframe['P'] = dataframe.groupby('vessel').apply(get_p)
        dataframe['Rae'] = dataframe.groupby('vessel').apply(get_rae)
        dataframe['LF_mm'] = dataframe.groupby('vessel').apply(lambda x: get_lf(x, 'manoeuvring', 'main'))
        dataframe['LF_hm'] = dataframe.groupby('vessel').apply(lambda x: get_lf(x, 'hoteling', 'main'))
        dataframe['LF_ma'] = dataframe.groupby('vessel').apply(lambda x: get_lf(x, 'manoeuvring', 'aux'))
        dataframe['LF_ha'] = dataframe.groupby('vessel').apply(lambda x: get_lf(x, 'hoteling', 'aux'))
        dataframe['T_m'] = dataframe.groupby('vessel').apply(lambda x: get_t(x, 'manoeuvring'))
        dataframe['T_h'] = dataframe.groupby('vessel').apply(lambda x: get_t(x, 'hoteling'))
        for pollutant in self.source_pollutants:
            dataframe['EF_m_{0}'.format(pollutant)] = dataframe.groupby('vessel').apply(
                lambda x: get_ef(x, 'main', pollutant))
            dataframe['EF_a_{0}'.format(pollutant)] = dataframe.groupby('vessel').apply(
                lambda x: get_ef(x, 'aux', pollutant))

        return dataframe

    def calculate_yearly_emissions_by_port_vessel(self):
        constants = self.get_constants()
        manoeuvring = pd.DataFrame(index=constants.index)
        hoteling = pd.DataFrame(index=constants.index)
        for pollutant in self.source_pollutants:
            manoeuvring['{0}'.format(pollutant)] = \
                constants['P'] * constants['N'] * constants['LF_mm'] * constants['T_m'] * \
                constants['EF_m_{0}'.format(pollutant)]
            hoteling['{0}'.format(pollutant)] = \
                constants['P'] * constants['N'] * constants['LF_hm'] * constants['T_h'] * \
                constants['EF_m_{0}'.format(pollutant)]
            manoeuvring['{0}'.format(pollutant)] += \
                constants['P'] * constants['Rae'] * constants['N'] * constants['LF_ma'] * constants['T_m'] * \
                constants['EF_a_{0}'.format(pollutant)]
            hoteling['{0}'.format(pollutant)] += \
                constants['P'] * constants['Rae'] * constants['N'] * constants['LF_ha'] * constants['T_h'] * \
                constants['EF_a_{0}'.format(pollutant)]

        return [manoeuvring, hoteling]

    def dates_to_month_weekday_hour(self, dataframe):
        dataframe['month'] = dataframe['date'].dt.month
        dataframe['weekday'] = dataframe['date'].dt.weekday
        dataframe['hour'] = dataframe['date'].dt.hour

        # dataframe.drop('date', axis=1, inplace=True)
        return dataframe

    def calculate_monthly_emissions_by_port(self, dataframe):
        def get_mf(df):
            vessel = df.name[0]
            month = df.name[1]

            if vessel not in list(np.unique(self.monthly_profiles['type'].values)):
                vessel = 'default'
            mf_df = self.monthly_profiles.loc[self.monthly_profiles['type'] == vessel, ['code', month]]
            mf_df.rename(columns={month: 'MF'}, inplace=True)
            mf_df.set_index('code', inplace=True)
            df = df.join(mf_df, how='inner')

            return df.loc[:, ['MF']]
        dataframe['MF'] = dataframe.groupby(['vessel', 'month']).apply(get_mf)
        dataframe[self.source_pollutants] = dataframe[self.source_pollutants].multiply(dataframe['MF'], axis=0)
        dataframe.drop(['MF', 'month'], axis=1, inplace=True)

        operations = {x: 'sum' for x in self.source_pollutants}
        operations['weekday'] = 'max'
        operations['hour'] = 'max'
        operations['date'] = 'max'
        dataframe = dataframe.groupby(level=['code', 'tstep']).agg(operations)

        return dataframe

    def calculate_hourly_emissions_by_port(self, dataframe):
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

        dataframe['date_as_date'] = dataframe['date'].dt.date

        dataframe['WF'] = dataframe.groupby('date_as_date').apply(get_wf)
        dataframe[self.source_pollutants] = dataframe[self.source_pollutants].multiply(dataframe['WF'], axis=0)
        dataframe.drop(columns=['weekday', 'date', 'date_as_date', 'WF'], inplace=True)

        dataframe['HF'] = dataframe.groupby('hour').apply(get_hf)
        dataframe[self.source_pollutants] = dataframe[self.source_pollutants].multiply(dataframe['HF'], axis=0)
        dataframe.drop(columns=['hour', 'HF'], inplace=True)

        return dataframe

    def to_port_geometry(self, dataframe, shapefile_path):
        def normalize_weight(df):
            df['Weight'] = df['Weight'] / df['Weight'].sum()
            return df.loc[:, ['Weight']]

        shapefile = gpd.read_file(shapefile_path)
        shapefile = shapefile.loc[shapefile['Weight'] > 0, ['code', 'Weight', 'geometry']]

        shapefile['Weight'] = shapefile.groupby('code').apply(normalize_weight)

        shapefile.set_index('code', inplace=True)
        dataframe.reset_index(inplace=True)
        dataframe.set_index('code', inplace=True)

        dataframe = shapefile.join(dataframe, how='outer')

        dataframe[self.source_pollutants] = dataframe[self.source_pollutants].multiply(dataframe['Weight'], axis=0)
        dataframe.drop(columns=['Weight'], inplace=True)

        return dataframe

    def to_grid_geometry(self, dataframe):
        dataframe.reset_index(inplace=True)
        dataframe.drop(columns=['code'], inplace=True)

        dataframe.to_crs(self.grid_shp.crs, inplace=True)
        dataframe['src_inter_fraction'] = dataframe.geometry.area
        dataframe = self.spatial_overlays(dataframe, self.grid_shp, how='intersection')
        dataframe['src_inter_fraction'] = dataframe.geometry.area / dataframe['src_inter_fraction']

        dataframe[self.source_pollutants] = dataframe[self.source_pollutants].multiply(dataframe["src_inter_fraction"],
                                                                                       axis=0)
        dataframe.rename(columns={'idx2': 'FID'}, inplace=True)

        dataframe.drop(columns=['src_inter_fraction', 'idx1', 'geometry'], inplace=True)
        dataframe['layer'] = 0

        dataframe = dataframe.groupby(['FID', 'layer', 'tstep']).sum()

        return dataframe

    def calculate_emissions(self):
        [manoeuvring, hoteling] = self.calculate_yearly_emissions_by_port_vessel()
        # print manoeuvring.reset_index().groupby('code').sum()
        # print hoteling.reset_index().groupby('code').sum()
        manoeuvring = self.add_timezone(manoeuvring, self.maneuvering_shapefile_path)
        hoteling = self.add_timezone(hoteling, self.hoteling_shapefile_path)

        manoeuvring = self.add_dates(manoeuvring)
        hoteling = self.add_dates(hoteling)

        manoeuvring = self.dates_to_month_weekday_hour(manoeuvring)
        hoteling = self.dates_to_month_weekday_hour(hoteling)

        manoeuvring = self.calculate_monthly_emissions_by_port(manoeuvring)
        hoteling = self.calculate_monthly_emissions_by_port(hoteling)

        manoeuvring = self.calculate_hourly_emissions_by_port(manoeuvring)
        hoteling = self.calculate_hourly_emissions_by_port(hoteling)

        manoeuvring = self.to_port_geometry(manoeuvring, self.maneuvering_shapefile_path)
        hoteling = self.to_port_geometry(hoteling, self.hoteling_shapefile_path)

        manoeuvring = self.to_grid_geometry(manoeuvring)
        hoteling = self.to_grid_geometry(hoteling)

        dataframe = pd.concat([manoeuvring, hoteling])
        dataframe = dataframe.groupby(['FID', 'layer', 'tstep']).sum()

        dataframe = self.speciate(dataframe, 'default')

        return dataframe
