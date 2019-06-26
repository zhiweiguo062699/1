#!/usr/bin/env python
import sys
import os
import timeit

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.ops import nearest_points
import warnings
from hermesv3_bu.logger.log import Log
from hermesv3_bu.sectors.sector import Sector

MIN_RAIN = 0.254  # After USEPA (2011)
RECOVERY_RATIO = 0.0872  # After Amato et al. (2012)


aerosols = ['oc', 'ec', 'pno3', 'pso4', 'pmfine', 'pmc', 'poa', 'poc', 'pec', 'pcl', 'pnh4', 'pna', 'pmg', 'pk', 'pca',
            'pncom', 'pfe', 'pal', 'psi', 'pti', 'pmn', 'ph2o', 'pmothr']
pmc_list = ['pmc', 'PMC']
rline_shp = False


class TrafficSector(Sector):
    # TODO MARC -> to revise these descriptions
    """
    The traffic class does have all the necessary functions to calculate the traffic emission in bottom-up mode.

    Part of the traffic emissions are calculated by roadlink (hot, cold, road wear, tyre wear, brake wear and
        resuspension) differentiating by vehicle type.
        The other emissions (other cities and evaporative) are calculated by cell instead of by road link.

    To calculate the traffic emissions some input files are needed as the shapefile that contains the information and
        geolocalization of each road link, the temporal proxies, the emission factors files and also the information
        relative to the timesteps.
    """

    def __init__(self, comm, logger, auxiliary_dir, grid_shp, clip, date_array, source_pollutants, vertical_levels,
                 road_link_path, fleet_compo_path, speed_hourly_path, monthly_profiles_path, weekly_profiles_path,
                 hourly_mean_profiles_path, hourly_weekday_profiles_path, hourly_saturday_profiles_path,
                 hourly_sunday_profiles_path, ef_common_path, vehicle_list=None, load=0.5, speciation_map_path=None,
                 hot_cold_speciation=None, tyre_speciation=None, road_speciation=None, brake_speciation=None,
                 resuspension_speciation=None, temp_common_path=None, output_dir=None, molecular_weights_path=None,
                 resuspension_correction=True, precipitation_path=None, do_hot=True, do_cold=True, do_tyre_wear=True,
                 do_brake_wear=True, do_road_wear=True, do_resuspension=True, write_rline=False):

        spent_time = timeit.default_timer()
        logger.write_log('===== TRAFFIC SECTOR =====')
        super(TrafficSector, self).__init__(
            comm, logger, auxiliary_dir, grid_shp, clip, date_array, source_pollutants, vertical_levels,
            monthly_profiles_path, weekly_profiles_path, None, speciation_map_path, None, molecular_weights_path)

        self.resuspension_correction = resuspension_correction
        self.precipitation_path = precipitation_path

        self.output_dir = output_dir

        self.link_to_grid_csv = os.path.join(auxiliary_dir, 'traffic', 'link_grid.csv')
        self.crs = None   # crs is the projection of the road links and it is set on the read_road_links function.
        self.write_rline = write_rline
        self.road_links = self.read_road_links(road_link_path)
        self.load = load
        self.ef_common_path = ef_common_path
        self.temp_common_path = temp_common_path
        # TODO use only date_array
        self.timestep_num = len(self.date_array)
        self.timestep_freq = 1
        self.starting_date = self.date_array[0]
        self.add_local_date(self.date_array[0])

        self.hot_cold_speciation = hot_cold_speciation
        self.tyre_speciation = tyre_speciation
        self.road_speciation = road_speciation
        self.brake_speciation = brake_speciation
        self.resuspension_speciation = resuspension_speciation

        self.fleet_compo = self.read_fleet_compo(fleet_compo_path, vehicle_list)
        self.speed_hourly = self.read_speed_hourly(speed_hourly_path)

        self.hourly_profiles = pd.concat([
            pd.read_csv(hourly_mean_profiles_path),
            pd.read_csv(hourly_weekday_profiles_path),
            pd.read_csv(hourly_saturday_profiles_path),
            pd.read_csv(hourly_sunday_profiles_path)
        ]).reset_index()

        self.expanded = self.expand_road_links('hourly', len(self.date_array), 1)

        del self.fleet_compo, self.speed_hourly, self.monthly_profiles, self.weekly_profiles, self.hourly_profiles

        self.do_hot = do_hot
        self.do_cold = do_cold
        self.do_tyre_wear = do_tyre_wear
        self.do_brake_wear = do_brake_wear
        self.do_road_wear = do_road_wear
        self.do_resuspension = do_resuspension

        self.logger.write_time_log('TrafficSector', '__init__', timeit.default_timer() - spent_time)

    def read_speciation_map(self, path):
        """
        Read the speciation map.

        The speciation map is the CSV file that contains the relation from the output pollutant and the correspondent
        input pollutant associated. That file also contains a short description of the output pollutant and the units to
        be stored.

        e.g.:
        dst,src,description,units
        NOx,nox_no2,desc_no,mol.s-1
        SOx,so2,desc_so2,mol.s-1
        CO,co,desc_co,mol.s-1
        CO2,co2,desc_co2,mol.s-1
        NMVOC,nmvoc,desc_nmvoc,g.s-1
        PM10,pm10,desc_pm10,g.s-1
        PM25,pm25,desc_pm25,g.s-1
        PMC,,desc_pmc,g.s-1

        :param path: Path to the speciation map file.
        :type path: str

        :return: Dictionary with the output pollutant as key and the input pollutant as value.
        :rtype: dict
        """
        spent_time = timeit.default_timer()
        dataframe = pd.read_csv(path)
        # input_pollutants = list(self.source_pollutants)
        input_pollutants = ['nmvoc' if x == 'voc' else x for x in list(self.source_pollutants)]
        if 'PMC' in dataframe['dst'].values and all(element in input_pollutants for element in ['pm']):
            dataframe_aux = dataframe.loc[dataframe['src'].isin(input_pollutants), :]
            dataframe = pd.concat([dataframe_aux, dataframe.loc[dataframe['dst'] == 'PMC', :]])
        else:
            dataframe = dataframe.loc[dataframe['src'].isin(input_pollutants), :]

        dataframe = dict(zip(dataframe['dst'], dataframe['src']))

        if 'pm' in self.source_pollutants:
            dataframe['PM10'] = 'pm10'
            dataframe['PM25'] = 'pm25'
        self.logger.write_time_log('TrafficSector', 'read_speciation_map', timeit.default_timer() - spent_time)

        return dataframe

    def add_local_date(self, utc_date):
        """
        Adds to the road links the starting date in local time.
        This new column is called 'start_date'.

        :param utc_date: Starting date in UTC.
        """
        import pytz
        spent_time = timeit.default_timer()

        self.add_timezones()
        self.road_links.loc[:, 'utc'] = utc_date
        self.road_links['start_date'] = self.road_links.groupby('timezone')['utc'].apply(
            lambda x: pd.to_datetime(x).dt.tz_localize(pytz.utc).dt.tz_convert(x.name).dt.tz_localize(None))

        del self.road_links['utc'], self.road_links['timezone']

        self.logger.write_time_log('TrafficSector', 'add_local_date', timeit.default_timer() - spent_time)
        return True

    def add_timezones(self):
        """
        Finds and sets the timezone for each road link.
        """
        spent_time = timeit.default_timer()
        # TODO calculate timezone from the centroid of each roadlink.

        self.road_links['timezone'] = 'Europe/Madrid'

        self.logger.write_time_log('TrafficSector', 'add_timezones', timeit.default_timer() - spent_time)
        return True

    def read_speed_hourly(self, path):
        # TODO complete description
        """
        Reads the speed hourly file.

        :param path: Path to the speed hourly file.
        :type path: str:

        :return: ...
        :rtype: Pandas.DataFrame
        """
        spent_time = timeit.default_timer()

        df = pd.read_csv(path, sep=',', dtype=np.float32)
        df['P_speed'] = df['P_speed'].astype(int)
        # df.set_index('P_speed', inplace=True)
        self.logger.write_time_log('TrafficSector', 'read_speed_hourly', timeit.default_timer() - spent_time)
        return df

    def read_fleet_compo(self, path, vehicle_list):
        spent_time = timeit.default_timer()
        df = pd.read_csv(path, sep=',')
        if vehicle_list is not None:
            df = df.loc[df['Code'].isin(vehicle_list), :]
        self.logger.write_time_log('TrafficSector', 'read_fleet_compo', timeit.default_timer() - spent_time)
        return df

    def read_road_links(self, path):
        def chunk_road_links(df, nprocs):
            def index_marks(nrows, nprocs):
                max_len = int(nrows // nprocs) + 1
                min_len = max_len - 1
                max_num = nrows % nprocs
                min_num = nprocs - max_num
                index_list = []
                prev = 0
                for i in xrange(max_num):
                    prev += max_len
                    index_list.append(prev)
                if min_num > 0:
                    for i in xrange(min_num - 1):
                        prev += min_len
                        index_list.append(prev)

                return index_list

            def split(dfm, nprocs):
                indices = index_marks(dfm.shape[0], nprocs)
                return np.split(dfm, indices)

            chunks_aux = split(df, nprocs)
            return chunks_aux

        spent_time = timeit.default_timer()

        if self.comm.Get_rank() == 0:
            df = gpd.read_file(path)

            df = gpd.sjoin(df, self.clip.shapefile.to_crs(df.crs), how="inner", op='intersects')

            # Filtering road links to CONSiderate.
            df['CONS'] = df['CONS'].astype(np.int16)
            df = df[df['CONS'] != 0]
            df = df[df['aadt'] > 0]

            # TODO Manu update shapefile replacing NULL values on 'aadt_m-mn' column
            df = df.loc[df['aadt_m_mn'] != 'NULL', :]

            # Adding identificator of road link
            df['Link_ID'] = xrange(len(df))

            del df['Adminis'], df['CCAA'], df['CONS'], df['NETWORK_ID']
            del df['Province'], df['Road_name']

            # Deleting unused columns
            del df['aadt_m_sat'], df['aadt_m_sun'], df['aadt_m_wd'], df['Source']

            chunks = chunk_road_links(df, self.comm.Get_size())
        else:
            chunks = None
        self.comm.Barrier()

        df = self.comm.scatter(chunks, root=0)
        df = df.to_crs({'init': 'epsg:4326'})

        self.crs = df.crs

        # Correcting percentages
        df['PcMoto'] = df['PcMoto'] / 100
        df['PcHeavy'] = df['PcHeavy'] / 100
        df['PcMoped'] = df['PcMoped'] / 100
        df['PcLight'] = 1 - (df['PcMoto'] + df['PcHeavy'] + df['PcMoped'])

        # Road_type int to string
        df['Road_type'] = df['Road_type'].astype(str)
        df.loc[df['Road_type'] == '0', 'Road_type'] = 'Highway'
        df.loc[df['Road_type'] == '1', 'Road_type'] = 'Rural'
        df.loc[df['Road_type'] == '2', 'Road_type'] = 'Urban Off Peak'
        df.loc[df['Road_type'] == '3', 'Road_type'] = 'Urban Peak'

        # TODO Read with units types
        df['road_grad'] = df['road_grad'].astype(float)

        # Check if percents are ok
        if len(df[df['PcLight'] < 0]) is not 0:
            print 'ERROR: PcLight < 0'
            exit(1)

        if self.write_rline:
            self.write_rline_roadlinks(df)

        self.logger.write_time_log('TrafficSector', 'read_road_links', timeit.default_timer() - spent_time)

        return df

    def read_ef(self, emission_type, pollutant_name):
        """
        Reads the file that contains the necessary emission factor for the current pollutant and emission type.

        Depending on the emission tyme the file contain different columns.

        :param emission_type: Type of the emission. It can be hot, cold, tyre, road, brake or resuspension.
        :type emission_type: str

        :param pollutant_name: Name of the pollutant to read their emission factor.
        :type pollutant_name:str

        :return: Returns the readed emission factor in DataFrame mode.
        :rtype: Pandas.DataFrame
        """
        spent_time = timeit.default_timer()

        ef_path = os.path.join(self.ef_common_path, '{0}_{1}.csv'.format(emission_type, pollutant_name))
        df = self.read_profiles(ef_path)

        # Pollutants different to NH3
        if pollutant_name != 'nh3':
            del df['Copert_V_name']

            # For hot emission factors
            if emission_type == 'hot':
                df = df[(df['Load'] == self.load) | (df['Load'].isnull())]

                df.loc[df['Technology'].isnull(), 'Technology'] = ''
                df = df[df['Technology'] != 'EGR']

                del df['Technology'], df['Load']

                # Split the EF file into small DataFrames divided by column Road.Slope and Mode restrictions.
                df_code_slope_road = df[df['Road.Slope'].notnull() & df['Mode'].notnull()]
                df_code_slope = df[df['Road.Slope'].notnull() & (df['Mode'].isnull())]
                df_code_road = df[df['Road.Slope'].isnull() & (df['Mode'].notnull())]
                df_code = df[df['Road.Slope'].isnull() & (df['Mode'].isnull())]

                # Checks that the splited DataFrames contain the full DataFrame
                if (len(df_code_slope_road) + len(df_code_slope) + len(df_code_road) + len(df_code)) != len(df):
                    # TODO check that error
                    print 'ERROR in blablavbla'

                return df_code_slope_road, df_code_slope, df_code_road, df_code
            elif emission_type == 'cold' or emission_type == 'tyre' or emission_type == 'road' or \
                    emission_type == 'brake' or emission_type == 'resuspension':
                return df
        # NH3 pollutant
        else:
            del df['Copert_V_name']
            # Specific case for cold NH3 emission factors that needs the hot emission factors and the cold ones.
            if emission_type == 'cold':
                df_hot = self.read_ef('hot', pollutant_name)
                df_hot.columns = [x + '_hot' for x in df_hot.columns.values]

                df = df.merge(df_hot, left_on=['Code', 'Mode'], right_on=['Code_hot', 'Mode_hot'],
                              how='left')

                del df['Cmileage_hot'], df['Mode_hot'], df['Code_hot']

            return df

        self.logger.write_time_log('TrafficSector', 'read_ef', timeit.default_timer() - spent_time)
        return None

    def read_mcorr_file(self, pollutant_name):
        spent_time = timeit.default_timer()
        try:
            df_path = os.path.join(self.ef_common_path, 'mcorr_{0}.csv'.format(pollutant_name))

            df = pd.read_csv(df_path, sep=',')
            if 'Copert_V_name' in list(df.columns.values):
                df.drop(columns=['Copert_V_name'], inplace=True)
        except IOError:
            self.logger.write_log('WARNING! No mileage correction applied to {0}'.format(pollutant_name))
            warnings.warn('No mileage correction applied to {0}'.format(pollutant_name))
            df = None

        self.logger.write_time_log('TrafficSector', 'read_ef', timeit.default_timer() - spent_time)
        return df

    def read_temperature(self, lon_min, lon_max, lat_min, lat_max, temp_dir, date, tstep_num, tstep_freq):
        """
        Reads the temperature from the ERA5 tas value.
        It will return only the involved cells of the NetCDF in DataFrame format.

        To clip the global NetCDF to the desired region it is needed the minimum and maximum value of the latitudes and
        longitudes of the centroids of all the road links.

        :param lon_min: Minimum longitude of the centroid of the road links.
        :type lon_min: float

        :param lon_max: Maximum longitude of the centroid of the road links.
        :type lon_max: float

        :param lat_min: Minimum latitude of the centroid of the road links.
        :type lat_min: float

        :param lat_max: Maximum latitude of the centroid of the road links.
        :type lat_max: float

        :return: Temperature, centroid of the cell and cell identificator (REC).
            Each time step is each column with the name t_<timestep>.
        :rtype: GeoDataFrame
        """
        from netCDF4 import Dataset
        import cf_units
        from shapely.geometry import Point
        from datetime import timedelta
        spent_time = timeit.default_timer()

        path = os.path.join(temp_dir, 'tas_{0}{1}.nc'.format(date.year, str(date.month).zfill(2)))
        self.logger.write_log('Getting temperature from {0}'.format(path), message_level=2)

        nc = Dataset(path, mode='r')
        lat_o = nc.variables['latitude'][:]
        lon_o = nc.variables['longitude'][:]
        time = nc.variables['time']
        # From time array to list of dates.
        time_array = cf_units.num2date(time[:], time.units,  cf_units.CALENDAR_STANDARD)
        i_time = np.where(time_array == date)[0][0]

        # Correction to set the longitudes from -180 to 180 instead of from 0 to 360.
        if lon_o.max() > 180:
            lon_o[lon_o > 180] -= 360

        # Finds the array positions for the clip.
        i_min, i_max, j_min, j_max = self.find_index(lon_o, lat_o, lon_min, lon_max, lat_min, lat_max)

        # Clips the lat lons
        lon_o = lon_o[i_min:i_max]
        lat_o = lat_o[j_min:j_max]

        # From 1D to 2D
        lat = np.array([lat_o[:]] * len(lon_o[:])).T.flatten()
        lon = np.array([lon_o[:]] * len(lat_o[:])).flatten()
        del lat_o, lon_o

        # Reads the tas variable of the xone and the times needed.
        tas = nc.variables['tas'][i_time:i_time + (tstep_num*tstep_freq): tstep_freq, j_min:j_max, i_min:i_max]

        nc.close()
        # That condition is fot the cases that the needed temperature is in a different NetCDF.
        while len(tas) < tstep_num:
            aux_date = date + timedelta(hours=len(tas) + 1)
            path = os.path.join(temp_dir, 'tas_{0}{1}.nc'.format(aux_date.year, str(aux_date.month).zfill(2)))
            self.logger.write_log('Getting temperature from {0}'.format(path), message_level=2)
            nc = Dataset(path, mode='r')
            i_time = 0
            new_tas = nc.variables['tas'][i_time:i_time + ((tstep_num - len(tas))*tstep_freq): tstep_freq, j_min:j_max,
                                          i_min:i_max]

            tas = np.concatenate([tas, new_tas])

            nc.close()

        # From Kelvin to Celsius degrees
        tas = (tas - 273.15).reshape((tas.shape[0], tas.shape[1] * tas.shape[2]))
        # Creates the GeoDataFrame
        df = gpd.GeoDataFrame(tas.T, geometry=[Point(xy) for xy in zip(lon, lat)])
        df.columns = ['t_{0}'.format(x) for x in df.columns.values[:-1]] + ['geometry']
        df.loc[:, 'REC'] = df.index

        self.logger.write_time_log('TrafficSector', 'read_temperature', timeit.default_timer() - spent_time)
        return df

    def get_precipitation(self, lon_min, lon_max, lat_min, lat_max, precipitation_dir):
        from datetime import timedelta
        from netCDF4 import Dataset
        import cf_units
        from shapely.geometry import Point
        spent_time = timeit.default_timer()

        dates_to_extract = [self.date_array[0] + timedelta(hours=x - 47) for x in range(47)] + self.date_array

        path = os.path.join(precipitation_dir, 'prlr_{0}{1}.nc'.format(
            dates_to_extract[0].year, str(dates_to_extract[0].month).zfill(2)))
        self.logger.write_log('Getting precipitation from {0}'.format(path), message_level=2)

        nc = Dataset(path, mode='r')
        lat_o = nc.variables['latitude'][:]
        lon_o = nc.variables['longitude'][:]
        time = nc.variables['time']
        # From time array to list of dates.
        time_array = cf_units.num2date(time[:], time.units,  cf_units.CALENDAR_STANDARD)
        i_time = np.where(time_array == dates_to_extract[0])[0][0]

        # Correction to set the longitudes from -180 to 180 instead of from 0 to 360.
        if lon_o.max() > 180:
            lon_o[lon_o > 180] -= 360

        # Finds the array positions for the clip.
        i_min, i_max, j_min, j_max = self.find_index(lon_o, lat_o, lon_min, lon_max, lat_min, lat_max)

        # Clips the lat lons
        lon_o = lon_o[i_min:i_max]
        lat_o = lat_o[j_min:j_max]

        # From 1D to 2D
        lat = np.array([lat_o[:]] * len(lon_o[:])).T.flatten()
        lon = np.array([lon_o[:]] * len(lat_o[:])).flatten()
        del lat_o, lon_o

        # Reads the tas variable of the xone and the times needed.
        prlr = nc.variables['prlr'][i_time:i_time + len(dates_to_extract), j_min:j_max, i_min:i_max]
        nc.close()
        # That condition is fot the cases that the needed temperature is in a different NetCDF.
        while len(prlr) < len(dates_to_extract):
            aux_date = dates_to_extract[len(prlr)]
            path = os.path.join(precipitation_dir, 'prlr_{0}{1}.nc'.format(aux_date.year, str(aux_date.month).zfill(2)))
            # path = os.path.join(precipitation_dir, 'prlr_{0}{1}.nc'.format(
            #     dates_to_extract[len(prlr)].year, str(dates_to_extract[len(prlr)].month).zfill(2)))
            self.logger.write_log('Getting precipitation from {0}'.format(path), message_level=2)
            nc = Dataset(path, mode='r')
            i_time = 0
            new_prlr = nc.variables['prlr'][i_time:i_time + (len(dates_to_extract) - len(prlr)),
                                            j_min:j_max, i_min:i_max]

            prlr = np.concatenate([prlr, new_prlr])

            nc.close()

        # From m/s to mm/h
        prlr = prlr * (3600 * 1000)
        prlr = prlr <= MIN_RAIN
        dst = np.empty(prlr.shape)
        last = np.zeros((prlr.shape[-2], prlr.shape[-1]))
        for time in xrange(prlr.shape[0]):
            dst[time, :] = (last + prlr[time, :]) * prlr[time, :]
            last = dst[time, :]

        dst = dst[47:, :]
        dst = 1 - np.exp(- RECOVERY_RATIO * dst)
        # It is assumed that after 48 h without rain the potential emission is equal to one
        dst[dst >= (1 - np.exp(- RECOVERY_RATIO * 48))] = 1.
        dst = dst.reshape((dst.shape[0], dst.shape[1] * dst.shape[2]))
        # Creates the GeoDataFrame
        df = gpd.GeoDataFrame(dst.T, geometry=[Point(xy) for xy in zip(lon, lat)])
        df.columns = ['PR_{0}'.format(x) for x in df.columns.values[:-1]] + ['geometry']

        df.loc[:, 'REC'] = df.index

        self.logger.write_time_log('TrafficSector', 'get_precipitation', timeit.default_timer() - spent_time)
        return df

    def find_index(self, lon, lat, lon_min, lon_max, lat_min, lat_max):
        spent_time = timeit.default_timer()

        aux = lon - lon_min
        aux[aux > 0] = np.nan
        i_min = np.where(aux == np.nanmax(aux))[0][0]

        aux = lon - lon_max

        aux[aux < 0] = np.nan

        i_max = np.where(aux == np.nanmin(aux))[0][0]

        aux = lat - lat_min
        aux[aux > 0] = np.nan
        j_max = np.where(aux == np.nanmax(aux))[0][0]

        aux = lat - lat_max
        aux[aux < 0] = np.nan
        j_min = np.where(aux == np.nanmin(aux))[0][0]

        self.logger.write_time_log('TrafficSector', 'find_index', timeit.default_timer() - spent_time)
        return i_min, i_max+1, j_min, j_max+1

    def update_fleet_value(self, df):
        spent_time = timeit.default_timer()

        # Calculating fleet value by fleet class
        df.loc[:, 'Fleet_value'] = df['Fleet_value'] * df['aadt']

        df.loc[df['Fleet_Class'] == 'light_veh', 'Fleet_value'] = df['PcLight'] * df['Fleet_value']
        df.loc[df['Fleet_Class'] == 'heavy_veh', 'Fleet_value'] = df['PcHeavy'] * df['Fleet_value']
        df.loc[df['Fleet_Class'] == 'motos', 'Fleet_value'] = df['PcMoto'] * df['Fleet_value']
        df.loc[df['Fleet_Class'] == 'mopeds', 'Fleet_value'] = df['PcMoped'] * df['Fleet_value']

        for link_id, aux_df in df.groupby('Link_ID'):
            aadt = round(aux_df['aadt'].min(), 1)
            fleet_value = round(aux_df['Fleet_value'].sum(), 1)
            if aadt != fleet_value:
                self.logger.write_log('link_ID: {0} aadt: {1} sum_fleet: {2}'.format(link_id, aadt, fleet_value),
                                      message_level=2)

        # Drop 0 values
        df = df[df['Fleet_value'] > 0]

        # Deleting unused columns
        del df['aadt'], df['PcLight'], df['PcHeavy'], df['PcMoto'], df['PcMoped'], df['Fleet_Class']
        self.logger.write_time_log('TrafficSector', 'update_fleet_value', timeit.default_timer() - spent_time)
        return df

    def calculate_timedelta(self, timestep_type, num_tstep, timestep_freq):
        from datetime import timedelta
        spent_time = timeit.default_timer()

        delta = timedelta(hours=timestep_freq * num_tstep)

        self.logger.write_time_log('TrafficSector', 'calculate_timedelta', timeit.default_timer() - spent_time)
        return pd.Timedelta(delta)

    def calculate_hourly_speed(self, df):
        spent_time = timeit.default_timer()

        df = df.merge(self.speed_hourly, left_on='profile_id', right_on='P_speed', how='left')
        df['speed'] = df.groupby('hour').apply(lambda x: x[[str(x.name)]])

        self.logger.write_time_log('TrafficSector', 'calculate_hourly_speed', timeit.default_timer() - spent_time)
        return df['speed'] * df['speed_mean']

    def calculate_temporal_factor(self, df):
        spent_time = timeit.default_timer()

        def get_hourly_id_from_weekday(weekday):
            if weekday <= 4:
                return 'aadt_h_wd'
            elif weekday == 5:
                return 'aadt_h_sat'
            elif weekday == 6:
                return 'aadt_h_sun'
            else:
                print 'ERROR: Weekday not found'
                exit()

        # Monthly factor
        df = df.merge(self.monthly_profiles.reset_index(), left_on='aadt_m_mn', right_on='P_month', how='left')
        df['MF'] = df.groupby('month').apply(lambda x: x[[x.name]])
        df.drop(columns=range(1, 12 + 1), inplace=True)

        # Daily factor
        df = df.merge(self.weekly_profiles.reset_index(), left_on='aadt_week', right_on='P_week', how='left')

        df['WF'] = df.groupby('week_day').apply(lambda x: x[[x.name]])
        df.drop(columns=range(0, 7), inplace=True)

        # Hourly factor
        df['hourly_profile'] = df.groupby('week_day').apply(lambda x: x[[get_hourly_id_from_weekday(x.name)]])
        df.loc[df['hourly_profile'] == '', 'hourly_profile'] = df['aadt_h_mn']

        df['hourly_profile'] = df['hourly_profile'].astype(str)
        self.hourly_profiles['P_hour'] = self.hourly_profiles['P_hour'].astype(str)

        df = df.merge(self.hourly_profiles, left_on='hourly_profile', right_on='P_hour', how='left')
        df['HF'] = df.groupby('hour').apply(lambda x: x[[str(x.name)]])

        self.logger.write_time_log('TrafficSector', 'calculate_temporal_factor', timeit.default_timer() - spent_time)
        return df['MF'] * df['WF'] * df['HF']

    def calculate_time_dependent_values(self, df, timestep_type, timestep_num, timestep_freq):
        spent_time = timeit.default_timer()

        df.reset_index(inplace=True)
        for tstep in xrange(timestep_num):
            # Finding weekday
            # 0 -> Monday; 6 -> Sunday
            df.loc[:, 'month'] = (df['start_date'] + self.calculate_timedelta(
                timestep_type, tstep, timestep_freq)).dt.month
            df.loc[:, 'week_day'] = (df['start_date'] + self.calculate_timedelta(
                timestep_type, tstep, timestep_freq)).dt.weekday
            df.loc[:, 'hour'] = (df['start_date'] + self.calculate_timedelta(
                timestep_type, tstep, timestep_freq)).dt.hour

            # Selecting speed_mean
            df.loc[df['week_day'] <= 4, 'speed_mean'] = df['sp_wd']
            df.loc[df['week_day'] > 4, 'speed_mean'] = df['sp_we']

            # Selecting speed profile_id
            df.loc[df['week_day'] == 0, 'profile_id'] = df['sp_hour_mo']
            df.loc[df['week_day'] == 1, 'profile_id'] = df['sp_hour_tu']
            df.loc[df['week_day'] == 2, 'profile_id'] = df['sp_hour_we']
            df.loc[df['week_day'] == 3, 'profile_id'] = df['sp_hour_th']
            df.loc[df['week_day'] == 4, 'profile_id'] = df['sp_hour_fr']
            df.loc[df['week_day'] == 5, 'profile_id'] = df['sp_hour_sa']
            df.loc[df['week_day'] == 6, 'profile_id'] = df['sp_hour_su']

            df['profile_id'] = df['profile_id'].astype(int)

            # Selecting flat profile for 0 and nan's
            df.loc[df['profile_id'] == 0, 'profile_id'] = 1
            df.loc[df['profile_id'] == np.nan, 'profile_id'] = 1

            # Calculating speed by tstep
            speed_column_name = 'v_{0}'.format(tstep)
            df[speed_column_name] = self.calculate_hourly_speed(df.loc[:, ['hour', 'speed_mean', 'profile_id']])

            factor_column_name = 'f_{0}'.format(tstep)

            df.loc[:, factor_column_name] = self.calculate_temporal_factor(
                df.loc[:, ['month', 'week_day', 'hour', 'aadt_m_mn', 'aadt_week', 'aadt_h_mn', 'aadt_h_wd',
                           'aadt_h_sat', 'aadt_h_sun']])

        # Deleting time variables

        del df['month'], df['week_day'], df['hour'], df['profile_id'], df['speed_mean']
        del df['sp_wd'], df['sp_we'], df['index']
        del df['sp_hour_mo'], df['sp_hour_tu'], df['sp_hour_we'], df['sp_hour_th'], df['sp_hour_fr']
        del df['sp_hour_sa'], df['sp_hour_su']
        del df['aadt_m_mn'], df['aadt_h_mn'], df['aadt_h_wd'], df['aadt_h_sat'], df['aadt_h_sun'], df['aadt_week']
        del df['start_date']

        self.logger.write_time_log('TrafficSector', 'calculate_time_dependent_values',
                                   timeit.default_timer() - spent_time)

        return df

    def expand_road_links(self, timestep_type, timestep_num, timestep_freq):
        spent_time = timeit.default_timer()

        # Expands each road link by any vehicle type that the selected road link has.
        df_list = []
        road_link_aux = self.road_links.copy()

        del road_link_aux['geometry']
        for zone, compo_df in road_link_aux.groupby('fleet_comp'):
            fleet = self.find_fleet(zone)
            df_aux = pd.merge(compo_df, fleet, how='left', on='fleet_comp')
            df_list.append(df_aux)

        df = pd.concat(df_list, ignore_index=True)

        del df['fleet_comp']

        df = self.update_fleet_value(df)
        df = self.calculate_time_dependent_values(df, timestep_type, timestep_num, timestep_freq)

        self.logger.write_time_log('TrafficSector', 'expand_road_links', timeit.default_timer() - spent_time)

        return df

    def find_fleet(self, zone):
        spent_time = timeit.default_timer()

        try:
            fleet = self.fleet_compo[['Code', 'Class', zone]]
        except KeyError as e:
            raise KeyError(e.message + ' of the fleet_compo file')
        fleet.columns = ['Fleet_Code', 'Fleet_Class', 'Fleet_value']

        fleet = fleet[fleet['Fleet_value'] > 0]

        fleet['fleet_comp'] = zone

        self.logger.write_time_log('TrafficSector', 'find_fleet', timeit.default_timer() - spent_time)

        return fleet

    def calculate_hot(self):
        spent_time = timeit.default_timer()

        expanded_aux = self.expanded.copy().reset_index()

        for pollutant in self.source_pollutants:
            if pollutant != 'nh3':

                ef_code_slope_road, ef_code_slope, ef_code_road, ef_code = self.read_ef('hot', pollutant)
                df_code_slope_road = expanded_aux.merge(
                    ef_code_slope_road, left_on=['Fleet_Code', 'road_grad', 'Road_type'],
                    right_on=['Code', 'Road.Slope', 'Mode'], how='inner')
                df_code_slope = expanded_aux.merge(ef_code_slope, left_on=['Fleet_Code', 'road_grad'],
                                                   right_on=['Code', 'Road.Slope'], how='inner')
                df_code_road = expanded_aux.merge(ef_code_road, left_on=['Fleet_Code', 'Road_type'],
                                                  right_on=['Code', 'Mode'], how='inner')
                df_code = expanded_aux.merge(ef_code, left_on=['Fleet_Code'], right_on=['Code'], how='inner')

                del ef_code_slope_road, ef_code_slope, ef_code_road, ef_code

                expanded_aux = pd.concat([df_code_slope_road, df_code_slope, df_code_road, df_code])

                expanded_aux.drop(columns=['Code', 'Road.Slope', 'Mode'], inplace=True)
            else:
                ef_code_road = self.read_ef('hot', pollutant)
                expanded_aux = expanded_aux.merge(ef_code_road, left_on=['Fleet_Code', 'Road_type'],
                                                  right_on=['Code', 'Mode'], how='inner')

                del expanded_aux['Code'], expanded_aux['Mode']

            # Warnings and Errors
            original_ef_profile = self.expanded['Fleet_Code'].unique()
            calculated_ef_profiles = expanded_aux['Fleet_Code'].unique()
            resta_1 = [item for item in original_ef_profile if item not in calculated_ef_profiles]  # Warining
            resta_2 = [item for item in calculated_ef_profiles if item not in original_ef_profile]  # Error

            if len(resta_1) > 0:
                self.logger.write_log('WARNING! Exists some fleet codes that not appear on the EF file: {0}'.format(
                    resta_1))
                warnings.warn('Exists some fleet codes that not appear on the EF file: {0}'.format(resta_1), Warning)
            if len(resta_2) > 0:
                raise ImportError('Exists some fleet codes duplicateds on the EF file: {0}'.format(resta_2))

            m_corr = self.read_mcorr_file(pollutant)
            if m_corr is not None:
                expanded_aux = expanded_aux.merge(m_corr, left_on='Fleet_Code', right_on='Code', how='left')
                del expanded_aux['Code']

            for tstep in xrange(self.timestep_num):
                ef_name = 'ef_{0}_{1}'.format(pollutant, tstep)
                p_column = '{0}_{1}'.format(pollutant, tstep)
                if pollutant != 'nh3':
                    expanded_aux['v_aux'] = expanded_aux['v_{0}'.format(tstep)]
                    expanded_aux.loc[expanded_aux['v_aux'] < expanded_aux['Min.Speed'], 'v_aux'] = expanded_aux.loc[
                        expanded_aux['v_aux'] < expanded_aux['Min.Speed'], 'Min.Speed']
                    expanded_aux.loc[expanded_aux['v_aux'] > expanded_aux['Max.Speed'], 'v_aux'] = expanded_aux.loc[
                        expanded_aux['v_aux'] > expanded_aux['Max.Speed'], 'Max.Speed']

                    # EF
                    expanded_aux.loc[:, ef_name] = \
                        ((expanded_aux.Alpha * expanded_aux.v_aux**2 + expanded_aux.Beta * expanded_aux.v_aux +
                          expanded_aux.Gamma + (expanded_aux.Delta / expanded_aux.v_aux)) /
                         (expanded_aux.Epsilon * expanded_aux.v_aux**2 + expanded_aux.Zita * expanded_aux.v_aux +
                          expanded_aux.Hta)) * (1 - expanded_aux.RF) * \
                        (expanded_aux.PF * expanded_aux['T'] / expanded_aux.Q)
                else:
                    expanded_aux.loc[:, ef_name] = \
                        ((expanded_aux['a'] * expanded_aux['Cmileage'] + expanded_aux['b']) *
                         (expanded_aux['EFbase'] * expanded_aux['TF'])) / 1000

                # Mcorr
                if m_corr is not None:
                    expanded_aux.loc[expanded_aux['v_aux'] <= 19., 'Mcorr'] = \
                        expanded_aux.A_urban * expanded_aux['M'] + expanded_aux.B_urban
                    expanded_aux.loc[expanded_aux['v_aux'] >= 63., 'Mcorr'] = \
                        expanded_aux.A_road * expanded_aux['M'] + expanded_aux.B_road
                    expanded_aux.loc[(expanded_aux['v_aux'] > 19.) & (expanded_aux['v_aux'] < 63.), 'Mcorr'] = \
                        (expanded_aux.A_urban * expanded_aux['M'] + expanded_aux.B_urban) + \
                        ((expanded_aux.v_aux - 19) *
                         ((expanded_aux.A_road * expanded_aux['M'] + expanded_aux.B_road) -
                          (expanded_aux.A_urban * expanded_aux['M'] + expanded_aux.B_urban))) / 44.
                    expanded_aux.loc[expanded_aux['Mcorr'].isnull(), 'Mcorr'] = 1
                else:
                    expanded_aux.loc[:, 'Mcorr'] = 1

                # Full formula
                expanded_aux.loc[:, p_column] = \
                    expanded_aux['Fleet_value'] * expanded_aux[ef_name] * expanded_aux['Mcorr'] * \
                    expanded_aux['f_{0}'.format(tstep)]
                del expanded_aux[ef_name], expanded_aux['Mcorr']

            if pollutant != 'nh3':
                del expanded_aux['v_aux']
                del expanded_aux['Min.Speed'], expanded_aux['Max.Speed'], expanded_aux['Alpha'], expanded_aux['Beta']
                del expanded_aux['Gamma'], expanded_aux['Delta'], expanded_aux['Epsilon'], expanded_aux['Zita']
                del expanded_aux['Hta'], expanded_aux['RF'], expanded_aux['Q'], expanded_aux['PF'], expanded_aux['T']
            else:
                del expanded_aux['a'], expanded_aux['Cmileage'], expanded_aux['b'], expanded_aux['EFbase']
                del expanded_aux['TF']

            if m_corr is not None:
                del expanded_aux['A_urban'], expanded_aux['B_urban'], expanded_aux['A_road'], expanded_aux['B_road']
                del expanded_aux['M']

        del expanded_aux['road_grad']

        for tstep in xrange(self.timestep_num):
            del expanded_aux['f_{0}'.format(tstep)]

        self.logger.write_time_log('TrafficSector', 'calculate_hot', timeit.default_timer() - spent_time)
        return expanded_aux

    def calculate_cold(self, hot_expanded):
        spent_time = timeit.default_timer()

        cold_links = self.road_links.copy()

        del cold_links['aadt'], cold_links['PcHeavy'], cold_links['PcMoto'], cold_links['PcMoped'], cold_links['sp_wd']
        del cold_links['sp_we'], cold_links['sp_hour_su'], cold_links['sp_hour_mo'], cold_links['sp_hour_tu']
        del cold_links['sp_hour_we'], cold_links['sp_hour_th'], cold_links['sp_hour_fr'], cold_links['sp_hour_sa']
        del cold_links['Road_type'], cold_links['aadt_m_mn'], cold_links['aadt_h_mn'], cold_links['aadt_h_wd']
        del cold_links['aadt_h_sat'], cold_links['aadt_h_sun'], cold_links['aadt_week'], cold_links['fleet_comp']
        del cold_links['road_grad'], cold_links['PcLight'], cold_links['start_date']

        cold_links.loc[:, 'centroid'] = cold_links['geometry'].centroid
        link_lons = cold_links['geometry'].centroid.x
        link_lats = cold_links['geometry'].centroid.y

        temperature = self.read_temperature(link_lons.min(), link_lons.max(), link_lats.min(), link_lats.max(),
                                            self.temp_common_path, self.starting_date, self.timestep_num,
                                            self.timestep_freq)

        unary_union = temperature.unary_union
        cold_links['REC'] = cold_links.apply(self.nearest, geom_union=unary_union, df1=cold_links, df2=temperature,
                                             geom1_col='centroid', src_column='REC', axis=1)
        del cold_links['geometry'], cold_links['centroid'], temperature['geometry']

        cold_links = cold_links.merge(temperature, left_on='REC', right_on='REC', how='left')

        del cold_links['REC']

        c_expanded = hot_expanded.merge(cold_links, left_on='Link_ID', right_on='Link_ID', how='left')

        df_list = []
        for pollutant in self.source_pollutants:

            ef_cold = self.read_ef('cold', pollutant)

            if pollutant != 'nh3':
                ef_cold.loc[ef_cold['Tmin'].isnull(), 'Tmin'] = -999
                ef_cold.loc[ef_cold['Tmax'].isnull(), 'Tmax'] = 999
                ef_cold.loc[ef_cold['Min.Speed'].isnull(), 'Min.Speed'] = -999
                ef_cold.loc[ef_cold['Max.Speed'].isnull(), 'Max.Speed'] = 999

            c_expanded_p = c_expanded.merge(ef_cold, left_on=['Fleet_Code', 'Road_type'],
                                            right_on=['Code', 'Mode'], how='inner')
            cold_exp_p_aux = c_expanded_p.copy()

            del cold_exp_p_aux['index_right_x'], cold_exp_p_aux['Road_type'], cold_exp_p_aux['Fleet_value']
            del cold_exp_p_aux['Code']

            for tstep in xrange(self.timestep_num):
                v_column = 'v_{0}'.format(tstep)
                p_column = '{0}_{1}'.format(pollutant, tstep)
                t_column = 't_{0}'.format(tstep)
                if pollutant != 'nh3':
                    cold_exp_p_aux = cold_exp_p_aux.loc[cold_exp_p_aux[t_column] >= cold_exp_p_aux['Tmin'], :]
                    cold_exp_p_aux = cold_exp_p_aux.loc[cold_exp_p_aux[t_column] < cold_exp_p_aux['Tmax'], :]
                    cold_exp_p_aux = cold_exp_p_aux.loc[cold_exp_p_aux[v_column] >= cold_exp_p_aux['Min.Speed'], :]
                    cold_exp_p_aux = cold_exp_p_aux.loc[cold_exp_p_aux[v_column] < cold_exp_p_aux['Max.Speed'], :]

                # Beta
                cold_exp_p_aux.loc[:, 'Beta'] = \
                    (0.6474 - (0.02545 * cold_exp_p_aux['ltrip']) - (0.00974 - (0.000385 * cold_exp_p_aux['ltrip'])) *
                     cold_exp_p_aux[t_column]) * cold_exp_p_aux['bc']
                if pollutant != 'nh3':
                    cold_exp_p_aux.loc[:, 'cold_hot'] = \
                        cold_exp_p_aux['A'] * cold_exp_p_aux[v_column] + cold_exp_p_aux['B'] * \
                        cold_exp_p_aux[t_column] + cold_exp_p_aux['C']

                else:
                    cold_exp_p_aux.loc[:, 'cold_hot'] = \
                        ((cold_exp_p_aux['a'] * cold_exp_p_aux['Cmileage'] + cold_exp_p_aux['b']) *
                         cold_exp_p_aux['EFbase'] * cold_exp_p_aux['TF']) / \
                        ((cold_exp_p_aux['a_hot'] * cold_exp_p_aux['Cmileage'] + cold_exp_p_aux['b_hot']) *
                         cold_exp_p_aux['EFbase_hot'] * cold_exp_p_aux['TF_hot'])
                cold_exp_p_aux.loc[cold_exp_p_aux['cold_hot'] < 1, 'cold_hot'] = 1

                # Formula Cold emissions
                cold_exp_p_aux.loc[:, p_column] = \
                    cold_exp_p_aux[p_column] * cold_exp_p_aux['Beta'] * (cold_exp_p_aux['cold_hot'] - 1)
                df_list.append((cold_exp_p_aux.loc[:, ['Link_ID', 'Fleet_Code', p_column]]).set_index(
                    ['Link_ID', 'Fleet_Code']))

        try:
            cold_df = pd.concat(df_list, axis=1, ).reset_index()
        except Exception:
            error_fleet_code = []
            for df in df_list:
                orig = list(df.index.values)
                uni = list(np.unique(df.index.values))

                for o in orig:
                    try:
                        uni.remove(o)
                    except Exception:
                        error_fleet_code.append(o)
            raise IndexError('There are duplicated values for {0} codes in the cold EF files.'.format(error_fleet_code))

        for tstep in xrange(self.timestep_num):
            if 'pm' in self.source_pollutants:
                cold_df.loc[:, 'pm10_{0}'.format(tstep)] = cold_df['pm_{0}'.format(tstep)]
                cold_df.loc[:, 'pm25_{0}'.format(tstep)] = cold_df['pm_{0}'.format(tstep)]
                del cold_df['pm_{0}'.format(tstep)]

            if 'voc' in self.source_pollutants and 'ch4' in self.source_pollutants:
                cold_df.loc[:, 'nmvoc_{0}'.format(tstep)] = \
                    cold_df['voc_{0}'.format(tstep)] - cold_df['ch4_{0}'.format(tstep)]
                del cold_df['voc_{0}'.format(tstep)]
            else:
                self.logger.write_log("WARNING! nmvoc emissions cannot be estimated because voc or ch4 are not " +
                                      "selected in the pollutant list.")
                warnings.warn("nmvoc emissions cannot be estimated because voc or ch4 are not selected in the " +
                              "pollutant list.")

        cold_df = self.speciate_traffic(cold_df, self.hot_cold_speciation)

        self.logger.write_time_log('TrafficSector', 'calculate_cold', timeit.default_timer() - spent_time)
        return cold_df

    def compact_hot_expanded(self, expanded):
        spent_time = timeit.default_timer()

        columns_to_delete = ['Road_type', 'Fleet_value'] + ['v_{0}'.format(x) for x in xrange(self.timestep_num)]
        for column_name in columns_to_delete:
            del expanded[column_name]

        for tstep in xrange(self.timestep_num):
            if 'pm' in self.source_pollutants:
                expanded.loc[:, 'pm10_{0}'.format(tstep)] = expanded['pm_{0}'.format(tstep)]
                expanded.loc[:, 'pm25_{0}'.format(tstep)] = expanded['pm_{0}'.format(tstep)]
                del expanded['pm_{0}'.format(tstep)]

            if 'voc' in self.source_pollutants and 'ch4' in self.source_pollutants:
                expanded.loc[:, 'nmvoc_{0}'.format(tstep)] = expanded['voc_{0}'.format(tstep)] - \
                                                             expanded['ch4_{0}'.format(tstep)]
                del expanded['voc_{0}'.format(tstep)]
            else:
                self.logger.write_log("nmvoc emissions cannot be estimated because voc or ch4 are not selected in " +
                                      "the pollutant list.")
                warnings.warn(
                    "nmvoc emissions cannot be estimated because voc or ch4 are not selected in the pollutant list.")

        compacted = self.speciate_traffic(expanded, self.hot_cold_speciation)

        self.logger.write_time_log('TrafficSector', 'compact_hot_expanded', timeit.default_timer() - spent_time)
        return compacted

    def calculate_tyre_wear(self):
        spent_time = timeit.default_timer()

        pollutants = ['pm']
        for pollutant in pollutants:
            ef_tyre = self.read_ef('tyre', pollutant)
            df = self.expanded.merge(ef_tyre, left_on='Fleet_Code', right_on='Code', how='inner')
            del df['road_grad'], df['Road_type'], df['Code']
            for tstep in xrange(self.timestep_num):
                p_column = '{0}_{1}'.format(pollutant, tstep)
                f_column = 'f_{0}'.format(tstep)
                v_column = 'v_{0}'.format(tstep)
                df.loc[df[v_column] < 40, p_column] = df['Fleet_value'] * df['EFbase'] * df[f_column] * 1.39
                df.loc[(df[v_column] >= 40) & (df[v_column] <= 90), p_column] = \
                    df['Fleet_value'] * df['EFbase'] * df[f_column] * (-0.00974 * df[v_column] + 1.78)
                df.loc[df[v_column] > 90, p_column] = df['Fleet_value'] * df['EFbase'] * df[f_column] * 0.902

                # from PM to PM10 & PM2.5
                if pollutant == 'pm':
                    df.loc[:, 'pm10_{0}'.format(tstep)] = df[p_column] * 0.6
                    df.loc[:, 'pm25_{0}'.format(tstep)] = df[p_column] * 0.42
                    del df[p_column]

        # Cleaning df
        columns_to_delete = ['f_{0}'.format(x) for x in xrange(self.timestep_num)] + ['v_{0}'.format(x) for x in xrange(
            self.timestep_num)]
        columns_to_delete += ['Fleet_value', 'EFbase']
        for column in columns_to_delete:
            del df[column]
        df = self.speciate_traffic(df, self.tyre_speciation)

        self.logger.write_time_log('TrafficSector', 'calculate_tyre_wear', timeit.default_timer() - spent_time)
        return df

    def calculate_brake_wear(self):
        spent_time = timeit.default_timer()

        pollutants = ['pm']
        for pollutant in pollutants:
            ef_tyre = self.read_ef('brake', pollutant)
            df = self.expanded.merge(ef_tyre, left_on='Fleet_Code', right_on='Code', how='inner')
            del df['road_grad'], df['Road_type'], df['Code']
            for tstep in xrange(self.timestep_num):
                p_column = '{0}_{1}'.format(pollutant, tstep)
                f_column = 'f_{0}'.format(tstep)
                v_column = 'v_{0}'.format(tstep)
                df.loc[df[v_column] < 40, p_column] = df['Fleet_value'] * df['EFbase'] * df[f_column] * 1.67
                df.loc[(df[v_column] >= 40) & (df[v_column] <= 95), p_column] = \
                    df['Fleet_value'] * df['EFbase'] * df[f_column] * (-0.027 * df[v_column] + 2.75)
                df.loc[df[v_column] > 95, p_column] = df['Fleet_value'] * df['EFbase'] * df[f_column] * 0.185

                # from PM to PM10 & PM2.5
                if pollutant == 'pm':
                    df.loc[:, 'pm10_{0}'.format(tstep)] = df[p_column] * 0.98
                    df.loc[:, 'pm25_{0}'.format(tstep)] = df[p_column] * 0.39
                    del df[p_column]

        # Cleaning df
        columns_to_delete = ['f_{0}'.format(x) for x in xrange(self.timestep_num)] + ['v_{0}'.format(x) for x in xrange(
            self.timestep_num)]
        columns_to_delete += ['Fleet_value', 'EFbase']
        for column in columns_to_delete:
            del df[column]

        df = self.speciate_traffic(df, self.brake_speciation)

        self.logger.write_time_log('TrafficSector', 'calculate_brake_wear', timeit.default_timer() - spent_time)
        return df

    def calculate_road_wear(self):
        spent_time = timeit.default_timer()

        pollutants = ['pm']
        for pollutant in pollutants:
            ef_tyre = self.read_ef('road', pollutant)
            df = self.expanded.merge(ef_tyre, left_on='Fleet_Code', right_on='Code', how='inner')
            del df['road_grad'], df['Road_type'], df['Code']
            for tstep in xrange(self.timestep_num):
                p_column = '{0}_{1}'.format(pollutant, tstep)
                f_column = 'f_{0}'.format(tstep)
                df.loc[:, p_column] = df['Fleet_value'] * df['EFbase'] * df[f_column]

                # from PM to PM10 & PM2.5
                if pollutant == 'pm':
                    df.loc[:, 'pm10_{0}'.format(tstep)] = df[p_column] * 0.5
                    df.loc[:, 'pm25_{0}'.format(tstep)] = df[p_column] * 0.27
                    del df[p_column]

        # Cleaning df
        columns_to_delete = ['f_{0}'.format(x) for x in xrange(self.timestep_num)] + ['v_{0}'.format(x) for x in xrange(
            self.timestep_num)]
        columns_to_delete += ['Fleet_value', 'EFbase']
        for column in columns_to_delete:
            del df[column]

        df = self.speciate_traffic(df, self.road_speciation)

        self.logger.write_time_log('TrafficSector', 'calculate_road_wear', timeit.default_timer() - spent_time)
        return df

    def calculate_resuspension(self):
        spent_time = timeit.default_timer()

        if self.resuspension_correction:
            road_link_aux = self.road_links.loc[:, ['Link_ID', 'geometry']].copy()

            road_link_aux.loc[:, 'centroid'] = road_link_aux['geometry'].centroid
            link_lons = road_link_aux['geometry'].centroid.x
            link_lats = road_link_aux['geometry'].centroid.y
            p_factor = self.get_precipitation(link_lons.min(), link_lons.max(), link_lats.min(), link_lats.max(),
                                              self.precipitation_path)

            unary_union = p_factor.unary_union
            road_link_aux['REC'] = road_link_aux.apply(self.nearest, geom_union=unary_union, df1=road_link_aux,
                                                       df2=p_factor, geom1_col='centroid', src_column='REC', axis=1)
            del road_link_aux['centroid'], p_factor['geometry']

            road_link_aux = road_link_aux.merge(p_factor, left_on='REC', right_on='REC', how='left')

            del road_link_aux['REC']

        pollutants = ['pm']
        for pollutant in pollutants:
            ef_tyre = self.read_ef('resuspension', pollutant)
            df = self.expanded.merge(ef_tyre, left_on='Fleet_Code', right_on='Code', how='inner')
            if self.resuspension_correction:
                df = df.merge(road_link_aux, left_on='Link_ID', right_on='Link_ID', how='left')

            del df['road_grad'], df['Road_type'], df['Code']
            for tstep in xrange(self.timestep_num):
                p_column = '{0}_{1}'.format(pollutant, tstep)
                f_column = 'f_{0}'.format(tstep)
                if self.resuspension_correction:
                    pr_column = 'PR_{0}'.format(tstep)
                    df.loc[:, p_column] = df['Fleet_value'] * df['EFbase'] * df[pr_column] * df[f_column]
                else:
                    df.loc[:, p_column] = df['Fleet_value'] * df['EFbase'] * df[f_column]

                # from PM to PM10 & PM2.5
                if pollutant == 'pm':
                    df.loc[:, 'pm10_{0}'.format(tstep)] = df[p_column]
                    # TODO Check fraction of pm2.5
                    df.loc[:, 'pm25_{0}'.format(tstep)] = df[p_column] * 0.5
                    del df[p_column]

        # Cleaning df
        columns_to_delete = ['f_{0}'.format(x) for x in xrange(self.timestep_num)] + ['v_{0}'.format(x) for x in
                                                                                      xrange(self.timestep_num)]
        columns_to_delete += ['Fleet_value', 'EFbase']
        for column in columns_to_delete:
            del df[column]

        df = self.speciate_traffic(df, self.resuspension_speciation)

        self.logger.write_time_log('TrafficSector', 'calculate_resuspension', timeit.default_timer() - spent_time)
        return df

    def transform_df(self, df):
        spent_time = timeit.default_timer()

        df_list = []
        for tstep in xrange(self.timestep_num):
            pollutants_to_rename = [p for p in list(df.columns.values) if p.endswith('_{0}'.format(tstep))]
            pollutants_renamed = []
            for p_name in pollutants_to_rename:
                p_name_new = p_name.replace('_{0}'.format(tstep), '')
                df.rename(columns={p_name: p_name_new}, inplace=True)
                pollutants_renamed.append(p_name_new)

            df_aux = pd.DataFrame(df.loc[:, ['Link_ID', 'Fleet_Code'] + pollutants_renamed])
            df_aux['tstep'] = tstep

            df_list.append(df_aux)
            df.drop(columns=pollutants_renamed, inplace=True)

        df = pd.concat(df_list, ignore_index=True)
        self.logger.write_time_log('TrafficSector', 'transform_df', timeit.default_timer() - spent_time)
        return df

    def speciate_traffic(self, df, speciation):
        spent_time = timeit.default_timer()

        # Reads speciation profile
        speciation = self.read_profiles(speciation)
        del speciation['Copert_V_name']

        # Transform dataset into timestep rows instead of timestep columns
        df = self.transform_df(df)

        in_list = list(df.columns.values)

        in_columns = ['Link_ID', 'Fleet_Code', 'tstep']
        for in_col in in_columns:
            in_list.remove(in_col)

        df_out_list = []

        # PMC
        if not set(speciation.columns.values).isdisjoint(pmc_list):
            out_p = set(speciation.columns.values).intersection(pmc_list).pop()
            speciation_by_in_p = speciation.loc[:, [out_p] + ['Code']]

            speciation_by_in_p.rename(columns={out_p: 'f_{0}'.format(out_p)}, inplace=True)
            df_aux = df.loc[:, ['pm10', 'pm25', 'Fleet_Code', 'tstep', 'Link_ID']]
            df_aux = df_aux.merge(speciation_by_in_p, left_on='Fleet_Code', right_on='Code', how='left')
            df_aux.drop(columns=['Code'], inplace=True)

            df_aux.loc[:, out_p] = df_aux['pm10'] - df_aux['pm25']
            # from g/km.h to g/km.s
            # df_aux.loc[:, out_p] = df_aux.loc[:, out_p] / 3600
            # if self.output_type == 'R-LINE':
            #     # from g/km.h to g/m.s
            #     df_aux.loc[:, out_p] = df_aux.loc[:, out_p] / (1000 * 3600)
            # elif self.output_type == 'CMAQ':
            #     # from g/km.h to mol/km.s
            #     df_aux.loc[:, out_p] = df_aux.loc[:, out_p] / 3600
            # elif self.output_type == 'MONARCH':
            #     # from g/km.h to Kg/km.s
            #     df_aux.loc[:, out_p] = df_aux.loc[:, out_p] / (1000 * 3600)

            df_out_list.append(df_aux.loc[:, [out_p] + ['tstep', 'Link_ID']].groupby(['tstep', 'Link_ID']).sum())
            del df_aux[out_p]
        for in_p in in_list:
            involved_out_pollutants = [key for key, value in self.speciation_map.iteritems() if value == in_p]
            # Selecting only necessary speciation profiles
            speciation_by_in_p = speciation.loc[:, involved_out_pollutants + ['Code']]

            # Adding "f_" in the formula column names
            for p in involved_out_pollutants:
                speciation_by_in_p.rename(columns={p: 'f_{0}'.format(p)}, inplace=True)
            # Getting a slice of the full dataset to be merged
            df_aux = df.loc[:, [in_p] + ['Fleet_Code', 'tstep', 'Link_ID']]
            df_aux = df_aux.merge(speciation_by_in_p, left_on='Fleet_Code', right_on='Code', how='left')
            df_aux.drop(columns=['Code'], inplace=True)

            # Renaming pollutant columns by adding "old_" to the beginning.
            df_aux.rename(columns={in_p: 'old_{0}'.format(in_p)}, inplace=True)
            for p in involved_out_pollutants:
                if in_p is not np.nan:
                    if in_p != 0:
                        df_aux.loc[:, p] = df_aux['old_{0}'.format(in_p)].multiply(df_aux['f_{0}'.format(p)])
                        try:
                            if in_p == 'nmvoc':
                                mol_w = 1.0
                            else:
                                mol_w = self.molecular_weights[in_p]
                        except KeyError:
                            raise AttributeError('{0} not found in the molecular weights file.'.format(in_p))
                        # from g/km.h to mol/km.h or g/km.h (aerosols)
                        df_aux.loc[:, p] = df_aux.loc[:, p] / mol_w

                        # if self.output_type == 'R-LINE':
                        #     # from g/km.h to g/m.s
                        #     df_aux.loc[:, p] = df_aux.loc[:, p] / (1000 * 3600)
                        # elif self.output_type == 'CMAQ':
                        #     # from g/km.h to mol/km.s or g/km.s (aerosols)
                        #     df_aux.loc[:, p] = df_aux.loc[:, p] / (3600 * mol_w)
                        # elif self.output_type == 'MONARCH':
                        #     if p.lower() in aerosols:
                        #         # from g/km.h to kg/km.s
                        #         df_aux.loc[:, p] = df_aux.loc[:, p] / (1000 * 3600 * mol_w)
                        #     else:
                        #         # from g/km.h to mol/km.s
                        #         df_aux.loc[:, p] = df_aux.loc[:, p] / (3600 * mol_w)
                    else:
                        df_aux.loc[:, p] = 0

                df_out_list.append(df_aux.loc[:, [p] + ['tstep', 'Link_ID']].groupby(['tstep', 'Link_ID']).sum())
                del df_aux[p]
            del df_aux
            del df[in_p]

        df_out = pd.concat(df_out_list, axis=1)

        self.logger.write_time_log('TrafficSector', 'speciate_traffic', timeit.default_timer() - spent_time)
        return df_out

    def calculate_emissions(self):
        spent_time = timeit.default_timer()

        self.logger.write_log('\t\tCalculating yearly emissions', message_level=2)
        df_accum = pd.DataFrame()
        if self.do_hot:
            df_accum = pd.concat([df_accum, self.compact_hot_expanded(self.calculate_hot())]).groupby(
                ['tstep', 'Link_ID']).sum()
        if self.do_cold:
            df_accum = pd.concat([df_accum, self.calculate_cold(self.calculate_hot())]).groupby(
                ['tstep', 'Link_ID']).sum()
        if self.do_tyre_wear:
            df_accum = pd.concat([df_accum, self.calculate_tyre_wear()]).groupby(['tstep', 'Link_ID']).sum()
        if self.do_brake_wear:
            df_accum = pd.concat([df_accum, self.calculate_brake_wear()]).groupby(['tstep', 'Link_ID']).sum()
        if self.do_road_wear:
            df_accum = pd.concat([df_accum, self.calculate_road_wear()]).groupby(['tstep', 'Link_ID']).sum()
        if self.do_resuspension:
            df_accum = pd.concat([df_accum, self.calculate_resuspension()]).groupby(['tstep', 'Link_ID']).sum()

        df_accum = df_accum.reset_index().merge(self.road_links.loc[:, ['Link_ID', 'geometry']], left_on='Link_ID',
                                                right_on='Link_ID', how='left')
        df_accum = gpd.GeoDataFrame(df_accum, crs=self.crs)
        df_accum.set_index(['Link_ID', 'tstep'], inplace=True)

        if self.write_rline:
            self.write_rline_output(df_accum.copy())

        df_accum = self.links_to_grid(df_accum)

        self.logger.write_log('\t\tTraffic emissions calculated', message_level=2)
        self.logger.write_time_log('TrafficSector', 'calculate_emissions', timeit.default_timer() - spent_time)
        return df_accum

    def links_to_grid(self, link_emissions):
        spent_time = timeit.default_timer()

        link_emissions.reset_index(inplace=True)
        if not os.path.exists(self.link_to_grid_csv):
            link_emissions_aux = link_emissions.loc[link_emissions['tstep'] == 0, :]
            link_emissions_aux = link_emissions_aux.to_crs(self.grid_shp.crs)

            link_emissions_aux = gpd.sjoin(link_emissions_aux, self.grid_shp.reset_index(),
                                           how="inner", op='intersects')
            link_emissions_aux = link_emissions_aux.loc[:, ['Link_ID', 'geometry', 'FID']]

            link_emissions_aux = link_emissions_aux.merge(self.grid_shp.reset_index().loc[:, ['FID', 'geometry']],
                                                          on='FID', how='left')

            length_list = []
            link_id_list = []
            fid_list = []
            count = 1
            for i, line in link_emissions_aux.iterrows():
                count += 1
                aux = line.get('geometry_x').intersection(line.get('geometry_y'))
                if not aux.is_empty:
                    link_id_list.append(line.get('Link_ID'))
                    fid_list.append(line.get('FID'))
                    length_list.append(aux.length / 1000)

            link_grid = pd.DataFrame({'Link_ID': link_id_list, 'FID': fid_list, 'length': length_list})
            # data = self.comm.gather(link_grid, root=0)
            # if self.comm.Get_rank() == 0:
            #     data = pd.concat(data)
            #     data.to_csv(self.link_to_grid_csv)
        else:
            link_grid = pd.read_csv(self.link_to_grid_csv)

        del link_emissions['geometry']

        link_grid = link_grid.merge(link_emissions, left_on='Link_ID', right_on='Link_ID')
        # link_grid.drop(columns=['Unnamed: 0'], inplace=True)

        cols_to_update = list(link_grid.columns.values)
        cols_to_update.remove('length')
        cols_to_update.remove('tstep')
        cols_to_update.remove('FID')
        for col in cols_to_update:
            link_grid.loc[:, col] = link_grid[col] * link_grid['length']
        del link_grid['length']
        link_grid.drop(columns=['Link_ID'], inplace=True)
        link_grid['layer'] = 0
        link_grid = link_grid.groupby(['FID', 'layer', 'tstep']).sum()
        # link_grid.reset_index(inplace=True)
        #
        # link_grid_list = self.comm.gather(link_grid, root=0)
        # if self.comm.Get_rank() == 0:
        #     link_grid = pd.concat(link_grid_list)
        #     link_grid = link_grid.groupby(['tstep', 'FID']).sum()
        #     # link_grid.sort_index(inplace=True)
        #     link_grid.reset_index(inplace=True)
        #
        #     emission_list = []
        #     out_poll_names = list(link_grid.columns.values)
        #     out_poll_names.remove('tstep')
        #     out_poll_names.remove('FID')
        #
        #     for p in out_poll_names:
        #         data = np.zeros((self.timestep_num, len(grid_shape)))
        #         for tstep in xrange(self.timestep_num):
        #             data[tstep, link_grid.loc[link_grid['tstep'] == tstep, 'FID']] = \
        #                 link_grid.loc[link_grid['tstep'] == tstep, p]
        #
        #         dict_aux = {
        #             'name': p,
        #             'units': None,
        #             'data': data
        #         }
        #
        #         if self.output_type == 'R-LINE':
        #             # from g/km.h to g/m.s
        #             pass
        #         elif self.output_type == 'CMAQ':
        #             # from g/km.h to mol/km.s
        #             if p.lower() in aerosols:
        #                 dict_aux['units'] = '0.001 kg.s-1'
        #             else:
        #                 dict_aux['units'] = 'kat'
        #         elif self.output_type == 'MONARCH':
        #             if p.lower() in aerosols:
        #                 dict_aux['units'] = 'kg.s-1'
        #             else:
        #                 dict_aux['units'] = 'kat'
        #         emission_list.append(dict_aux)
        #
        self.logger.write_time_log('TrafficSector', 'links_to_grid', timeit.default_timer() - spent_time)

        return link_grid

    def write_rline_output(self, emissions):
        from datetime import timedelta
        spent_time = timeit.default_timer()

        emissions.drop(columns=['geometry'], inplace=True)
        for poll in emissions.columns.values:
            mol_w = self.molecular_weights[self.speciation_map[poll]]
            emissions.loc[:, poll] = emissions.loc[:, poll] * mol_w / (1000 * 3600)

        emissions.reset_index(inplace=True)

        emissions_list = self.comm.gather(emissions, root=0)
        if self.comm.Get_rank() == 0:
            emissions = pd.concat(emissions_list)
            p_list = list(emissions.columns.values)
            p_list.remove('tstep')
            p_list.remove('Link_ID')
            for p in p_list:
                link_list = ['L_{0}'.format(x) for x in list(pd.unique(emissions['Link_ID']))]
                out_df = pd.DataFrame(columns=["Year", "Mon", "Day", "JDay", "Hr"] + link_list)
                for tstep, aux in emissions.loc[:, ['tstep', 'Link_ID', p]].groupby('tstep'):
                    aux_date = self.date_array[0] + timedelta(hours=tstep)
                    out_df.loc[tstep, 'Year'] = aux_date.strftime('%y')
                    out_df.loc[tstep, 'Mon'] = aux_date.month
                    out_df.loc[tstep, 'Day'] = aux_date.day
                    out_df.loc[tstep, 'JDay'] = aux_date.strftime('%j')
                    out_df.loc[tstep, 'Hr'] = aux_date.hour
                    out_df.loc[tstep, link_list] = aux.loc[:, [p]].transpose().values

                out_df.to_csv(os.path.join(self.output_dir, 'rline_{1}_{0}.csv'.format(
                    p, self.date_array[0].strftime('%Y%m%d'))), index=False)

        self.comm.Barrier()

        self.logger.write_time_log('TrafficSector', 'write_rline_output', timeit.default_timer() - spent_time)
        return True

    def write_rline_roadlinks(self, df_in):
        spent_time = timeit.default_timer()

        df_in_list = self.comm.gather(df_in, root=0)
        if self.comm.Get_rank() == 0:
            df_in = pd.concat(df_in_list)

            df_out = pd.DataFrame(
                columns=['Group', 'X_b', 'Y_b', 'Z_b', 'X_e', 'Y_e', 'Z_e', 'dCL', 'sigmaz0', '#lanes',
                         'lanewidth', 'Emis', 'Hw1', 'dw1', 'Hw2', 'dw2', 'Depth', 'Wtop', 'Wbottom',
                         'l_bh2sw', 'l_avgbh', 'l_avgbdensity', 'l_bhdev', 'X0_af', 'X45_af',
                         'X90_af', 'X135_af', 'X180_af', 'X225_af', 'X270_af', 'X315_af', 'l_maxbh', 'Link_ID'])
            df_err_list = []

            df_in = df_in.to_crs({u'units': u'm', u'no_defs': True, u'ellps': u'intl', u'proj': u'utm', u'zone': 31})
            if rline_shp:
                gpd.GeoDataFrame().to_file
                df_in.to_file(os.path.join(self.output_dir, 'roads.shp'))

            count = 0
            for i, line in df_in.iterrows():
                try:
                    df_out.loc[count] = pd.Series({
                        'Group': 'G1',
                        'X_b': round(line.get('geometry').coords[0][0], 3),
                        'Y_b': round(line.get('geometry').coords[0][1], 3),
                        'Z_b': 1,
                        'X_e': round(line.get('geometry').coords[-1][0], 3),
                        'Y_e': round(line.get('geometry').coords[-1][1], 3),
                        'Z_e': 1,
                        'dCL': 0,
                        'sigmaz0': 2,
                        '#lanes': 3,
                        'lanewidth': 2.5,
                        'Emis': 1,
                        'Hw1': 0,
                        'dw1': 0,
                        'Hw2': 0,
                        'dw2': 0,
                        'Depth': 0,
                        'Wtop': 0,
                        'Wbottom': 0,
                        'l_bh2sw': round(line.get('bh_2_sw'), 3),
                        'l_avgbh': round(line.get('mean_heigh'), 3),
                        'l_avgbdensity': round(line.get('area_densi'), 3),
                        'l_bhdev': round(line.get('sd_height'), 3),
                        'X0_af': round(line.get('af_0'), 3),
                        'X45_af': round(line.get('af_45'), 3),
                        'X90_af': round(line.get('af_90'), 3),
                        'X135_af': round(line.get('af_135'), 3),
                        'X180_af': round(line.get('af_180'), 3),
                        'X225_af': round(line.get('af_225'), 3),
                        'X270_af': round(line.get('af_270'), 3),
                        'X315_af': round(line.get('af_315'), 3),
                        'l_maxbh': round(line.get('max_height'), 3),
                        'Link_ID': line.get('Link_ID'),
                    })
                    count += 1
                except Exception:
                    # df_err_list.append(line)
                    pass

            df_out.set_index('Link_ID', inplace=True)
            df_out.sort_index(inplace=True)
            df_out.to_csv(os.path.join(self.output_dir, 'roads.txt'), index=False, sep=' ')
        self.comm.Barrier()

        self.logger.write_time_log('TrafficSector', 'write_rline_roadlinks', timeit.default_timer() - spent_time)
        return True
