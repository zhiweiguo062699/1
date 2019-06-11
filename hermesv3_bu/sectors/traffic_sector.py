#!/usr/bin/env python
import sys
import os
import timeit

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.ops import nearest_points
import warnings

MIN_RAIN = 0.254  # After USEPA (2011)
RECOVERY_RATIO = 0.0872  # After Amato et al. (2012)


aerosols = ['oc', 'ec', 'pno3', 'pso4', 'pmfine', 'pmc', 'poa', 'poc', 'pec', 'pcl', 'pnh4', 'pna', 'pmg', 'pk', 'pca',
            'pncom', 'pfe', 'pal', 'psi', 'pti', 'pmn', 'ph2o', 'pmothr']
pmc_list = ['pmc', 'PMC']
rline_shp = False


class Traffic(object):
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

    def __init__(self, auxiliar_path, clipping, date_array, road_link_path, fleet_compo_path, speed_hourly_path,
                 monthly_profile_path, daily_profile_path, hourly_mean_profile_path, hourly_weekday_profile_path,
                 hourly_saturday_profile_path, hourly_sunday_profile_path, ef_common_path, pollutants_list, date,
                 grid, vehicle_list=None, load=0.5, timestep_type='hourly', timestep_num=1, timestep_freq=1, speciation_map=None,
                 hot_cold_speciation=None, tyre_speciation=None, road_speciation=None, brake_speciation=None,
                 resuspension_speciation=None, temp_common_path=None, output_type=None, output_dir=None,
                 molecular_weights_path=None, resuspension_correction=True, precipitation_path=None):

        spent_time = timeit.default_timer()

        if timestep_type != 'hourly':
            raise AttributeError('Traffic emissions are only developed for hourly timesteps. ' +
                                 '\"{0}\" timestep type found.'.format(timestep_type))

        self.resuspension_correction = resuspension_correction
        self.precipitation_path = precipitation_path
        self.date_array = date_array

        self.output_type = output_type
        self.output_dir = output_dir

        self.link_to_grid_csv = os.path.join(auxiliar_path, 'link_grid.csv')
        self.crs = None  # crs is the projection of the road links and it is set on the read_road_links function.
        self.road_links = self.read_road_links(road_link_path, clipping, grid)
        self.load = load
        self.ef_common_path = ef_common_path
        self.temp_common_path = temp_common_path
        self.pollutant_list = pollutants_list
        self.timestep_num = timestep_num
        self.timestep_freq = timestep_freq
        self.starting_date = date
        # print date
        self.add_local_date(date)

        self.speciation_map = speciation_map
        self.hot_cold_speciation = hot_cold_speciation
        self.tyre_speciation = tyre_speciation
        self.road_speciation = road_speciation
        self.brake_speciation = brake_speciation
        self.resuspension_speciation = resuspension_speciation

        self.fleet_compo = self.read_fleet_compo(fleet_compo_path, vehicle_list)
        self.speed_hourly = self.read_speed_hourly(speed_hourly_path)
        self.monthly_profiles = pd.read_csv(monthly_profile_path)
        self.daily_profiles = pd.read_csv(daily_profile_path)
        self.hourly_profiles = pd.concat([
            pd.read_csv(hourly_mean_profile_path),
            pd.read_csv(hourly_weekday_profile_path),
            pd.read_csv(hourly_saturday_profile_path),
            pd.read_csv(hourly_sunday_profile_path)
        ]).reset_index()

        self.expanded = self.expand_road_links(timestep_type, timestep_num, timestep_freq)

        del self.fleet_compo, self.speed_hourly, self.monthly_profiles, self.daily_profiles, self.hourly_profiles

        self.molecular_weigths = pd.read_csv(molecular_weights_path, sep=';')
        if settings.log_level_3:
            print 'TIME -> Traffic.__init__: {0} s'.format(round(gettime() - st_time, 2))

        return None

    def add_local_date(self, utc_date):
        """
        Adds to the road links the starting date in local time.
        This new column is called 'start_date'.

        :param utc_date: Starting date in UTC.
        """
        import pytz

        self.add_timezones()
        self.road_links.loc[:, 'utc'] = utc_date
        self.road_links['start_date'] = self.road_links.groupby('timezone')['utc'].apply(
            lambda x: pd.to_datetime(x).dt.tz_localize(pytz.utc).dt.tz_convert(x.name).dt.tz_localize(None))

        del self.road_links['utc'], self.road_links['timezone']

        return True

    def add_timezones(self):
        """
        Finds and sets the timezone for each road link.
        """
        # TODO calculate timezone from the centroid of each roadlink.

        self.road_links['timezone'] = 'Europe/Madrid'

        return True

    @staticmethod
    def read_speed_hourly(path):
        # TODO complete description
        """
        Reads the speed hourly file.

        :param path: Path to the speed hourly file.
        :type path: str:

        :return: ...
        :rtype: Pandas.DataFrame
        """
        df = pd.read_csv(path, sep=',', dtype=np.float32)
        return df

    @staticmethod
    def read_fleet_compo(path, vehicle_list):
        df = pd.read_csv(path, sep=',')
        if vehicle_list is not None:
            df = df.loc[df['Code'].isin(vehicle_list), :]
        return df

    @staticmethod
    def parse_clip(str_clip):
        import re
        from shapely.geometry import Point, Polygon
        # import FileNotFoundError
        if str_clip[0] == os.path.sep:
            if os.path.exists(str_clip):
                df_clip = gpd.read_file(str_clip)
                return df_clip
            else:
                warnings.warn(str_clip + ' file not found. Ignoring clipping.', Warning)
                return None
        else:
            str_clip = re.split(' , | ,|, |,', str_clip)
            lon_list = []
            lat_list = []
            for components in str_clip:
                components = re.split(' ', components)
                lon_list.append(float(components[0]))
                lat_list.append(float(components[1]))

            if not((lon_list[0] == lon_list[-1]) and (lat_list[0] == lat_list[-1])):
                lon_list.append(lon_list[0])
                lat_list.append(lat_list[0])

            df_clip = gpd.GeoDataFrame(geometry=[Polygon([[p.x, p.y] for p in [Point(xy) for xy in zip(lon_list, lat_list)]])], crs={'init': 'epsg:4326'})
            return df_clip
        return None

    def read_road_links(self, path, clipping, grid):
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

        if settings.log_level_3:
            st_time = gettime()
        else:
            st_time = None

        if settings.rank == 0:
            df = gpd.read_file(path)

            # Clipping
            if clipping is not None:
                clip = self.parse_clip(clipping)
                if clip is not None:
                    df = gpd.sjoin(df, clip.to_crs(df.crs), how="inner", op='intersects')
                    del clip
                else:
                    warnings.warn('Clipping type not found . Ignoring clipping.', Warning)
                    clipping = None

            if clipping is None:
                shape_grid = grid.to_shapefile()
                clip = gpd.GeoDataFrame(geometry=[shape_grid.unary_union], crs=shape_grid.crs)
                df = gpd.sjoin(df, clip.to_crs(df.crs), how="inner", op='intersects')

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

            chunks = chunk_road_links(df, settings.size)
        else:
            chunks = None
        settings.comm.Barrier()

        df = settings.comm.scatter(chunks, root=0)
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

        if self.output_type == 'R-LINE':
            self.write_rline_roadlinks(df)

        if settings.log_level_3:
            print 'TIME -> Traffic.read_road_links: {0} s'.format(round(gettime() - st_time, 2))

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
        ef_path = os.path.join(self.ef_common_path, '{0}_{1}.csv'.format(emission_type, pollutant_name))
        df = pd.read_csv(ef_path, sep=';')

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

                df = df.merge(df_hot, left_on=['CODE_HERMESv3', 'Mode'], right_on=['CODE_HERMESv3_hot', 'Mode_hot'], how='left')

                del df['Cmileage_hot'], df['Mode_hot'], df['CODE_HERMESv3_hot']

            return df
        return None

    def read_Mcorr_file(self, pollutant_name):
        try:
            df_path = os.path.join(self.ef_common_path, 'mcorr_{0}.csv'.format(pollutant_name))
            # print df_path
            df = pd.read_csv(df_path, sep=';')
            del df['Copert_V_name']
        except:
            warnings.warn('No mileage correction applied to {0}'.format(pollutant_name))
            # print 'WARNING: No mileage correction applied to {0}'.format(pollutant_name)
            return None
        return df

    @staticmethod
    def read_temperature(lon_min, lon_max, lat_min, lat_max, temp_dir, date, tstep_num, tstep_freq):
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
        :rtype: geopandas.GeoDataFrame
        """
        from netCDF4 import Dataset
        import cf_units
        from shapely.geometry import Point

        path = os.path.join(temp_dir, 'tas_{0}{1}.nc'.format(date.year, str(date.month).zfill(2)))
        print 'Getting temperature from {0}'.format(path)

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
        i_min, i_max, j_min, j_max = Traffic.find_index(lon_o, lat_o, lon_min, lon_max, lat_min, lat_max)

        # Clips the lat lons
        lon_o = lon_o[i_min:i_max]
        lat_o = lat_o[j_min:j_max]

        # From 1D to 2D
        lat = np.array([lat_o[:]] * len(lon_o[:])).T.flatten()
        lon = np.array([lon_o[:]] * len(lat_o[:])).flatten()
        del lat_o, lon_o

        # Reads the tas variable of the xone and the times needed.
        # tas = nc.variables['tas'][i_time:i_time + (self.timestep_num*self.timestep_freq): self.timestep_freq, i_min:i_max, j_min:j_max]
        tas = nc.variables['tas'][i_time:i_time + (tstep_num*tstep_freq): tstep_freq, j_min:j_max, i_min:i_max]

        nc.close()
        # That condition is fot the cases that the needed temperature is in a different NetCDF.
        while len(tas) < tstep_num:
            # TODO sum over year
            path = os.path.join(temp_dir, 'tas_{0}{1}.nc'.format(date.year, str(date.month + 1).zfill(2)))
            print 'Getting temperature from {0}'.format(path)
            nc = Dataset(path, mode='r')
            # TODO timestep_freq != 1
            i_time = 0
            new_tas = nc.variables['tas'][i_time:i_time + ((tstep_num - len(tas))*tstep_freq): tstep_freq, j_min:j_max, i_min:i_max]

            tas = np.concatenate([tas, new_tas])

            nc.close()

        # From Kelvin to Celsius degrees
        tas = (tas - 273.15).reshape((tas.shape[0], tas.shape[1] * tas.shape[2]))
        # Creates the GeoDataFrame
        df = gpd.GeoDataFrame(tas.T, geometry=[Point(xy) for xy in zip(lon, lat)])
        df.columns = ['t_{0}'.format(x) for x in df.columns.values[:-1]] + ['geometry']
        df.loc[:, 'REC'] = df.index

        return df

    def get_precipitation(self, lon_min, lon_max, lat_min, lat_max, precipitation_dir):
        from datetime import timedelta
        from netCDF4 import Dataset
        import cf_units
        from shapely.geometry import Point

        dates_to_extract = [self.date_array[0] + timedelta(hours=x - 47) for x in range(47)] + self.date_array

        # print dates_to_extract
        # sys.exit()

        path = os.path.join(precipitation_dir, 'prlr_{0}{1}.nc'.format(
            dates_to_extract[0].year, str(dates_to_extract[0].month).zfill(2)))
        print 'Getting precipitation from {0}'.format(path)

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
            path = os.path.join(precipitation_dir, 'prlr_{0}{1}.nc'.format(
                dates_to_extract[len(prlr)].year, str(dates_to_extract[len(prlr)].month).zfill(2)))
            print 'Getting precipitation from {0}'.format(path)
            nc = Dataset(path, mode='r')
            # TODO timestep_freq != 1
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

        dst = dst[-24:, :]
        dst = 1 - np.exp(- RECOVERY_RATIO * dst)
        # It is assumed that after 48 h without rain the potential emission is equal to one
        dst[dst >= (1 - np.exp(- RECOVERY_RATIO * 48))] = 1.
        dst = dst.reshape((dst.shape[0], dst.shape[1] * dst.shape[2]))
        # Creates the GeoDataFrame
        df = gpd.GeoDataFrame(dst.T, geometry=[Point(xy) for xy in zip(lon, lat)])
        df.columns = ['PR_{0}'.format(x) for x in df.columns.values[:-1]] + ['geometry']
        df.loc[:, 'REC'] = df.index

        return df

    @ staticmethod
    def find_index(lon, lat, lon_min, lon_max, lat_min, lat_max):
        # print lon, lat, lon_min, lon_max, lat_min, lat_max

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

        return i_min, i_max+1, j_min, j_max+1

    @staticmethod
    def update_fleet_value(df):
        if settings.log_level_3:
            st_time = gettime()
        else:
            st_time = None

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
                print 'link_ID: {0} aadt: {1} sum_fleet: {2}'.format(link_id, aadt, fleet_value)

        # Drop 0 values
        df = df[df['Fleet_value'] > 0]

        # Deleting unused columns
        del df['aadt'], df['PcLight'], df['PcHeavy'], df['PcMoto'], df['PcMoped'], df['Fleet_Class']
        if settings.log_level_3:
            print 'TIME -> Traffic.update_fleet_value: {0} s'.format(round(gettime() - st_time, 2))
        return df

    @staticmethod
    def calculate_timedelta(timestep_type, num_tstep, timestep_freq):
        from datetime import timedelta

        if timestep_type == 'hourly':
            delta = timedelta(hours=timestep_freq * num_tstep)
        else:
            print 'ERROR: only hourly emission permited'
            sys.exit(1)
        return pd.Timedelta(delta)

    def calculate_hourly_speed(self, df):

        # speed_aux = pd.DataFrame(self.speed_hourly.loc[self.speed_hourly['PROFILE_ID'].isin(np.unique(df['profile_id'].values))])

        df = df.merge(self.speed_hourly, left_on='profile_id', right_on='PROFILE_ID', how='left')
        df['speed'] = df.groupby('hour').apply(lambda x: x[[str(x.name)]])

        # df.loc[df['profile_id'] != 1, 'speed'] = df.groupby('hour').apply(lambda x: x[[str(x.name)]])
        # df.loc[df['profile_id'] != 1, 'speed'] = df['speed'] * df['speed_mean']
        # df.loc[df['profile_id'] == 1, 'speed'] = df['speed_mean']
        # # df.reset_index()
        return df['speed'] * df['speed_mean']

    def calculate_temporal_factor(self, df):
        import calendar

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
        df = df.merge(self.monthly_profiles, left_on='aadt_m_mn', right_on='PROFILE_ID', how='left')
        df['FM'] = df.groupby('month').apply(lambda x: x[[calendar.month_abbr[x.name].upper()]])
        # del df['JAN'], df['FEB'], df['MAR'], df['APR'], df['MAY'], df['JUN']
        # del df['JUL'], df['AUG'], df['SEP'], df['OCT'], df['NOV'], df['DEC']
        # del df['month'], df['PROFILE_ID'], df['aadt_m_mn']

        # print df

        # Daily factor
        df = df.merge(self.daily_profiles, left_on='aadt_week', right_on='PROFILE_ID', how='left')
        df['FD'] = df.groupby('week_day').apply(lambda x: x[[calendar.day_name[x.name].upper()]])
        # del df['MONDAY'], df['TUESDAY'], df['WEDNESDAY'], df['THURSDAY'], df['FRIDAY']
        # del df['SATURDAY'], df['SUNDAY']
        # del df['PROFILE_ID'], df['aadt_week']

        # print df

        # Hourly factor
        # print self.hourly_profiles
        df['hourly_profile'] = df.groupby('week_day').apply(lambda x: x[[get_hourly_id_from_weekday(x.name)]])
        df.loc[df['hourly_profile'] == '', 'hourly_profile'] = df['aadt_h_mn']

        df['hourly_profile'] = df['hourly_profile'].astype(str)
        self.hourly_profiles['PROFILE_ID'] = self.hourly_profiles['PROFILE_ID'].astype(str)

        df = df.merge(self.hourly_profiles, left_on='hourly_profile', right_on='PROFILE_ID', how='left')
        df['FH'] = df.groupby('hour').apply(lambda x: x[[str(x.name)]])

        return df['FM'] * df['FD'] * df['FH']

    def calculate_time_dependent_values(self, df, timestep_type, timestep_num, timestep_freq):
        if settings.log_level_3:
            st_time = gettime()
        else:
            st_time = None

        df.reset_index(inplace=True)
        for tstep in xrange(timestep_num):
            # Finding weekday
            # 0 -> Monday; 6 -> Sunday
            df.loc[:, 'month'] = (df['start_date'] + self.calculate_timedelta(timestep_type, tstep, timestep_freq)).dt.month
            df.loc[:, 'week_day'] = (df['start_date'] + self.calculate_timedelta(timestep_type, tstep, timestep_freq)).dt.weekday
            df.loc[:, 'hour'] = (df['start_date'] + self.calculate_timedelta(timestep_type, tstep, timestep_freq)).dt.hour

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

            # Selecting flat profile for 0 and nan's
            df.loc[df['profile_id'] == 0, 'profile_id'] = 1
            df.loc[df['profile_id'] == np.nan, 'profile_id'] = 1

            # Calculating speed by tstep
            speed_column_name = 'v_{0}'.format(tstep)
            df[speed_column_name] = self.calculate_hourly_speed(df.loc[:, ['hour', 'speed_mean', 'profile_id']])

            factor_column_name = 'f_{0}'.format(tstep)

            df.loc[:, factor_column_name] = self.calculate_temporal_factor(
                df.loc[:, ['month', 'week_day', 'hour', 'aadt_m_mn', 'aadt_week', 'aadt_h_mn', 'aadt_h_wd', 'aadt_h_sat', 'aadt_h_sun']])

        # Deleting time variables

        del df['month'], df['week_day'], df['hour'], df['profile_id'], df['speed_mean']
        del df['sp_wd'], df['sp_we'], df['index']
        del df['sp_hour_mo'], df['sp_hour_tu'], df['sp_hour_we'], df['sp_hour_th'], df['sp_hour_fr']
        del df['sp_hour_sa'], df['sp_hour_su']
        del df['aadt_m_mn'], df['aadt_h_mn'], df['aadt_h_wd'], df['aadt_h_sat'], df['aadt_h_sun'], df['aadt_week']
        del df['start_date']

        if settings.log_level_3:
            print 'TIME -> Traffic.calculate_time_dependent_values: {0} s'.format(round(gettime() - st_time, 2))

        return df

    def expand_road_links(self, timestep_type, timestep_num, timestep_freq):
        if settings.log_level_3:
            st_time = gettime()
        else:
            st_time = None
        # Expands each road link by any vehicle type that the selected road link has.
        df_list = []
        road_link_aux = self.road_links.copy()

        # if self.resuspension_correction:
        #     road_link_aux.loc[:, 'centroid'] = road_link_aux['geometry'].centroid
        #     link_lons = road_link_aux['geometry'].centroid.x
        #     link_lats = road_link_aux['geometry'].centroid.y
        #     p_factor = self.get_precipitation(link_lons.min(), link_lons.max(), link_lats.min(), link_lats.max(), self.precipitation_path)
        #
        #     unary_union = p_factor.unary_union
        #     road_link_aux['REC'] = road_link_aux.apply(self.nearest, geom_union=unary_union, df1=road_link_aux,
        #                                                df2=p_factor, geom1_col='centroid', src_column='REC', axis=1)
        #     del road_link_aux['centroid'], p_factor['geometry']
        #
        #     road_link_aux = road_link_aux.merge(p_factor, left_on='REC', right_on='REC', how='left')
        #
        #     del road_link_aux['REC']

        del road_link_aux['geometry']
        for zone, compo_df in road_link_aux.groupby('fleet_comp'):
            fleet = self.find_fleet(zone)
            df_aux = pd.merge(compo_df, fleet, how='left', on='fleet_comp')
            df_list.append(df_aux)

        df = pd.concat(df_list, ignore_index=True)

        del df['fleet_comp']

        # df.to_csv('/home/Earth/ctena/Models/HERMESv3/OUT/2_pre_expanded.csv')
        df = self.update_fleet_value(df)
        df = self.calculate_time_dependent_values(df, timestep_type, timestep_num, timestep_freq)

        if settings.log_level_3:
            print 'TIME -> Traffic.expand_road_links: {0} s'.format(round(gettime() - st_time, 2))

        return df

    def find_fleet(self, zone):

        # print self.fleet_compo
        try:
            fleet = self.fleet_compo[['Code', 'Class', zone]]
        except KeyError as e:
            raise KeyError(e.message + ' of the fleet_compo file')
        fleet.columns = ['Fleet_Code', 'Fleet_Class', 'Fleet_value']

        fleet = fleet[fleet['Fleet_value'] > 0]

        fleet['fleet_comp'] = zone

        return fleet

    def calculate_hot(self):
        if settings.log_level_3:
            st_time = gettime()
        else:
            st_time = None
        expanded_aux = self.expanded.copy().reset_index()

        for pollutant in self.pollutant_list:
            if pollutant != 'nh3':

                ef_code_slope_road, ef_code_slope, ef_code_road, ef_code = self.read_ef('hot', pollutant)

                df_code_slope_road = expanded_aux.merge(ef_code_slope_road, left_on=['Fleet_Code', 'road_grad', 'Road_type'], right_on=['CODE_HERMESv3', 'Road.Slope', 'Mode'], how='inner')
                df_code_slope = expanded_aux.merge(ef_code_slope, left_on=['Fleet_Code', 'road_grad'], right_on=['CODE_HERMESv3', 'Road.Slope'], how='inner')
                df_code_road = expanded_aux.merge(ef_code_road, left_on=['Fleet_Code', 'Road_type'], right_on=['CODE_HERMESv3', 'Mode'], how='inner')
                df_code = expanded_aux.merge(ef_code, left_on=['Fleet_Code'], right_on=['CODE_HERMESv3'], how='inner')

                del ef_code_slope_road, ef_code_slope, ef_code_road, ef_code

                expanded_aux = pd.concat([df_code_slope_road, df_code_slope, df_code_road, df_code])

                del expanded_aux['CODE_HERMESv3'], expanded_aux['Road.Slope'], expanded_aux['Mode']
                try:
                    del expanded_aux['index']
                except:
                    pass
            else:
                ef_code_road = self.read_ef('hot', pollutant)
                expanded_aux = expanded_aux.merge(ef_code_road, left_on=['Fleet_Code', 'Road_type'], right_on=['CODE_HERMESv3', 'Mode'], how='inner')

                del expanded_aux['CODE_HERMESv3'], expanded_aux['Mode']

            # Warnings and Errors
            original_ef_profile = self.expanded['Fleet_Code'].unique()
            calculated_ef_profiles = expanded_aux['Fleet_Code'].unique()
            resta_1 = [item for item in original_ef_profile if item not in calculated_ef_profiles]  # Warining
            resta_2 = [item for item in calculated_ef_profiles if item not in original_ef_profile]  # Error

            if len(resta_1) > 0:
                warnings.warn('Exists some fleet codes that not appear on the EF file: {0}'.format(resta_1), Warning)
            if len(resta_2) > 0:
                raise ImportError('Exists some fleet codes duplicateds on the EF file: {0}'.format(resta_2))

            m_corr = self.read_Mcorr_file(pollutant)
            if m_corr is not None:
                expanded_aux = expanded_aux.merge(m_corr, left_on='Fleet_Code', right_on='CODE_HERMESv3', how='left')
                del expanded_aux['CODE_HERMESv3']
            # print expanded_aux
            for tstep in xrange(self.timestep_num):
                ef_name = 'ef_{0}_{1}'.format(pollutant, tstep)
                p_column = '{0}_{1}'.format(pollutant, tstep)
                if pollutant != 'nh3':
                    expanded_aux['v_aux'] = expanded_aux['v_{0}'.format(tstep)]
                    # print tstep, expanded_aux.loc[:, ['v_aux', 'Min.Speed']]
                    # print len(expanded_aux.loc[expanded_aux['v_aux'] < expanded_aux['Min.Speed'], 'v_aux'])
                    # print expanded_aux
                    # print expanded_aux.loc[expanded_aux['v_aux'] < expanded_aux['Min.Speed'], 'v_aux']
                    # print expanded_aux.loc[expanded_aux['v_aux'] < expanded_aux['Min.Speed'], 'Min.Speed'].index
                    expanded_aux.loc[expanded_aux['v_aux'] < expanded_aux['Min.Speed'], 'v_aux'] = expanded_aux.loc[expanded_aux['v_aux'] < expanded_aux['Min.Speed'], 'Min.Speed']
                    expanded_aux.loc[expanded_aux['v_aux'] > expanded_aux['Max.Speed'], 'v_aux'] = expanded_aux.loc[expanded_aux['v_aux'] > expanded_aux['Max.Speed'], 'Max.Speed']

                    # EF
                    expanded_aux.loc[:, ef_name] = ((expanded_aux.Alpha * expanded_aux.v_aux**2 + expanded_aux.Beta*expanded_aux.v_aux + expanded_aux.Gamma + (expanded_aux.Delta/expanded_aux.v_aux))/(expanded_aux.Epsilon*expanded_aux.v_aux**2 + expanded_aux.Zita*expanded_aux.v_aux + expanded_aux.Hta))*(1 - expanded_aux.RF)*(expanded_aux.PF*expanded_aux['T']/expanded_aux.Q)
                else:
                    expanded_aux.loc[:, ef_name] = ((expanded_aux['a'] * expanded_aux['Cmileage'] + expanded_aux['b'])*(expanded_aux['EFbase'] * expanded_aux['TF']))/1000


                # Mcorr
                # m_corr = self.read_Mcorr_file(pollutant)
                if m_corr is not None:
                    # expanded_aux = expanded_aux.merge(m_corr)

                    expanded_aux.loc[expanded_aux['v_aux'] <= 19., 'Mcorr'] = expanded_aux.A_urban*expanded_aux['M'] + expanded_aux.B_urban
                    expanded_aux.loc[expanded_aux['v_aux'] >= 63., 'Mcorr'] = expanded_aux.A_road * expanded_aux['M'] + expanded_aux.B_road
                    expanded_aux.loc[(expanded_aux['v_aux'] > 19.) & (expanded_aux['v_aux'] < 63.), 'Mcorr'] = (expanded_aux.A_urban*expanded_aux['M'] + expanded_aux.B_urban) +((expanded_aux.v_aux - 19)*((expanded_aux.A_road * expanded_aux['M'] + expanded_aux.B_road) - (expanded_aux.A_urban*expanded_aux['M'] + expanded_aux.B_urban)))/44.
                    expanded_aux.loc[expanded_aux['Mcorr'].isnull(), 'Mcorr'] = 1
                else:
                    expanded_aux.loc[:, 'Mcorr'] = 1

                # Full formula
                expanded_aux.loc[:, p_column] = expanded_aux['Fleet_value'] * expanded_aux[ef_name] * expanded_aux['Mcorr'] * expanded_aux['f_{0}'.format(tstep)]
                # expanded_aux.to_csv('/home/Earth/ctena/Models/HERMESv3/OUT/hot_expanded_{0}_{1}.csv'.format(pollutant,tstep))
                del expanded_aux[ef_name], expanded_aux['Mcorr']

            if pollutant != 'nh3':
                del expanded_aux['v_aux']
                del expanded_aux['Min.Speed'], expanded_aux['Max.Speed'], expanded_aux['Alpha'], expanded_aux['Beta']
                del expanded_aux['Gamma'], expanded_aux['Delta'], expanded_aux['Epsilon'], expanded_aux['Zita']
                del expanded_aux['Hta'], expanded_aux['RF'], expanded_aux['Q'], expanded_aux['PF'], expanded_aux['T']
            else:
                del expanded_aux['a'], expanded_aux['Cmileage'], expanded_aux['b'], expanded_aux['EFbase'], expanded_aux['TF']

            if m_corr is not None:
                del expanded_aux['A_urban'], expanded_aux['B_urban'], expanded_aux['A_road'], expanded_aux['B_road'], expanded_aux['M']

            # del expanded_aux['Fleet_value'], expanded_aux[ef_name], expanded_aux['Mcorr'], expanded_aux['f_{0}'.format(tstep)]

        del expanded_aux['road_grad']

        for tstep in xrange(self.timestep_num):
            del expanded_aux['f_{0}'.format(tstep)]

        if settings.log_level_3:
            print 'TIME -> Traffic.calculate_hot: {0} s'.format(round(gettime() - st_time, 2))
        return expanded_aux

    def calculate_cold(self, hot_expanded):
        if settings.log_level_3:
            st_time = gettime()
        else:
            st_time = None
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

        temperature = self.read_temperature(link_lons.min(), link_lons.max(), link_lats.min(), link_lats.max(), self.temp_common_path, self.starting_date, self.timestep_num, self.timestep_freq)

        print 'Nearest time ...',
        st_time = gettime()
        unary_union = temperature.unary_union
        cold_links['REC'] = cold_links.apply(self.nearest, geom_union=unary_union, df1=cold_links, df2=temperature,
                                             geom1_col='centroid', src_column='REC', axis=1)
        del cold_links['geometry'], cold_links['centroid'], temperature['geometry']

        cold_links = cold_links.merge(temperature, left_on='REC', right_on='REC', how='left')

        del cold_links['REC']

        print ' {0} s'.format(round(gettime() - st_time, 2))

        c_expanded = hot_expanded.merge(cold_links, left_on='Link_ID', right_on='Link_ID', how='left')

        # cold_df = c_expanded.loc[:, ['Link_ID', 'Fleet_Code']]
        # cold_df.set_index('Link_ID', inplace=True)
        df_list = []
        for pollutant in self.pollutant_list:

            ef_cold = self.read_ef('cold', pollutant)

            if pollutant != 'nh3':
                ef_cold.loc[ef_cold['Tmin'].isnull(), 'Tmin'] = -999
                ef_cold.loc[ef_cold['Tmax'].isnull(), 'Tmax'] = 999
                ef_cold.loc[ef_cold['Min.Speed'].isnull(), 'Min.Speed'] = -999
                ef_cold.loc[ef_cold['Max.Speed'].isnull(), 'Max.Speed'] = 999

            c_expanded_p = c_expanded.merge(ef_cold, left_on=['Fleet_Code', 'Road_type'],
                                            right_on=['CODE_HERMESv3', 'Mode'], how='inner')
            cold_exp_p_aux = c_expanded_p.copy()

            del cold_exp_p_aux['index_right_x'], cold_exp_p_aux['Road_type'], cold_exp_p_aux['Fleet_value']
            del cold_exp_p_aux['CODE_HERMESv3']
            # df_list_aux = []
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
                cold_exp_p_aux.loc[:, 'Beta'] = (0.6474 - 0.02545*cold_exp_p_aux['ltrip'] - (0.00974 - 0.000385*cold_exp_p_aux['ltrip'])*cold_exp_p_aux[t_column])*cold_exp_p_aux['bc']
                if pollutant != 'nh3':
                    cold_exp_p_aux.loc[:, 'cold_hot'] = cold_exp_p_aux['A'] * cold_exp_p_aux[v_column] + cold_exp_p_aux['B'] * cold_exp_p_aux[t_column] + cold_exp_p_aux['C']

                else:
                    cold_exp_p_aux.loc[:, 'cold_hot'] = ((cold_exp_p_aux['a'] * cold_exp_p_aux['Cmileage'] + cold_exp_p_aux['b']) * cold_exp_p_aux['EFbase'] * cold_exp_p_aux['TF'])/((cold_exp_p_aux['a_hot'] * cold_exp_p_aux['Cmileage'] + cold_exp_p_aux['b_hot']) * cold_exp_p_aux['EFbase_hot'] * cold_exp_p_aux['TF_hot'])
                cold_exp_p_aux.loc[cold_exp_p_aux['cold_hot'] < 1, 'cold_hot'] = 1

                # Formula Cold emissions
                cold_exp_p_aux.loc[:, p_column] = cold_exp_p_aux[p_column] * cold_exp_p_aux['Beta'] * (cold_exp_p_aux['cold_hot'] - 1)
                # print pollutant
                df_list.append((cold_exp_p_aux.loc[:, ['Link_ID', 'Fleet_Code', p_column]]).set_index(['Link_ID', 'Fleet_Code']))

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
                    except:
                        error_fleet_code.append(o)
            raise IndexError('There are duplicated values for {0} codes in the cold EF files.'.format(error_fleet_code))

        for tstep in xrange(self.timestep_num):
            if 'pm' in self.pollutant_list:
                cold_df.loc[:, 'pm10_{0}'.format(tstep)] = cold_df['pm_{0}'.format(tstep)]
                cold_df.loc[:, 'pm25_{0}'.format(tstep)] = cold_df['pm_{0}'.format(tstep)]
                del cold_df['pm_{0}'.format(tstep)]

            if 'voc' in self.pollutant_list and 'ch4' in self.pollutant_list:
                cold_df.loc[:, 'nmvoc_{0}'.format(tstep)] = cold_df['voc_{0}'.format(tstep)] - cold_df['ch4_{0}'.format(tstep)]
                del cold_df['voc_{0}'.format(tstep)]
            else:
                warnings.warn("nmvoc emissions cannot be estimated because voc or ch4 are not selected in the pollutant list.")

        cold_df = self.speciate_traffic(cold_df, self.hot_cold_speciation)

        # del cold_df['Fleet_Code']
        #
        # cold_df = cold_df.groupby(['tstep', 'Link_ID']).sum()

        if settings.log_level_3:
            print 'TIME -> Traffic.calculate_cold: {0} s'.format(round(gettime() - st_time, 2))

        return cold_df

    def compact_hot_expanded(self, expanded):
        if settings.log_level_3:
            st_time = gettime()
        else:
            st_time = None
        columns_to_delete = ['Road_type', 'Fleet_value'] + ['v_{0}'.format(x) for x in xrange(self.timestep_num)]
        for column_name in columns_to_delete:
            del expanded[column_name]

        for tstep in xrange(self.timestep_num):
            if 'pm' in self.pollutant_list:
                expanded.loc[:, 'pm10_{0}'.format(tstep)] = expanded['pm_{0}'.format(tstep)]
                expanded.loc[:, 'pm25_{0}'.format(tstep)] = expanded['pm_{0}'.format(tstep)]
                del expanded['pm_{0}'.format(tstep)]

            if 'voc' in self.pollutant_list and 'ch4' in self.pollutant_list:
                expanded.loc[:, 'nmvoc_{0}'.format(tstep)] = expanded['voc_{0}'.format(tstep)] - expanded['ch4_{0}'.format(tstep)]
                del expanded['voc_{0}'.format(tstep)]
            else:
                warnings.warn(
                    "nmvoc emissions cannot be estimated because voc or ch4 are not selected in the pollutant list.")

        #expanded = self.speciate_traffic_old(expanded, self.hot_cold_speciation)
        compacted = self.speciate_traffic(expanded, self.hot_cold_speciation)

        # del expanded['Fleet_Code']
        #
        # df = expanded.groupby(['tstep', 'Link_ID']).sum()

        if settings.log_level_3:
            print 'TIME -> Traffic.compact_hot_expanded: {0} s'.format(round(gettime() - st_time, 2))

        return compacted

    def calculate_tyre_wear(self):
        if settings.log_level_3:
            st_time = gettime()
        else:
            st_time = None
        pollutants = ['pm']
        for pollutant in pollutants:
            ef_tyre = self.read_ef('tyre', pollutant)
            df = self.expanded.merge(ef_tyre, left_on='Fleet_Code', right_on='CODE_HERMESv3', how='inner')
            del df['road_grad'], df['Road_type'], df['CODE_HERMESv3']
            for tstep in xrange(self.timestep_num):
                p_column = '{0}_{1}'.format(pollutant, tstep)
                f_column = 'f_{0}'.format(tstep)
                v_column = 'v_{0}'.format(tstep)
                df.loc[df[v_column] < 40, p_column] = df['Fleet_value'] * df['EFbase'] * df[f_column] * 1.39
                df.loc[(df[v_column] >= 40) & (df[v_column] <= 90), p_column] = df['Fleet_value'] * df['EFbase'] * df[f_column] * (-0.00974* df[v_column]+1.78)
                df.loc[df[v_column] > 90, p_column] = df['Fleet_value'] * df['EFbase'] * df[f_column] * 0.902

                # from PM to PM10 & PM2.5
                if pollutant == 'pm':
                    df.loc[:, 'pm10_{0}'.format(tstep)] = df[p_column] * 0.6
                    df.loc[:, 'pm25_{0}'.format(tstep)] = df[p_column] * 0.42
                    del df[p_column]

        # Cleaning df
        columns_to_delete = ['f_{0}'.format(x) for x in xrange(self.timestep_num)] + ['v_{0}'.format(x) for x in xrange(self.timestep_num)]
        columns_to_delete += ['Fleet_value', 'EFbase']
        for column in columns_to_delete:
            del df[column]

        df = self.speciate_traffic(df, self.tyre_speciation)

        #del df['Fleet_Code']

        if settings.log_level_3:
            print 'TIME -> Traffic.calculate_tyre_wear: {0} s'.format(round(gettime() - st_time, 2))

        return df #.groupby(['tstep', 'Link_ID']).sum()

    def calculate_brake_wear(self):
        if settings.log_level_3:
            st_time = gettime()
        else:
            st_time = None
        pollutants = ['pm']
        for pollutant in pollutants:
            ef_tyre = self.read_ef('brake', pollutant)
            df = self.expanded.merge(ef_tyre, left_on='Fleet_Code', right_on='CODE_HERMESv3', how='inner')
            del df['road_grad'], df['Road_type'], df['CODE_HERMESv3']
            for tstep in xrange(self.timestep_num):
                p_column = '{0}_{1}'.format(pollutant, tstep)
                f_column = 'f_{0}'.format(tstep)
                v_column = 'v_{0}'.format(tstep)
                df.loc[df[v_column] < 40, p_column] = df['Fleet_value'] * df['EFbase'] * df[f_column] * 1.67
                df.loc[(df[v_column] >= 40) & (df[v_column] <= 95), p_column] = df['Fleet_value'] * df['EFbase'] * df[f_column] * (-0.027 * df[v_column] + 2.75)
                df.loc[df[v_column] > 95, p_column] = df['Fleet_value'] * df['EFbase'] * df[f_column] * 0.185

                # from PM to PM10 & PM2.5
                if pollutant == 'pm':
                    df.loc[:, 'pm10_{0}'.format(tstep)] = df[p_column] * 0.98
                    df.loc[:, 'pm25_{0}'.format(tstep)] = df[p_column] * 0.39
                    del df[p_column]

        # Cleaning df
        columns_to_delete = ['f_{0}'.format(x) for x in xrange(self.timestep_num)] + ['v_{0}'.format(x) for x in xrange(self.timestep_num)]
        columns_to_delete += ['Fleet_value', 'EFbase']
        for column in columns_to_delete:
            del df[column]

        df = self.speciate_traffic(df, self.brake_speciation)

        # del df['Fleet_Code']

        if settings.log_level_3:
            print 'TIME -> Traffic.calculate_brake_wear: {0} s'.format(round(gettime() - st_time, 2))

        return df #.groupby(['tstep', 'Link_ID']).sum()

    def calculate_road_wear(self):
        if settings.log_level_3:
            st_time = gettime()
        else:
            st_time = None
        pollutants = ['pm']
        for pollutant in pollutants:
            ef_tyre = self.read_ef('road', pollutant)
            df = self.expanded.merge(ef_tyre, left_on='Fleet_Code', right_on='CODE_HERMESv3', how='inner')
            del df['road_grad'], df['Road_type'], df['CODE_HERMESv3']
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
        columns_to_delete = ['f_{0}'.format(x) for x in xrange(self.timestep_num)] + ['v_{0}'.format(x) for x in xrange(self.timestep_num)]
        columns_to_delete += ['Fleet_value', 'EFbase']
        for column in columns_to_delete:
            del df[column]

        df = self.speciate_traffic(df, self.road_speciation)

        # del df['Fleet_Code']

        if settings.log_level_3:
            print 'TIME -> Traffic.calculate_road_wear: {0} s'.format(round(gettime() - st_time, 2))

        return df  # .groupby(['tstep', 'Link_ID']).sum()

    def calculate_resuspension(self):
        if settings.log_level_3:
            st_time = gettime()
        else:
            st_time = None

        if self.resuspension_correction:
            # print self.road_links
            # link_lons = self.road_links['geometry'].centroid.x
            # link_lats = self.road_links['geometry'].centroid.y
            # p_factor = self.get_precipitation(link_lons.min(), link_lons.max(), link_lats.min(), link_lats.max(), self.precipitation_path)

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
            df = self.expanded.merge(ef_tyre, left_on='Fleet_Code', right_on='CODE_HERMESv3', how='inner')
            if self.resuspension_correction:
                df = df.merge(road_link_aux, left_on='Link_ID', right_on='Link_ID', how='left')

            del df['road_grad'], df['Road_type'], df['CODE_HERMESv3']
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

        # del df['Fleet_Code']

        if settings.log_level_3:
            print 'TIME -> Traffic.calculate_resuspension: {0} s'.format(round(gettime() - st_time, 2))

        return df  # .groupby(['tstep', 'Link_ID']).sum()

    def transform_df(self, df):

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
            for p_name in pollutants_renamed:
                del df[p_name]

        df = pd.concat(df_list, ignore_index=True)
        return df

    def speciate_traffic_old(self, df, speciation):
        df_map = pd.read_csv(self.speciation_map, sep=';')

        if settings.log_level_3:
            st_time = gettime()
        else:
            st_time = None

        df = self.transform_df(df)

        speciation = pd.read_csv(speciation, sep=';')
        del speciation['Copert_V_name']
        in_p_list = list(df.columns.values)
        in_columns = ['Link_ID', 'Fleet_Code', 'tstep']
        for in_col in in_columns:
            try:
                in_p_list.remove(in_col)
            except:
                print 'ERROR', in_col
        for in_col in in_p_list:
            df.rename(columns={in_col: 'old_{0}'.format(in_col)}, inplace=True)

        out_p_list = list(speciation.columns.values)
        out_p_list.remove('CODE_HERMESv3')
        for p in out_p_list:

            speciation.rename(columns={p: 'f_{0}'.format(p)}, inplace=True)

        df = df.merge(speciation, left_on='Fleet_Code', right_on='CODE_HERMESv3', how='left')
        try:
            del df['CODE_HERMESv3']
            del df['index_right']
        except:
            pass

        # print df_map.columns.values
        for p in out_p_list:
            if p == 'pmc':
                df.loc[:, p] = df['old_pm10'] - df['old_pm25']
                if self.output_type == 'R-LINE':
                    # from g/km.h to g/m.s
                    df.loc[:, p] = df.loc[:, p] / (1000 * 3600)
                elif self.output_type == 'CMAQ':
                    # from g/km.h to mol/km.s
                    df.loc[:, p] = df.loc[:, p] / 3600
                elif self.output_type == 'MONARCH':
                    # from g/km.h to Kg/km.s
                    df.loc[:, p] = df.loc[:, p] / (1000 * 3600)
            else:
                try:
                    in_p = df_map.loc[df_map['dst'] == p, 'src'].values[0]
                except IndexError:
                    raise ValueError('The pollutant {0} does not appear in the traffic_speciation_map file'.format(p))

                if in_p is not np.nan:
                    if in_p != 0:
                        df.loc[:, p] = df['old_{0}'.format(in_p)].multiply(df['f_{0}'.format(p)])
                        try:
                            mol_w = self.molecular_weigths.loc[self.molecular_weigths['Specie'] == in_p, 'MW'].values[0]
                        except IndexError:
                            raise AttributeError('{0} not found in the molecular weights file.'.format(in_p))

                        if self.output_type == 'R-LINE':
                            # from g/km.h to g/m.s
                            df.loc[:, p] = df.loc[:, p] / (1000 * 3600)
                        elif self.output_type == 'CMAQ':
                            # from g/km.h to mol/km.s or g/km.s (aerosols)
                            df.loc[:, p] = df.loc[:, p] / (3600 * mol_w)
                        elif self.output_type == 'MONARCH':
                            if p.lower() in aerosols:
                                # from g/km.h to kg/km.s
                                df.loc[:, p] = df.loc[:, p] / (1000 * 3600 * mol_w)
                            else:
                                # from g/km.h to mol/km.s
                                df.loc[:, p] = df.loc[:, p] / (3600 * mol_w)

                    else:
                        df.loc[:, p] = 0

        tot_cols = list(df.columns.values)
        for col in tot_cols:
            if col.startswith('old_') or col.startswith('f_'):
                del df[col]

        if settings.log_level_3:
            print 'TIME -> Traffic.speciate_traffic: {0} s'.format(round(gettime() - st_time, 2))
        return df

    def speciate_traffic(self, df, speciation):

        if settings.log_level_3:
            st_time = gettime()
        else:
            st_time = None

        # Reads maps dst to src pollutants
        map = pd.read_csv(self.speciation_map, sep=';')

        # Reads speciation profile
        speciation = pd.read_csv(speciation, sep=';')
        del speciation['Copert_V_name']

        # Transform dataset into timestep rows instead of timestep columns
        df = self.transform_df(df)

        # Makes input list from input dataframe
        in_list = list(df.columns.values)
        in_columns = ['Link_ID', 'Fleet_Code', 'tstep']
        for in_col in in_columns:
            try:
                in_list.remove(in_col)
            except:
                print 'ERROR', in_col

        df_out_list = []

        # PMC
        if not set(speciation.columns.values).isdisjoint(pmc_list):
            out_p = set(speciation.columns.values).intersection(pmc_list).pop()
            speciation_by_in_p = speciation.loc[:, [out_p] + ['CODE_HERMESv3']]

            speciation_by_in_p.rename(columns={out_p: 'f_{0}'.format(out_p)}, inplace=True)
            df_aux = df.loc[:, ['pm10', 'pm25', 'Fleet_Code', 'tstep', 'Link_ID']]
            df_aux = df_aux.merge(speciation_by_in_p, left_on='Fleet_Code', right_on='CODE_HERMESv3', how='left')
            try:
                del df['CODE_HERMESv3']
                del df['index_right']
            except:
                pass

            df_aux.loc[:, out_p] = df_aux['pm10'] - df_aux['pm25']
            if self.output_type == 'R-LINE':
                # from g/km.h to g/m.s
                df_aux.loc[:, out_p] = df_aux.loc[:, out_p] / (1000 * 3600)
            elif self.output_type == 'CMAQ':
                # from g/km.h to mol/km.s
                df_aux.loc[:, out_p] = df_aux.loc[:, out_p] / 3600
            elif self.output_type == 'MONARCH':
                # from g/km.h to Kg/km.s
                df_aux.loc[:, out_p] = df_aux.loc[:, out_p] / (1000 * 3600)

            df_out_list.append(df_aux.loc[:, [out_p] + ['tstep', 'Link_ID']].groupby(['tstep', 'Link_ID']).sum())
            del df_aux[out_p]

        for in_p in in_list:
            # Get output list involved on that input pollutant
            out_list = list(map.loc[map['src'] == in_p, 'dst'].unique())
            # Selecting only necessary speciation profiles
            speciation_by_in_p = speciation.loc[:, out_list + ['CODE_HERMESv3']]

            # Adding "f_" in the formula column names
            for p in out_list:
                speciation_by_in_p.rename(columns={p: 'f_{0}'.format(p)}, inplace=True)
            # Getting a slice of the full dataset to be merged
            df_aux = df.loc[:, [in_p] + ['Fleet_Code', 'tstep', 'Link_ID']]
            df_aux = df_aux.merge(speciation_by_in_p, left_on='Fleet_Code', right_on='CODE_HERMESv3', how='left')
            try:
                # Cleaning dataframe
                del df['CODE_HERMESv3']
                del df['index_right']
            except:
                pass
            # Renaming pollutant columns by adding "old_" to the beginning.
            df_aux.rename(columns={in_p: 'old_{0}'.format(in_p)}, inplace=True)
            for p in out_list:
                if in_p is not np.nan:
                    if in_p != 0:
                        df_aux.loc[:, p] = df_aux['old_{0}'.format(in_p)].multiply(df_aux['f_{0}'.format(p)])
                        try:
                            mol_w = self.molecular_weigths.loc[self.molecular_weigths['Specie'] == in_p, 'MW'].values[0]
                        except IndexError:
                            raise AttributeError('{0} not found in the molecular weights file.'.format(in_p))

                        if self.output_type == 'R-LINE':
                            # from g/km.h to g/m.s
                            df_aux.loc[:, p] = df_aux.loc[:, p] / (1000 * 3600)
                        elif self.output_type == 'CMAQ':
                            # from g/km.h to mol/km.s or g/km.s (aerosols)
                            df_aux.loc[:, p] = df_aux.loc[:, p] / (3600 * mol_w)
                        elif self.output_type == 'MONARCH':
                            if p.lower() in aerosols:
                                # from g/km.h to kg/km.s
                                df_aux.loc[:, p] = df_aux.loc[:, p] / (1000 * 3600 * mol_w)
                            else:
                                # from g/km.h to mol/km.s
                                df_aux.loc[:, p] = df_aux.loc[:, p] / (3600 * mol_w)
                    else:
                        df_aux.loc[:, p] = 0

                df_out_list.append(df_aux.loc[:, [p] + ['tstep', 'Link_ID']].groupby(['tstep', 'Link_ID']).sum())
                del df_aux[p]
            del df_aux
            del df[in_p]

        df_out = pd.concat(df_out_list, axis=1)
        return df_out

    def calculate_traffic_line_emissions(self, do_hot=True, do_cold=True, do_tyre_wear=True, do_brake_wear=True, do_road_wear=True,
                                         do_resuspension=True, do_evaporative=False, do_other_cities=False):
        df_accum = pd.DataFrame()

        if do_hot:
            df_accum = pd.concat([df_accum, self.compact_hot_expanded(self.calculate_hot())]).groupby(['tstep', 'Link_ID']).sum()
            # df_accum = pd.concat([df_accum, self.compact_hot_expanded(self.calculate_hot())]).groupby(['tstep', 'Link_ID']).sum()
        if do_cold:
            df_accum = pd.concat([df_accum, self.calculate_cold(self.calculate_hot())]).groupby(['tstep', 'Link_ID']).sum()
        if do_tyre_wear:
            df_accum = pd.concat([df_accum, self.calculate_tyre_wear()]).groupby(['tstep', 'Link_ID']).sum()
        if do_brake_wear:
            df_accum = pd.concat([df_accum, self.calculate_brake_wear()]).groupby(['tstep', 'Link_ID']).sum()
        if do_road_wear:
            df_accum = pd.concat([df_accum, self.calculate_road_wear()]).groupby(['tstep', 'Link_ID']).sum()
        if do_resuspension:
            df_accum = pd.concat([df_accum, self.calculate_resuspension()]).groupby(['tstep', 'Link_ID']).sum()

        df_accum = df_accum.reset_index().merge(self.road_links.loc[:, ['Link_ID', 'geometry']], left_on='Link_ID', right_on='Link_ID', how='left')
        df_accum = gpd.GeoDataFrame(df_accum, crs=self.crs)
        df_accum.set_index(['Link_ID', 'tstep'], inplace=True)
        return df_accum

    def links_to_grid(self, link_emissions, grid_shape):

        if settings.log_level_3:
            st_time = gettime()
        else:
            st_time = None
        link_emissions.reset_index(inplace=True)
        if not os.path.exists(self.link_to_grid_csv):
            link_emissions_aux = link_emissions.loc[link_emissions['tstep'] == 0, :]
            link_emissions_aux = link_emissions_aux.to_crs(grid_shape.crs)

            link_emissions_aux = gpd.sjoin(link_emissions_aux, grid_shape, how="inner", op='intersects')
            link_emissions_aux = link_emissions_aux.loc[:, ['Link_ID', 'geometry', 'FID']]
            link_emissions_aux = link_emissions_aux.merge(grid_shape.loc[:, ['FID', 'geometry']], left_on='FID', right_on='FID', how='left')

            length_list = []
            link_id_list = []
            fid_list = []
            count = 1
            for i, line in link_emissions_aux.iterrows():
                # print "{0}/{1}".format(count, len(link_emissions_aux))
                count += 1
                aux = line.get('geometry_x').intersection(line.get('geometry_y'))
                if not aux.is_empty:
                    link_id_list.append(line.get('Link_ID'))
                    fid_list.append(line.get('FID'))
                    length_list.append(aux.length / 1000)

            link_grid = pd.DataFrame({'Link_ID': link_id_list, 'FID': fid_list, 'length': length_list})

            # link_grid.to_csv(self.link_to_grid_csv)
        else:
            link_grid = pd.read_csv(self.link_to_grid_csv)

        del grid_shape['geometry'], link_emissions['geometry']

        link_grid = link_grid.merge(link_emissions, left_on='Link_ID', right_on='Link_ID')
        try:
            del link_grid['Unnamed: 0']
        except:
            pass
        del link_grid['Link_ID']

        # print link_grid

        cols_to_update = list(link_grid.columns.values)
        cols_to_update.remove('length')
        cols_to_update.remove('tstep')
        cols_to_update.remove('FID')
        for col in cols_to_update:
            # print col
            link_grid.loc[:, col] = link_grid[col] * link_grid['length']
        del link_grid['length']

        link_grid = link_grid.groupby(['tstep', 'FID']).sum()
        link_grid.reset_index(inplace=True)

        link_grid_list = settings.comm.gather(link_grid, root=0)
        if settings.rank == 0:
            link_grid = pd.concat(link_grid_list)
            link_grid = link_grid.groupby(['tstep', 'FID']).sum()
            # link_grid.sort_index(inplace=True)
            link_grid.reset_index(inplace=True)

            emission_list = []
            out_poll_names = list(link_grid.columns.values)
            out_poll_names.remove('tstep')
            out_poll_names.remove('FID')

            for p in out_poll_names:
                # print p
                data = np.zeros((self.timestep_num, len(grid_shape)))
                for tstep in xrange(self.timestep_num):
                    data[tstep, link_grid.loc[link_grid['tstep'] == tstep, 'FID']] = \
                        link_grid.loc[link_grid['tstep'] == tstep, p]
                    # data[tstep, link_grid.index] = link_grid['{0}_{1}'.format(p, tstep)]
                # print p, data.sum()
                # TODO Check units MARC
                dict_aux = {
                    'name': p,
                    'units': None,
                    'data': data
                }

                if self.output_type == 'R-LINE':
                    # from g/km.h to g/m.s
                    pass
                elif self.output_type == 'CMAQ':
                    # from g/km.h to mol/km.s
                    if p.lower() in aerosols:
                        dict_aux['units'] = '0.001 kg.s-1'
                    else:
                        dict_aux['units'] = 'kat'
                elif self.output_type == 'MONARCH':
                    if p.lower() in aerosols:
                        dict_aux['units'] = 'kg.s-1'
                    else:
                        dict_aux['units'] = 'kat'
                emission_list.append(dict_aux)
            if settings.log_level_3:
                print 'TIME -> Traffic.links_to_grid: {0} s'.format(round(gettime() - st_time, 2))

            return emission_list
        else:
            return None

    def links_to_grid_new(self, link_emissions, grid_shape):

        if settings.log_level_3:
            st_time = gettime()
        else:
            st_time = None


        if not os.path.exists(self.link_to_grid_csv):
            link_emissions.reset_index(inplace=True)
            link_emissions_aux = link_emissions.loc[link_emissions['tstep'] == 0, :]
            link_emissions_aux = link_emissions_aux.to_crs(grid_shape.crs)

            link_emissions_aux = gpd.sjoin(link_emissions_aux, grid_shape, how="inner", op='intersects')
            link_emissions_aux = link_emissions_aux.loc[:, ['Link_ID', 'geometry', 'FID']]
            link_emissions_aux = link_emissions_aux.merge(grid_shape.loc[:, ['FID', 'geometry']], left_on='FID', right_on='FID', how='left')

            length_list = []
            link_id_list = []
            fid_list = []
            count = 1
            for i, line in link_emissions_aux.iterrows():
                # print "{0}/{1}".format(count, len(link_emissions_aux))
                count += 1
                aux = line.get('geometry_x').intersection(line.get('geometry_y'))
                if not aux.is_empty:
                    link_id_list.append(line.get('Link_ID'))
                    fid_list.append(line.get('FID'))
                    length_list.append(aux.length / 1000)

            link_grid = pd.DataFrame({'Link_ID': link_id_list, 'FID': fid_list, 'length': length_list})

            # link_grid.to_csv(self.link_to_grid_csv)
        else:
            link_grid = pd.read_csv(self.link_to_grid_csv)

        del grid_shape['geometry'], link_emissions['geometry']

        link_grid = link_grid.merge(link_emissions, left_on='Link_ID', right_on='Link_ID')

        try:
            del link_grid['Unnamed: 0']
        except:
            pass
        del link_grid['Link_ID']

        p_list = [e for e in list(link_grid.columns.values) if e not in ('length', 'tstep', 'FID')]

        link_grid.loc[:, p_list] = link_grid[p_list].multiply(link_grid['length'], axis=0)

        del link_grid['length']

        link_grid = link_grid.groupby(['FID', 'tstep']).sum()

        return link_grid

    @staticmethod
    def nearest(row, geom_union, df1, df2, geom1_col='geometry', geom2_col='geometry', src_column=None):
        """Finds the nearest point and return the corresponding value from specified column.
        https://automating-gis-processes.github.io/2017/lessons/L3/nearest-neighbour.html#nearest-points-using-geopandas
        """

        # Find the geometry that is closest
        nearest = df2[geom2_col] == nearest_points(row[geom1_col], geom_union)[1]
        # Get the corresponding value from df2 (matching is based on the geometry)
        value = df2[nearest][src_column].get_values()[0]
        return value

    @staticmethod
    def write_rline(emissions, output_dir, start_date):
        from datetime import timedelta
        # emissions = emissions.head(5)
        # print emissions
        # print len(emissions)

        emissions.reset_index(inplace=True)

        emissions_list = settings.comm.gather(emissions, root=0)
        if settings.rank == 0:
            emissions = pd.concat(emissions_list)
            p_list = list(emissions.columns.values)
            p_list.remove('tstep')
            p_list.remove('Link_ID')
            p_list.remove('geometry')
            for p in p_list:
                link_list = ['L_{0}'.format(x) for x in list(pd.unique(emissions['Link_ID']))]
                out_df = pd.DataFrame(columns=["Year", "Mon", "Day", "JDay", "Hr"] + link_list)
                for tstep, aux in emissions.loc[:, ['tstep', 'Link_ID', p]].groupby('tstep'):
                    # out_ds = pd.Series(
                    #     columns=["Year", "Mon", "Day", "JDay", "Hr"] + list(pd.unique(emissions['Link_ID'])))
                    # aux_df = aux.copy()
                    aux_date = start_date + timedelta(hours=tstep)
                    # print out_df
                    out_df.loc[tstep, 'Year'] = aux_date.strftime('%y')
                    out_df.loc[tstep, 'Mon'] = aux_date.month
                    out_df.loc[tstep, 'Day'] = aux_date.day
                    out_df.loc[tstep, 'JDay'] = aux_date.strftime('%j')
                    out_df.loc[tstep, 'Hr'] = aux_date.hour
                    out_df.loc[tstep, link_list] = aux.loc[:, [p]].transpose().values

                out_df.to_csv(os.path.join(output_dir, 'rline_{1}_{0}.csv'.format(p, start_date.strftime('%Y%m%d'))), index=False)

        settings.comm.Barrier()
        return True

    def write_rline_roadlinks(self, df_in):
        # df_out = pd.DataFrame()

        df_in_list = settings.comm.gather(df_in, root=0)
        if settings.rank == 0:
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
                except:
                    # df_err_list.append(line)
                    pass

            df_out.set_index('Link_ID', inplace=True)
            df_out.sort_index(inplace=True)
            df_out.to_csv(os.path.join(self.output_dir, 'roads.txt'), index=False, sep=' ')
        settings.comm.Barrier()

        return True


if __name__ == '__main__':
    from datetime import datetime

    t = Traffic('/home/Earth/ctena/Models/HERMESv3/IN/data/traffic/road_links/BCN/road_links_BCN.shp',
                '/home/Earth/ctena/Models/HERMESv3/IN/data/traffic/fleet_compo',
                '/home/Earth/ctena/Models/HERMESv3/IN/data/profiles/temporal/traffic/speed_hourly.csv',
                '/home/Earth/ctena/Models/HERMESv3/IN/data/profiles/temporal/traffic/aadt_m_mn.csv',
                '/home/Earth/ctena/Models/HERMESv3/IN/data/profiles/temporal/traffic/aadt_week.csv',
                '/home/Earth/ctena/Models/HERMESv3/IN/data/profiles/temporal/traffic/aadt_h_mn.csv',
                '/home/Earth/ctena/Models/HERMESv3/IN/data/profiles/temporal/traffic/aadt_h_wd.csv',
                '/home/Earth/ctena/Models/HERMESv3/IN/data/profiles/temporal/traffic/aadt_h_sat.csv',
                '/home/Earth/ctena/Models/HERMESv3/IN/data/profiles/temporal/traffic/aadt_h_sun.csv',
                '/home/Earth/ctena/Models/HERMESv3/IN/data/traffic/ef',
                #['nox_no2', 'nh3'],
                ['nox_no2'],
                datetime(year=2015, month=01, day=31), load=0.5, timestep_type='hourly', timestep_num=2, timestep_freq=1,
                temp_common_path='/esarchive/recon/ecmwf/era5/1hourly/tas/')

    t.calculate_traffic_line_emissions()
    print t.tyre_wear
    print t.brake_wear
    print t.road_wear
    # del hot_expanded['geometry']
    # hot_expanded = hot_expanded.loc[(hot_expanded['Fleet_Code'] == 'PCG_11') | (hot_expanded['Fleet_Code'] == 'PCG_12'), :]
    # hot_expanded.to_csv('/home/Earth/ctena/Models/HERMESv3/OUT/testing.csv')

    # cold_links = t.road_links.copy()
    # print cold_links.columns.values
    #
    # cold_links.loc[:, 'centroid'] = cold_links['geometry'].centroid
    #
    # temperature = t.read_temperature()
    #
    # unary_union = temperature.unary_union
    #
    # cold_links['nearest_id'] = cold_links.apply(t.nearest, geom_union=unary_union, df1=cold_links, df2=temperature, geom1_col='centroid',
    #                               src_column='t_0', axis=1)
    #
    # print cold_links
    # print temperature
