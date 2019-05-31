#!/usr/bin/env python

import os
import timeit
import pandas as pd
import numpy as np

from hermesv3_bu.sectors.agricultural_sector import AgriculturalSector
from hermesv3_bu.io_server.io_shapefile import IoShapefile
from hermesv3_bu.logger.log import Log


class AgriculturalCropOperationsSector(AgriculturalSector):
    def __init__(self, comm_agr, comm, logger, auxiliary_dir, grid_shp, clip, date_array, source_pollutants,
                 vertical_levels, nut_shapefile_path, crop_list, land_uses_path, ef_dir, monthly_profiles_path,
                 weekly_profiles_path, hourly_profiles_path, speciation_map_path, speciation_profiles_path,
                 molecular_weights_path, landuse_by_nut, crop_by_nut, crop_from_landuse):
        """

        :param auxiliary_dir: Path to the directory where the necessary auxiliary files will be created if them are
            not created yet.
        :type auxiliary_dir: str

        :param grid_shp: Shapefile that contains the destination grid. It must contains the 'FID' (cell num).
        :type grid_shp: GeoPandas.GeoDataframe

        :param clip: Path to the shapefile that contains the region of interest.
        :type clip: str

        :param date_array: List of datetimes.
        :type date_array: list(datetime.datetime, ...)

        :param nut_shapefile_path: Path to the shapefile that contain the NUT polygons. The shapefile must contain
            the 'ORDER06' information with the NUT_code.
        :type nut_shapefile_path: str

        :param source_pollutants: List of input pollutants to take into account. Agricultural livestock module can
            calculate emissions derived from the next source pollutants: NH3, NOx expressed as PM10 and PM2.5
            ['pm10', 'pm25']
        :type source_pollutants: list

        :param crop_list: Crop list to take into account for the emission calculation. [barley, oats, rye, wheat]
        :type crop_list: list

        :param land_uses_path: Path to the shapefile that contains all the land-uses.
        :type land_uses_path: str.

        :param ef_dir: Path to the folder that contains all the CSV files with the information to calculate the
            emissions. Each pollutant have to be in separated files (pm10.csv, pm25.csv):
            Columns: [crop, operation, EF_pm10] and [crop, operation, EF_pm25].
        :type ef_dir: str

        :param monthly_profiles_path: Path to the CSV file that contains all the monthly profiles. The CSV file must
            contain the following columns [P_month, January, February, ..., November, December]
            The P_month code have to be the input pollutant.
        :type monthly_profiles_path: str

        :param weekly_profiles_path: Path to the CSV file that contains all the weekly profiles. The CSV file must
            contain the following columns [P_week, Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday]
            The P_week code have to be the input pollutant.
        :type weekly_profiles_path: str

        :param hourly_profiles_path: Path to the CSV file that contains all the hourly profiles. The CSV file must
            contain the following columns [P_hour, 0, 1, 2, 3, ..., 22, 23]
            The P_week code have to be the input pollutant.
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

        :param landuse_by_nut:
        :param crop_by_nut:

        :param nut_shapefile_path: Path to the shapefile that contain the NUT polygons. The shapefile must contain
            the 'ORDER07' information with the NUT_code.
        :type nut_shapefile_path: str
        """
        spent_time = timeit.default_timer()
        logger.write_log('===== AGRICULTURAL CROP OPERATIONS SECTOR =====')
        super(AgriculturalCropOperationsSector, self).__init__(
            comm_agr, comm, logger, auxiliary_dir, grid_shp, clip, date_array, nut_shapefile_path, source_pollutants,
            vertical_levels, crop_list, land_uses_path, ef_dir, monthly_profiles_path, weekly_profiles_path,
            hourly_profiles_path, speciation_map_path, speciation_profiles_path, molecular_weights_path)

        self.landuse_by_nut = landuse_by_nut
        self.crop_by_nut = crop_by_nut
        self.crop_from_landuse = self.get_crop_from_land_uses(crop_from_landuse)

        self.months = self.get_date_array_by_month()

        self.logger.write_time_log('AgriculturalCropOperationsSector', '__init__', timeit.default_timer() - spent_time)

    def get_date_array_by_month(self):
        spent_time = timeit.default_timer()

        month_array = [hour.date().month for hour in self.date_array]
        month_list, num_days = np.unique(month_array, return_counts=True)

        month_dict = {}
        for month in month_list:
            month_dict[month] = np.array(self.date_array)[month_array == month]

        self.logger.write_time_log('AgriculturalCropOperationsSector', 'get_date_array_by_month',
                                   timeit.default_timer() - spent_time)

        return month_dict

    def calculate_distribution_by_month(self, month):
        spent_time = timeit.default_timer()

        month_distribution = self.crop_distribution.loc[:, ['FID', 'timezone', 'geometry']].copy()
        for pollutant in self.source_pollutants:
            month_distribution[pollutant] = 0

            emission_factors = pd.read_csv(os.path.join(self.ef_files_dir, '{0}.csv'.format(pollutant)))
            for crop in self.crop_list:
                ef_c = emission_factors.loc[
                    (emission_factors['crop'] == crop) & (emission_factors['operation'] == 'soil_cultivation'),
                    'EF_{0}'.format(pollutant)].values[0]
                ef_h = emission_factors.loc[
                    (emission_factors['crop'] == crop) & (emission_factors['operation'] == 'harvesting'),
                    'EF_{0}'.format(pollutant)].values[0]
                m_c = self.monthly_profiles.loc[
                    (self.monthly_profiles['P_month'] == crop) & (self.monthly_profiles['operation'] ==
                                                                  'soil_cultivation'), month].values[0]
                m_h = self.monthly_profiles.loc[
                    (self.monthly_profiles['P_month'] == crop) & (self.monthly_profiles['operation'] == 'harvesting'),
                    month].values[0]
                factor = ef_c * m_c + ef_h * m_h
                # From Kg to g
                factor *= 1000.0
                month_distribution[pollutant] += self.crop_distribution[crop].multiply(factor)
        self.logger.write_time_log('AgriculturalCropOperationsSector', 'calculate_distribution_by_month',
                                   timeit.default_timer() - spent_time)

        return month_distribution

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
        self.logger.write_time_log('AgriculturalCropOperationsSector', 'add_dates', timeit.default_timer() - spent_time)

        return dataframe_by_day

    def calculate_hourly_emissions(self, ):
        spent_time = timeit.default_timer()

        self.crop_distribution['weekday'] = self.crop_distribution['date'].dt.weekday
        self.crop_distribution['hour'] = self.crop_distribution['date'].dt.hour

        for pollutant in self.source_pollutants:
            # hourly_profile = self.hourly_profiles.loc[self.hourly_profiles['P_hour'] == pollutant, :].to_dict()
            daily_profile = self.calculate_rebalance_factor(
                self.weekly_profiles.loc[self.weekly_profiles['P_week'] == pollutant, :].values[0],
                np.unique(self.crop_distribution['date']))

            self.crop_distribution[pollutant] = self.crop_distribution.groupby('weekday')[pollutant].apply(
                lambda x: x.multiply(daily_profile[x.name]))
            self.crop_distribution[pollutant] = self.crop_distribution.groupby('hour')[pollutant].apply(
                lambda x: x.multiply(self.hourly_profiles.loc[self.hourly_profiles['P_hour'] == pollutant,
                                                              x.name].values[0]))
        self.crop_distribution.drop(['weekday', 'hour'], axis=1, inplace=True)

        self.logger.write_time_log('AgriculturalCropOperationsSector', 'calculate_hourly_emissions',
                                   timeit.default_timer() - spent_time)

        return self.crop_distribution

    def calculate_emissions(self):
        spent_time = timeit.default_timer()
        self.logger.write_log('\tCalculating emissions')

        distribution_by_month = {}
        for month in self.months.iterkeys():
            distribution_by_month[month] = self.calculate_distribution_by_month(month)

        self.crop_distribution = self.add_dates(distribution_by_month)
        self.crop_distribution.drop('date_utc', axis=1, inplace=True)
        self.crop_distribution = self.calculate_hourly_emissions()
        self.crop_distribution = self.speciate(self.crop_distribution)

        self.logger.write_log('\t\tCrop operations emissions calculated', message_level=2)
        self.logger.write_time_log('AgriculturalCropOperationsSector', 'calculate_emissions',
                                   timeit.default_timer() - spent_time)

        return self.crop_distribution
