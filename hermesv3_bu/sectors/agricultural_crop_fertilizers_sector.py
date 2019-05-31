#!/usr/bin/env python

import os
import timeit
import pandas as pd
import numpy as np

from hermesv3_bu.sectors.agricultural_sector import AgriculturalSector
from hermesv3_bu.io_server.io_raster import IoRaster
from hermesv3_bu.io_server.io_shapefile import IoShapefile
from hermesv3_bu.logger.log import Log

formula = True


class AgriculturalCropFertilizersSector(AgriculturalSector):
    def __init__(self, comm, logger, auxiliary_dir, grid_shp, clip, date_array, source_pollutants, vertical_levels,
                 crop_list, nut_shapefile, land_uses_path, hourly_profiles_path, speciation_map_path,
                 speciation_profiles_path, molecular_weights_path, landuse_by_nut, crop_by_nut, crop_from_landuse,
                 cultivated_ratio, fertilizer_rate, crop_f_parameter, crop_f_fertilizers, gridded_ph, gridded_cec,
                 fertilizer_denominator_yearly_factor_path, crop_calendar,
                 temperature_path, wind_speed_path, crop_growing_degree_day_path):
        spent_time = timeit.default_timer()
        logger.write_log('===== AGRICULTURAL CROP FERTILIZERS SECTOR =====')
        super(AgriculturalCropFertilizersSector, self).__init__(
            comm, logger, auxiliary_dir, grid_shp, clip, date_array, nut_shapefile, source_pollutants,
            vertical_levels, crop_list, land_uses_path, None, None, None,
            hourly_profiles_path, speciation_map_path, speciation_profiles_path, molecular_weights_path)

        self.day_dict = self.calculate_num_days()

        self.landuse_by_nut = landuse_by_nut
        self.crop_by_nut = crop_by_nut
        self.crop_from_landuse = self.get_crop_from_land_uses(crop_from_landuse)

        self.cultivated_ratio = pd.read_csv(cultivated_ratio)
        self.fertilizer_rate = pd.read_csv(fertilizer_rate)
        self.crop_f_parameter = pd.read_csv(crop_f_parameter)
        self.crop_f_fertilizers = pd.read_csv(crop_f_fertilizers)

        if self.comm.Get_rank() == 0:
            self.crop_distribution = self.get_crops_by_dst_cell(os.path.join(auxiliary_dir, 'crops', 'crops.shp'))

            self.gridded_constants = self.get_gridded_constants(
                os.path.join(auxiliary_dir, 'fertilizers', 'gridded_constants.shp'),
                gridded_ph,
                os.path.join(auxiliary_dir, 'fertilizers', 'gridded_ph.tiff'),
                gridded_cec,
                os.path.join(auxiliary_dir, 'fertilizers', 'gridded_cec.tiff'))
            self.ef_by_crop = self.get_ef_by_crop()
        else:
            self.crop_distribution = None
            self.gridded_constants = None
            self.ef_by_crop = None

        self.crop_distribution = IoShapefile().split_shapefile(self.crop_distribution)
        self.gridded_constants = IoShapefile().split_shapefile(self.gridded_constants)
        self.ef_by_crop = IoShapefile().split_shapefile(self.ef_by_crop)

        self.fertilizer_denominator_yearly_factor_path = fertilizer_denominator_yearly_factor_path
        self.crop_calendar = pd.read_csv(crop_calendar)

        self.temperature_path = temperature_path
        self.wind_speed_path = wind_speed_path
        self.crop_growing_degree_day_path = crop_growing_degree_day_path
        self.logger.write_time_log('AgriculturalCropFertilizersSector', '__init__', timeit.default_timer() - spent_time)

    def get_ftype_fcrop_fmode_by_nut(self, crop, nut_list):
        spent_time = timeit.default_timer()
        filtered_crop_f_parameter = self.crop_f_parameter.loc[(self.crop_f_parameter['code'].isin(nut_list)) &
                                                              (self.crop_f_parameter['crop'] == crop), :].copy()
        filtered_crop_f_parameter.rename(columns={'code': 'nut_code'}, inplace=True)
        filtered_crop_f_parameter.set_index('nut_code', inplace=True)

        element_list = []
        for i, element in self.crop_f_fertilizers.iterrows():
            element_list.append(element['fertilizer_type'])
            filtered_crop_f_parameter[element['fertilizer_type']] *= element['values']

        f_by_nut = pd.concat([filtered_crop_f_parameter.loc[:, element_list].sum(axis=1),
                              filtered_crop_f_parameter['f_crop'],
                              filtered_crop_f_parameter['f_mode']], axis=1).reset_index()
        f_by_nut.rename(columns={0: 'f_type'}, inplace=True)
        self.logger.write_time_log('AgriculturalCropFertilizersSector', 'get_ftype_fcrop_fmode_by_nut',
                                   timeit.default_timer() - spent_time)

        return f_by_nut

    def get_ef_by_crop(self):
        spent_time = timeit.default_timer()
        total_crop_df = self.gridded_constants.loc[:, ['FID', 'geometry', 'nut_code']]
        for crop in self.crop_list:
            crop_ef = self.gridded_constants.loc[:, ['FID', 'geometry', 'nut_code']].copy()
            # f_ph
            if formula:
                # After Zhang et al. (2018)
                crop_ef['f_ph'] = (0.067 * self.gridded_constants['ph'] ** 2) - \
                                  (0.69 * self.gridded_constants['ph']) + 0.68
            else:
                crop_ef['f_ph'] = 0
                crop_ef.loc[self.gridded_constants['ph'] <= 5.5, 'f_ph'] = -1.072
                crop_ef.loc[(self.gridded_constants['ph'] > 5.5) & (self.gridded_constants['ph'] <= 7.3), 'f_ph'] = \
                    -0.933
                crop_ef.loc[(self.gridded_constants['ph'] > 7.3) & (self.gridded_constants['ph'] <= 8.5), 'f_ph'] = \
                    -0.608
                crop_ef.loc[self.gridded_constants['ph'] > 8.5, 'f_ph'] = 0
            # f_cec
            crop_ef['f_cec'] = 0
            crop_ef.loc[self.gridded_constants['cec'] <= 16, 'f_cec'] = 0.088
            crop_ef.loc[(self.gridded_constants['cec'] > 16) & (self.gridded_constants['cec'] <= 24), 'f_cec'] = 0.012
            crop_ef.loc[(self.gridded_constants['cec'] > 24) & (self.gridded_constants['cec'] <= 32), 'f_cec'] = 0.163
            crop_ef.loc[self.gridded_constants['cec'] > 32, 'f_cec'] = 0
            # f_type
            # f_crop
            # f_mode
            f_by_nut = self.get_ftype_fcrop_fmode_by_nut(crop, np.unique(crop_ef['nut_code'].values))

            crop_ef = pd.merge(crop_ef, f_by_nut, how='left', on='nut_code')
            crop_ef.set_index('FID', inplace=True, drop=False)

            crop_ef['f_sum'] = np.exp(crop_ef['f_ph'] + crop_ef['f_cec'] + crop_ef['f_type'] + crop_ef['f_crop'] +
                                      crop_ef['f_mode'])

            total_crop_df['EF_{0}'.format(crop)] = crop_ef['f_sum']

        self.logger.write_time_log('AgriculturalCropFertilizersSector', 'get_ef_by_crop',
                                   timeit.default_timer() - spent_time)
        return total_crop_df

    def to_dst_resolution(self, src_shapefile, value):
        spent_time = timeit.default_timer()

        intersection = self.spatial_overlays(src_shapefile.to_crs(self.grid_shp.crs), self.grid_shp)
        intersection['area'] = intersection.geometry.area
        dst_shapefile = self.grid_shp.copy()
        dst_shapefile['involved_area'] = intersection.groupby('FID')['area'].sum()
        intersection_with_dst_areas = pd.merge(intersection, dst_shapefile.loc[:, ['FID', 'involved_area']],
                                               how='left', on='FID')
        intersection_with_dst_areas['involved_area'] = \
            intersection_with_dst_areas['area'] / intersection_with_dst_areas['involved_area']

        intersection_with_dst_areas[value] = \
            intersection_with_dst_areas[value] * intersection_with_dst_areas['involved_area']
        dst_shapefile[value] = intersection_with_dst_areas.groupby('FID')[value].sum()
        dst_shapefile.drop('involved_area', axis=1, inplace=True)
        self.logger.write_time_log('AgriculturalCropFertilizersSector', 'to_dst_resolution',
                                   timeit.default_timer() - spent_time)

        return dst_shapefile

    def get_gridded_constants(self, gridded_ph_cec_path, ph_path, clipped_ph_path, cec_path, clipped_cec_path):
        spent_time = timeit.default_timer()
        if not os.path.exists(gridded_ph_cec_path):
            IoRaster().clip_raster_with_shapefile_poly(ph_path, self.clipping, clipped_ph_path, nodata=255)
            ph_gridded = IoRaster().to_shapefile(clipped_ph_path, nodata=255)
            ph_gridded.rename(columns={'data': 'ph'}, inplace=True)
            # To correct input data
            ph_gridded['ph'] = ph_gridded['ph'] / 10
            ph_gridded = self.to_dst_resolution(ph_gridded, value='ph')

            IoRaster().clip_raster_with_shapefile_poly_serie(cec_path, self.clipping, clipped_cec_path, nodata=-32768)
            cec_gridded = IoRaster().to_shapefile_serie(clipped_cec_path, nodata=-32768)
            cec_gridded.rename(columns={'data': 'cec'}, inplace=True)
            cec_gridded = self.to_dst_resolution(cec_gridded, value='cec')

            gridded_ph_cec = ph_gridded
            gridded_ph_cec['cec'] = cec_gridded['cec']

            gridded_ph_cec.dropna(inplace=True)

            gridded_ph_cec = self.add_nut_code(gridded_ph_cec, self.nut_shapefile)

            # Selecting only PH and CEC cells that have also some crop.
            gridded_ph_cec = gridded_ph_cec.loc[gridded_ph_cec['FID'].isin(self.crop_distribution['FID'].values), :]

            IoShapefile().write_serial_shapefile(gridded_ph_cec, gridded_ph_cec_path)
        else:
            gridded_ph_cec = IoShapefile().read_serial_shapefile(gridded_ph_cec_path)
        gridded_ph_cec.set_index('FID', inplace=True, drop=False)

        self.logger.write_time_log('AgriculturalCropFertilizersSector', 'get_gridded_constants',
                                   timeit.default_timer() - spent_time)
        return gridded_ph_cec

    def get_daily_inputs(self, yearly_emissions):
        spent_time = timeit.default_timer()
        daily_inputs = {}
        geometry_shp = yearly_emissions.loc[:, ['FID', 'geometry']].to_crs({'init': 'epsg:4326'})
        geometry_shp['c_lat'] = geometry_shp.centroid.y
        geometry_shp['c_lon'] = geometry_shp.centroid.x
        geometry_shp['centroid'] = geometry_shp.centroid
        geometry_shp.drop(columns='geometry', inplace=True)

        for day in self.day_dict.keys():
            aux_df = yearly_emissions.copy()
            meteo_df = self.get_data_from_netcdf(
                os.path.join(self.temperature_path, 'tas_{0}{1}.nc'.format(day.year, str(day.month).zfill(2))),
                'tas', 'daily', day, geometry_shp)
            meteo_df['tas'] = meteo_df['tas'] - 273.15
            meteo_df['sfcWind'] = self.get_data_from_netcdf(
                os.path.join(self.wind_speed_path, 'sfcWind_{0}{1}.nc'.format(day.year, str(day.month).zfill(2))),
                'sfcWind', 'daily', day, geometry_shp).loc[:, 'sfcWind']

            for crop in self.crop_list:
                meteo_df['d_{0}'.format(crop)] = self.get_data_from_netcdf(
                    self.fertilizer_denominator_yearly_factor_path.replace('<crop>', crop).replace(
                        '<year>', str(day.year)), 'FD', 'yearly', day, geometry_shp).loc[:, 'FD']

            meteo_df['winter'] = self.get_data_from_netcdf(
                self.crop_growing_degree_day_path.replace('<season>', 'winter').replace('<year>', str(day.year)),
                'Tsum', 'yearly', day, geometry_shp).loc[:, 'Tsum'].astype(np.int16)
            meteo_df['spring'] = self.get_data_from_netcdf(
                self.crop_growing_degree_day_path.replace('<season>', 'spring').replace('<year>', str(day.year)),
                'Tsum', 'yearly', day, geometry_shp).loc[:, 'Tsum'].astype(np.int16)

            aux_df = aux_df.to_crs({'init': 'epsg:4326'})
            aux_df['centroid'] = aux_df.centroid

            aux_df['REC'] = aux_df.apply(self.nearest, geom_union=meteo_df.unary_union, df1=aux_df,
                                         df2=meteo_df, geom1_col='centroid', src_column='REC', axis=1)
            aux_df = pd.merge(aux_df, meteo_df, how='left', on='REC')

            aux_df.drop(columns=['centroid', 'REC', 'geometry_y'], axis=1, inplace=True)
            aux_df.rename(columns={'geometry_x': 'geometry'}, inplace=True)

            daily_inputs[day] = aux_df

        self.logger.write_time_log('AgriculturalCropFertilizersSector', 'get_daily_inputs',
                                   timeit.default_timer() - spent_time)
        return daily_inputs

    def calculate_yearly_emissions(self):
        spent_time = timeit.default_timer()
        self.crop_distribution = pd.merge(self.crop_distribution, self.ef_by_crop.loc[:, ['FID', 'nut_code']],
                                          how='left', on='FID')

        self.crop_distribution.set_index('FID', drop=False, inplace=True)
        self.ef_by_crop.set_index('FID', drop=False, inplace=True)
        self.ef_by_crop = self.ef_by_crop.loc[self.crop_distribution.index, :]

        for crop in self.crop_list:
            self.crop_distribution[crop] = self.crop_distribution.groupby('nut_code')[crop].apply(
                lambda x: x.multiply(np.float64(self.cultivated_ratio.loc[0, crop]) *
                                     self.fertilizer_rate.loc[self.fertilizer_rate['code'] == x.name, crop].values[0]))
            self.crop_distribution[crop] = self.crop_distribution[crop] * self.ef_by_crop['EF_{0}'.format(crop)]

        self.logger.write_time_log('AgriculturalCropFertilizersSector', 'calculate_yearly_emissions',
                                   timeit.default_timer() - spent_time)
        return self.crop_distribution

    def calculate_nh3_emissions(self, day, daily_inputs):
        import math
        spent_time = timeit.default_timer()
        daily_inputs['exp'] = np.exp(daily_inputs['tas'].multiply(0.0223) + daily_inputs['sfcWind'].multiply(0.0419))
        daily_inputs.drop(['tas', 'sfcWind'], axis=1, inplace=True)

        for crop in self.crop_list:
            beta_1 = self.crop_calendar.loc[self.crop_calendar['crop'] == crop, 'beta_1'].values[0]
            beta_2 = self.crop_calendar.loc[self.crop_calendar['crop'] == crop, 'beta_2'].values[0]
            beta_3 = self.crop_calendar.loc[self.crop_calendar['crop'] == crop, 'beta_3'].values[0]
            thau_1 = self.crop_calendar.loc[self.crop_calendar['crop'] == crop, 'thau_1'].values[0]
            thau_2 = self.crop_calendar.loc[self.crop_calendar['crop'] == crop, 'thau_2'].values[0]
            thau_3 = self.crop_calendar.loc[self.crop_calendar['crop'] == crop, 'thau_3'].values[0]
            sigma_1 = self.crop_calendar.loc[self.crop_calendar['crop'] == crop, 'sigma_1'].values[0]
            sigma_2 = self.crop_calendar.loc[self.crop_calendar['crop'] == crop, 'sigma_2'].values[0]
            sigma_3 = self.crop_calendar.loc[self.crop_calendar['crop'] == crop, 'sigma_3'].values[0]

            try:
                thau_2 = float(thau_2)
                sum = (beta_1 / (sigma_1 * math.sqrt(2 * math.pi))) * math.exp(
                    (float(int(day.strftime('%j')) - thau_1) ** 2) / (-2 * (sigma_1 ** 2)))
                sum += (beta_2 / (sigma_2 * math.sqrt(2 * math.pi))) * math.exp(
                    (float(int(day.strftime('%j')) - thau_2) ** 2) / (-2 * (sigma_2 ** 2)))
                sum += (beta_3 / (sigma_3 * math.sqrt(2 * math.pi))) * math.exp(
                    (float(int(day.strftime('%j')) - thau_3) ** 2) / (-2 * (sigma_3 ** 2)))
            except ValueError:
                aux = (beta_1 / (sigma_1 * math.sqrt(2 * math.pi))) * math.exp(
                    (float(int(day.strftime('%j')) - thau_1) ** 2) / (-2 * (sigma_1 ** 2)))
                aux += (beta_3 / (sigma_3 * math.sqrt(2 * math.pi))) * math.exp(
                    (float(int(day.strftime('%j')) - thau_3) ** 2) / (-2 * (sigma_3 ** 2)))
                sum = (beta_2 / (sigma_2 * math.sqrt(2 * math.pi))) * np.exp(
                    ((int(day.strftime('%j')) - daily_inputs[thau_2]).astype(np.float64)) ** 2 / (-2 * (sigma_2 ** 2)))
                sum += aux
            daily_inputs['FD_{0}'.format(crop)] = daily_inputs['exp'].multiply(sum)

        for crop in self.crop_list:
            daily_inputs[crop] = daily_inputs[crop].multiply(
                daily_inputs['FD_{0}'.format(crop)] / daily_inputs['d_{0}'.format(crop)])
        daily_emissions = daily_inputs.loc[:, ['FID', 'timezone', 'geometry', 'nut_code']].copy()
        daily_emissions['nh3'] = daily_inputs.loc[:, self.crop_list].sum(axis=1)
        # From kg NH3-N to g NH3
        daily_emissions['nh3'] = daily_emissions['nh3'].multiply((17. / 14.) * 1000.)

        self.logger.write_time_log('AgriculturalCropFertilizersSector', 'calculate_nh3_emissions',
                                   timeit.default_timer() - spent_time)
        return daily_emissions

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

        self.logger.write_time_log('AgriculturalCropFertilizersSector', 'add_dates',
                                   timeit.default_timer() - spent_time)
        return dataframe_by_day

    def calculate_daily_emissions(self, emissions):
        spent_time = timeit.default_timer()
        df_by_day = self.get_daily_inputs(emissions)

        for day, daily_inputs in df_by_day.iteritems():

            df_by_day[day] = self.calculate_nh3_emissions(day, daily_inputs)
        self.logger.write_time_log('AgriculturalCropFertilizersSector', 'calculate_daily_emissions',
                                   timeit.default_timer() - spent_time)

        return df_by_day

    def calculate_hourly_emissions(self, emissions):
        spent_time = timeit.default_timer()
        emissions['hour'] = emissions['date'].dt.hour
        emissions['nh3'] = emissions.groupby('hour')['nh3'].apply(
            lambda x: x.multiply(self.hourly_profiles.loc[self.hourly_profiles['P_hour'] == 'nh3', x.name].values[0]))

        emissions['date'] = emissions['date_utc']
        emissions.drop(columns=['hour', 'date_utc'], axis=1, inplace=True)

        self.logger.write_time_log('AgriculturalCropFertilizersSector', 'calculate_hourly_emissions',
                                   timeit.default_timer() - spent_time)
        return emissions

    def calculate_emissions(self):
        spent_time = timeit.default_timer()
        self.logger.write_log('\tCalculating emissions')

        emissions = self.calculate_yearly_emissions()
        df_by_day = self.calculate_daily_emissions(emissions)
        emissions = self.add_dates(df_by_day)
        emissions = self.calculate_hourly_emissions(emissions)
        emissions = self.speciate(emissions)

        self.logger.write_log('\t\tCrop fertilizers emissions calculated', message_level=2)
        self.logger.write_time_log('AgriculturalCropFertilizersSector', 'calculate_emissions',
                                   timeit.default_timer() - spent_time)
        return emissions
