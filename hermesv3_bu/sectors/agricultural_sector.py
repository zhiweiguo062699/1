#!/usr/bin/env python

import sys
import os
import timeit

import numpy as np
import pandas as pd
import geopandas as gpd

from hermesv3_bu.sectors.sector import Sector
from hermesv3_bu.io_server.io_shapefile import IoShapefile
from hermesv3_bu.logger.log import Log


class AgriculturalSector(Sector):
    def __init__(self, comm, logger, auxiliary_dir, grid_shp, clip, date_array, nut_shapefile, source_pollutants,
                 vertical_levels, crop_list, land_uses_path, ef_files_dir, monthly_profiles_path, weekly_profiles_path,
                 hourly_profiles_path, speciation_map_path, speciation_profiles_path, molecular_weights_path):

        spent_time = timeit.default_timer()
        logger.write_log('===== AGRICULTURAL SECTOR =====')
        super(AgriculturalSector, self).__init__(
            comm, logger, auxiliary_dir, grid_shp, clip, date_array, source_pollutants, vertical_levels,
            monthly_profiles_path, weekly_profiles_path, hourly_profiles_path, speciation_map_path,
            speciation_profiles_path, molecular_weights_path)

        self.nut_shapefile = nut_shapefile
        self.crop_list = crop_list
        self.land_uses_path = land_uses_path
        self.ef_files_dir = ef_files_dir
        self.logger.write_time_log('AgriculturalSector', '__init__', timeit.default_timer() - spent_time)

    def involved_grid_cells(self, src_shp):
        spent_time = timeit.default_timer()
        grid_shp = IoShapefile(self.comm).split_shapefile(self.grid_shp)
        src_union = src_shp.to_crs(grid_shp.crs).geometry.unary_union
        grid_shp = grid_shp.loc[grid_shp.intersects(src_union), :]

        grid_shp_list = self.comm.gather(grid_shp, root=0)
        animal_dist_list = []
        if self.comm.Get_rank() == 0:
            for small_grid in grid_shp_list:
                animal_dist_list.append(src_shp.loc[src_shp.intersects(
                    small_grid.to_crs(src_shp.crs).geometry.unary_union), :])
            grid_shp = pd.concat(grid_shp_list)
            grid_shp = np.array_split(grid_shp, self.comm.Get_size())
        else:
            grid_shp = None
            animal_dist_list = None

        grid_shp = self.comm.scatter(grid_shp, root=0)

        animal_dist = self.comm.scatter(animal_dist_list, root=0)

        self.logger.write_time_log('AgriculturalSector', 'involved_grid_cells', timeit.default_timer() - spent_time)

        return grid_shp, animal_dist

    def calculate_num_days(self):
        spent_time = timeit.default_timer()

        day_array = [hour.date() for hour in self.date_array]
        days, num_days = np.unique(day_array, return_counts=True)

        day_dict = {}
        for key, value in zip(days, num_days):
            day_dict[key] = value
        self.logger.write_time_log('AgriculturalSector', 'calculate_num_days', timeit.default_timer() - spent_time)
        return day_dict

    def get_crop_from_land_uses(self, crop_from_landuse_path):
        import re
        spent_time = timeit.default_timer()

        crop_from_landuse = pd.read_csv(crop_from_landuse_path, sep=';')
        crop_dict = {}
        for i, element in crop_from_landuse.iterrows():
            if element.crop in self.crop_list:
                land_uses = list(map(str, re.split(' , |, | ,|,| ', element.land_use)))
                weights = list(map(str, re.split(' , |, | ,|,| ', element.weight)))
                crop_dict[element.crop] = zip(land_uses, weights)
        self.logger.write_time_log('AgriculturalSector', 'get_crop_from_land_uses', timeit.default_timer() - spent_time)

        return crop_dict

    def get_involved_land_uses(self):
        spent_time = timeit.default_timer()

        land_uses_list = []
        for land_use_and_weight_list in self.crop_from_landuse.itervalues():
            for land_use_and_weight in land_use_and_weight_list:
                land_use = int(land_use_and_weight[0])
                if land_use not in land_uses_list:
                    land_uses_list.append(land_use)
        self.logger.write_time_log('AgriculturalSector', 'get_involved_land_uses', timeit.default_timer() - spent_time)

        return land_uses_list

    def get_land_use_src_by_nut(self, land_uses):
        spent_time = timeit.default_timer()

        df_land_use_with_nut = gpd.read_file(self.land_uses_path)

        df_land_use_with_nut.rename(columns={'CODE': 'NUT', 'gridcode': 'land_use'}, inplace=True)

        df_land_use_with_nut = df_land_use_with_nut.loc[df_land_use_with_nut['land_use'].isin(land_uses), :]

        df_land_use_with_nut = self.spatial_overlays(df_land_use_with_nut,
                                                     self.clip.shapefile.to_crs(df_land_use_with_nut.crs))

        self.logger.write_time_log('AgriculturalSector', 'get_land_use_src_by_nut', timeit.default_timer() - spent_time)
        return df_land_use_with_nut

    def get_tot_land_use_by_nut(self, land_uses):
        spent_time = timeit.default_timer()
        df = pd.read_csv(self.landuse_by_nut)
        df = df.loc[df['land_use'].isin(land_uses), :]
        self.logger.write_time_log('AgriculturalSector', 'get_tot_land_use_by_nut', timeit.default_timer() - spent_time)

        return df

    def get_land_use_by_nut_csv(self, land_use_distribution_src_nut, land_uses, first=False):
        spent_time = timeit.default_timer()
        land_use_distribution_src_nut['area'] = land_use_distribution_src_nut.area

        land_use_by_nut = land_use_distribution_src_nut.groupby(['NUT', 'land_use']).sum().reset_index()
        land_use_by_nut = land_use_by_nut.loc[land_use_by_nut['land_use'].isin(land_uses), :]
        self.logger.write_time_log('AgriculturalSector', 'get_land_use_by_nut_csv', timeit.default_timer() - spent_time)

        return land_use_by_nut

    def land_use_to_crop_by_nut(self, land_use_by_nut, nuts=None):
        spent_time = timeit.default_timer()
        if nuts is not None:
            land_use_by_nut = land_use_by_nut.loc[land_use_by_nut['NUT'].isin(nuts), :]
        new_dict = pd.DataFrame()
        for nut in np.unique(land_use_by_nut['NUT']):
            aux_dict = {'NUT': [nut]}
            for crop, landuse_weight_list in self.crop_from_landuse.iteritems():
                aux = 0
                for landuse, weight in landuse_weight_list:
                    try:
                        aux += land_use_by_nut.loc[(land_use_by_nut['land_use'] == int(landuse)) &
                                                   (land_use_by_nut['NUT'] == nut), 'area'].values[0] * float(weight)
                    except IndexError:
                        # TODO understand better that error
                        pass
                aux_dict[crop] = [aux]
            new_dict = new_dict.append(pd.DataFrame.from_dict(aux_dict), ignore_index=True)
        self.logger.write_time_log('AgriculturalSector', 'land_use_to_crop_by_nut', timeit.default_timer() - spent_time)

        return new_dict

    def get_crop_shape_by_nut(self, crop_by_nut, tot_crop_by_nut):
        spent_time = timeit.default_timer()
        crop_share_by_nut = crop_by_nut.copy()
        crop_share_by_nut[self.crop_list] = 0
        for crop in self.crop_list:
            crop_share_by_nut[crop] = crop_by_nut[crop] / tot_crop_by_nut[crop]
        self.logger.write_time_log('AgriculturalSector', 'get_crop_shape_by_nut', timeit.default_timer() - spent_time)

        return crop_share_by_nut

    def get_crop_area_by_nut(self, crop_share_by_nut):
        spent_time = timeit.default_timer()

        self.crop_by_nut = pd.read_csv(self.crop_by_nut)
        self.crop_by_nut['code'] = self.crop_by_nut['code'].astype(np.int16)
        self.crop_by_nut = self.crop_by_nut.loc[self.crop_by_nut['code'].isin(np.unique(crop_share_by_nut['NUT'])),
                                                ['code'] + self.crop_list].reset_index()

        crop_area_by_nut = crop_share_by_nut.copy()
        crop_area_by_nut[self.crop_list] = 0
        for crop in self.crop_list:
            crop_area_by_nut[crop] = crop_share_by_nut[crop] * self.crop_by_nut[crop]
        self.logger.write_time_log('AgriculturalSector', 'get_crop_area_by_nut', timeit.default_timer() - spent_time)

        return crop_area_by_nut

    def calculate_crop_distribution_src(self, crop_area_by_nut, land_use_distribution_src_nut):
        spent_time = timeit.default_timer()
        crop_distribution_src = land_use_distribution_src_nut.loc[:, ['NUT', 'geometry']]
        for crop, landuse_weight_list in self.crop_from_landuse.iteritems():
            crop_distribution_src[crop] = 0
            for landuse, weight in landuse_weight_list:
                crop_distribution_src.loc[land_use_distribution_src_nut['land_use'] == int(landuse), crop] += \
                    land_use_distribution_src_nut.loc[land_use_distribution_src_nut['land_use'] == int(landuse),
                                                      'area'] * float(weight)

        for nut in np.unique(crop_distribution_src['NUT']):
            for crop in self.crop_list:
                crop_distribution_src.loc[crop_distribution_src['NUT'] == nut, crop] /= \
                    crop_distribution_src.loc[crop_distribution_src['NUT'] == nut, crop].sum()
        for nut in np.unique(crop_distribution_src['NUT']):
            for crop in self.crop_list:
                crop_distribution_src.loc[crop_distribution_src['NUT'] == nut, crop] *= \
                    crop_area_by_nut.loc[crop_area_by_nut['NUT'] == nut, crop].values[0]
        self.logger.write_time_log('AgriculturalSector', 'calculate_crop_distribution_src',
                                   timeit.default_timer() - spent_time)

        return crop_distribution_src

    def get_crop_distribution_in_dst_cells(self, crop_distribution):
        spent_time = timeit.default_timer()
        crop_distribution = crop_distribution.to_crs(self.grid_shp.crs)
        crop_distribution['src_inter_fraction'] = crop_distribution.geometry.area
        crop_distribution = self.spatial_overlays(crop_distribution, self.grid_shp, how='intersection')
        crop_distribution['src_inter_fraction'] = \
            crop_distribution.geometry.area / crop_distribution['src_inter_fraction']

        crop_distribution[self.crop_list] = crop_distribution.loc[:, self.crop_list].multiply(
            crop_distribution["src_inter_fraction"], axis="index")

        crop_distribution = crop_distribution.loc[:, self.crop_list + ['FID']].groupby('FID').sum()

        crop_distribution = gpd.GeoDataFrame(crop_distribution, crs=self.grid_shp.crs,
                                             geometry=self.grid_shp.loc[crop_distribution.index, 'geometry'])
        crop_distribution.reset_index(inplace=True)
        self.logger.write_time_log('AgriculturalSector', 'get_crop_distribution_in_dst_cells',
                                   timeit.default_timer() - spent_time)
        return crop_distribution

    def get_crops_by_dst_cell(self, file_path):
        spent_time = timeit.default_timer()
        if not os.path.exists(file_path):
            involved_land_uses = self.get_involved_land_uses()
            land_use_distribution_src_nut = self.get_land_use_src_by_nut(involved_land_uses)

            land_use_by_nut = self.get_land_use_by_nut_csv(land_use_distribution_src_nut, involved_land_uses)
            tot_land_use_by_nut = self.get_tot_land_use_by_nut(involved_land_uses)

            crop_by_nut = self.land_use_to_crop_by_nut(land_use_by_nut)
            tot_crop_by_nut = self.land_use_to_crop_by_nut(tot_land_use_by_nut, nuts=np.unique(land_use_by_nut['NUT']))

            crop_share_by_nut = self.get_crop_shape_by_nut(crop_by_nut, tot_crop_by_nut)
            crop_area_by_nut = self.get_crop_area_by_nut(crop_share_by_nut)

            crop_distribution_src = self.calculate_crop_distribution_src(crop_area_by_nut,
                                                                         land_use_distribution_src_nut)

            crop_distribution_dst = self.get_crop_distribution_in_dst_cells(crop_distribution_src)

            crop_distribution_dst = self.add_timezone(crop_distribution_dst)

            IoShapefile().write_serial_shapefile(crop_distribution_dst, file_path)

        else:
            crop_distribution_dst = IoShapefile().read_serial_shapefile(file_path)
        crop_distribution_dst.set_index('FID', inplace=True, drop=False)

        self.logger.write_time_log('AgriculturalSector', 'get_crops_by_dst_cell', timeit.default_timer() - spent_time)
        return crop_distribution_dst
