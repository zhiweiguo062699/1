#!/usr/bin/env python

import sys
import os
import timeit

import numpy as np
import pandas as pd
import geopandas as gpd
from mpi4py import MPI

from hermesv3_bu.sectors.sector import Sector
from hermesv3_bu.io_server.io_shapefile import IoShapefile
from hermesv3_bu.io_server.io_raster import IoRaster
from hermesv3_bu.tools.checker import error_exit
from hermesv3_bu.logger.log import Log
from geopandas import GeoDataFrame
from pandas import DataFrame


class AgriculturalSector(Sector):
    def __init__(self, comm_agr, comm, logger, auxiliary_dir, grid, clip, date_array, nut_shapefile,
                 source_pollutants, vertical_levels, crop_list, land_uses_path, land_use_by_nut, crop_by_nut,
                 crop_from_landuse_path, ef_files_dir, monthly_profiles_path, weekly_profiles_path,
                 hourly_profiles_path, speciation_map_path, speciation_profiles_path, molecular_weights_path):
        """
        Initialise the common class for agricultural sectors (fertilizers, crop operations and machinery)

        :param comm_agr: Common communicator for all the agricultural sectors.
        :type comm_agr: MPI.Comm

        :param comm: Comunicator for the current sector.
        :type comm: MPI.Comm

        :param logger: Logger
        :type logger: Log

        :param auxiliary_dir: Path to the directory where the necessary auxiliary files will be created if them are not
            created yet.
        :type auxiliary_dir: str

        :param grid_shp: Shapefile with the grid horizontal distribution.
        :type grid_shp: GeoDataFrame

        :param date_array: List of datetimes.
        :type date_array: list(datetime.datetime, ...)

        :param source_pollutants: List of input pollutants to take into account.
        :type source_pollutants: list

        :param vertical_levels: List of top level of each vertical layer.
        :type vertical_levels: list

        :param nut_shapefile: Shapefile path to the one that have the NUT_codes.
        :type nut_shapefile: str

        :param crop_list: List of crops to take into account for that sector.
        :type crop_list: list

        :param land_uses_path: Path to the shapefile that contains all the land uses.
        :type land_uses_path: str

        :param land_use_by_nut: Path to the DataFrame with the area for each land use of each NUT code.
            columns: NUT, land_use, area
        :type land_use_by_nut: str

        :param crop_by_nut: Path to the DataFrame with the amount of crops for each NUT code.
            That DataFrame have the 'code' column with the NUT code and as many columns as crops.
        :type crop_by_nut: str

        :param crop_from_landuse_path: Path to the DataFrame with the mapping between crops and land uses.
            That CSV have as value separator a semicolon and a comma between elements of the same column.
            There are needed the following columns: crop, land_use and weight.
            The land_use and weight columns can have as elements as needed, separated by commas, but both have to have
            the same length.
            The land_use column contains the list, or unique value, of the land use that contains that crop.
            The weight column contains each weight of each selected land use.
        :type crop_from_landuse_path: str

        :param ef_files_dir: Path to the folder that contains all the Emission Factors.
        :type ef_files_dir: str

        :param monthly_profiles_path: Path to the CSV file that contains all the monthly profiles. The CSV file must
            contain the following columns [P_month, January, February, March, April, May, June, July, August, September,
            October, November, December]
        :type monthly_profiles_path: str

        :param weekly_profiles_path: Path to the CSV file that contains all the weekly profiles. The CSV file must
            contain the following columns [P_week, Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday]
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
        spent_time = timeit.default_timer()

        super(AgriculturalSector, self).__init__(
            comm, logger, auxiliary_dir, grid, clip, date_array, source_pollutants, vertical_levels,
            monthly_profiles_path, weekly_profiles_path, hourly_profiles_path, speciation_map_path,
            speciation_profiles_path, molecular_weights_path)

        self.comm_agr = comm_agr
        self.nut_shapefile = nut_shapefile
        self.crop_list = crop_list
        self.land_uses_path = land_uses_path
        self.ef_files_dir = ef_files_dir
        self.land_use_by_nut = land_use_by_nut
        self.crop_by_nut = crop_by_nut
        self.crop_from_landuse = self.get_crop_from_land_uses(crop_from_landuse_path)
        self.crop_distribution = self.get_crops_by_dst_cell(
            os.path.join(auxiliary_dir, 'agriculture', 'crops', 'crops.shp'))
        self.logger.write_time_log('AgriculturalSector', '__init__', timeit.default_timer() - spent_time)

    def involved_grid_cells(self, src_shp):
        spent_time = timeit.default_timer()
        grid_shp = IoShapefile(self.comm).split_shapefile(self.grid.shapefile)
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

    def get_crop_from_land_uses(self, crop_from_land_use_path):
        """
        Get the involved land uses and their weight for each crop.

        :param crop_from_land_use_path: Path to the file that contains the crops and their involved land uses with the
            weights.
        :type crop_from_land_use_path: str

        :return: Dictionary with the crops as keys and a list as value. That list have as many elements as involved
            land uses in that crop. Each element of that list is a tuple with the land use as first element and their
            weight on the second place.
        :rtype: dict
        """
        import re
        spent_time = timeit.default_timer()

        crop_from_landuse = pd.read_csv(crop_from_land_use_path, sep=';')
        crop_dict = {}
        for i, element in crop_from_landuse.iterrows():
            # if element.crop in self.crop_list:
            land_uses = list(map(int, re.split(' , |, | ,|,| ', element.land_use)))
            weights = list(map(float, re.split(' , |, | ,|,| ', element.weight)))
            crop_dict[element.crop] = list(zip(land_uses, weights))

        self.logger.write_time_log('AgriculturalSector', 'get_crop_from_land_uses', timeit.default_timer() - spent_time)
        return crop_dict

    def get_involved_land_uses(self):
        """
        Generate the list of involved land uses.

        :return: List of land uses involved in the selected crops
        :rtype: list
        """
        spent_time = timeit.default_timer()

        land_uses_list = []
        for land_use_and_weight_list in self.crop_from_landuse.values():
            for land_use_and_weight in land_use_and_weight_list:
                land_use = int(land_use_and_weight[0])
                if land_use not in land_uses_list:
                    land_uses_list.append(land_use)
        self.logger.write_time_log('AgriculturalSector', 'get_involved_land_uses', timeit.default_timer() - spent_time)

        return land_uses_list

    def get_land_use_src_by_nut(self, land_uses):
        """
        Create a shapefile with the involved source cells from the input raster and only for the given land uses.

        :param land_uses: List of land uses to use.
        :type land_uses: list

        :return: Shapefile with the land use and nut_code of each source cell. Index: CELL_ID
        :rtype: GeoDataFrame
        """
        spent_time = timeit.default_timer()

        land_uses_clipped = os.path.join(self.auxiliary_dir, 'agriculture', 'land_uses', 'land_uses_clip.tif')
        if self.comm_agr.Get_rank() == 0:
            land_uses_clipped = IoRaster(self.comm_agr).clip_raster_with_shapefile_poly(
                self.land_uses_path, self.clip.shapefile, land_uses_clipped, values=land_uses)

        land_use_src_by_nut = IoRaster(self.comm_agr).to_shapefile_parallel(land_uses_clipped)
        land_use_src_by_nut.rename(columns={'data': 'land_use'}, inplace=True)
        land_use_src_by_nut['land_use'] = land_use_src_by_nut['land_use'].astype(np.int16)

        land_use_src_by_nut = self.add_nut_code(land_use_src_by_nut, self.nut_shapefile, nut_value='nuts2_id')
        land_use_src_by_nut = land_use_src_by_nut[land_use_src_by_nut['nut_code'] != -999]

        land_use_src_by_nut = IoShapefile(self.comm_agr).balance(land_use_src_by_nut)
        land_use_src_by_nut.set_index('CELL_ID', inplace=True)

        self.logger.write_time_log('AgriculturalSector', 'get_land_use_src_by_nut', timeit.default_timer() - spent_time)

        return land_use_src_by_nut

    def get_tot_land_use_by_nut(self, land_uses):
        """
        Get the total amount of land use area by NUT of the involved land uses.

        :param land_uses: Involved land uses.
        :type land_uses: list

        :return: Total amount of land use area by NUT.
        :rtype: DataFrame
        """
        spent_time = timeit.default_timer()

        df = pd.read_csv(self.land_use_by_nut, dtype={'nuts2_id': str})
        df.rename(columns={'nuts2_id': 'nut_code'}, inplace=True)
        df = df.loc[df['land_use'].isin(land_uses), :]
        df['nut_code'] = df['nut_code'].astype(np.int32)
        df.set_index(['nut_code', 'land_use'], inplace=True)

        self.logger.write_time_log('AgriculturalSector', 'get_tot_land_use_by_nut', timeit.default_timer() - spent_time)
        return df

    def get_land_use_by_nut_csv(self, land_use_distribution_src_nut, land_uses):
        """
        Get the involved area of land use by involved NUT.

        :param land_use_distribution_src_nut: Shapefile with the polygons of all the land uses for each NUT.
        :type land_use_distribution_src_nut: GeoDataFrame

        :param land_uses: Land uses to take into account.
        :type land_uses: list

        :return: DataFrame with the total amount of land use area by involved NUT.
        :rtype: DataFrame
        """
        spent_time = timeit.default_timer()

        land_use_by_nut = pd.DataFrame(index=pd.MultiIndex.from_product(
            [np.unique(land_use_distribution_src_nut['nut_code'].astype(np.int64)),
             np.unique(land_uses).astype(np.int16)], names=['nut_code', 'land_use']))
        land_use_by_nut['area'] = 0.0
        land_use_distribution_src_nut['area'] = land_use_distribution_src_nut.area
        land_use_by_nut['area'] += land_use_distribution_src_nut.groupby(['nut_code', 'land_use'])['area'].sum()
        land_use_by_nut.fillna(0.0, inplace=True)

        self.logger.write_time_log('AgriculturalSector', 'get_land_use_by_nut_csv', timeit.default_timer() - spent_time)
        return land_use_by_nut

    def land_use_to_crop_by_nut(self, land_use_by_nut, nuts=None):
        """
        Get the amount of crop by involved NUT.

        :param land_use_by_nut: Area of each land use for each NUT
        :type land_use_by_nut: DataFrame

        :param nuts: NUT list to take into account. None for all of them.
        :type nuts: list

        :return: Amount of crop by NUT.
        :rtype: DataFrame
        """
        spent_time = timeit.default_timer()

        if nuts is not None:
            land_use_by_nut = land_use_by_nut.iloc[land_use_by_nut.index.get_level_values('nut_code').isin(nuts)]

        new_df = pd.DataFrame(index=np.unique(land_use_by_nut.index.get_level_values('nut_code')),
                              columns=self.crop_from_landuse.keys())
        new_df.fillna(0, inplace=True)

        for crop, land_use_weight_list in self.crop_from_landuse.items():
            for land_use, weight in land_use_weight_list:
                new_df[crop] += land_use_by_nut.xs(land_use, level='land_use')['area'] * weight

        self.logger.write_time_log('AgriculturalSector', 'land_use_to_crop_by_nut', timeit.default_timer() - spent_time)

        return new_df

    def get_crop_shape_by_nut(self, crop_by_nut, tot_crop_by_nut):
        """
        Calculate the fraction of crop for each NUT involved on the simulated domain.

        :param crop_by_nut: Amount of crop by NUT on the simulated domain.
        :type crop_by_nut: DataFrame

        :param tot_crop_by_nut: Total amount of crop by NUT.
        :type tot_crop_by_nut: DataFrame

        :return: Fraction of involved crop for NUT.
        :rtype: DataFrame(
        """
        spent_time = timeit.default_timer()

        crop_share_by_nut = crop_by_nut.copy()
        for crop in crop_by_nut.columns:
            crop_share_by_nut[crop] = crop_by_nut[crop] / tot_crop_by_nut[crop]

        self.logger.write_time_log('AgriculturalSector', 'get_crop_shape_by_nut', timeit.default_timer() - spent_time)

        return crop_share_by_nut

    def get_crop_area_by_nut(self, crop_share_by_nut):
        """
        Calculate the amount of crop for each NUT.

        :param crop_share_by_nut: GeoDataFrame with the fraction of crop for each NUT. That fraction means the quantity
            of the NUT crop involved on the simulation. If the complete NUT is fulfilled on the domain is it 1.
        :type crop_share_by_nut: DataFrame

        :return: Amount of crop for each NUT.
        :rtype: DataFrame
        """
        spent_time = timeit.default_timer()

        crop_by_nut = pd.read_csv(self.crop_by_nut, dtype={'nuts2_id': str})
        crop_by_nut.drop(columns='nuts2_na', inplace=True)
        crop_by_nut.rename(columns={'nuts2_id': 'nut_code'}, inplace=True)

        crop_by_nut['nut_code'] = crop_by_nut['nut_code'].astype(np.int64)
        crop_by_nut.set_index('nut_code', inplace=True)

        crop_by_nut = crop_by_nut.loc[crop_share_by_nut.index, :]
        crop_area_by_nut = crop_share_by_nut * crop_by_nut

        self.logger.write_time_log('AgriculturalSector', 'get_crop_area_by_nut', timeit.default_timer() - spent_time)
        return crop_area_by_nut

    def calculate_crop_distribution_src(self, crop_area_by_nut, land_use_distribution_src_nut):
        """
        Calculate the crop distribution on the source resolution.

        :param crop_area_by_nut: Amount of crop on each NUT.
        :type crop_area_by_nut: DataFrame

        :param land_use_distribution_src_nut: Source distribution land uses with their calculated areas.
        :type land_use_distribution_src_nut: GeoDataFrame

        :return: Crop distribution on the source resolution.
        :rtype: GeoDataFrame
        """
        spent_time = timeit.default_timer()

        crop_distribution_src = land_use_distribution_src_nut.loc[:, ['nut_code', 'geometry']]

        for crop, landuse_weight_list in self.crop_from_landuse.items():
            crop_distribution_src[crop] = 0
            for landuse, weight in landuse_weight_list:
                crop_distribution_src.loc[land_use_distribution_src_nut['land_use'] == int(landuse), crop] += \
                    land_use_distribution_src_nut.loc[land_use_distribution_src_nut['land_use'] == int(landuse),
                                                      'area'] * float(weight)
        for nut in np.unique(crop_distribution_src['nut_code']):
            for crop in crop_area_by_nut.columns.values:
                crop_distribution_src.loc[crop_distribution_src['nut_code'] == nut, crop] /= crop_distribution_src.loc[
                    crop_distribution_src['nut_code'] == nut, crop].sum()
        for nut in np.unique(crop_distribution_src['nut_code']):
            for crop in crop_area_by_nut.columns.values:

                crop_distribution_src.loc[crop_distribution_src['nut_code'] == nut, crop] *= \
                    crop_area_by_nut.loc[nut, crop]
        self.logger.write_time_log('AgriculturalSector', 'calculate_crop_distribution_src',
                                   timeit.default_timer() - spent_time)
        crop_distribution_src = IoShapefile(self.comm_agr).balance(crop_distribution_src)
        return crop_distribution_src

    def get_crop_distribution_in_dst_cells(self, crop_distribution):
        """
        Regrid the crop distribution in the source resolution to the grid resolution.

        :param crop_distribution: Crop distribution in source resolution.
        :type crop_distribution: pandas.GeoDataFrame

        :return: Crop by grid cell.
        :rtype: pandas.GeoDataFrame
        """
        spent_time = timeit.default_timer()
        crop_list = list(np.setdiff1d(crop_distribution.columns.values, ['NUT', 'geometry']))

        crop_distribution = crop_distribution.to_crs(self.grid.shapefile.crs)
        crop_distribution['src_inter_fraction'] = crop_distribution.geometry.area
        crop_distribution = self.spatial_overlays(crop_distribution.reset_index(), self.grid.shapefile.reset_index(),
                                                  how='intersection')

        crop_distribution = IoShapefile(self.comm_agr).balance(crop_distribution)
        crop_distribution['src_inter_fraction'] = \
            crop_distribution.geometry.area / crop_distribution['src_inter_fraction']

        crop_distribution[crop_list] = crop_distribution.loc[:, crop_list].multiply(
            crop_distribution["src_inter_fraction"], axis="index")

        crop_distribution = crop_distribution.loc[:, crop_list + ['FID']].groupby('FID').sum()

        crop_distribution = gpd.GeoDataFrame(crop_distribution, crs=self.grid.shapefile.crs,
                                             geometry=self.grid.shapefile.loc[crop_distribution.index, 'geometry'])
        crop_distribution.reset_index(inplace=True)
        crop_distribution.set_index('FID', inplace=True)

        self.logger.write_time_log('AgriculturalSector', 'get_crop_distribution_in_dst_cells',
                                   timeit.default_timer() - spent_time)
        return crop_distribution

    def get_crops_by_dst_cell(self, file_path):
        """
        Create, or read if it is already created, the crop distribution over the grid cells.

        The created crop distribution file contains all the available crops, but the returned shapefile only contains
        the involved crops on that sector.

        :param file_path: Path to the auxiliary file where is stored the crop distribution, or will be stored.
        :type file_path: str

        :return: GeoDataFrame with the crop distribution over the grid cells.
        :rtype: GeoDataFrame
        """
        spent_time = timeit.default_timer()
        if not os.path.exists(file_path):
            self.logger.write_log('Creating the crop distribution shapefile.', message_level=2)

            self.logger.write_log('Creating land use distribution on the source resolution.', message_level=3)
            involved_land_uses = self.get_involved_land_uses()
            land_use_distribution_src_nut = self.get_land_use_src_by_nut(involved_land_uses)

            land_use_by_nut = self.get_land_use_by_nut_csv(land_use_distribution_src_nut, involved_land_uses)

            tot_land_use_by_nut = self.get_tot_land_use_by_nut(involved_land_uses)

            self.logger.write_log('Creating the crop distribution on the source resolution.', message_level=3)
            crop_by_nut = self.land_use_to_crop_by_nut(land_use_by_nut)
            tot_crop_by_nut = self.land_use_to_crop_by_nut(
                tot_land_use_by_nut, nuts=list(np.unique(land_use_by_nut.index.get_level_values('nut_code'))))
            crop_shape_by_nut = self.get_crop_shape_by_nut(crop_by_nut, tot_crop_by_nut)
            crop_area_by_nut = self.get_crop_area_by_nut(crop_shape_by_nut)
            crop_distribution_src = self.calculate_crop_distribution_src(
                crop_area_by_nut, land_use_distribution_src_nut)

            self.logger.write_log('Creating the crop distribution on the grid resolution.', message_level=3)
            crop_distribution_dst = self.get_crop_distribution_in_dst_cells(crop_distribution_src)
            self.logger.write_log('Creating the crop distribution shapefile.', message_level=3)
            crop_distribution_dst = IoShapefile(self.comm_agr).gather_shapefile(crop_distribution_dst.reset_index())
            if self.comm_agr.Get_rank() == 0:
                crop_distribution_dst = crop_distribution_dst.groupby('FID').sum()
                crop_distribution_dst = GeoDataFrame(
                    crop_distribution_dst,
                    geometry=self.grid.shapefile.loc[crop_distribution_dst.index.get_level_values('FID'),
                                                     'geometry'].values,
                    crs=self.grid.shapefile.crs)
            else:
                crop_distribution_dst = None

            self.logger.write_log('Adding timezone to the shapefile.', message_level=3)
            crop_distribution_dst = IoShapefile(self.comm_agr).split_shapefile(crop_distribution_dst)
            crop_distribution_dst = self.add_timezone(crop_distribution_dst)

            self.logger.write_log('Writing the crop distribution shapefile.', message_level=3)
            IoShapefile(self.comm_agr).write_shapefile_parallel(crop_distribution_dst, file_path)

        crop_distribution_dst = IoShapefile(self.comm).read_shapefile_parallel(file_path)
        crop_distribution_dst.set_index('FID', inplace=True, drop=True)
        # Filtering crops by used on the sub-sector (operations, fertilizers, machinery)
        crop_distribution_dst = crop_distribution_dst.loc[:, self.crop_list + ['timezone', 'geometry']]

        self.logger.write_time_log('AgriculturalSector', 'get_crops_by_dst_cell', timeit.default_timer() - spent_time)
        return crop_distribution_dst

    @staticmethod
    def get_agricultural_processor_list(sector_dict):
        """
        Select the common ranks for that ones that will work on some agricultural sector.

        The agricultural sectors are 'crop_operations', 'crop_fertilizers' and 'agricultural_machinery'.

        :param sector_dict: Rank distribution for all the sectors.
        :type sector_dict: dict

        :return: List of ranks involved on some agricultural sector.
        :rtype: list
        """
        rank_list = []

        for sector, sector_procs in sector_dict.items():
            if sector in ['crop_operations', 'crop_fertilizers', 'agricultural_machinery']:
                rank_list += sector_procs
        rank_list = sorted(rank_list)
        return rank_list
