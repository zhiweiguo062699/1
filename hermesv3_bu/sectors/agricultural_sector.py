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
from hermesv3_bu.logger.log import Log


class AgriculturalSector(Sector):
    def __init__(self, comm_agr, comm, logger, auxiliary_dir, grid_shp, clip, date_array, nut_shapefile,
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
            comm, logger, auxiliary_dir, grid_shp, clip, date_array, source_pollutants, vertical_levels,
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
            # if element.crop in self.crop_list:
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
        df = pd.read_csv(self.land_use_by_nut)
        df = df.loc[df['land_use'].isin(land_uses), :]
        self.logger.write_time_log('AgriculturalSector', 'get_tot_land_use_by_nut', timeit.default_timer() - spent_time)

        return df

    def get_land_use_by_nut_csv(self, land_use_distribution_src_nut, land_uses):
        """

        :param land_use_distribution_src_nut: Shapefile with the polygons of all the land uses for each NUT.
        :type land_use_distribution_src_nut: GeoDataFrame

        :param land_uses: Land uses to take into account.
        :type land_uses: list

        :return:
        """
        spent_time = timeit.default_timer()

        land_use_distribution_src_nut['area'] = land_use_distribution_src_nut.area
        land_use_by_nut = land_use_distribution_src_nut.groupby(['NUT', 'land_use']).sum().reset_index()
        land_use_by_nut = land_use_by_nut.loc[land_use_by_nut['land_use'].isin(land_uses), :]

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
        new_dict.set_index('NUT', inplace=True)

        self.logger.write_time_log('AgriculturalSector', 'land_use_to_crop_by_nut', timeit.default_timer() - spent_time)
        return new_dict

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

        crop_by_nut = pd.read_csv(self.crop_by_nut)
        crop_by_nut.drop(columns='name', inplace=True)

        crop_by_nut['code'] = crop_by_nut['code'].astype(np.int16)
        crop_by_nut.set_index('code', inplace=True)
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

        crop_distribution_src = land_use_distribution_src_nut.loc[:, ['NUT', 'geometry']]
        for crop, landuse_weight_list in self.crop_from_landuse.iteritems():
            crop_distribution_src[crop] = 0
            for landuse, weight in landuse_weight_list:
                crop_distribution_src.loc[land_use_distribution_src_nut['land_use'] == int(landuse), crop] += \
                    land_use_distribution_src_nut.loc[land_use_distribution_src_nut['land_use'] == int(landuse),
                                                      'area'] * float(weight)
        for nut in np.unique(crop_distribution_src['NUT']):
            for crop in crop_area_by_nut.columns.values:
                crop_distribution_src.loc[crop_distribution_src['NUT'] == nut, crop] /= crop_distribution_src.loc[
                    crop_distribution_src['NUT'] == nut, crop].sum()
        for nut in np.unique(crop_distribution_src['NUT']):
            for crop in crop_area_by_nut.columns.values:

                crop_distribution_src.loc[crop_distribution_src['NUT'] == nut, crop] *= \
                    crop_area_by_nut.loc[nut, crop]
        self.logger.write_time_log('AgriculturalSector', 'calculate_crop_distribution_src',
                                   timeit.default_timer() - spent_time)

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

        crop_distribution = crop_distribution.to_crs(self.grid_shp.crs)
        crop_distribution['src_inter_fraction'] = crop_distribution.geometry.area
        crop_distribution = self.spatial_overlays(crop_distribution, self.grid_shp, how='intersection')
        crop_distribution['src_inter_fraction'] = \
            crop_distribution.geometry.area / crop_distribution['src_inter_fraction']

        crop_distribution[crop_list] = crop_distribution.loc[:, crop_list].multiply(
            crop_distribution["src_inter_fraction"], axis="index")

        crop_distribution = crop_distribution.loc[:, crop_list + ['FID']].groupby('FID').sum()

        crop_distribution = gpd.GeoDataFrame(crop_distribution, crs=self.grid_shp.crs,
                                             geometry=self.grid_shp.loc[crop_distribution.index, 'geometry'])
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
            if self.comm_agr.Get_rank() == 0:
                self.logger.write_log('Creating the crop distribution shapefile on the grid resolution.',
                                      message_level=2)
                involved_land_uses = self.get_involved_land_uses()

                land_use_distribution_src_nut = self.get_land_use_src_by_nut(involved_land_uses)

                land_use_by_nut = self.get_land_use_by_nut_csv(land_use_distribution_src_nut, involved_land_uses)
                tot_land_use_by_nut = self.get_tot_land_use_by_nut(involved_land_uses)

                crop_by_nut = self.land_use_to_crop_by_nut(land_use_by_nut)
                tot_crop_by_nut = self.land_use_to_crop_by_nut(
                    tot_land_use_by_nut, nuts=list(np.unique(land_use_by_nut['NUT'])))

                crop_shape_by_nut = self.get_crop_shape_by_nut(crop_by_nut, tot_crop_by_nut)
                crop_area_by_nut = self.get_crop_area_by_nut(crop_shape_by_nut)

                crop_distribution_src = self.calculate_crop_distribution_src(
                    crop_area_by_nut, land_use_distribution_src_nut)

                crop_distribution_dst = self.get_crop_distribution_in_dst_cells(crop_distribution_src)

                crop_distribution_dst = self.add_timezone(crop_distribution_dst)
                IoShapefile(self.comm).write_shapefile_serial(crop_distribution_dst, file_path)
            else:
                self.logger.write_log('Waiting for the master process that creates the crop distribution shapefile.',
                                      message_level=2)
                crop_distribution_dst = None
            self.comm_agr.Barrier()
            if self.comm.Get_rank() == 0 and self.comm_agr.Get_rank() != 0:
                # Every master rank read the created crop distribution shapefile.
                crop_distribution_dst = IoShapefile(self.comm).read_shapefile_serial(file_path)
            self.comm.Barrier()

            crop_distribution_dst = IoShapefile(self.comm).split_shapefile(crop_distribution_dst)
        else:
            crop_distribution_dst = IoShapefile(self.comm).read_shapefile_parallel(file_path)
        crop_distribution_dst.set_index('FID', inplace=True, drop=True)
        # Filtering crops by used on the subsector (operations, fertilizers, machinery)
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

        for sector, sector_procs in sector_dict.iteritems():
            if sector in ['crop_operations', 'crop_fertilizers', 'agricultural_machinery']:
                rank_list += sector_procs
        rank_list = sorted(rank_list)
        return rank_list
