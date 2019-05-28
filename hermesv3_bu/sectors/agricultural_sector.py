#!/usr/bin/env python

import sys
import os
from timeit import default_timer as gettime

import numpy as np
import pandas as pd
import geopandas as gpd

from hermesv3_bu.io_server.io_shapefile import IoShapefile


class AgriculturalSector(object):
    def __init__(self, comm, grid_shp, clipping, date_array, nut_shapefile, source_pollutants, element_list, input_file,
                 ef_files_dir, monthly_profiles, weekly_profiles, hourly_profiles, speciation_map, speciation_profiles,
                 molecular_weights):

        # self.grid_shp = IoShapefile().split_shapefile(grid_shp)
        self.grid_shp = grid_shp

        self.clipping = self.get_clip(clipping)
        self.date_array = date_array

        self.nut_shapefile = nut_shapefile
        self.source_pollutants = source_pollutants
        self.element_list = element_list
        self.input_file = input_file
        self.ef_files_dir = ef_files_dir
        self.monthly_profiles = self.read_monthly_profiles(monthly_profiles)
        self.weekly_profiles = self.read_weekly_profiles(weekly_profiles)
        self.hourly_profiles = self.read_hourly_profiles(hourly_profiles)
        self.speciation_map = speciation_map
        self.speciation_profiles = speciation_profiles
        self.molecular_weights = pd.read_csv(molecular_weights, sep=';')

        self.output_pollutants = None

    def involved_grid_cells(self, src_shp):

        grid_shp = IoShapefile(self.comm).split_shapefile(self.grid_shp)
        src_union = src_shp.to_crs(grid_shp.crs).geometry.unary_union
        grid_shp = grid_shp.loc[grid_shp.intersects(src_union), :]

        grid_shp_list = self.comm.gather(grid_shp, root=0)
        animal_dist_list = []
        if self.rank == 0:
            for small_grid in grid_shp_list:
                animal_dist_list.append(src_shp.loc[src_shp.intersects(small_grid.to_crs(src_shp.crs).geometry.unary_union), :])
            grid_shp = pd.concat(grid_shp_list)
            grid_shp = np.array_split(grid_shp, self.size)
        else:
            grid_shp = None
            animal_dist_list = None

        grid_shp = self.comm.scatter(grid_shp, root=0)

        animal_dist = self.comm.scatter(animal_dist_list, root=0)

        return grid_shp, animal_dist

    def calculate_num_days(self):
        import numpy as np
        from datetime import date


        day_array = [hour.date() for hour in self.date_array]
        days, num_days = np.unique(day_array, return_counts=True)

        day_dict = {}
        for key, value in zip(days, num_days):
            day_dict[key] = value

        return day_dict

    def get_clip(self, clip_path):

        if clip_path is not None:
            if os.path.exists(clip_path):
                clip = gpd.read_file(clip_path)
            else:
                raise IOError("The shapefile '{0}' does not exists to make the clip.".format(clip_path))
        else:
            clip = gpd.GeoDataFrame(geometry=[self.grid_shp.unary_union], crs=self.grid_shp.crs)

        return clip

    @staticmethod
    def read_monthly_profiles(path):
        """
        Read the Dataset of the monthly profiles with the month number as columns.

        :param path: Path to the file that contains the monthly profiles.
        :type path: str

        :return: Dataset od the monthly profiles.
        :rtype: pandas.DataFrame
        """
        import pandas as pd

        if path is None:
            return None
        profiles = pd.read_csv(path)

        profiles.rename(columns={'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7,
                                 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12},
                        inplace=True)
        return profiles

    @staticmethod
    def read_weekly_profiles(path):
        """
        Read the Dataset of the weekly profiles with the weekdays as numbers (Monday: 0 - Sunday:6) as columns.


        :param path: Path to the file that contains the weekly profiles.
        :type path: str

        :return: Dataset od the weekly profiles.
        :rtype: pandas.DataFrame
        """
        import pandas as pd

        if path is None:
            return None
        profiles = pd.read_csv(path)

        profiles.rename(columns={'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5,
                                 'Sunday': 6, }, inplace=True)

        return profiles

    @staticmethod
    def read_hourly_profiles(path):
        """
        Read the Dataset of the hourly profiles with the hours (int) as columns.

        :param path: Path to the file that contains the monthly profiles.
        :type path: str

        :return: Dataset od the monthly profiles.
        :rtype: pandas.DataFrame
        """
        import pandas as pd

        if path is None:
            return None
        profiles = pd.read_csv(path)
        profiles.rename(columns={'P_hour': -1, '00': 0, '01': 1, '02': 2, '03': 3, '04': 4, '05': 5, '06': 6, '07': 7,
                                 '08': 8, '09': 9, '10': 10, '11': 11, '12': 12, '13': 13, '14': 14, '15': 15, '16': 16,
                                 '17': 17, '18': 18, '19': 19, '20': 20, '21': 21, '22': 22, '23': 23}, inplace=True)
        profiles.columns = profiles.columns.astype(int)
        profiles.rename(columns={-1: 'P_hour'}, inplace=True)

        return profiles

    def get_molecular_weight(self, input_pollutant):

        moleculat_weight = self.molecular_weights.loc[self.molecular_weights['Specie'] == input_pollutant,
                                                      'MW'].values[0]

        return moleculat_weight

    def get_clipping_shapefile_path(self, auxiliary_dir):
        import geopandas as gpd

        if self.clipping is None:
            grid_shape_path = os.path.join(auxiliary_dir, 'shapefile', 'boundary_grid.shp')
            if self.comm.Get_rank() == 0:
                grid_shape_aux = gpd.read_file(os.path.join(auxiliary_dir, 'shapefile', 'grid_shapefile.shp'))
                grid_shape_aux = gpd.GeoDataFrame(geometry=[grid_shape_aux.unary_union], crs=grid_shape_aux.crs)

                grid_shape_aux.to_file(grid_shape_path)
        else:
            grid_shape_path = self.clipping

        return grid_shape_path

    @staticmethod
    def add_nut_code(shapefile, nut_shapefile_path, nut_value='ORDER06'):
        """
        Add 'nut_code' column into the shapefile based on the 'nut_value' column of the 'nut_shapefile_path' shapefile.

        The elements that are not into any NUT will be dropped.
        If an element belongs to two NUTs will be set the fist one that appear in the 'nut_shapefile_path' shapefile.

        :param shapefile: Shapefile where add the NUT code.
        :type shapefile: geopandas.GeoDataframe

        :param nut_shapefile_path: Path to the shapefile with the polygons that contains the NUT code into the
            'nut_value' column.
        :type nut_shapefile_path: str

        :param nut_value: Column name of the NUT codes.
        :type nut_value: str

        :return: Shapefile with the 'nut_code' column set.
        :rtype: geopandas.GeoDataframe
        """

        nut_shapefile = gpd.read_file(nut_shapefile_path).to_crs(shapefile.crs)
        shapefile = gpd.sjoin(shapefile, nut_shapefile.loc[:, [nut_value, 'geometry']], how='left', op='intersects')

        shapefile = shapefile[~shapefile.index.duplicated(keep='first')]
        shapefile.drop('index_right', axis=1, inplace=True)

        shapefile.rename(columns={nut_value: 'nut_code'}, inplace=True)
        shapefile.loc[shapefile['nut_code'].isna(), 'nut_code'] = -999
        shapefile['nut_code'] = shapefile['nut_code'].astype(np.int16)

        return shapefile

    def get_crop_from_land_uses(self, crop_from_landuse_path):
        import re

        crop_from_landuse = pd.read_csv(crop_from_landuse_path, sep=';')
        crop_dict = {}
        for i, element in crop_from_landuse.iterrows():
            if element.crop in self.element_list:
                land_uses = list(map(str, re.split(' , |, | ,|,| ', element.land_use)))
                weights = list(map(str, re.split(' , |, | ,|,| ', element.weight)))
                crop_dict[element.crop] = zip(land_uses, weights)

        return crop_dict

    def get_involved_land_uses(self):

        # return [12, 13]
        land_uses_list = []
        for land_use_and_weight_list in self.crop_from_landuse.itervalues():
            for land_use_and_weight in land_use_and_weight_list:
                land_use = int(land_use_and_weight[0])
                if land_use not in land_uses_list:
                    land_uses_list.append(land_use)

        return land_uses_list

    def get_land_use_src_by_nut(self, land_uses):

        # clip = gpd.read_file(self.clipping)

        df_land_use_with_nut = gpd.read_file(self.input_file)

        df_land_use_with_nut.rename(columns={'CODE': 'NUT', 'gridcode': 'land_use'}, inplace=True)

        df_land_use_with_nut = df_land_use_with_nut.loc[df_land_use_with_nut['land_use'].isin(land_uses), :]
        # clip = clip.to_crs(df_land_use_with_nut.crs)
        # df_land_use_with_nut = gpd.overlay(df_land_use_with_nut, clip, how='intersection')
        self.clipping = self.clipping.to_crs(df_land_use_with_nut.crs)

        df_land_use_with_nut = self.spatial_overlays(df_land_use_with_nut, self.clipping)
        # sys.exit()

        return df_land_use_with_nut

    def get_tot_land_use_by_nut(self, land_uses):

        df = pd.read_csv(self.landuse_by_nut)
        df = df.loc[df['land_use'].isin(land_uses), :]

        return df

    def get_land_use_by_nut_csv(self, land_use_distribution_src_nut, land_uses, first=False):

        land_use_distribution_src_nut['area'] = land_use_distribution_src_nut.area

        land_use_by_nut = land_use_distribution_src_nut.groupby(['NUT', 'land_use']).sum().reset_index()
        # land_use_by_nut['NUT'] = land_use_distribution_src_nut.groupby('NUT').apply(lambda x: str(x.name).zfill(2))
        # if first:
        #     land_use_by_nut.to_csv(self.landuse_by_nut, index=False)
        land_use_by_nut = land_use_by_nut.loc[land_use_by_nut['land_use'].isin(land_uses), :]

        return land_use_by_nut

    def land_use_to_crop_by_nut(self, land_use_by_nut, nuts=None):

        if nuts is not None:
            land_use_by_nut = land_use_by_nut.loc[land_use_by_nut['NUT'].isin(nuts), :]
        new_dict = pd.DataFrame()
        for nut in np.unique(land_use_by_nut['NUT']):
            aux_dict = {'NUT': [nut]}
            for crop, landuse_weight_list in self.crop_from_landuse.iteritems():
                aux = 0
                for landuse, weight in landuse_weight_list:
                    try:
                        aux += land_use_by_nut.loc[(land_use_by_nut['land_use'] == int(landuse)) & (land_use_by_nut['NUT'] == nut), 'area'].values[0] * float(weight)
                    except IndexError:
                        # TODO understand better that error
                        pass
                aux_dict[crop] = [aux]
            new_dict = new_dict.append(pd.DataFrame.from_dict(aux_dict), ignore_index=True)

        return new_dict

    def get_crop_shape_by_nut(self, crop_by_nut, tot_crop_by_nut):

        crop_share_by_nut = crop_by_nut.copy()
        crop_share_by_nut[self.element_list] = 0
        for crop in self.element_list:
            crop_share_by_nut[crop] = crop_by_nut[crop] / tot_crop_by_nut[crop]

        return crop_share_by_nut

    def get_crop_area_by_nut(self, crop_share_by_nut):

        self.crop_by_nut = pd.read_csv(self.crop_by_nut)
        self.crop_by_nut['code'] = self.crop_by_nut['code'].astype(np.int16)
        self.crop_by_nut = self.crop_by_nut.loc[self.crop_by_nut['code'].isin(np.unique(crop_share_by_nut['NUT'])),
                                                ['code'] + self.element_list].reset_index()

        crop_area_by_nut = crop_share_by_nut.copy()
        crop_area_by_nut[self.element_list] = 0
        for crop in self.element_list:
            crop_area_by_nut[crop] = crop_share_by_nut[crop] * self.crop_by_nut[crop]

        return crop_area_by_nut

    def calculate_crop_distribution_src(self, crop_area_by_nut, land_use_distribution_src_nut):

        crop_distribution_src = land_use_distribution_src_nut.loc[:, ['NUT', 'geometry']]
        for crop, landuse_weight_list in self.crop_from_landuse.iteritems():
            crop_distribution_src[crop] = 0
            for landuse, weight in landuse_weight_list:
                crop_distribution_src.loc[land_use_distribution_src_nut['land_use'] == int(landuse), crop] += \
                    land_use_distribution_src_nut.loc[land_use_distribution_src_nut['land_use'] == int(landuse),
                                                      'area'] * float(weight)

        for nut in np.unique(crop_distribution_src['NUT']):
            for crop in self.element_list:
                crop_distribution_src.loc[crop_distribution_src['NUT'] == nut, crop] /= \
                    crop_distribution_src.loc[crop_distribution_src['NUT'] == nut, crop].sum()
        for nut in np.unique(crop_distribution_src['NUT']):
            for crop in self.element_list:
                crop_distribution_src.loc[crop_distribution_src['NUT'] == nut, crop] *= \
                    crop_area_by_nut.loc[crop_area_by_nut['NUT'] == nut, crop].values[0]

        return crop_distribution_src

    def get_crop_distribution_in_dst_cells(self, crop_distribution):

        crop_distribution = crop_distribution.to_crs(self.grid_shp.crs)
        crop_distribution['src_inter_fraction'] = crop_distribution.geometry.area
        crop_distribution = self.spatial_overlays(crop_distribution, self.grid_shp, how='intersection')
        crop_distribution['src_inter_fraction'] = crop_distribution.geometry.area / \
                                                  crop_distribution['src_inter_fraction']

        crop_distribution[self.element_list] = crop_distribution.loc[:, self.element_list].multiply(
            crop_distribution["src_inter_fraction"], axis="index")

        crop_distribution = crop_distribution.loc[:, self.element_list + ['FID']].groupby('FID').sum()

        crop_distribution = gpd.GeoDataFrame(crop_distribution, crs=self.grid_shp.crs,
                                             geometry=self.grid_shp.loc[crop_distribution.index, 'geometry'])
        crop_distribution.reset_index(inplace=True)

        return crop_distribution

    def get_crops_by_dst_cell(self, file_path, grid_shapefile, clipped_tiff_path):

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

        return crop_distribution_dst

    @staticmethod
    def find_lonlat_index(lon, lat, lon_min, lon_max, lat_min, lat_max):
        """
        Find the NetCDF index to extract all the data avoiding the maximum of unused data.

        :param lon: Longitudes array from the NetCDF.
        :type lon: numpy.array

        :param lat: Latitude array from the NetCDF.
        :type lat: numpy.array

        :param lon_min: Minimum longitude of the point for the needed date.
        :type lon_min float

        :param lon_max: Maximum longitude of the point for the needed date.
        :type lon_max: float

        :param lat_min: Minimum latitude of the point for the needed date.
        :type lat_min: float

        :param lat_max: Maximum latitude of the point for the needed date.
        :type lat_max: float

        :return: Tuple with the four index of the NetCDF
        :rtype: tuple
        """
        import numpy as np

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

        return i_min, i_max + 1, j_min, j_max + 1

    @staticmethod
    def nearest(row, geom_union, df1, df2, geom1_col='geometry', geom2_col='geometry', src_column=None):
        """Finds the nearest point and return the corresponding value from specified column.
        https://automating-gis-processes.github.io/2017/lessons/L3/nearest-neighbour.html#nearest-points-using-geopandas
        """
        from shapely.ops import nearest_points

        # Find the geometry that is closest
        nearest = df2[geom2_col] == nearest_points(row[geom1_col], geom_union)[1]
        # Get the corresponding value from df2 (matching is based on the geometry)
        value = df2[nearest][src_column].get_values()[0]

        return value

    def get_data_from_netcdf(self, netcdf_path, var_name, date_type, date, geometry_df):
        """
        Read for extract a NetCDF variable in the desired points.

        :param netcdf_path: Path to the NetCDF that contains the data to extract.
        :type netcdf_path: str

        :param var_name: Name of the NetCDF variable to extract.
        :type var_name: str

        :param date_type: Option to set if we want to extract a 'daily' variable or a 'yearly' one.
        :type date_type: str

        :param date: Date of the day to extract.
        :type date: datetime.date

        :param geometry_df: GeoDataframe with the point where extract the variables.
        :type geometry_df: geopandas.GeoDataframe

        :return: GeoDataframe with the data in the desired points.
        :rtype: geopandas.GeoDataframe
        """
        from netCDF4 import Dataset
        from shapely.geometry import Point
        from cf_units import num2date, CALENDAR_STANDARD

        nc = Dataset(netcdf_path, mode='r')
        lat_o = nc.variables['latitude'][:]
        lon_o = nc.variables['longitude'][:]

        if date_type == 'daily':
            time = nc.variables['time']
            # From time array to list of dates.
            time_array = num2date(time[:], time.units, CALENDAR_STANDARD)
            time_array = np.array([aux.date() for aux in time_array])
            i_time = np.where(time_array == date)[0][0]
        elif date_type == 'yearly':
            i_time = 0

        # Find the index to read all the necessary information but avoiding to read as many unused data as we can
        i_min, i_max, j_min, j_max = self.find_lonlat_index(
            lon_o, lat_o, geometry_df['c_lon'].min(), geometry_df['c_lon'].max(),
            geometry_df['c_lat'].min(), geometry_df['c_lat'].max())

        # Clips the lat lons
        lon_o = lon_o[i_min:i_max]
        lat_o = lat_o[j_min:j_max]

        # From 1D to 2D
        lat = np.array([lat_o[:]] * len(lon_o[:])).T.flatten()
        lon = np.array([lon_o[:]] * len(lat_o[:])).flatten()
        del lat_o, lon_o

        # Reads the tas variable of the xone and the times needed.
        var = nc.variables[var_name][i_time, j_min:j_max, i_min:i_max]
        nc.close()

        var_df = gpd.GeoDataFrame(var.flatten().T, columns=[var_name], crs={'init': 'epsg:4326'},
                                  geometry=[Point(xy) for xy in zip(lon, lat)])
        var_df.loc[:, 'REC'] = var_df.index


        return var_df

    def to_grid(self, data, grid):
        import pandas as pd

        data = self.comm.gather(data, root=0)
        if self.comm.Get_rank() == 0:
            data = pd.concat(data)

            emission_list = []
            for out_p in self.output_pollutants:
                aux_data = data.loc[:, [out_p, 'tstep', 'FID']]
                aux_data = aux_data.loc[aux_data[out_p] > 0, :]
                dict_aux = {
                    'name': out_p,
                    'units': '',
                    'data': aux_data
                }
                # print dict_aux
                emission_list.append(dict_aux)
        else:
            emission_list = None

        return emission_list
