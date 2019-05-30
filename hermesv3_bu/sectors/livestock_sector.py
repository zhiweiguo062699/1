#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import timeit
from hermesv3_bu.logger.log import Log
from warnings import warn

from hermesv3_bu.sectors.sector import Sector
from hermesv3_bu.io_server.io_shapefile import IoShapefile
from hermesv3_bu.io_server.io_raster import IoRaster
from hermesv3_bu.io_server.io_netcdf import IoNetcdf

# Constants for grassing daily factor estimation
SIGMA = 60
TAU = 170


class LivestockSector(Sector):
    """
    Class that contains all the information and methods to calculate the livestock emissions.
    """
    def __init__(self, comm, logger, auxiliary_dir, grid_shp, clip, date_array, source_pollutants, vertical_levels,
                 animal_list, gridded_livestock_path, correction_split_factors_path, temperature_dir, wind_speed_dir,
                 denominator_yearly_factor_dir, ef_dir, monthly_profiles_path, weekly_profiles_path,
                 hourly_profiles_path, speciation_map_path, speciation_profiles_path, molecular_weights_path,
                 nut_shapefile_path):
        """
        :param comm: MPI Communicator

        :param logger: Logger
        :type logger: Log

        :param auxiliary_dir: Path to the directory where the necessary auxiliary files will be created if them are
            not created yet.
        :type auxiliary_dir: str

        :param source_pollutants: List of input pollutants to take into account. Agricultural livestock module can
            calculate emissions derived from the next source pollutants: NH3, NOx expressed as NO, NMVOC, PM10 and
            PM2.5
            ['nox_no', 'nh3', 'nmvoc', 'pm10', 'pm25']
        :type source_pollutants: list

        :param grid_shp: Shapefile that contains the destination grid. It must contains the 'FID' (cell num).
        :type grid_shp: GeoPandas.GeoDataframe

        :param clip: Clip.
        :type clip: Clip

        :param animal_list: List of animals to take into account.
        :type animal_list: list

        :param gridded_livestock_path: Path to the Raster that contains the animal distribution.
            '<animal>' will be replaced by each animal of the animal list.
        :type gridded_livestock_path: str

        :param correction_split_factors_path: Path to the CSV file that contains the correction factors and the
            splitting factors to discretizise each animal into theirs different animal types.
            '<animal>' will be replaced by each animal of the animal list.
            The CSV file must contain the following columns ["NUT", "nut_code", "<animal>_fact", "<animal>_01",
            ...]
            "nut_code" column must contain the NUT ID
        :type correction_split_factors_path: str

        :param date_array: List of datetimes.
        :type date_array: list(datetime.datetime, ...)

        :param temperature_dir: Path to the directory that contains the needed temperature files. The temperature
            file names have to follow the 'tas_<YYYYMM>.nc' convention where YYYY is the year and MM the month.
            (e.g. 'tas_201601.nc')
            That NetCDF file have to contain:
                - 'time', 'longitude' and 'latitude' dimensions.
                - As many times as days of the month.
                - 'latitude' variable
                - 'longitude' variable
                - 'tas' variable (time, latitude, longitude), 2m temperature, Kelvins
        :type temperature_dir: str

        :param wind_speed_dir: Path to the directory that contains the needed wind speed files. The wind speed file
            names have to follow the 'sfcWind_<YYYYMM>.nc' convention where YYYY is the year and MM the month.
            (e.g. 'scfWind_201601.nc')
            That NetCDF file have to contain:
                - 'time', 'longitude' and 'latitude' dimensions.
                - As many times as days of the month.
                - 'latitude' variable
                - 'longitude' variable
                - 'sfcWind' variable (time, latitude, longitude), 10 m wind speed, m/s
        :type wind_speed_dir: str

        :param denominator_yearly_factor_dir: Path to the directory that contains the needed denominator files.
            The denominator file names have to follow the 'grassing_<YYYY>.nc' convention where YYYY is the year.
            Have to contains grassing, house_closed, house_open and storage denominators files.
            (e.g. 'grassing_2016.nc')
            That NetCDF file have to contain:
            - 'time', 'longitude' and 'latitude' dimensions.
            - One time value
            - 'latitude' variable
            - 'longitude' variable
            - 'FD' variable (time, latitude, longitude)
        :type denominator_yearly_factor_dir: str

        :param ef_dir: Path to the CSV file that contains all the information to calculate the emissions for each
            input pollutant.
            - PM10 (pm10) & PM2.5 (pm25) use the same emission factor file 'pm.csv' with the following columns
                ["Code", "Xhousing", "EF_pm10", "EF_pm25"]
            - NH3 'nh3.csv' with the following columns ["Code", "Nex", "Xtan", "Xhousing", "Xyards", "Xgraz",
                "Xslurry", "Xsolid", "EF_hous_slurry", "EF_hous_solid", "EF_yard", "f_imm", "m_bedding_N",
                "x_store_slurry", "x_store_FYM", "f_min", "EF_storage_slurry_NH3", "EF_storage_slurry_N20",
                "EF_storage_slurry_NO", "EF_storage_slurry_N2", "EF_storage_solid_NH3", "EF_storage_solid_N2O",
                "EF_storage_solid_NO", "EF_storage_solid_N2", "EF_graz"]
            - NMVOC 'nmvoc.csv' with the following columns ["Code", "Xhousing", "EF_nmvoc"]
            - NOx 'nox_no.csv' with the following columns [Code, Nex, Xtan, Xhousing, Xyards, Xgraz, Xslurry,
                Xsolid, EF_hous_slurry, EF_hous_solid, EF_yard, f_imm, m_bedding_N, x_store_slurry, x_store_FYM,
                f_min, EF_storage_slurry_NO, EF_storage_solid_NO]
            Each csv file have to contain as many rows as animal types with their codes on the "Code" column.
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

        :param nut_shapefile_path: Path to the shapefile that contain the NUT polygons. The shapefile must contain
            the 'ORDER07' information with the NUT_code.
        :type nut_shapefile_path: str
        """
        spent_time = timeit.default_timer()
        logger.write_log('===== LIVESTOCK SECTOR =====')
        super(LivestockSector, self).__init__(
            comm, logger, auxiliary_dir, grid_shp, clip, date_array, source_pollutants, vertical_levels,
            monthly_profiles_path, weekly_profiles_path, hourly_profiles_path, speciation_map_path,
            speciation_profiles_path, molecular_weights_path)

        # Common
        self.animal_list = animal_list
        self.day_dict = self.calculate_num_days()

        # Paths
        self.ef_dir = ef_dir
        self.paths = {
            'temperature_dir': temperature_dir,
            'wind_speed_dir': wind_speed_dir,
            'denominator_dir': denominator_yearly_factor_dir,
            'ef_dir': ef_dir,
        }

        # Creating dst resolution shapefile with the amount of animals
        self.animals_df = self.create_animals_distribution(gridded_livestock_path, nut_shapefile_path,
                                                           correction_split_factors_path)
        self.logger.write_time_log('LivestockSector', '__init__', timeit.default_timer() - spent_time)

    def create_animals_distribution(self, gridded_livestock_path, nut_shapefile_path, correction_split_factors_path):
        """
        Get and distribute the animal distribution between the MPI process.

        The creation of the shapefile belong to the master process.

        :param gridded_livestock_path: Path to the Raster (TIFF) that contains the animal distribution.
            '<animal>' will be replaced by each animal of the animal list.
        :type gridded_livestock_path: str

        :param nut_shapefile_path: Path to the shapefile that contain the NUT polygons. The shapefile must contain
            the 'ORDER07' information with the NUT ID.
        :type nut_shapefile_path: str

        :param correction_split_factors_path: Path to the CSV file that contains the correction factors and the
            splitting factors to discretizise each animal into theirs different animal types.
            '<animal>' will be replaced by each animal of the animal list.

            The CSV file must contain the following columns ["NUT", "nut_code", "<animal>_fact", "<animal>_01", ...]
            "nut_code" column must contain the NUT ID.
        :type correction_split_factors_path: str

        :return: GeoDataframe with the amount of each animal subtype by destiny cell (FID)
            Columns:
            'FID', 'cattle_01', 'cattle_02', 'cattle_03' 'cattle_04', 'cattle_05', 'cattle_06', 'cattle_07',
            'cattle_08', 'cattle_09', 'cattle_10', 'cattle_11', 'chicken_01', 'chicken_02', 'goats_01', 'goats_02',
            'goats_03', goats_04', 'goats_05',  'goats_06', 'pigs_01', 'pigs_02', 'pigs_03', 'pigs_04', 'pigs_05',
            'pigs_06', 'pigs_07', 'pigs_08', 'pigs_09', 'pigs_10', 'timezone',  'geometry'
        :rtype: geopandas.GeoDataframe
        """
        spent_time = timeit.default_timer()
        # Work for master MPI process
        if self.comm.Get_rank() == 0:
            animals_df = self.create_animals_shapefile(gridded_livestock_path)
            animals_df = self.animal_distribution_by_category(animals_df, nut_shapefile_path,
                                                              correction_split_factors_path)
        else:
            animals_df = None

        # Split distribution, in a balanced way, between MPI process
        animals_df = IoShapefile(self.comm).split_shapefile(animals_df)
        self.logger.write_log('Animal distribution done', message_level=2)
        self.logger.write_time_log('LivestockSector', 'create_animals_distribution',
                                   timeit.default_timer() - spent_time)

        return animals_df

    def calculate_num_days(self):
        """
        Create a dictionary with the day as key and num oh hours as value.

        :return: Dictionary with the day as key and num oh hours as value.
        :rtype: dict
        """
        spent_time = timeit.default_timer()
        day_array = [hour.date() for hour in self.date_array]
        days, num_days = np.unique(day_array, return_counts=True)

        day_dict = {}
        for key, value in zip(days, num_days):
            day_dict[key] = value
        self.logger.write_time_log('LivestockSector', 'calculate_num_days', timeit.default_timer() - spent_time)

        return day_dict

    def create_animals_shapefile_src_resolution(self, gridded_livestock_path):
        """
        Create the animal shapefile in the same resolution as the raster have.

        It will return a complete shapefile with the amount of animals of the animal list.
        That shapefile will contain as many columns as animal types of the list ant the 'CELL_ID' as index.

        Each one of the animal distributions will be stored separately in folders following the example path
        <auxiliary_dir>/livestock/animal_distribution/<animal>/<animal>.shp

        Will be created also the clipped raster (TIFF) following the example path
        <auxiliary_dir>/livestock/animal_distribution/<animal>/<animal>_clip.tiff

        :param gridded_livestock_path: Path to the Raster (TIFF) that contains the animal distribution.
            '<animal>' will be replaced by each animal of the animal list.
        :type gridded_livestock_path: str

        :return: Shapefile with the amount of each animal of the animal list in the source resolution.
        :rtype: geopandas.GeoDataframe
        """
        spent_time = timeit.default_timer()
        animal_distribution = None
        # For each one of the animals of the animal list
        for animal in self.animal_list:
            # Each one of the animal distributions will be stored separately
            animal_distribution_path = os.path.join(self.auxiliary_dir, 'livestock', 'animal_distribution', animal,
                                                    '{0}.shp'.format(animal))
            if not os.path.exists(animal_distribution_path):
                # Create clipped raster file
                clipped_raster_path = IoRaster(self.comm).clip_raster_with_shapefile_poly(
                    gridded_livestock_path.replace('<animal>', animal), self.clip.shapefile,
                    os.path.join(self.auxiliary_dir, 'livestock', 'animal_distribution', animal,
                                 '{0}_clip.tiff'.format(animal)))

                animal_df = IoRaster(self.comm).to_shapefile(clipped_raster_path, animal_distribution_path, write=True)
            else:
                animal_df = IoShapefile(self.comm).read_shapefile(animal_distribution_path)

            animal_df.rename(columns={'data': animal}, inplace=True)
            animal_df.set_index('CELL_ID', inplace=True)

            # Creating full animal shapefile
            if animal_distribution is None:
                # First animal type of the list
                animal_distribution = animal_df
            else:
                # Adding new animal distribution values
                animal_distribution = pd.concat([animal_distribution, animal_df.loc[:, animal]], axis=1)
                # Adding new cell geometries that have not appear in the previous animals
                animal_distribution['geometry'] = animal_distribution['geometry'].fillna(animal_df['geometry'])

        # Removing empty data
        animal_distribution = animal_distribution.loc[(animal_distribution[self.animal_list] != 0).any(axis=1), :]
        self.logger.write_time_log('LivestockSector', 'create_animals_shapefile_src_resolution',
                                   timeit.default_timer() - spent_time)

        return animal_distribution

    def animals_shapefile_to_dst_resolution(self, animal_distribution):
        """
        Interpolates the source distribution into the destiny grid.

        :param animal_distribution: Animal distribution shapefile in the source resolution.
        :type animal_distribution: geopandas.GeoDataframe

        :return: Animal distribution shapefile in the destiny resolution.
        :rtype: geopandas.GeoDataframe
        """
        spent_time = timeit.default_timer()
        self.grid_shp.reset_index(inplace=True, drop=True)
        # Changing coordinates sistem to the grid one
        animal_distribution.to_crs(self.grid_shp.crs, inplace=True)
        # Getting src area
        animal_distribution['src_inter_fraction'] = animal_distribution.geometry.area

        # Making the intersection between the src distribution and the destiny grid
        animal_distribution = self.spatial_overlays(animal_distribution, self.grid_shp, how='intersection')
        # Getting proportion of intersecion in the src cell (src_area/portion_area)
        animal_distribution['src_inter_fraction'] = \
            animal_distribution.geometry.area / animal_distribution['src_inter_fraction']
        # Applying proportion to src distribution
        animal_distribution[self.animal_list] = animal_distribution.loc[:, self.animal_list].multiply(
            animal_distribution["src_inter_fraction"], axis="index")
        # Sum by destiny cell
        animal_distribution = animal_distribution.loc[:, self.animal_list + ['FID']].groupby('FID').sum()

        self.grid_shp.set_index('FID', drop=False, inplace=True)
        # Adding geometry and coordinates system from the destiny grid shapefile
        animal_distribution = gpd.GeoDataFrame(animal_distribution, crs=self.grid_shp.crs,
                                               geometry=self.grid_shp.loc[animal_distribution.index, 'geometry'])
        animal_distribution.reset_index(inplace=True)
        self.logger.write_time_log('LivestockSector', 'animals_shapefile_to_dst_resolution',
                                   timeit.default_timer() - spent_time)

        return animal_distribution

    def create_animals_shapefile(self, gridded_livestock_path):
        """
        Create the animal distribution shapefile into the destiny resolution grid.

        That auxiliary file will be stored in '<auxiliary_dir>/livestock/animal_distribution/animal_distribution.shp'
        path.

        Work done on master only once. In the rest of simulations, master will read the work done previously.

        :param gridded_livestock_path: Path to the Raster (TIFF) that contains the animal distribution.
            '<animal>' will be replaced by each animal of the animal list.
        :type gridded_livestock_path: str

        :return:
        """
        spent_time = timeit.default_timer()
        animal_distribution_path = os.path.join(self.auxiliary_dir, 'livestock', 'animal_distribution',
                                                'animal_distribution.shp')

        if not os.path.exists(animal_distribution_path):
            dataframe = self.create_animals_shapefile_src_resolution(gridded_livestock_path)
            dataframe = self.animals_shapefile_to_dst_resolution(dataframe)
            IoShapefile().write_serial_shapefile(dataframe, animal_distribution_path)
        else:
            dataframe = IoShapefile().read_serial_shapefile(animal_distribution_path)
        self.logger.write_time_log('LivestockSector', 'create_animals_shapefile', timeit.default_timer() - spent_time)

        return dataframe

    def get_splitting_factors(self, correction_split_factors_path):
        """
        Gather all the splitting factors for each sub animal.

        It will multiply each splitting factor by their correction factor.
        The result dataframe have to have the nut_code column and all the animal subtype percentages.

        :param correction_split_factors_path: Path to the CSV file that contains the correction factors and the
            splitting factors to discretizise each animal into theirs different animal types.
            '<animal>' will be replaced by each animal of the animal list.

            The CSV file must contain the following columns ["NUT", "nut_code", "<animal>_fact", "<animal>_01", ...]
            "nut_code" column must contain the NUT ID.
        :type correction_split_factors_path: str

        :return: Dataframe with the nut_code column and all the animal subtype percentages.
        :rtype : pandas.Dataframe
        """
        spent_time = timeit.default_timer()
        splitting_factors_list = []
        for animal in self.animal_list:
            correction_split_factors = pd.read_csv(correction_split_factors_path.replace('<animal>', animal))
            correction_split_factors.set_index('nut_code', inplace=True)

            categories = list(correction_split_factors.columns.values)
            categories = [e for e in categories if e not in ['NUT', 'nut_code', '{0}_fact'.format(animal)]]

            correction_split_factors[categories] = correction_split_factors.loc[:, categories].multiply(
                correction_split_factors['{0}_fact'.format(animal)], axis='index')
            correction_split_factors.drop(columns=['NUT', '{0}_fact'.format(animal)], inplace=True)
            splitting_factors_list.append(correction_split_factors)
        splitting_factors = pd.concat(splitting_factors_list, axis=1)

        splitting_factors.reset_index(inplace=True)
        splitting_factors['nut_code'] = splitting_factors['nut_code'].astype(np.int16)
        self.logger.write_time_log('LivestockSector', 'get_splitting_factors', timeit.default_timer() - spent_time)

        return splitting_factors

    def animal_distribution_by_category(self, dataframe, nut_shapefile_path, correction_split_factors_path):
        """
        Split the animal categories into as many categories as each animal type has.

        :param dataframe: GeoDataframe with the animal distribution by animal type.
        :type dataframe: geopandas.GeoDataframe

        :param nut_shapefile_path: Path to the shapefile that contain the NUT polygons. The shapefile must contain
            the 'ORDER07' information with the NUT_code.
        :type nut_shapefile_path: str

        :param correction_split_factors_path: Path to the CSV file that contains the correction factors and the
            splitting factors to discretizise each animal into theirs different animal types.
            '<animal>' will be replaced by each animal of the animal list.
            The CSV file must contain the following columns ["NUT", "nut_code", "<animal>_fact", "<animal>_01",
            ...]
            "nut_code" column must contain the NUT ID
        :type correction_split_factors_path: str

        :return: GeoDataframe with the amount of each animal subtype by destiny cell (FID)
            Columns:
            'FID', 'cattle_01', 'cattle_02', 'cattle_03' 'cattle_04', 'cattle_05', 'cattle_06', 'cattle_07',
            'cattle_08', 'cattle_09', 'cattle_10', 'cattle_11', 'chicken_01', 'chicken_02', 'goats_01', 'goats_02',
            'goats_03', goats_04', 'goats_05',  'goats_06', 'pigs_01', 'pigs_02', 'pigs_03', 'pigs_04', 'pigs_05',
            'pigs_06', 'pigs_07', 'pigs_08', 'pigs_09', 'pigs_10', 'timezone',  'geometry'
        :rtype: geopandas.GeoDataframe
        """
        spent_time = timeit.default_timer()
        animal_distribution_path = os.path.join(self.auxiliary_dir, 'livestock', 'animal_distribution',
                                                'animal_distribution_by_cat.shp')

        if not os.path.exists(animal_distribution_path):
            dataframe = self.add_nut_code(dataframe, nut_shapefile_path, nut_value='ORDER07')

            splitting_factors = self.get_splitting_factors(correction_split_factors_path)

            # Adding the splitting factors by NUT code
            dataframe = pd.merge(dataframe, splitting_factors, how='left', on='nut_code')

            dataframe.drop(columns=['nut_code'], inplace=True)

            for animal in self.animal_list:
                animal_types = [i for i in list(dataframe.columns.values) if i.startswith(animal)]
                dataframe.loc[:, animal_types] = dataframe.loc[:, animal_types].multiply(dataframe[animal],
                                                                                         axis='index')
                dataframe.drop(columns=[animal], inplace=True)

            dataframe = self.add_timezone(dataframe)
            IoShapefile().write_serial_shapefile(dataframe, animal_distribution_path)
        else:
            dataframe = IoShapefile().read_serial_shapefile(animal_distribution_path)
        self.logger.write_time_log('LivestockSector', 'animal_distribution_by_category',
                                   timeit.default_timer() - spent_time)

        return dataframe

    def get_daily_factors(self, animal_shp, day):
        """
        Calculate the daily factors necessaries.

        This function returns a shapefile with the following columns:
        - 'REC': ID number of the destiny cell.
        - 'FD_housing_open': Daily factor for housing open emissions.
        - 'FD_housing_closed': Daily factor for housing close emissions.
        - 'FD_storage': Daily factor for storage emissions.
        - 'FD_grassing': Daily factor for grassing emissions.
        - 'geometry': Destiny cell geometry

        :param animal_shp: GeoDataframe with the amount of each animal subtype by destiny cell (FID)
            Columns:
            'FID', 'cattle_01', 'cattle_02', 'cattle_03' 'cattle_04', 'cattle_05', 'cattle_06', 'cattle_07',
            'cattle_08', 'cattle_09', 'cattle_10', 'cattle_11', 'chicken_01', 'chicken_02', 'goats_01', 'goats_02',
            'goats_03', goats_04', 'goats_05',  'goats_06', 'pigs_01', 'pigs_02', 'pigs_03', 'pigs_04', 'pigs_05',
            'pigs_06', 'pigs_07', 'pigs_08', 'pigs_09', 'pigs_10', 'timezone',  'geometry'
        :type animal_shp: geopandas.GeoDataframe

        :param day: Date of the day to generate.
        :type day: datetime.date

        :return: Shapefile with the daily factors.
        :rtype: geopandas.GeoDataframe
        """
        import math
        spent_time = timeit.default_timer()
        # Extract the points where we want meteorological parameters
        geometry_shp = animal_shp.loc[:, ['FID', 'geometry']].to_crs({'init': 'epsg:4326'})
        geometry_shp['c_lat'] = geometry_shp.centroid.y
        geometry_shp['c_lon'] = geometry_shp.centroid.x
        geometry_shp['centroid'] = geometry_shp.centroid
        geometry_shp.drop(columns='geometry', inplace=True)

        # Extracting temperature
        meteo = IoNetcdf(self.comm).get_data_from_netcdf(
            os.path.join(self.paths['temperature_dir'], 'tas_{0}{1}.nc'.format(day.year, str(day.month).zfill(2))),
            'tas', 'daily', day, geometry_shp)
        meteo['tas'] = meteo['tas'] - 273.15  # From Celsius to Kelvin degrees
        # Extracting wind speed
        meteo['sfcWind'] = IoNetcdf(self.comm).get_data_from_netcdf(
            os.path.join(self.paths['wind_speed_dir'], 'sfcWind_{0}{1}.nc'.format(day.year, str(day.month).zfill(2))),
            'sfcWind', 'daily', day, geometry_shp).loc[:, 'sfcWind']

        # Extracting denominators already calculated for all the emission types
        meteo['D_grassing'] = IoNetcdf(self.comm).get_data_from_netcdf(
            os.path.join(self.paths['denominator_dir'], 'grassing_{0}.nc'.format(day.year)),
            'FD', 'yearly', day, geometry_shp).loc[:, 'FD']
        meteo['D_housing_closed'] = IoNetcdf(self.comm).get_data_from_netcdf(
            os.path.join(self.paths['denominator_dir'], 'housing_closed_{0}.nc'.format(day.year)),
            'FD', 'yearly', day, geometry_shp).loc[:, 'FD']
        meteo['D_housing_open'] = IoNetcdf(self.comm).get_data_from_netcdf(
            os.path.join(self.paths['denominator_dir'], 'housing_open_{0}.nc'.format(day.year)),
            'FD', 'yearly', day, geometry_shp).loc[:, 'FD']
        meteo['D_storage'] = IoNetcdf(self.comm).get_data_from_netcdf(
            os.path.join(self.paths['denominator_dir'], 'storage_{0}.nc'.format(day.year)),
            'FD', 'yearly', day, geometry_shp).loc[:, 'FD']

        # ===== Daily Factor for housing open =====
        meteo.loc[meteo['tas'] < 1, 'FD_housing_open'] = ((4.0 ** 0.89) * (0.228 ** 0.26)) / (meteo['D_housing_open'])
        meteo.loc[meteo['tas'] >= 1, 'FD_housing_open'] = (((meteo['tas'] + 3.0) ** 0.89) * (0.228 ** 0.26)) / (
            meteo['D_housing_open'])

        # ===== Daily Factor for housing closed =====
        meteo.loc[meteo['tas'] < 0, 'FD_housing_closed'] = \
            ((np.maximum([0], 18.0 + meteo['tas'].multiply(0.5)) ** 0.89) * (0.2 ** 0.26)) / (meteo['D_housing_closed'])
        meteo.loc[(meteo['tas'] >= 0) & (meteo['tas'] <= 12.5), 'FD_housing_closed'] = \
            ((18.0 ** 0.89) * ((0.2 + meteo['tas'].multiply((0.38 - 0.2) / 12.5)) ** 0.26)) / (
                meteo['D_housing_closed'])
        meteo.loc[meteo['tas'] > 12.5, 'FD_housing_closed'] = \
            (((18.0 + (meteo['tas'] - 12.5).multiply(0.77)) ** 0.89) * (0.38 ** 0.26)) / (meteo['D_housing_closed'])

        # ===== Daily Factor for storage =====
        meteo.loc[meteo['tas'] < 1, 'FD_storage'] = ((1 ** 0.89) * (meteo['sfcWind'] ** 0.26)) / (meteo['D_storage'])
        meteo.loc[meteo['tas'] >= 1, 'FD_storage'] = \
            ((meteo['tas'] ** 0.89) * (meteo['sfcWind'] ** 0.26)) / (meteo['D_storage'])

        # ===== Daily Factor for grassing =====
        meteo.loc[:, 'FD_grassing'] = \
            (np.exp(meteo['tas'].multiply(0.0223)) * np.exp(meteo['sfcWind'].multiply(0.0419))) / (meteo['D_grassing'])
        meteo.loc[:, 'FD_grassing'] = \
            meteo.loc[:, 'FD_grassing'].multiply((1 / (SIGMA * math.sqrt(2 * math.pi))) * math.exp(
                (float(int(day.strftime('%j')) - TAU) ** 2) / (-2 * (SIGMA ** 2))))

        self.logger.write_time_log('LivestockSector', 'get_daily_factors', timeit.default_timer() - spent_time)

        return meteo.loc[:, ['REC', 'FD_housing_open', 'FD_housing_closed', 'FD_storage', 'FD_grassing', 'geometry']]

    def get_nh3_ef(self):
        """
        Calculate the emission factor for yarding, grazing, housing and storage emissions

        :return: Dataframe with the Emission factors as columns and animal subtypes as 'Code'
        """
        spent_time = timeit.default_timer()
        ef_df = pd.read_csv(os.path.join(self.ef_dir, 'nh3.csv'))

        new_df = ef_df.loc[:, ['Code']]

        new_df['EF_yarding'] = ef_df['Nex'] * ef_df['Xtan'] * ef_df['Xyards'] * ef_df['EF_yard']
        new_df['EF_grazing'] = ef_df['Nex'] * ef_df['Xtan'] * ef_df['Xgraz'] * ef_df['EF_graz']
        new_df['EF_housing'] = \
            ef_df['Nex'] * ef_df['Xtan'] * ef_df['Xhousing'] * \
            ((ef_df['EF_hous_slurry'] * ef_df['Xslurry']) + (ef_df['EF_hous_solid'] * ef_df['Xsolid']))

        new_df['Estorage_sd_l'] = \
            ((((ef_df['Nex'] * ef_df['Xhousing'] * ef_df['Xtan'] * ef_df['Xsolid']) -
               (ef_df['Nex'] * ef_df['Xhousing'] * ef_df['Xtan'] * ef_df['Xsolid'] * ef_df['EF_hous_solid'])) *
              (1 - ef_df['f_imm'])) * ef_df['x_store_FYM']) * ef_df['EF_storage_solid_NH3']
        new_df['Mstorage_slurry_TAN'] = \
            (((ef_df['Nex'] * ef_df['Xhousing'] * ef_df['Xtan'] * ef_df['Xslurry']) -
              (ef_df['Nex'] * ef_df['Xhousing'] * ef_df['Xtan'] * ef_df['Xslurry'] * ef_df['EF_hous_slurry'])) +
             ((ef_df['Nex'] * ef_df['Xyards'] * ef_df['Xtan']) -
              (ef_df['Nex'] * ef_df['Xyards'] * ef_df['Xtan'] * ef_df['EF_yard']))) * ef_df['x_store_slurry']
        new_df['Mstorage_slurry_N'] = \
            (((ef_df['Nex'] * ef_df['Xhousing'] * ef_df['Xslurry']) -
              (ef_df['Nex'] * ef_df['Xhousing'] * ef_df['Xslurry'] * ef_df['Xtan'] * ef_df['EF_hous_slurry'])) +
             ((ef_df['Nex'] * ef_df['Xyards']) - (ef_df['Nex'] * ef_df['Xyards'] * ef_df['Xtan'] *
                                                  ef_df['EF_yard']))) * ef_df['x_store_slurry']

        new_df['Estorage_sl_l'] = \
            (new_df['Mstorage_slurry_TAN'] + ((new_df['Mstorage_slurry_N'] - new_df['Mstorage_slurry_TAN']) *
                                              ef_df['f_min'])) * ef_df['EF_storage_slurry_NH3']
        new_df.drop(['Mstorage_slurry_TAN', 'Mstorage_slurry_N'], axis=1, inplace=True)

        new_df['EF_storage'] = new_df['Estorage_sd_l'] + new_df['Estorage_sl_l']
        new_df.drop(['Estorage_sd_l', 'Estorage_sl_l'], axis=1, inplace=True)
        self.logger.write_time_log('LivestockSector', 'get_nh3_ef', timeit.default_timer() - spent_time)

        return new_df

    def get_nox_no_ef(self):
        """
        Calculate the emission factor for storage emissions

        :return: Dataframe with the Emission factors as columns and animal subtypes as 'Code'
        """
        spent_time = timeit.default_timer()
        ef_df = pd.read_csv(os.path.join(self.ef_dir, 'nox_no.csv'))

        new_df = ef_df.loc[:, ['Code']]

        new_df['Estorage_sd_l'] = \
            ((((ef_df['Nex'] * ef_df['Xhousing'] * ef_df['Xtan'] * ef_df['Xsolid']) -
               (ef_df['Nex'] * ef_df['Xhousing'] * ef_df['Xtan'] * ef_df['Xsolid'] * ef_df['EF_hous_solid'])) *
              (1 - ef_df['f_imm'])) * ef_df['x_store_FYM']) * ef_df['EF_storage_solid_NO']
        new_df['Mstorage_slurry_TAN'] = \
            (((ef_df['Nex'] * ef_df['Xhousing'] * ef_df['Xtan'] * ef_df['Xslurry']) -
              (ef_df['Nex'] * ef_df['Xhousing'] * ef_df['Xtan'] * ef_df['Xslurry'] * ef_df['EF_hous_slurry'])) +
             ((ef_df['Nex'] * ef_df['Xyards'] * ef_df['Xtan']) -
              (ef_df['Nex'] * ef_df['Xyards'] * ef_df['Xtan'] * ef_df['EF_yard']))) * ef_df['x_store_slurry']
        new_df['Mstorage_slurry_N'] = \
            (((ef_df['Nex'] * ef_df['Xhousing'] * ef_df['Xslurry']) -
              (ef_df['Nex'] * ef_df['Xhousing'] * ef_df['Xslurry'] * ef_df['Xtan'] * ef_df['EF_hous_slurry'])) +
             ((ef_df['Nex'] * ef_df['Xyards']) -
              (ef_df['Nex'] * ef_df['Xyards'] * ef_df['Xtan'] * ef_df['EF_yard']))) * ef_df['x_store_slurry']

        new_df['Estorage_sl_l'] = \
            (new_df['Mstorage_slurry_TAN'] + ((new_df['Mstorage_slurry_N'] - new_df['Mstorage_slurry_TAN']) *
                                              ef_df['f_min'])) * ef_df['EF_storage_slurry_NO']
        new_df.drop(['Mstorage_slurry_TAN', 'Mstorage_slurry_N'], axis=1, inplace=True)

        new_df['EF_storage'] = new_df['Estorage_sd_l'] + new_df['Estorage_sl_l']
        new_df.drop(['Estorage_sd_l', 'Estorage_sl_l'], axis=1, inplace=True)
        self.logger.write_time_log('LivestockSector', 'get_nox_no_ef', timeit.default_timer() - spent_time)

        return new_df

    def add_daily_factors_to_animal_distribution(self, animals_df, daily_factors):
        """
        Add to the animal distribution the daily factors.

        :param animals_df: GeoDataframe with the amount of each animal subtype by destiny cell (FID)
            Columns:
            'FID', 'cattle_01', 'cattle_02', 'cattle_03' 'cattle_04', 'cattle_05', 'cattle_06', 'cattle_07',
            'cattle_08', 'cattle_09', 'cattle_10', 'cattle_11', 'chicken_01', 'chicken_02', 'goats_01', 'goats_02',
            'goats_03', goats_04', 'goats_05',  'goats_06', 'pigs_01', 'pigs_02', 'pigs_03', 'pigs_04', 'pigs_05',
            'pigs_06', 'pigs_07', 'pigs_08', 'pigs_09', 'pigs_10', 'timezone',  'geometry'
        :type animals_df: geopandas.GeoDataframe

        :param daily_factors: GeoDataframe with the daily factors.
            Columns:
            'REC', 'geometry', 'FD_housing_open', 'FD_housing_closed, 'FD_storage', 'FD_grassing'
        :type daily_factors: geopandas.GeoDataframe

        :return: Animal distribution with the daily factors.
        :rtype: geopandas.GeoDataframe
        """
        spent_time = timeit.default_timer()
        animals_df = animals_df.to_crs({'init': 'epsg:4326'})
        animals_df['centroid'] = animals_df.centroid

        animals_df['REC'] = animals_df.apply(self.nearest, geom_union=daily_factors.unary_union, df1=animals_df,
                                             df2=daily_factors, geom1_col='centroid', src_column='REC', axis=1)

        animals_df = pd.merge(animals_df, daily_factors, how='left', on='REC')

        animals_df.drop(columns=['centroid', 'REC', 'geometry_y'], axis=1, inplace=True)
        animals_df.rename(columns={'geometry_x': 'geometry'}, inplace=True)
        self.logger.write_time_log('LivestockSector', 'add_daily_factors_to_animal_distribution',
                                   timeit.default_timer() - spent_time)

        return animals_df

    def calculate_day_emissions(self, animals_df, day):
        """
        Calculate the emissions, already speciated, corresponding to the given day.

        :param animals_df: GeoDataframe with the amount of each animal subtype by destiny cell (FID)
            Columns:
            'FID', 'cattle_01', 'cattle_02', 'cattle_03' 'cattle_04', 'cattle_05', 'cattle_06', 'cattle_07',
            'cattle_08', 'cattle_09', 'cattle_10', 'cattle_11', 'chicken_01', 'chicken_02', 'goats_01', 'goats_02',
            'goats_03', goats_04', 'goats_05',  'goats_06', 'pigs_01', 'pigs_02', 'pigs_03', 'pigs_04', 'pigs_05',
            'pigs_06', 'pigs_07', 'pigs_08', 'pigs_09', 'pigs_10', 'timezone',  'geometry'
        :type animals_df: geopandas.GeoDataframe

        :param day: Date of the day to generate.
        :type day: datetime.date

        :return: GeoDataframe with the daily emissions by destiny cell.
        :rtype: geopandas.GeoDataframe
        """
        spent_time = timeit.default_timer()
        daily_factors = self.get_daily_factors(animals_df.loc[:, ['FID', 'geometry']], day)
        animals_df = self.add_daily_factors_to_animal_distribution(animals_df, daily_factors)

        out_df = animals_df.loc[:, ['FID', 'timezone', 'geometry']]

        # ===== NH3 =====
        if 'nh3' in [x.lower() for x in self.source_pollutants]:
            # get_list out_pollutants from speciation map -> NH3
            out_pollutants = self.get_output_pollutants('nh3')
            for out_p in out_pollutants:
                self.logger.write_log('\t\t\tCalculating {0} emissions'.format(out_p), message_level=3)
                out_df[out_p] = 0
                if out_p not in self.output_pollutants:
                    self.output_pollutants.append(out_p)
                for i, animal in self.get_nh3_ef().iterrows():
                    # Iterating by animal subtype
                    if animal.Code.startswith(tuple(self.animal_list)):
                        # Housing emissions
                        if animal.Code.startswith(('cattle', 'sheep', 'goats')):
                            # Housing open emissions
                            out_df.loc[:, out_p] += \
                                (animals_df[animal['Code']] * animals_df['FD_housing_open']).multiply(
                                    animal['EF_housing'])
                        elif animal.Code.startswith(('chicken', 'pigs')):
                            # Housing close emissions
                            out_df.loc[:, out_p] += \
                                (animals_df[animal['Code']] * animals_df['FD_housing_closed']).multiply(
                                    animal['EF_housing'])
                        else:
                            raise KeyError('Animal {0} not found on the nh3 emission factors file.'.format(animal.Code))
                        # Storage emissions
                        out_df.loc[:, out_p] += \
                            (animals_df[animal['Code']] * animals_df['FD_storage']).multiply(animal['EF_yarding'])
                        # Grassing emissions
                        out_df.loc[:, out_p] += \
                            (animals_df[animal['Code']] * animals_df['FD_grassing']).multiply(animal['EF_grazing'])
                        # Storage emissions
                        out_df.loc[:, out_p] += \
                            (animals_df[animal['Code']] * animals_df['FD_storage']).multiply(animal['EF_storage'])

                # From kg NH3-N to mol NH3
                out_df.loc[:, out_p] = out_df.loc[:, out_p].multiply(
                    (17. / 14.) * 1000. * (1. / self.molecular_weights['nh3']))

        # ===== NMVOC =====
        if 'nmvoc' in [x.lower() for x in self.source_pollutants]:
            # get_list out_pollutants from speciation map -> PAR, OLE, TOL ... (15 species)
            out_pollutants = self.get_output_pollutants('nmvoc')
            for out_p in out_pollutants:
                self.logger.write_log('\t\t\tCalculating {0} emissions'.format(out_p), message_level=3)
                out_df[out_p] = 0
                if out_p not in self.output_pollutants:
                    self.output_pollutants.append(out_p)
                for i, animal in pd.read_csv(os.path.join(self.ef_dir, 'nmvoc.csv')).iterrows():
                    # Iterating by animal subtype
                    if animal.Code.startswith(tuple(self.animal_list)):
                        # Housing emissions
                        if animal.Code.startswith(('cattle',)):
                            out_df.loc[:, out_p] += animals_df[animal['Code']].multiply(
                                animal['Xhousing'] * animal['EF_nmvoc'] * self.speciation_profile.loc['cattle', out_p])
                        elif animal.Code.startswith(('pigs',)):
                            out_df.loc[:, out_p] += animals_df[animal['Code']].multiply(
                                animal['Xhousing'] * animal['EF_nmvoc'] * self.speciation_profile.loc['pigs', out_p])
                        elif animal.Code.startswith(('chicken',)):
                            out_df.loc[:, out_p] += animals_df[animal['Code']].multiply(
                                animal['Xhousing'] * animal['EF_nmvoc'] * self.speciation_profile.loc['chicken', out_p])
                        elif animal.Code.startswith(('sheep',)):
                            out_df.loc[:, out_p] += animals_df[animal['Code']].multiply(
                                animal['Xhousing'] * animal['EF_nmvoc'] * self.speciation_profile.loc['sheep', out_p])
                        elif animal.Code.startswith(('goats',)):
                            out_df.loc[:, out_p] += animals_df[animal['Code']].multiply(
                                animal['Xhousing'] * animal['EF_nmvoc'] * self.speciation_profile.loc['goats', out_p])

                out_df.loc[:, out_p] = out_df.loc[:, out_p].multiply(1000. * (1. / self.molecular_weights['nmvoc']))

        # ===== PM10 =====
        if 'pm10' in [x.lower() for x in self.source_pollutants]:
            out_pollutants = self.get_output_pollutants('pm10')
            for out_p in out_pollutants:
                self.logger.write_log('\t\t\tCalculating {0} emissions'.format(out_p), message_level=3)
                out_df[out_p] = 0
                if out_p not in self.output_pollutants:
                    self.output_pollutants.append(out_p)
                for i, animal in pd.read_csv(os.path.join(self.ef_dir, 'pm.csv')).iterrows():
                    # Iterating by animal subtype
                    if animal.Code.startswith(tuple(self.animal_list)):
                        if animal.Code.startswith(('cattle',)):
                            out_df.loc[:, out_p] += animals_df[animal['Code']].multiply(
                                animal['Xhousing'] * animal['EF_pm10'] * self.speciation_profile.loc['cattle', out_p])
                        elif animal.Code.startswith(('pigs',)):
                            out_df.loc[:, out_p] += animals_df[animal['Code']].multiply(
                                animal['Xhousing'] * animal['EF_pm10'] * self.speciation_profile.loc['pigs', out_p])
                        elif animal.Code.startswith(('chicken',)):
                            out_df.loc[:, out_p] += animals_df[animal['Code']].multiply(
                                animal['Xhousing'] * animal['EF_pm10'] * self.speciation_profile.loc['chicken', out_p])
                        elif animal.Code.startswith(('sheep',)):
                            out_df.loc[:, out_p] += animals_df[animal['Code']].multiply(
                                animal['Xhousing'] * animal['EF_pm10'] * self.speciation_profile.loc['sheep', out_p])
                        elif animal.Code.startswith(('goats',)):
                            out_df.loc[:, out_p] += animals_df[animal['Code']].multiply(
                                animal['Xhousing'] * animal['EF_pm10'] * self.speciation_profile.loc['goats', out_p])
                out_df.loc[:, out_p] = out_df.loc[:, out_p].multiply(1000. * (1. / self.molecular_weights['pm10']))

            # Preparing PM10 for PMC
            if 'pmc' in [x.lower() for x in self.speciation_map.iterkeys()]:
                out_df['aux_pm10'] = 0
                for i, animal in pd.read_csv(os.path.join(self.ef_dir, 'pm.csv')).iterrows():
                    # Iterating by animal subtype
                    if animal.Code.startswith(tuple(self.animal_list)):
                        if animal.Code.startswith(('cattle',)):
                            out_df.loc[:, 'aux_pm10'] += animals_df[animal['Code']].multiply(
                                animal['Xhousing'] * animal['EF_pm10'])
                        elif animal.Code.startswith(('pigs',)):
                            out_df.loc[:, 'aux_pm10'] += animals_df[animal['Code']].multiply(
                                animal['Xhousing'] * animal['EF_pm10'])
                        elif animal.Code.startswith(('chicken',)):
                            out_df.loc[:, 'aux_pm10'] += animals_df[animal['Code']].multiply(
                                animal['Xhousing'] * animal['EF_pm10'])
                        elif animal.Code.startswith(('sheep',)):
                            out_df.loc[:, 'aux_pm10'] += animals_df[animal['Code']].multiply(
                                animal['Xhousing'] * animal['EF_pm10'])
                        elif animal.Code.startswith(('goats',)):
                            out_df.loc[:, 'aux_pm10'] += animals_df[animal['Code']].multiply(
                                animal['Xhousing'] * animal['EF_pm10'])
                out_df.loc[:, 'aux_pm10'] = out_df.loc[:, 'aux_pm10'].multiply(
                    1000. * (1. / self.molecular_weights['pm10']))

        # ===== PM2.5 =====
        if 'pm25' in [x.lower() for x in self.source_pollutants]:
            out_pollutants = self.get_output_pollutants('pm25')
            for out_p in out_pollutants:
                self.logger.write_log('\t\t\tCalculating {0} emissions'.format(out_p), message_level=3)
                out_df[out_p] = 0
                if out_p not in self.output_pollutants:
                    self.output_pollutants.append(out_p)
                for i, animal in pd.read_csv(os.path.join(self.ef_dir, 'pm.csv')).iterrows():
                    if animal.Code.startswith(tuple(self.animal_list)):
                        # Iterating by animal subtype
                        if animal.Code.startswith(('cattle',)):
                            out_df.loc[:, out_p] += animals_df[animal['Code']].multiply(
                                animal['Xhousing'] * animal['EF_pm25'] * self.speciation_profile.loc['cattle', out_p])
                        elif animal.Code.startswith(('pigs',)):
                            out_df.loc[:, out_p] += animals_df[animal['Code']].multiply(
                                animal['Xhousing'] * animal['EF_pm25'] * self.speciation_profile.loc['pigs', out_p])
                        elif animal.Code.startswith(('chicken',)):
                            out_df.loc[:, out_p] += animals_df[animal['Code']].multiply(
                                animal['Xhousing'] * animal['EF_pm25'] * self.speciation_profile.loc['chicken', out_p])
                        elif animal.Code.startswith(('sheep',)):
                            out_df.loc[:, out_p] += animals_df[animal['Code']].multiply(
                                animal['Xhousing'] * animal['EF_pm25'] * self.speciation_profile.loc['sheep', out_p])
                        elif animal.Code.startswith(('goats',)):
                            out_df.loc[:, out_p] += animals_df[animal['Code']].multiply(
                                animal['Xhousing'] * animal['EF_pm25'] * self.speciation_profile.loc['goats', out_p])
                out_df.loc[:, out_p] = out_df.loc[:, out_p].multiply(1000. * (1. / self.molecular_weights['pm25']))

            # Preparing PM2.5 for PMC
            if 'pmc' in [x.lower() for x in self.speciation_map.iterkeys()]:
                out_df['aux_pm25'] = 0
                for i, animal in pd.read_csv(os.path.join(self.ef_dir, 'pm.csv')).iterrows():
                    if animal.Code.startswith(tuple(self.animal_list)):
                        # Iterating by animal subtype
                        if animal.Code.startswith(('cattle',)):
                            out_df.loc[:, 'aux_pm25'] += animals_df[animal['Code']].multiply(
                                animal['Xhousing'] * animal['EF_pm25'])
                        elif animal.Code.startswith(('pigs',)):
                            out_df.loc[:, 'aux_pm25'] += animals_df[animal['Code']].multiply(
                                animal['Xhousing'] * animal['EF_pm25'])
                        elif animal.Code.startswith(('chicken',)):
                            out_df.loc[:, 'aux_pm25'] += animals_df[animal['Code']].multiply(
                                animal['Xhousing'] * animal['EF_pm25'])
                        elif animal.Code.startswith(('sheep',)):
                            out_df.loc[:, 'aux_pm25'] += animals_df[animal['Code']].multiply(
                                animal['Xhousing'] * animal['EF_pm25'])
                        elif animal.Code.startswith(('goats',)):
                            out_df.loc[:, 'aux_pm25'] += animals_df[animal['Code']].multiply(
                                animal['Xhousing'] * animal['EF_pm25'])
                out_df.loc[:, 'aux_pm25'] = out_df.loc[:, 'aux_pm25'].multiply(
                    1000. * (1. / self.molecular_weights['pm25']))

        # ===== NOX_NO =====
        if 'nox_no' in [x.lower() for x in self.source_pollutants]:
            out_pollutants = self.get_output_pollutants('nox_no')
            for out_p in out_pollutants:
                self.logger.write_log('\t\t\tCalculating {0} emissions'.format(out_p), message_level=3)
                out_df[out_p] = 0
                if out_p not in self.output_pollutants:
                    self.output_pollutants.append(out_p)

                for i, animal in self.get_nox_no_ef().iterrows():
                    # Iterating by animal subtype
                    if animal.Code.startswith(tuple(self.animal_list)):
                        # Storage emissions
                        out_df.loc[:, out_p] += (animals_df[animal['Code']] * animals_df['FD_storage']).multiply(
                            animal['EF_storage'])

                # From kg NOX-N to mol NO
                out_df.loc[:, out_p] = out_df.loc[:, out_p].multiply(
                    (30. / 14.) * 1000. * (1. / self.molecular_weights['nox_no']))

        # ===== PMC =====
        if 'pmc' in [x.lower() for x in self.speciation_map.iterkeys()]:
            pmc_name = 'PMC'
            self.logger.write_log('\t\t\tCalculating {0} emissions'.format(pmc_name), message_level=3)
            if all(x in [x.lower() for x in self.source_pollutants] for x in ['pm10', 'pm25']):
                if pmc_name not in self.output_pollutants:
                    self.output_pollutants.append(pmc_name)
                out_df[pmc_name] = out_df['aux_pm10'] - out_df['aux_pm25']
                out_df.drop(columns=['aux_pm10', 'aux_pm25'], axis=1, inplace=True)
            else:
                warn("WARNING: '{0}' cannot be calculated because 'pm10' or/and 'pm25' ".format(pmc_name) +
                     "are not in the livestock_source_pollutants list")

        not_pollutants = [poll for poll in self.source_pollutants
                          if poll not in ['nh3', 'nox_no', 'nh3', 'nmvoc', 'pm10', 'pm25']]
        if len(not_pollutants) > 0:
            if self.comm.Get_rank() == 0:
                warn('The pollutants {0} cannot be calculated on the Livestock sector'.format(not_pollutants))
        self.logger.write_time_log('LivestockSector', 'calculate_day_emissions', timeit.default_timer() - spent_time)

        return out_df

    def calculate_daily_emissions_dict(self, animals_df):
        """
        Calculate the daily emissions setting it in a dictionary with the day as key.

        :param animals_df: GeoDataframe with the amount of each animal subtype by destiny cell (FID)
            Columns:
            'FID', 'cattle_01', 'cattle_02', 'cattle_03' 'cattle_04', 'cattle_05', 'cattle_06', 'cattle_07',
            'cattle_08', 'cattle_09', 'cattle_10', 'cattle_11', 'chicken_01', 'chicken_02', 'goats_01', 'goats_02',
            'goats_03', goats_04', 'goats_05',  'goats_06', 'pigs_01', 'pigs_02', 'pigs_03', 'pigs_04', 'pigs_05',
            'pigs_06', 'pigs_07', 'pigs_08', 'pigs_09', 'pigs_10', 'timezone',  'geometry'
        :type animals_df: geopandas.GeoDataframe

        :return: Dictionary with the day as key (same key as self.day_dict) and the daily emissions as value.
        :rtype: dict
        """
        spent_time = timeit.default_timer()
        daily_emissions = {}
        for day in self.day_dict.keys():
            daily_emissions[day] = self.calculate_day_emissions(animals_df, day)
        self.logger.write_time_log('LivestockSector', 'calculate_daily_emissions_dict',
                                   timeit.default_timer() - spent_time)

        return daily_emissions

    def add_dates(self, df_by_day):
        """
        Expand each daily dataframe into a single dataframe with all the time steps.

        :param df_by_day: Dictionary with the daily emissions for each day.
        :type df_by_day: dict

        :return: GeoDataframe with all the time steps (each time step have the daily emission)
        :rtype: geopandas.GeoDataframe
        """
        spent_time = timeit.default_timer()
        df_list = []
        for tstep, date in enumerate(self.date_array):
            df_aux = df_by_day[date.date()].copy()
            df_aux['date'] = pd.to_datetime(date, utc=True)
            df_aux['date_utc'] = pd.to_datetime(date, utc=True)
            df_aux['tstep'] = tstep
            df_list.append(df_aux)
        dataframe_by_day = pd.concat(df_list, ignore_index=True)

        dataframe_by_day = self.to_timezone(dataframe_by_day)
        self.logger.write_time_log('LivestockSector', 'add_dates', timeit.default_timer() - spent_time)

        return dataframe_by_day

    def calculate_hourly_distribution(self, dict_by_day):
        """
        Calculate the hourly distribution for all the emissions.

        The NH3 & NOX_NO emissions have to be also monthly and weekly distributed.

        :param dict_by_day: Dictionary with the day as key (same key as self.day_dict) and the daily emissions as value.
        :type dict_by_day: dict

        :return: GeoDataframe with the hourly distribution.
        :rtype: geopandas.GeoDataframe
        """
        spent_time = timeit.default_timer()

        def distribute_weekly(df):
            import datetime
            date_np = df.head(1)['date'].values[0]
            date = datetime.datetime.utcfromtimestamp(date_np.astype(int) * 1e-9)
            profile = self.calculate_rebalanced_weekly_profile(self.weekly_profiles.loc[in_p, :].to_dict(), date)

            df[out_p] = df[out_p].multiply(profile[df.name[1]])
            return df.loc[:, [out_p]]

        # Create unique dataframe
        distribution = self.add_dates(dict_by_day)

        distribution['hour'] = distribution['date'].dt.hour
        for out_p in self.output_pollutants:
            self.logger.write_log('\t\tDistributing {0} emissions to hourly resolution'.format(out_p), message_level=3)
            if out_p.lower() == 'pmc':
                in_p = 'pmc'
            else:
                in_p = self.speciation_map[out_p]

            # NH3 & NOX_NO emissions have to be also monthly and weekly distributed
            if in_p.lower() not in ['nh3', 'nox_no']:
                # Monthly distribution
                distribution['month'] = distribution['date'].dt.month
                distribution[out_p] = distribution.groupby('month')[out_p].apply(lambda x: x.multiply(
                    self.monthly_profiles.loc[in_p, x.name]))

                # Weekday distribution
                distribution['weekday'] = distribution['date'].dt.weekday

                distribution[out_p] = distribution.groupby(['month', 'weekday'])['date', out_p].apply(distribute_weekly)

                distribution.drop(columns=['month', 'weekday'], axis=1, inplace=True)
            # Hourly distribution
            distribution[out_p] = distribution.groupby('hour')[out_p].apply(lambda x: x.multiply(
                self.hourly_profiles.loc[in_p, x.name]))

        distribution['date'] = distribution['date_utc']
        distribution.drop(columns=['hour', 'date_utc'], axis=1, inplace=True)
        self.logger.write_time_log('LivestockSector', 'calculate_hourly_distribution',
                                   timeit.default_timer() - spent_time)

        return distribution

    def calculate_emissions(self):
        """
        Calculate the livestock emissions hourly distributed.

        :return: GeoDataframe with all the emissions.
        :rtype: geopandas.GeoDataframe
        """
        spent_time = timeit.default_timer()
        self.logger.write_log('\tCalculating emissions')

        self.logger.write_log('\t\tCalculating Daily emissions', message_level=2)
        df_by_day = self.calculate_daily_emissions_dict(self.animals_df)
        self.logger.write_log('\t\tCalculating hourly emissions', message_level=2)
        animals_df = self.calculate_hourly_distribution(df_by_day)

        animals_df.drop(columns=['geometry'], inplace=True)
        animals_df['layer'] = 0

        animals_df = animals_df.groupby(['FID', 'layer', 'tstep']).sum()
        self.logger.write_log('\t\tLivestock emissions calculated', message_level=2)
        self.logger.write_time_log('LivestockSector', 'calculate_emissions', timeit.default_timer() - spent_time)

        return animals_df
