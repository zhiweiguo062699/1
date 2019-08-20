#!/usr/bin/env python

import sys
import os
import timeit
from hermesv3_bu.logger.log import Log
import numpy as np
import pandas as pd
import geopandas as gpd
from warnings import warn

from hermesv3_bu.sectors.sector import Sector

PHASE_TYPE = {'taxi_out': 'departure', 'pre-taxi_out': 'departure', 'takeoff': 'departure', 'climbout': 'departure',
              'approach': 'arrival', 'taxi_in': 'arrival', 'post-taxi_in': 'arrival', 'landing': 'arrival',
              'landing_wear': 'arrival'}
PHASE_EF_FILE = {'taxi_out': 'ef_taxi.csv', 'pre-taxi_out': 'ef_apu.csv', 'takeoff': 'ef_takeoff.csv',
                 'climbout': 'ef_climbout.csv', 'approach': 'ef_approach.csv', 'taxi_in': 'ef_taxi.csv',
                 'post-taxi_in': 'ef_apu.csv', 'landing': 'ef_approach.csv', 'landing_wear': 'ef_landing_wear.csv'}


class AviationSector(Sector):
    """
    The aviation module divide the emissions into 9 emission phases (4 for departure and 5 for arrival)
    - Departure:
        - Pre-taxi out
        - Taxi out
        - Take off
        - Climb out
    - Arrival:
        - Final approach
        - Landing
        - Landing wear
        - Taxi in
        - Post-taxi in
    """
    def __init__(self, comm, logger, auxiliary_dir, grid_shp, clip, date_array, source_pollutants, vertical_levels,
                 airport_list, plane_list, airport_shapefile_path, airport_runways_shapefile_path,
                 airport_runways_corners_shapefile_path, airport_trajectories_shapefile_path, operations_path,
                 planes_path, times_path, ef_dir, weekly_profiles_path, hourly_profiles_path, speciation_map_path,
                 speciation_profiles_path, molecular_weights_path):
        """
        :param comm: Communicator for the sector calculation.
        :type comm: MPI.COMM

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

        :param airport_list: List of airports to take into account.
        :type airport_list: list

        :param plane_list: List of planes to take into account.
        :type plane_list: list

        :param airport_shapefile_path: Path to the shapefile that contains the airport polygons.
        :type airport_shapefile_path: str

        :param airport_runways_shapefile_path: Path to the shapefile that contains the runways lines.
        :type airport_runways_shapefile_path: str

        :param airport_runways_corners_shapefile_path: Path to the shapefile that contains the runway starting points.
        :type airport_runways_corners_shapefile_path: str

        :param airport_trajectories_shapefile_path: Path to the shapefile that contains the trajectories lines.
        :type airport_trajectories_shapefile_path: str

        :param operations_path: Path to the CSV that contains the operations information by month.
        columns: plane_id, airport_id, operation, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
        :type operations_path: str

        :param planes_path: Path to the CSV that contains the planes information:
        columns: plane_id, engine_id, engine_n, mtow, apu_id, plane_type
        :type planes_path: str

        :param times_path: Path to the CSV that contains the times information.
        columns: airport_id, plane_type, taxi_out, taxi_in, takeoff, climbout, approach, landing, post-taxi_in,
        pre-taxi_out
        :type times_path: str

        :param ef_dir: Path to the directory that contains all the emission factors files. That folder must contain the
        following emission factor files: ef_approach.csv, ef_apu.csv, ef_climbout.csv, ef_landing.csv,
        ef_landing_wear.csv, ef_takeoff.csv and ef_taxi.csv.
        :type ef_dir: str

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
        """
        spent_time = timeit.default_timer()
        logger.write_log('===== AVIATION SECTOR =====')
        super(AviationSector, self).__init__(
            comm, logger, auxiliary_dir, grid_shp, clip, date_array, source_pollutants, vertical_levels, None,
            weekly_profiles_path, hourly_profiles_path, speciation_map_path, speciation_profiles_path,
            molecular_weights_path)

        if 'hc' in self.source_pollutants:
            for poll in ['nmvoc', 'ch4']:
                if poll not in self.source_pollutants:
                    self.source_pollutants.append(poll)
            self.source_pollutants.remove('hc')

        # self.ef_dir = ef_dir
        self.ef_files = self.read_ef_files(ef_dir)

        airport_trajectories_shapefile = self.read_trajectories_shapefile(
            airport_trajectories_shapefile_path, airport_runways_corners_shapefile_path, airport_runways_shapefile_path)
        self.airport_list_full = None  # only needed for master process
        self.airport_list = self.get_airport_list(airport_list, airport_trajectories_shapefile, operations_path)
        self.plane_list = plane_list

        full_airport_shapefile = gpd.read_file(airport_shapefile_path)
        full_airport_shapefile.drop(columns='airport_na', inplace=True)
        full_airport_shapefile.set_index('airport_id', inplace=True)
        self.airport_shapefile = full_airport_shapefile.loc[self.airport_list, ['geometry']]

        self.operations = self.read_operations_update_plane_list(operations_path)
        self.planes_info = self.read_planes(planes_path)
        self.times_info = self.read_times_info(times_path)

        runway_shapefile = self.read_runway_shapefile(airport_runways_shapefile_path)
        self.airport_distribution = self.calculate_airport_distribution(full_airport_shapefile)
        self.runway_arrival_distribution = self.calculate_runway_distribution(runway_shapefile, 'arrival')
        self.runway_departure_distribution = self.calculate_runway_distribution(runway_shapefile, 'departure')

        self.trajectory_departure_distribution = self.calculate_trajectories_distribution(
            airport_trajectories_shapefile, 'departure')
        self.trajectory_arrival_distribution = self.calculate_trajectories_distribution(
            airport_trajectories_shapefile, 'arrival')
        comm.Barrier()
        self.logger.write_time_log('AviationSector', '__init__', timeit.default_timer() - spent_time)

    def read_ef_files(self, ef_path):
        if self.comm.Get_rank() == 0:
            ef_files = {}
            for phase in PHASE_TYPE.keys():
                ef_files[phase] = pd.read_csv(os.path.join(ef_path, PHASE_EF_FILE[phase]))
        else:
            ef_files = None

        ef_files = self.comm.bcast(ef_files, root=0)

        return ef_files

    def read_trajectories_shapefile(self, trajectories_path, runways_corners_path, runways_path):
        """
        Create a shapefile with 2 geometries: trajectories & staring point

        :param trajectories_path: Path to the trajectories shapefile.
        :type trajectories_path: str

        :param runways_corners_path: Path to the trajectories starting point path.
        :type runways_corners_path: str

        :param runways_path: Path to the shapefile that contains the runways and their fraction of use.
        :type runways_path: str

        :return: GeoDataFrame with the trajectories information, their praction and staring point.
        :rtype: GeoDataFrame
        """
        spent_time = timeit.default_timer()
        trajectories = gpd.read_file(trajectories_path)

        corners = gpd.read_file(runways_corners_path).to_crs(trajectories.crs)
        corners.rename(columns={'geometry': 'start_point'}, inplace=True)

        runways = gpd.read_file(runways_path).to_crs(trajectories.crs)
        runways.rename(columns={'approach_f': 'arrival_f', 'climbout_f': 'departure_f'}, inplace=True)

        trajectories = trajectories.merge(corners[['runway_id', 'start_point']], on='runway_id', how='left')
        trajectories = trajectories.merge(runways[['runway_id', 'arrival_f', 'departure_f']], on='runway_id',
                                          how='left')
        trajectories.loc[trajectories['operation'] == 'departure', 'fraction'] = trajectories['departure_f']
        trajectories.loc[trajectories['operation'] == 'arrival', 'fraction'] = trajectories['arrival_f']

        trajectories.drop(columns=['arrival_f', 'departure_f'], inplace=True)
        trajectories.set_index(['runway_id', 'operation'], inplace=True)
        self.logger.write_time_log('AviationSector', 'read_trajectories_shapefile', timeit.default_timer() - spent_time)

        return trajectories

    def read_runway_shapefile(self, airport_runways_shapefile_path):
        """
        The master process reads the runway shapefile.

        :param airport_runways_shapefile_path: Path to the shapefile that contains the runways.
        :type airport_runways_shapefile_path: str

        :return: GeoDataFrame with the runways information.
        :rtype: GeoDataFrame, None
        """
        spent_time = timeit.default_timer()
        if self.comm.Get_rank() == 0:
            runway_shapefile = gpd.read_file(airport_runways_shapefile_path)
            runway_shapefile.set_index(['airport_id', 'runway_id'], inplace=True)
            runway_shapefile = runway_shapefile.loc[self.airport_list_full, :]
            runway_shapefile = runway_shapefile.loc[runway_shapefile['cons'] == 1,
                                                    ['approach_f', 'climbout_f', 'geometry']]
            runway_shapefile.rename(columns={'approach_f': 'arrival_f', 'climbout_f': 'departure_f'}, inplace=True)
        else:
            runway_shapefile = None
        self.logger.write_time_log('AviationSector', 'read_runway_shapefile', timeit.default_timer() - spent_time)

        return runway_shapefile

    def read_hourly_profiles(self, path):
        """
        Read the Dataset of the hourly profiles with the hours (int) as columns.

        Overwrites the super method.

        :param path: Path to the file that contains the monthly profiles.
        :type path: str

        :return: Dataset od the monthly profiles.
        :rtype: DataFrame
        """
        spent_time = timeit.default_timer()
        if path is None:
            profiles = None
        else:
            profiles = pd.read_csv(path)
            profiles.rename(
                columns={"operation": -3, "day_type": -2, 'P_hour': -1, '00': 0, '01': 1, '02': 2, '03': 3, '04': 4,
                         '05': 5, '06': 6, '07': 7, '08': 8, '09': 9, '10': 10, '11': 11, '12': 12, '13': 13, '14': 14,
                         '15': 15, '16': 16, '17': 17, '18': 18, '19': 19, '20': 20, '21': 21, '22': 22, '23': 23},
                inplace=True)
            profiles.columns = profiles.columns.astype(int)
            profiles.rename(columns={-1: 'P_hour', -3: "operation", -2: "day_type"}, inplace=True)
            profiles.set_index(["P_hour", "operation", "day_type"], inplace=True)

        self.logger.write_time_log('AviationSector', 'read_hourly_profiles', timeit.default_timer() - spent_time)

        return profiles

    def read_operations_update_plane_list(self, operations_csv_path):
        """
        Read the operations CSV file and update the plane_list argument.

        If plane_list is not set in the configuration file it will be set using the plane_codes of the selected
        airports.

        :param operations_csv_path: Path to the CSV that contains the operations information by plane, airport, and
            phase. The cSC must contain the following columns: [plane_id, airport_id, operation, 1, 2, 3, 4, 5, 6,
            7, 8, 9, 10, 11, 12] with the number of operations by month.
        :type operations_csv_path: str

        :return: DataFrame with the amount operations by month. The operations are detailed with the plane_code, airport
            and phase.
        :rtype: DataFrame
        """
        spent_time = timeit.default_timer()

        operations = pd.read_csv(operations_csv_path)

        if self.plane_list is None:
            self.plane_list = list(np.unique(operations['plane_id'].values))
        else:
            operations = operations.loc[operations['plane_id'].isin(self.plane_list), :]

        if len(operations) == 0:
            raise NameError("The plane/s defined in the plane_list do not exist.")
        operations = operations.loc[operations['airport_id'].isin(self.airport_list), :]
        operations.set_index(['airport_id', 'plane_id', 'operation'], inplace=True)
        operations.rename(columns={'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
                                   '11': 11, '12': 12}, inplace=True)
        self.logger.write_time_log('AviationSector', 'read_operations_update_plane_list',
                                   timeit.default_timer() - spent_time)

        return operations

    def read_planes(self, planes_path):
        """
        Read the CSV with the planes information.

        :param planes_path: Path to the CSV file that contains the planes information.
        :type planes_path: str

        :return: Dataframe with the planes information
        :rtype: DataFrame
        """
        spent_time = timeit.default_timer()

        dataframe = pd.read_csv(planes_path)
        dataframe = dataframe.loc[dataframe['plane_id'].isin(self.plane_list)]

        dataframe.set_index('plane_id', inplace=True)
        self.logger.write_time_log('AviationSector', 'read_planes', timeit.default_timer() - spent_time)

        return dataframe

    def read_times_info(self, times_path):
        """
        Read the CSV file that contains the time spent on each phase.

        :param times_path: Path to the CSV file that contains the time spent on each phase.
        :type times_path: str

        :return: Dataframe with the times of each phase
        :rtype: DataFrame
        """
        spent_time = timeit.default_timer()
        dataframe = pd.read_csv(times_path)
        dataframe = dataframe.loc[dataframe['airport_id'].isin(self.airport_list)]
        dataframe.set_index(['airport_id', 'plane_type'], inplace=True)
        self.logger.write_time_log('AviationSector', 'read_times_info', timeit.default_timer() - spent_time)

        return dataframe

    def get_airport_list(self, conf_airport_list, airport_shapefile, operations_file):
        """
        Get the airport list from the involved airports on the domain.

        It will select only the involved airports that are inside the grid.
        If the argument 'airport_list' is set in the configuration file it will use the ones of that list that are into
        the grid.

        :param conf_airport_list: List of airport codes from the configuration file (or None).
        :type conf_airport_list: list

        :param airport_shapefile: Shapefile with the 'ICAO' information.
        :type airport_shapefile:  GeoDataFrame

        :return: List with the airports to calculate.
        :rtype: list
        """
        spent_time = timeit.default_timer()
        if self.comm.Get_rank() == 0:
            airport_shapefile = airport_shapefile.reset_index()
            airport_shapefile = gpd.sjoin(airport_shapefile.to_crs(self.grid_shp.crs),
                                          self.clip.shapefile.to_crs(self.grid_shp.crs), how='inner', op='intersects')

            shp_airport_list = list(np.unique(airport_shapefile['airport_id'].values))

            if conf_airport_list is not None:
                shp_airport_list = list(set(conf_airport_list).intersection(shp_airport_list))

            if len(shp_airport_list) == 0:
                raise NameError("No airports intersect with the defined domain or the defined aiport/s in the " +
                                "airport_list do no exist ")

            airports_with_operations = np.unique(pd.read_csv(operations_file, usecols=['airport_id']).values)

            new_list = list(set(shp_airport_list) & set(airports_with_operations))
            if len(new_list) != len(shp_airport_list):
                warn('{0} airports have no operations. Ignoring them.'.format(
                    list(set(new_list) - set(shp_airport_list))))

            max_len = len(new_list)
            # Only for master (rank == 0)
            self.airport_list_full = new_list

            new_list = [new_list[i * len(new_list) // self.comm.size: (i + 1) * len(new_list) // self.comm.size]
                        for i in range(self.comm.size)]
            for sublist in new_list:
                if len(sublist) == 0:
                    raise ValueError("ERROR: The selected number of processors is to high. " +
                                     "The maximum number of processors accepted are {0}".format(max_len) +
                                     "(Maximum number of airports included in the working domain")
        else:
            new_list = None

        new_list = self.comm.scatter(new_list, root=0)
        self.logger.write_time_log('AviationSector', 'get_airport_list', timeit.default_timer() - spent_time)

        return new_list

    def calculate_airport_distribution(self, airport_shapefile):
        """
        Calculate the location and portion for the emissions that have to be distributed on the airport polygon.

        It only need to be calculated once. the second execution will read the auxiliary file already created.
        All the emissions that have to be distributed on the airport polygon goes to the surface layer.

        :param airport_shapefile: Shapefile with the airport polygon geometries.
        :type airport_shapefile: GeoDataFrame

        :return: DataFrame with the location (FID) and fraction for each airport.
        :rtype: DataFrame
        """
        spent_time = timeit.default_timer()
        self.logger.write_log('\t\tCalculating airport distribution', message_level=2)
        airport_distribution_path = os.path.join(self.auxiliary_dir, 'aviation', 'airport_distribution.csv')

        if not os.path.exists(airport_distribution_path):
            if self.comm.rank == 0:
                airport_shapefile = airport_shapefile.loc[self.airport_list_full, :].copy()
                if not os.path.exists(os.path.dirname(airport_distribution_path)):
                    os.makedirs(os.path.dirname(airport_distribution_path))
                airport_shapefile.to_crs(self.grid_shp.crs, inplace=True)
                airport_shapefile['area'] = airport_shapefile.area
                airport_distribution = self.spatial_overlays(airport_shapefile, self.grid_shp.reset_index(),
                                                             how='intersection')
                airport_distribution['fraction'] = airport_distribution.area / airport_distribution['area']
                airport_distribution.drop(columns=['idx2', 'area', 'geometry', 'cons'], inplace=True)
                airport_distribution.rename(columns={'idx1': 'airport_id'}, inplace=True)
                airport_distribution['layer'] = 0
                airport_distribution.set_index(['airport_id', 'FID', 'layer'], inplace=True)

                airport_distribution.to_csv(airport_distribution_path)
            else:
                airport_distribution = None
            airport_distribution = self.comm.bcast(airport_distribution, root=0)
        else:
            airport_distribution = pd.read_csv(airport_distribution_path)
            airport_distribution.set_index(['airport_id', 'FID', 'layer'], inplace=True)
        self.logger.write_time_log('AviationSector', 'calculate_airport_distribution',
                                   timeit.default_timer() - spent_time)

        return airport_distribution

    def calculate_runway_distribution(self, runway_shapefile, phase_type):
        """
        Calculate the location and portion for the emissions that have to be distributed on the runway lines.

        It only need to be calculated once. the second execution will read the auxiliary file already created.
        All the emissions that have to be distributed on the runway line goes to the surface layer.

        :param runway_shapefile: Shapefile with the runway line geometries.
        :type runway_shapefile: GeoDataFrame

        :param phase_type: Phase type to distribute. Arrival or Departure.
        :type phase_type: str

        :return: DataFrame with the location (FID) and fraction for each airport.
        :rtype: DataFrame
        """
        spent_time = timeit.default_timer()

        def get_intersection_length(row):
            intersection = row.get('geometry_x').intersection(row.get('geometry_y'))
            return intersection.length

        def normalize(df):
            total_fraction = df['{0}_f'.format(phase_type)].values.sum()
            df['{0}_f'.format(phase_type)] = df['{0}_f'.format(phase_type)] / total_fraction

            return df.loc[:, ['{0}_f'.format(phase_type)]]

        self.logger.write_log('\t\tCalculating runway distribution for {0}'.format(phase_type), message_level=2)

        runway_distribution_path = os.path.join(
            self.auxiliary_dir, 'aviation', 'runway_{0}_distribution.csv'.format(phase_type))

        if not os.path.exists(runway_distribution_path):
            if self.comm.rank == 0:
                runway_shapefile['{0}_f'.format(phase_type)] = runway_shapefile.groupby('airport_id').apply(normalize)

                runway_shapefile.to_crs(self.grid_shp.crs, inplace=True)
                runway_shapefile['length'] = runway_shapefile.length
                # duplicating each runway by involved cell
                runway_shapefile = gpd.sjoin(runway_shapefile.reset_index(), self.grid_shp.reset_index(), how="inner",
                                             op='intersects')
                # Adding cell geometry
                runway_shapefile = runway_shapefile.merge(self.grid_shp.reset_index(), on='FID',  how='left')
                # Intersection between line (roadway) and polygon (cell)
                # runway_shapefile['geometry'] = runway_shapefile.apply(do_intersection, axis=1)
                runway_shapefile['mini_length'] = runway_shapefile.apply(get_intersection_length, axis=1)

                runway_shapefile.drop(columns=['geometry_x', 'geometry_y', 'index_right'], inplace=True)

                runway_shapefile['fraction'] = runway_shapefile['{0}_f'.format(phase_type)].multiply(
                    runway_shapefile['mini_length'] / runway_shapefile['length'])

                runway_shapefile['layer'] = 0
                runway_shapefile = runway_shapefile[['airport_id', 'FID', 'layer', 'fraction']]
                runway_shapefile = runway_shapefile.groupby(['airport_id', 'FID', 'layer']).sum()
                # runway_shapefile.set_index(['airport_id', 'FID', 'layer'], inplace=True)
                if not os.path.exists(os.path.dirname(runway_distribution_path)):
                    os.makedirs(os.path.dirname(runway_distribution_path))
                runway_shapefile.to_csv(runway_distribution_path)
            else:
                runway_shapefile = None
            runway_shapefile = self.comm.bcast(runway_shapefile, root=0)
        else:
            runway_shapefile = pd.read_csv(runway_distribution_path)
            runway_shapefile.set_index(['airport_id', 'FID', 'layer'], inplace=True)
        self.logger.write_time_log('AviationSector', 'calculate_runway_distribution',
                                   timeit.default_timer() - spent_time)

        return runway_shapefile

    def calculate_trajectories_distribution(self, airport_trajectories_shapefile, phase_type):
        """
        Calculate the location and portion for the emissions that have to be distributed on the trajectories lines.

        It only need to be calculated once. the second execution will read the auxiliary file already created.
        That emissions have to be distributed also vertically.

        :param airport_trajectories_shapefile: Shapefile with the trajectories information.
        :type airport_trajectories_shapefile: GeoDataFrame

        :param phase_type: 'arrival' or 'departure' to indicate teh type of approach.
        :type phase_type: str

        :return: DataFrame with the location (FID & level) and fraction for each airport.
        :rtype: DataFrame
        """
        spent_time = timeit.default_timer()

        def get_vertical_intersection_length(row):
            circle = row.get('start_point').buffer(row.get('circle_radious'))
            return row.get('src_geometry').intersection(circle).length

        def get_horizontal_intersection_length(row):
            return row.get('geometry_x').intersection(row.get('geometry_y')).length

        def do_vertical_intersection(row):
            circle = row.get('start_point').buffer(row.get('circle_radious'))
            return row.get('src_geometry').intersection(circle)

        def do_horizontal_intersection(row):
            return row.get('geometry_x').intersection(row.get('geometry_y'))

        def do_difference(row):
            circle = row.get('start_point').buffer(row.get('circle_radious'))
            return row.get('src_geometry').difference(circle)

        def normalize(df):
            total_fraction = df['fraction'].values.sum()
            df['fraction'] = df['fraction'] / total_fraction
            return df.loc[:, ['fraction']]

        self.logger.write_log('\t\tCalculating trajectories distribution for {0}'.format(phase_type), message_level=2)
        trajectories_distribution_path = os.path.join(
            self.auxiliary_dir, 'aviation', 'trajectories_{0}_distribution.csv'.format(phase_type))

        if not os.path.exists(trajectories_distribution_path):
            if self.comm.rank == 0:
                if not os.path.exists(os.path.dirname(trajectories_distribution_path)):
                    os.makedirs(os.path.dirname(trajectories_distribution_path))
                # Filtering shapefile
                airport_trajectories_shapefile = airport_trajectories_shapefile.xs(phase_type, level='operation').copy()
                airport_trajectories_shapefile = airport_trajectories_shapefile.loc[
                                                 airport_trajectories_shapefile['airport_id'].isin(
                                                     self.airport_list_full),
                                                 :]
                airport_trajectories_shapefile['fraction'] = airport_trajectories_shapefile.groupby('airport_id').apply(
                    normalize)

                # VERTICAL DISTRIBUTION
                airport_trajectories_shapefile['length'] = airport_trajectories_shapefile['geometry'].length
                trajectories_distr = []
                for level, v_lev in enumerate(self.vertical_levels):
                    dataframe = airport_trajectories_shapefile.copy()
                    dataframe.rename(columns={'geometry': 'src_geometry'}, inplace=True)
                    dataframe['layer'] = level
                    dataframe['circle_radious'] = (float(v_lev) / 1000.) * dataframe['length']
                    dataframe['geometry'] = dataframe[['src_geometry', 'start_point', 'circle_radious']].apply(
                        do_vertical_intersection, axis=1)
                    trajectories_distr.append(dataframe[['airport_id', 'fraction', 'length', 'layer', 'geometry']])
                    airport_trajectories_shapefile['geometry'] = dataframe[
                        ['src_geometry', 'start_point', 'circle_radious']].apply(do_difference, axis=1)
                    if v_lev > 1000:
                        break
                trajectories_distr = gpd.GeoDataFrame(pd.concat(trajectories_distr), geometry='geometry',
                                                      crs=airport_trajectories_shapefile.crs)
                trajectories_distr.reset_index(inplace=True)

                # HORIZONTAL DISTRIBUTION
                aux_grid = self.grid_shp.to_crs(trajectories_distr.crs).reset_index()
                # trajectories_distr.to_crs(self.grid_shp.crs, inplace=True)
                # duplicating each runway by involved cell
                trajectories_distr = gpd.sjoin(trajectories_distr, aux_grid, how="inner", op='intersects')
                # Adding cell geometry
                trajectories_distr = trajectories_distr.merge(aux_grid.loc[:, ['FID', 'geometry']], on='FID',
                                                              how='left')
                # Intersection between line (roadway) and polygon (cell)
                trajectories_distr['geometry'] = trajectories_distr.apply(do_horizontal_intersection, axis=1)
                trajectories_distr['mini_h_length'] = trajectories_distr.apply(get_horizontal_intersection_length,
                                                                               axis=1)
                trajectories_distr.drop(columns=['geometry_x', 'geometry_y', 'index_right'], inplace=True)

                trajectories_distr['fraction'] = trajectories_distr['fraction'].multiply(
                    trajectories_distr['mini_h_length'] / trajectories_distr['length'])

                trajectories_distr = trajectories_distr[['airport_id', 'FID', 'layer', 'fraction']]
                trajectories_distr = trajectories_distr.groupby(['airport_id', 'FID', 'layer']).sum()

                trajectories_distr.to_csv(trajectories_distribution_path)
            else:
                trajectories_distr = None
            trajectories_distr = self.comm.bcast(trajectories_distr, root=0)
        else:
            trajectories_distr = pd.read_csv(trajectories_distribution_path)
            trajectories_distr.set_index(['airport_id', 'FID', 'layer'], inplace=True)
        self.logger.write_time_log('AviationSector', 'calculate_trajectories_distribution',
                                   timeit.default_timer() - spent_time)

        return trajectories_distr

    def get_main_engine_emission(self, phase):
        """
        Calculate the main engine emissions for the given phase.

        :param phase: Phase to calculate.
        :type phase: str

        :return: Dataframe with the emissions of the phase py airport.
        :rtype: DataFrame
        """
        spent_time = timeit.default_timer()

        def get_e(df):
            """
            Number of engines associated to each airport
            """
            df['E'] = self.planes_info.loc[df.name, 'engine_n']
            return df.loc[:, ['E']]

        def get_t(df):
            """
            Time spent by each aircraft to complete tha selected phase (s)
            """
            plane_type = self.planes_info.loc[df.name[1], 'plane_type']
            df['t'] = self.times_info.loc[(df.name[0], plane_type), phase]
            return df.loc[:, ['t']]

        def get_ef(df, poll):
            """
            Emission factor associated to phase and pollutant
            """
            engine = self.planes_info.loc[df.name, 'engine_id']
            # ef_dataframe = pd.read_csv(os.path.join(self.ef_dir, PHASE_EF_FILE[phase]))
            ef_dataframe = self.ef_files[phase].reset_index()
            ef_dataframe.set_index('engine_id', inplace=True)
            df['EF'] = ef_dataframe.loc[engine, poll]
            return df.loc[:, ['EF']]

        def get_n(df):
            """
            Number of monthly operations associated to each aircraft, phase and month
            """
            self.operations = self.operations.sort_index()
            df['N'] = self.operations.loc[(df.name[0], df.name[1], PHASE_TYPE[phase]), df.name[2]]
            return df.loc[:, ['N']]

        def get_wf(df):
            """
            Daily factor associated to weekday and airport
            """
            import datetime
            date_np = df.head(1)['date'].values[0]
            date = datetime.datetime.utcfromtimestamp(date_np.astype(int) * 1e-9)
            profile = self.calculate_rebalanced_weekly_profile(self.weekly_profiles.loc[df.name[0], :].to_dict(), date)
            for weekday in np.unique(df['weekday'].values):
                df.loc[df['weekday'] == weekday, 'WF'] = profile[weekday]
            return df.loc[:, ['WF']]

        def get_hf(df):
            """
            Hourly factor associated to hour,
            """
            operation = PHASE_TYPE[phase]
            if df.name[2] > 4:
                day_type = 'weekend'
            else:
                day_type = 'weekday'
            df['HF'] = self.hourly_profiles.loc[(df.name[0], operation, day_type), df.name[1]]
            return df.loc[:, ['HF']]

        # Merging operations with airport geometry
        dataframe = pd.DataFrame(index=self.operations.xs(PHASE_TYPE[phase], level='operation').index)

        dataframe = dataframe.reset_index().set_index('airport_id')
        dataframe = self.airport_shapefile.join(dataframe, how='inner')
        dataframe.index.name = 'airport_id'
        dataframe = dataframe.reset_index().set_index(['airport_id', 'plane_id'])

        dataframe['E'] = dataframe.groupby('plane_id').apply(get_e)
        dataframe['t'] = dataframe.groupby(['airport_id', 'plane_id']).apply(get_t)

        # Dates
        dataframe = self.add_dates(dataframe)
        dataframe['month'] = dataframe['date'].dt.month
        dataframe['weekday'] = dataframe['date'].dt.weekday
        dataframe['hour'] = dataframe['date'].dt.hour

        dataframe['N'] = dataframe.groupby(['airport_id', 'plane_id', 'month']).apply(get_n)
        dataframe['WF'] = dataframe.groupby(['airport_id', 'month']).apply(get_wf)
        dataframe['HF'] = dataframe.groupby(['airport_id', 'hour', 'weekday']).apply(get_hf)
        dataframe.drop(columns=['date', 'month', 'weekday', 'hour'], inplace=True)

        # Getting factor
        dataframe['f'] = dataframe['E'] * dataframe['t'] * dataframe['N'] * dataframe['WF'] * dataframe['HF']
        dataframe.drop(columns=['E', 't', 'N', 'WF', 'HF'], inplace=True)

        for pollutant in self.source_pollutants:
            if pollutant not in ['nmvoc', 'ch4']:
                dataframe[pollutant] = dataframe.groupby('plane_id').apply(lambda x: get_ef(x, pollutant))
                dataframe[pollutant] = dataframe[pollutant] * dataframe['f']

        dataframe.drop(columns=['f', 'plane_id', 'geometry'], inplace=True)
        dataframe = dataframe.groupby(['airport_id', 'tstep']).sum()
        self.logger.write_time_log('AviationSector', 'get_main_engine_emission', timeit.default_timer() - spent_time)

        return dataframe

    def get_tyre_and_brake_wear_emission(self, phase):
        """
        Calculate the tyre and brake wear emissions for the given phase.

        :param phase: Phase to calculate.
        :type phase: str

        :return: Dataframe with the emissions of the phase py airport.
        :rtype: DataFrame
        """
        spent_time = timeit.default_timer()

        def get_mtow(df):
            """
            Maximum take-off weight associated to aircraft
            """
            df['MTOW'] = self.planes_info.loc[df.name, 'mtow']
            return df.loc[:, ['MTOW']]

        def get_ef(poll):
            """
            Emission factor associated to phase and pollutant
            """
            # ef_dataframe = pd.read_csv(os.path.join(self.ef_dir, PHASE_EF_FILE[phase]))
            ef_dataframe = self.ef_files[phase].reset_index()
            ef_dataframe.set_index('plane_id', inplace=True)
            ef = ef_dataframe.loc['default', poll]
            return ef

        def get_n(df):
            """
            Number of monthly operations associated to each aircraft, phase and month
            """
            self.operations = self.operations.sort_index()
            df['N'] = self.operations.loc[(df.name[0], df.name[1], PHASE_TYPE[phase]), df.name[2]]
            return df.loc[:, ['N']]

        def get_wf(df):
            """
            Daily factor associated to weekday and airport
            """
            import datetime
            date_np = df.head(1)['date'].values[0]
            date = datetime.datetime.utcfromtimestamp(date_np.astype(int) * 1e-9)
            profile = self.calculate_rebalanced_weekly_profile(self.weekly_profiles.loc[df.name[0], :].to_dict(), date)
            for weekday in np.unique(df['weekday'].values):
                df.loc[df['weekday'] == weekday, 'WF'] = profile[weekday]
            return df.loc[:, ['WF']]

        def get_hf(df):
            """
            Hourly factor associated to hour,
            """
            operation = PHASE_TYPE[phase]
            if df.name[2] > 4:
                day_type = 'weekend'
            else:
                day_type = 'weekday'
            df['HF'] = self.hourly_profiles.loc[(df.name[0], operation, day_type), df.name[1]]
            return df.loc[:, ['HF']]

        # Merging operations with airport geometry
        dataframe = pd.DataFrame(index=self.operations.xs(PHASE_TYPE[phase], level='operation').index)
        dataframe = dataframe.reset_index().set_index('airport_id')
        dataframe = self.airport_shapefile.join(dataframe, how='inner')
        dataframe.index.name = 'airport_id'
        dataframe = dataframe.reset_index().set_index(['airport_id', 'plane_id'])

        dataframe['MTOW'] = dataframe.groupby('plane_id').apply(get_mtow)

        # Dates
        dataframe = self.add_dates(dataframe)
        dataframe['month'] = dataframe['date'].dt.month
        dataframe['weekday'] = dataframe['date'].dt.weekday
        dataframe['hour'] = dataframe['date'].dt.hour

        dataframe['N'] = dataframe.groupby(['airport_id', 'plane_id', 'month']).apply(get_n)
        dataframe['WF'] = dataframe.groupby(['airport_id', 'month']).apply(get_wf)
        dataframe['HF'] = dataframe.groupby(['airport_id', 'hour', 'weekday']).apply(get_hf)
        dataframe.drop(columns=['date', 'month', 'weekday', 'hour'], inplace=True)

        # Getting factor
        dataframe['f'] = dataframe['MTOW'] * dataframe['N']

        dataframe['f'] = dataframe['MTOW'] * dataframe['N'] * dataframe['WF'] * dataframe['HF']
        dataframe.drop(columns=['MTOW', 'N', 'WF', 'HF'], inplace=True)

        for pollutant in self.source_pollutants:
            if pollutant in ['pm10', 'pm25']:
                dataframe[pollutant] = get_ef(pollutant)
                dataframe[pollutant] = dataframe[pollutant] * dataframe['f']

        dataframe.drop(columns=['f', 'plane_id', 'geometry'], inplace=True)
        dataframe = dataframe.groupby(['airport_id', 'tstep']).sum()
        self.logger.write_time_log('AviationSector', 'get_tyre_and_brake_wear_emission',
                                   timeit.default_timer() - spent_time)

        return dataframe

    def get_auxiliary_power_unit_emission(self, phase):
        """
        Calculate the auxiliary power unit (APU) emissions for the given phase.

        :param phase: Phase to calculate.
        :type phase: str

        :return: Dataframe with the emissions of the phase py airport.
        :rtype: DataFrame
        """
        spent_time = timeit.default_timer()

        def get_t(df):
            """
            Time spent by each aircraft to complete tha selected phase (s)
            """
            plane_type = self.planes_info.loc[df.name[1], 'plane_type']
            df['t'] = self.times_info.loc[(df.name[0], plane_type), phase]
            return df.loc[:, ['t']]

        def get_ef(df, poll):
            """
            Emission factor associated to phase and pollutant
            """
            engine = self.planes_info.loc[df.name, 'apu_id']
            # ef_dataframe = pd.read_csv(os.path.join(self.ef_dir, PHASE_EF_FILE[phase]))
            ef_dataframe = self.ef_files[phase].reset_index()
            ef_dataframe.set_index('apu_id', inplace=True)
            try:
                df['EF'] = ef_dataframe.loc[engine, poll]
            except (TypeError, KeyError):
                # Occurs when the plane has not APU
                df['EF'] = 0
            return df.loc[:, ['EF']]

        def get_n(df):
            """
            Number of monthly operations associated to each aircraft, phase and month
            """
            self.operations = self.operations.sort_index()
            df['N'] = self.operations.loc[(df.name[0], df.name[1], PHASE_TYPE[phase]), df.name[2]]
            return df.loc[:, ['N']]

        def get_wf(df):
            """
            Daily factor associated to weekday and airport
            """
            import datetime
            date_np = df.head(1)['date'].values[0]
            date = datetime.datetime.utcfromtimestamp(date_np.astype(int) * 1e-9)
            profile = self.calculate_rebalanced_weekly_profile(self.weekly_profiles.loc[df.name[0], :].to_dict(), date)
            for weekday in np.unique(df['weekday'].values):
                df.loc[df['weekday'] == weekday, 'WF'] = profile[weekday]
            return df.loc[:, ['WF']]

        def get_hf(df):
            """
            Hourly factor associated to hour,
            """
            operation = PHASE_TYPE[phase]
            if df.name[2] > 4:
                day_type = 'weekend'
            else:
                day_type = 'weekday'
            df['HF'] = self.hourly_profiles.loc[(df.name[0], operation, day_type), df.name[1]]
            return df.loc[:, ['HF']]

        # Merging operations with airport geometry
        dataframe = pd.DataFrame(index=self.operations.xs(PHASE_TYPE[phase], level='operation').index)
        dataframe = dataframe.reset_index().set_index('airport_id')
        dataframe = self.airport_shapefile.join(dataframe, how='inner')
        dataframe.index.name = 'airport_id'
        dataframe = dataframe.reset_index().set_index(['airport_id', 'plane_id'])

        dataframe['t'] = dataframe.groupby(['airport_id', 'plane_id']).apply(get_t)

        # Dates
        dataframe = self.add_dates(dataframe)
        dataframe['month'] = dataframe['date'].dt.month
        dataframe['weekday'] = dataframe['date'].dt.weekday
        dataframe['hour'] = dataframe['date'].dt.hour

        dataframe['N'] = dataframe.groupby(['airport_id', 'plane_id', 'month']).apply(get_n)
        dataframe['WF'] = dataframe.groupby(['airport_id', 'month']).apply(get_wf)
        dataframe['HF'] = dataframe.groupby(['airport_id', 'hour', 'weekday']).apply(get_hf)
        dataframe.drop(columns=['date', 'month', 'weekday', 'hour'], inplace=True)

        # Getting factor
        dataframe['f'] = dataframe['t'] * dataframe['N'] * dataframe['WF'] * dataframe['HF']
        dataframe.drop(columns=['t', 'N', 'WF', 'HF'], inplace=True)

        for pollutant in self.source_pollutants:
            if pollutant not in ['nmvoc', 'ch4']:
                dataframe[pollutant] = dataframe.groupby('plane_id').apply(lambda x: get_ef(x, pollutant))
                dataframe[pollutant] = dataframe[pollutant] * dataframe['f']

        dataframe.drop(columns=['f', 'plane_id', 'geometry'], inplace=True)
        dataframe = dataframe.groupby(['airport_id', 'tstep']).sum()
        self.logger.write_time_log('AviationSector', 'get_auxiliary_power_unit_emission',
                                   timeit.default_timer() - spent_time)

        return dataframe

    def distribute(self, dataframe, distribution):
        """
        Distributes the airport emissions by the given distribution.

        :param dataframe: Emissions by airport.
        :type dataframe: DataFrame

        :param distribution: Involved cells by airport.
        :type distribution: DataFrame

        :return: Emissions distributed by cell (FID)
        :rtype: DataFrame
        """
        spent_time = timeit.default_timer()
        pollutants = dataframe.columns.values
        dataframe.reset_index(inplace=True)
        distribution.reset_index(inplace=True)

        dataframe = dataframe.merge(distribution, on='airport_id')

        dataframe[pollutants] = dataframe[pollutants].multiply(dataframe['fraction'], axis=0)
        dataframe.drop(columns=['airport_id', 'fraction'], inplace=True)
        dataframe = dataframe.groupby(['FID', 'layer', 'tstep']).sum()
        self.logger.write_time_log('AviationSector', 'distribute', timeit.default_timer() - spent_time)

        return dataframe

    def calculate_emissions(self):
        """
        Main function to calculate the emissions for the Landing and take off airport emissions.

        :return: Airport emissions distributed by cell (FID), layer and time step.
        :rtype: DataFrame
        """
        spent_time = timeit.default_timer()
        self.logger.write_log('\tCalculating emissions')

        taxi_out = self.get_main_engine_emission('taxi_out')
        self.logger.write_log('\t\tTaxi out emissions calculated.', message_level=2)
        taxi_in = self.get_main_engine_emission('taxi_in')
        self.logger.write_log('\t\tTaxi in emissions calculated.', message_level=2)
        takeoff = self.get_main_engine_emission('takeoff')
        self.logger.write_log('\t\tTake off emissions calculated.', message_level=2)
        climbout = self.get_main_engine_emission('climbout')
        self.logger.write_log('\t\tClimb out emissions calculated.', message_level=2)
        approach = self.get_main_engine_emission('approach')
        self.logger.write_log('\t\tApproach emissions calculated.', message_level=2)
        landing = self.get_main_engine_emission('landing')
        self.logger.write_log('\t\tLanding emissions calculated.', message_level=2)

        landing_wear = self.get_tyre_and_brake_wear_emission('landing_wear')
        self.logger.write_log('\t\tLanding wear emissions calculated.', message_level=2)

        post_taxi_in = self.get_auxiliary_power_unit_emission('post-taxi_in')
        self.logger.write_log('\t\tPost taxi in emissions calculated.', message_level=2)
        pre_taxi_out = self.get_auxiliary_power_unit_emission('pre-taxi_out')
        self.logger.write_log('\t\tPre taxi out emissions calculated.', message_level=2)

        airport_emissions = pd.concat([pre_taxi_out, taxi_out, taxi_in, post_taxi_in])
        airport_emissions = airport_emissions.groupby(['airport_id', 'tstep']).sum()
        airport_emissions = self.distribute(airport_emissions, self.airport_distribution)
        self.logger.write_log('\t\tAirport emissions distributed (pre_taxi_out, taxi_out, taxi_in, post_taxi_in)',
                              message_level=2)

        runway_departure_emissions = self.distribute(takeoff, self.runway_departure_distribution)
        runway_arrival_emissions = self.distribute(landing, self.runway_arrival_distribution)
        runway_arrival_emissions_wear = self.distribute(landing_wear, self.runway_arrival_distribution)
        self.logger.write_log('\t\tRunway emissions distributed (takeoff, landing, landing_wear)', message_level=2)

        trajectory_arrival_emissions = self.distribute(approach, self.trajectory_arrival_distribution)
        trajectory_departure_emisions = self.distribute(climbout, self.trajectory_departure_distribution)
        self.logger.write_log('\t\tTrajectory emissions distributed (approach, climb out)', message_level=2)

        emissions = pd.concat([airport_emissions, runway_departure_emissions, trajectory_arrival_emissions,
                               trajectory_departure_emisions, runway_arrival_emissions], sort=False)

        emissions = emissions.groupby(['FID', 'layer', 'tstep']).sum()
        runway_arrival_emissions_wear = runway_arrival_emissions_wear.groupby(['FID', 'layer', 'tstep']).sum()

        if 'hc' in self.source_pollutants:  # After Olivier (1991)
            emissions['nmvoc'] = 0.9 * emissions['hc']
            emissions['ch4'] = 0.1 * emissions['hc']

        # Speceiation
        runway_arrival_emissions_wear = self.speciate(runway_arrival_emissions_wear, 'landing_wear')
        emissions = self.speciate(emissions, 'default')

        emissions = pd.concat([emissions, runway_arrival_emissions_wear], sort=False)
        emissions = emissions[(emissions.T != 0).any()]
        emissions = emissions.groupby(['FID', 'layer', 'tstep']).sum()

        # From kmol/h or kg/h to mol/h or g/h
        emissions = emissions * 1000
        self.logger.write_log('\t\tAviation emissions calculated', message_level=2)
        self.logger.write_time_log('AviationSector', 'calculate_emissions', timeit.default_timer() - spent_time)
        return emissions
