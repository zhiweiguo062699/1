#!/usr/bin/env python

# Copyright 2018 Earth Sciences Department, BSC-CNS
#
# This file is part of HERMESv3_BU.
#
# HERMESv3_BU is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HERMESv3_BU is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HERMESv3_BU. If not, see <http://www.gnu.org/licenses/>.


from configargparse import ArgParser
import os
from mpi4py import MPI


class Config(ArgParser):
    """
    Configuration arguments class.
    """
    def __init__(self, new_date=None):
        """
        Read and parse all the arguments.

        :param new_date: Starting date for simulation loop day.
        :type new_date: datetime.datetime
        """
        self.new_date = new_date

        super(Config, self).__init__()
        self.arguments = self.read_arguments()

    def read_arguments(self):
        """
        Reads all the arguments from command line or from the configuration file.
        The value of an argument given by command line has high priority that the one that appear in the
        configuration file.

        :return: Arguments already parsed.
        :rtype: Namespace
        """
        from shutil import rmtree

        p = ArgParser()
        p.add_argument('-c', '--my-config', required=False, is_config_file=True, help='Path to the configuration file.')
        # ===== GENERAL =====
        p.add_argument('--log_level', required=True, help='Level of detail of the running process information.',
                       type=int, choices=[1, 2, 3])

        p.add_argument('--input_dir', required=True, help='Path to the input directory of the model.')
        p.add_argument('--data_path', required=True, help='Path to the data necessary for the model.')
        p.add_argument('--output_dir', required=True, help='Path to the output directory of the model.')
        p.add_argument('--output_name', required=True,
                       help="Name of the output file. You can add the string '<date>' that will be substitute by the " +
                            "starting date of the simulation day.")
        p.add_argument('--emission_summary', required=False, type=str, default='False',
                       help='Indicates if you want to create the emission summary files.')
        p.add_argument('--start_date', required=True, help='Starting Date to simulate (UTC)')
        p.add_argument('--end_date', required=False, default=None,
                       help='If you want to simulate more than one day you have to specify the ending date of ' +
                            'simulation in this parameter. If it is not set end_date = start_date.')
        p.add_argument('--output_timestep_num', required=True, help='Number of timesteps to simulate.', type=int)
        p.add_argument('--auxiliary_files_path', required=True,
                       help='Path to the directory where the necessary auxiliary files will be created if them are ' +
                            'not created yet.')
        p.add_argument('--erase_auxiliary_files', required=False, default='False', type=str,
                       help='Indicates if you want to start from scratch removing the auxiliary files already created.')
        p.add_argument('--molecular_weights', required=True,
                       help='Path to the file that contains the molecular weights of the input pollutants.')

        # ===== DOMAIN =====
        p.add_argument('--output_model', required=True, help='Name of the output model.',
                       choices=['MONARCH', 'CMAQ', 'WRF_CHEM', 'DEFAULT'])
        p.add_argument('--writing_processors', required=False, type=int,
                       help='Number of processors dedicated to write. ' +
                            'Maximum number accepted is the number of rows of the destiny grid.')
        p.add_argument('--output_attributes', required=False,
                       help='Path to the file that contains the global attributes.')

        p.add_argument('--domain_type', required=True, help='Type of domain to simulate.',
                       choices=['lcc', 'rotated', 'mercator', 'regular'])

        p.add_argument('--vertical_description', required=True,
                       help='Path to the file that contains the vertical description of the desired output.')

        # Rotated options
        p.add_argument('--centre_lat', required=False, type=float,
                       help='Central geographic latitude of grid (non-rotated degrees). Corresponds to the TPH0D ' +
                            'parameter in NMMB-MONARCH.')
        p.add_argument('--centre_lon', required=False, type=float,
                       help='Central geographic longitude of grid (non-rotated degrees, positive east). Corresponds ' +
                            'to the TLM0D parameter in NMMB-MONARCH.')
        p.add_argument('--west_boundary', required=False, type=float,
                       help="Grid's western boundary from center point (rotated degrees). Corresponds to the WBD " +
                            "parameter in NMMB-MONARCH.")
        p.add_argument('--south_boundary', required=False, type=float,
                       help="Grid's southern boundary from center point (rotated degrees). Corresponds to the SBD " +
                            "parameter in NMMB-MONARCH.")
        p.add_argument('--inc_rlat', required=False, type=float,
                       help='Latitudinal grid resolution (rotated degrees). Corresponds to the DPHD parameter in ' +
                            'NMMB-MONARCH.')
        p.add_argument('--inc_rlon', required=False, type=float,
                       help='Longitudinal grid resolution (rotated degrees). Corresponds to the DLMD parameter  ' +
                            'in NMMB-MONARCH.')

        # Lambert conformal conic options
        p.add_argument('--lat_1', required=False, type=float,
                       help='Standard parallel 1 (in deg). Corresponds to the P_ALP parameter of the GRIDDESC file.')
        p.add_argument('--lat_2', required=False, type=float,
                       help='Standard parallel 2 (in deg). Corresponds to the P_BET parameter of the GRIDDESC file.')
        p.add_argument('--lon_0', required=False, type=float,
                       help='Longitude of the central meridian (degrees). Corresponds to the P_GAM parameter of ' +
                            'the GRIDDESC file.')
        p.add_argument('--lat_0', required=False, type=float,
                       help='Latitude of the origin of the projection (degrees). Corresponds to the Y_CENT  ' +
                            'parameter of the GRIDDESC file.')
        p.add_argument('--nx', required=False, type=int,
                       help='Number of grid columns. Corresponds to the NCOLS parameter of the GRIDDESC file.')
        p.add_argument('--ny', required=False, type=int,
                       help='Number of grid rows. Corresponds to the NROWS parameter of the GRIDDESC file.')
        p.add_argument('--inc_x', required=False, type=float,
                       help='X-coordinate cell dimension (meters). Corresponds to the XCELL parameter of the ' +
                            'GRIDDESC file.')
        p.add_argument('--inc_y', required=False, type=float,
                       help='Y-coordinate cell dimension (meters). Corresponds to the YCELL parameter of the ' +
                            'GRIDDESC file.')
        p.add_argument('--x_0', required=False, type=float,
                       help='X-coordinate origin of grid (meters). Corresponds to the XORIG parameter of the ' +
                            'GRIDDESC file.')
        p.add_argument('--y_0', required=False, type=float,
                       help='Y-coordinate origin of grid (meters). Corresponds to the YORIG parameter of the ' +
                            'GRIDDESC file.')

        # Mercator
        p.add_argument('--lat_ts', required=False, type=float, help='...')

        # Regular lat-lon options:
        p.add_argument('--lat_orig', required=False, type=float, help='Latitude of the corner of the first cell.')
        p.add_argument('--lon_orig', required=False, type=float, help='Longitude of the corner of the first cell.')
        p.add_argument('--n_lat', required=False, type=int, help='Number of latitude elements.')
        p.add_argument('--n_lon', required=False, type=int, help='Number of longitude elements.')
        p.add_argument('--inc_lat', required=False, type=float, help='Latitude grid resolution.')
        p.add_argument('--inc_lon', required=False, type=float, help='Longitude grid resolution.')

        # ===== SECTOR SELECTION =====
        p.add_argument('--traffic_processors', required=True, type=int)
        p.add_argument('--traffic_area_processors', required=True, type=int)
        p.add_argument('--aviation_processors', required=True, type=int)
        p.add_argument('--point_sources_processors', required=True, type=int)
        p.add_argument('--recreational_boats_processors', required=True, type=int)
        p.add_argument('--shipping_port_processors', required=True, type=int)
        p.add_argument('--residential_processors', required=True, type=int)
        p.add_argument('--livestock_processors', required=True, type=int)
        p.add_argument('--crop_operations_processors', required=True, type=int)
        p.add_argument('--crop_fertilizers_processors', required=True, type=int)
        p.add_argument('--agricultural_machinery_processors', required=True, type=int)

        p.add_argument('--speciation_map', required=False, help='...')

        # ===== SHAPEFILES =====
        p.add_argument('--nut_shapefile_prov', required=False, type=str, default='True')
        p.add_argument('--nut_shapefile_ccaa', required=False, type=str, default='True')

        p.add_argument('--clipping', required=False, type=str, default=None,
                       help='To clip the domain into an specific zone. ' +
                            'It can be a shapefile path, a list of points to make a polygon or nothing to use ' +
                            'the default clip: domain extension')

        # ===== METEO PATHS =====
        p.add_argument('--temperature_hourly_files_path', required=False, type=str, default='True')
        p.add_argument('--temperature_daily_files_path', required=False, type=str, default='True')
        p.add_argument('--wind_speed_daily_files_path', required=False, type=str, default='True')
        p.add_argument('--precipitation_files_path', required=False, type=str, default='True')
        p.add_argument('--temperature_4d_dir', required=False, type=str, default='True')
        p.add_argument('--temperature_sfc_dir', required=False, type=str, default='True')
        p.add_argument('--u_wind_speed_4d_dir', required=False, type=str, default='True')
        p.add_argument('--v_wind_speed_4d_dir', required=False, type=str, default='True')
        p.add_argument('--u10_wind_speed_dir', required=False, type=str, default='True')
        p.add_argument('--v10_wind_speed_dir', required=False, type=str, default='True')
        p.add_argument('--friction_velocity_dir', required=False, type=str, default='True')
        p.add_argument('--pblh_dir', required=False, type=str, default='True')
        p.add_argument('--obukhov_length_dir', required=False, type=str, default='True')
        p.add_argument('--layer_thickness_dir', required=False, type=str, default='True')

        # ***** AVIATION SECTOR *****
        p.add_argument('--aviation_source_pollutants', required=False, help='...')
        p.add_argument('--airport_list', required=False, help='...')
        p.add_argument('--plane_list', required=False, help='...')
        p.add_argument('--airport_shapefile_path', required=False, help='...')
        p.add_argument('--airport_runways_shapefile_path', required=False, help='...')
        p.add_argument('--airport_runways_corners_shapefile_path', required=False, help='...')
        p.add_argument('--airport_trajectories_shapefile_path', required=False, help='...')
        p.add_argument('--airport_operations_path', required=False, help='...')
        p.add_argument('--planes_path', required=False, help='...')
        p.add_argument('--airport_times_path', required=False, help='...')
        p.add_argument('--airport_ef_dir', required=False, help='...')
        p.add_argument('--aviation_weekly_profiles', required=False, help='...')
        p.add_argument('--aviation_hourly_profiles', required=False, help='...')
        p.add_argument('--aviation_speciation_profiles', required=False, help='...')

        # ***** SHIPPING PORT SECTOR *****
        p.add_argument('--shipping_port_source_pollutants', required=False, help='...')
        p.add_argument('--vessel_list', required=False, help='...')
        p.add_argument('--port_list', required=False, help='...')
        p.add_argument('--hoteling_shapefile_path', required=False, help='...')
        p.add_argument('--maneuvering_shapefile_path', required=False, help='...')
        p.add_argument('--shipping_port_ef_path', required=False, help='...')
        p.add_argument('--shipping_port_engine_percent_path', required=False, help='...')
        p.add_argument('--shipping_port_tonnage_path', required=False, help='...')
        p.add_argument('--shipping_port_load_factor_path', required=False, help='...')
        p.add_argument('--shipping_port_power_path', required=False, help='...')
        p.add_argument('--shipping_port_monthly_profiles', required=False, help='...')
        p.add_argument('--shipping_port_weekly_profiles', required=False, help='...')
        p.add_argument('--shipping_port_hourly_profiles', required=False, help='...')
        p.add_argument('--shipping_port_speciation_profiles', required=False, help='...')

        # ***** LIVESTOCK SECTOR *****
        p.add_argument('--livestock_source_pollutants', required=False, help='...')
        p.add_argument('--animal_list', required=False, help='...')
        p.add_argument('--gridded_livestock', required=False, help='...')
        p.add_argument('--correction_split_factors', required=False, help='...')
        p.add_argument('--denominator_yearly_factor_dir', required=False, help='...')
        p.add_argument('--livestock_ef_files_dir', required=False, help='...')
        p.add_argument('--livestock_monthly_profiles', required=False, help='...')
        p.add_argument('--livestock_weekly_profiles', required=False, help='...')
        p.add_argument('--livestock_hourly_profiles', required=False, help='...')
        p.add_argument('--livestock_speciation_profiles', required=False, help='...')

        # ***** AGRICULTURAL SECTOR*****
        p.add_argument('--land_uses_path', required=False, help='...')
        p.add_argument('--land_use_by_nut_path', required=False, help='...')
        p.add_argument('--crop_by_nut_path', required=False, help='...')
        p.add_argument('--crop_from_landuse_path', required=False, help='...')

        # ***** CROP OPERATIONS SECTOR
        p.add_argument('--crop_operations_source_pollutants', required=False, help='...')
        p.add_argument('--crop_operations_list', required=False, help='...')
        p.add_argument('--crop_operations_ef_files_dir', required=False, help='...')
        p.add_argument('--crop_operations_monthly_profiles', required=False, help='...')
        p.add_argument('--crop_operations_weekly_profiles', required=False, help='...')
        p.add_argument('--crop_operations_hourly_profiles', required=False, help='...')
        p.add_argument('--crop_operations_speciation_profiles', required=False, help='...')

        # ***** CROP FERTILIZERS SECTOR *****
        p.add_argument('--crop_fertilizers_source_pollutants', required=False, help='...')
        p.add_argument('--crop_fertilizers_list', required=False, help='...')
        p.add_argument('--cultivated_ratio', required=False, help='...')
        p.add_argument('--fertilizers_rate', required=False, help='...')
        p.add_argument('--crop_f_parameter', required=False, help='...')
        p.add_argument('--crop_f_fertilizers', required=False, help='...')
        p.add_argument('--gridded_ph', required=False, help='...')
        p.add_argument('--gridded_cec', required=False, help='...')
        p.add_argument('--fertilizers_denominator_yearly_factor_path', required=False, help='...')
        p.add_argument('--crop_calendar', required=False, help='...')
        p.add_argument('--crop_fertilizers_hourly_profiles', required=False, help='...')
        p.add_argument('--crop_fertilizers_speciation_profiles', required=False, help='...')
        p.add_argument('--crop_growing_degree_day_path', required=False, help='...')

        # ***** CROP MACHINERY SECTOR *****
        p.add_argument('--crop_machinery_source_pollutants', required=False, help='...')
        p.add_argument('--crop_machinery_list', required=False, help='...')
        p.add_argument('--machinery_list', required=False, help='...')
        p.add_argument('--crop_machinery_deterioration_factor_path', required=False, help='...')
        p.add_argument('--crop_machinery_load_factor_path', required=False, help='...')
        p.add_argument('--crop_machinery_vehicle_ratio_path', required=False, help='...')
        p.add_argument('--crop_machinery_vehicle_units_path', required=False, help='...')
        p.add_argument('--crop_machinery_vehicle_workhours_path', required=False, help='...')
        p.add_argument('--crop_machinery_vehicle_power_path', required=False, help='...')
        p.add_argument('--crop_machinery_ef_path', required=False, help='...')
        p.add_argument('--crop_machinery_monthly_profiles', required=False, help='...')
        p.add_argument('--crop_machinery_weekly_profiles', required=False, help='...')
        p.add_argument('--crop_machinery_hourly_profiles', required=False, help='...')
        p.add_argument('--crop_machinery_speciation_map', required=False, help='...')
        p.add_argument('--crop_machinery_speciation_profiles', required=False, help='...')
        p.add_argument('--crop_machinery_by_nut', required=False, help='...')

        # ***** RESIDENTIAL SECTOR *****
        p.add_argument('--fuel_list', required=False, help='...')
        p.add_argument('--residential_source_pollutants', required=False, help='...')
        p.add_argument('--population_density_map', required=False, help='...')
        p.add_argument('--population_type_map', required=False, help='...')
        p.add_argument('--population_type_by_ccaa', required=False, help='...')
        p.add_argument('--population_type_by_prov', required=False, help='...')
        p.add_argument('--energy_consumption_by_prov', required=False, help='...')
        p.add_argument('--energy_consumption_by_ccaa', required=False, help='...')
        p.add_argument('--residential_spatial_proxies', required=False, help='...')
        p.add_argument('--residential_ef_files_path', required=False, help='...')
        p.add_argument('--residential_heating_degree_day_path', required=False, help='...')
        p.add_argument('--residential_hourly_profiles', required=False, help='...')
        p.add_argument('--residential_speciation_profiles', required=False, help='...')

        # ***** RECREATIONAL BOATS SECTOR *****
        p.add_argument('--recreational_boats_source_pollutants', required=False, help='...')
        p.add_argument('--recreational_boats_list', required=False, help='...')
        p.add_argument('--recreational_boats_density_map', required=False, help='...')
        p.add_argument('--recreational_boats_by_type', required=False, help='...')
        p.add_argument('--recreational_boats_ef_path', required=False, help='...')
        p.add_argument('--recreational_boats_monthly_profiles', required=False, help='...')
        p.add_argument('--recreational_boats_weekly_profiles', required=False, help='...')
        p.add_argument('--recreational_boats_hourly_profiles', required=False, help='...')
        p.add_argument('--recreational_boats_speciation_profiles', required=False, help='...')

        # ***** POINT SOURCE SECTOR *****
        p.add_argument('--point_source_pollutants', required=False, help='...')
        p.add_argument('--plume_rise', required=False, help='...')
        p.add_argument('--point_source_snaps', required=False, help='...')
        p.add_argument('--point_source_catalog', required=False, help='...')
        p.add_argument('--point_source_monthly_profiles', required=False, help='...')
        p.add_argument('--point_source_weekly_profiles', required=False, help='...')
        p.add_argument('--point_source_hourly_profiles', required=False, help='...')
        p.add_argument('--point_source_speciation_profiles', required=False, help='...')
        p.add_argument('--point_source_measured_emissions', required=False, help='...')

        # ***** TRAFFIC SECTOR *****
        p.add_argument('--do_hot', required=False, help='...')
        p.add_argument('--do_cold', required=False, help='...')
        p.add_argument('--do_tyre_wear', required=False, help='...')
        p.add_argument('--do_brake_wear', required=False, help='...')
        p.add_argument('--do_road_wear', required=False, help='...')
        p.add_argument('--do_resuspension', required=False, help='...')
        p.add_argument('--resuspension_correction', required=False, help='...')
        p.add_argument('--write_rline', required=False, help='...')

        p.add_argument('--traffic_pollutants', required=False, help='...')
        p.add_argument('--vehicle_types', required=False, help='...')
        p.add_argument('--load', type=float, required=False, help='...')
        p.add_argument('--road_link_path', required=False, help='...')
        p.add_argument('--fleet_compo_path', required=False, help='...')
        p.add_argument('--traffic_ef_path', required=False, help='...')
        p.add_argument('--traffic_speed_hourly_path', required=False, help='...')
        p.add_argument('--traffic_monthly_profiles', required=False, help='...')
        p.add_argument('--traffic_weekly_profiles', required=False, help='...')
        p.add_argument('--traffic_hourly_profiles_mean', required=False, help='...')
        p.add_argument('--traffic_hourly_profiles_weekday', required=False, help='...')
        p.add_argument('--traffic_hourly_profiles_saturday', required=False, help='...')
        p.add_argument('--traffic_hourly_profiles_sunday', required=False, help='...')
        p.add_argument('--traffic_speciation_profile_hot_cold', required=False, help='...')
        p.add_argument('--traffic_speciation_profile_tyre', required=False, help='...')
        p.add_argument('--traffic_speciation_profile_road', required=False, help='...')
        p.add_argument('--traffic_speciation_profile_brake', required=False, help='...')
        p.add_argument('--traffic_speciation_profile_resuspension', required=False, help='...')

        # ***** TRAFFIC AREA SECTOR *****
        p.add_argument('--traffic_area_pollutants', required=False, help='...')
        p.add_argument('--do_evaporative', required=False, help='...')
        p.add_argument('--traffic_area_gas_path', required=False, help='...')
        p.add_argument('--population_by_municipality', required=False, help='...')
        p.add_argument('--traffic_area_speciation_profiles_evaporative', required=False, help='...')
        p.add_argument('--traffic_area_evaporative_ef_file', required=False, help='...')
        p.add_argument('--do_small_cities', required=False, help='...')
        p.add_argument('--traffic_area_small_cities_path', required=False, help='...')
        p.add_argument('--traffic_area_speciation_profiles_small_cities', required=False, help='...')
        p.add_argument('--traffic_area_small_cities_ef_file', required=False, help='...')
        p.add_argument('--small_cities_hourly_profile', required=False, help='...')
        p.add_argument('--small_cities_weekly_profile', required=False, help='...')
        p.add_argument('--small_cities_monthly_profile', required=False, help='...')

        arguments = p.parse_args()

        for item in vars(arguments):
            is_str = isinstance(arguments.__dict__[item], str)
            if is_str:
                arguments.__dict__[item] = arguments.__dict__[item].replace('<data_path>', arguments.data_path)
                arguments.__dict__[item] = arguments.__dict__[item].replace('<input_dir>', arguments.input_dir)
                arguments.__dict__[item] = arguments.__dict__[item].replace('<domain_type>', arguments.domain_type)

                if arguments.domain_type == 'regular':
                    arguments.__dict__[item] = arguments.__dict__[item].replace('<resolution>', '{1}_{2}'.format(
                        item, arguments.inc_lat, arguments.inc_lon))
                elif arguments.domain_type == 'rotated':
                    arguments.__dict__[item] = arguments.__dict__[item].replace('<resolution>', '{1}_{2}'.format(
                        item, arguments.inc_rlat, arguments.inc_rlon))
                elif arguments.domain_type == 'lcc' or arguments.domain_type == 'mercator':
                    arguments.__dict__[item] = arguments.__dict__[item].replace('<resolution>', '{1}_{2}'.format(
                        item, arguments.inc_x, arguments.inc_y))

        arguments.emission_summary = self._parse_bool(arguments.emission_summary)
        arguments.start_date = self._parse_start_date(arguments.start_date)
        arguments.end_date = self._parse_end_date(arguments.end_date, arguments.start_date)
        arguments.output_name = self.get_output_name(arguments)

        arguments.erase_auxiliary_files = self._parse_bool(arguments.erase_auxiliary_files)
        self.create_dir(arguments.output_dir)

        if arguments.erase_auxiliary_files:
            if os.path.exists(arguments.auxiliary_files_path):
                comm = MPI.COMM_WORLD
                if comm.Get_rank() == 0:
                    rmtree(arguments.auxiliary_files_path)
                comm.Barrier()
        self.create_dir(arguments.auxiliary_files_path)

        arguments.do_traffic = arguments.traffic_processors > 0
        arguments.do_traffic_area = arguments.traffic_area_processors > 0
        arguments.do_aviation = arguments.aviation_processors > 0
        arguments.do_point_sources = arguments.point_sources_processors > 0
        arguments.do_recreational_boats = arguments.recreational_boats_processors > 0
        arguments.do_shipping_port = arguments.shipping_port_processors > 0
        arguments.do_residential = arguments.residential_processors > 0
        arguments.do_livestock = arguments.livestock_processors > 0
        arguments.do_crop_operations = arguments.crop_operations_processors > 0
        arguments.do_crop_fertilizers = arguments.crop_fertilizers_processors > 0
        arguments.do_agricultural_machinery = arguments.agricultural_machinery_processors > 0

        # Aviation lists
        arguments.airport_list = self._parse_list(arguments.airport_list)
        arguments.plane_list = self._parse_list(arguments.plane_list)
        arguments.aviation_source_pollutants = self._parse_list(arguments.aviation_source_pollutants)

        # Shipping Port lists
        arguments.shipping_port_source_pollutants = self._parse_list(arguments.shipping_port_source_pollutants)
        arguments.vessel_list = self._parse_list(arguments.vessel_list)
        arguments.port_list = self._parse_list(arguments.port_list)

        # Livestock lists
        arguments.livestock_source_pollutants = self._parse_list(arguments.livestock_source_pollutants)
        arguments.animal_list = self._parse_list(arguments.animal_list)

        # Crop operations lists
        arguments.crop_operations_source_pollutants = self._parse_list(arguments.crop_operations_source_pollutants)
        arguments.crop_operations_list = self._parse_list(arguments.crop_operations_list)

        # Crop fertilizers lists
        arguments.crop_fertilizers_source_pollutants = self._parse_list(arguments.crop_fertilizers_source_pollutants)
        arguments.crop_fertilizers_list = self._parse_list(arguments.crop_fertilizers_list)

        # Crop machinery lists
        arguments.crop_machinery_source_pollutants = self._parse_list(arguments.crop_machinery_source_pollutants)
        arguments.crop_machinery_list = self._parse_list(arguments.crop_machinery_list)
        arguments.machinery_list = self._parse_list(arguments.machinery_list)

        # Residential lists
        arguments.fuel_list = self._parse_list(arguments.fuel_list)
        arguments.residential_source_pollutants = self._parse_list(arguments.residential_source_pollutants)

        # Recreational Boats lists
        arguments.recreational_boats_source_pollutants = self._parse_list(
            arguments.recreational_boats_source_pollutants)
        arguments.recreational_boats_list = self._parse_list(arguments.recreational_boats_list)

        # Point Source bools
        arguments.plume_rise = self._parse_bool(arguments.plume_rise)

        # Point Source lists
        arguments.point_source_pollutants = self._parse_list(arguments.point_source_pollutants)
        arguments.point_source_snaps = self._parse_list(arguments.point_source_snaps)

        # Traffic bools
        arguments.do_hot = self._parse_bool(arguments.do_hot)
        arguments.do_cold = self._parse_bool(arguments.do_cold)
        arguments.do_tyre_wear = self._parse_bool(arguments.do_tyre_wear)
        arguments.do_brake_wear = self._parse_bool(arguments.do_brake_wear)
        arguments.do_road_wear = self._parse_bool(arguments.do_road_wear)
        arguments.do_resuspension = self._parse_bool(arguments.do_resuspension)
        arguments.resuspension_correction = self._parse_bool(arguments.resuspension_correction)
        arguments.write_rline = self._parse_bool(arguments.write_rline)

        # Traffic lists
        arguments.traffic_pollutants = self._parse_list(arguments.traffic_pollutants)
        arguments.vehicle_types = self._parse_list(arguments.vehicle_types)

        # Traffic area bools
        arguments.do_evaporative = self._parse_bool(arguments.do_evaporative)
        arguments.do_small_cities = self._parse_bool(arguments.do_small_cities)

        # Traffic area lists
        arguments.traffic_area_pollutants = self._parse_list(arguments.traffic_area_pollutants)

        return arguments

    @staticmethod
    def get_output_name(arguments):
        """
        Generates the full path of the output replacing <date> by YYYYMMDDHH.

        :param arguments: Config file arguments.
        :type arguments: Namespace

        :return: Complete path to the output file.
        :rtype: str
        """
        import os
        file_name = arguments.output_name.replace('<date>', arguments.start_date.strftime('%Y%m%d%H'))

        full_path = os.path.join(arguments.output_dir, file_name)
        return full_path

    @staticmethod
    def create_dir(path):
        """
        Create the given folder if it is not created yet.

        :param path: Path to create.
        :type path: str
        """
        import os
        from mpi4py import MPI
        icomm = MPI.COMM_WORLD
        comm = icomm.Split(color=0, key=0)
        rank = comm.Get_rank()

        if rank == 0:
            if not os.path.exists(path):
                os.makedirs(path)

        comm.Barrier()

    @staticmethod
    def _parse_bool(str_bool):
        """
        Parse the giving string into a boolean.
        The accepted options for a True value are: 'True', 'true', 'T', 't', 'Yes', 'yes', 'Y', 'y', '1'
        The accepted options for a False value are: 'False', 'false', 'F', 'f', 'No', 'no', 'N', 'n', '0'

        If the sting is not in the options it will release a WARNING and the return value will be False.

        :param str_bool: String to convert to boolean.
        :return: bool
        """
        true_options = ['True', 'true', 'T', 't', 'Yes', 'yes', 'Y', 'y', '1', 1, True]
        false_options = ['False', 'false', 'F', 'f', 'No', 'no', 'N', 'n', '0', 0, False, None]

        if str_bool in true_options:
            return True
        elif str_bool in false_options:
            return False
        else:
            print('WARNING: Boolean value not contemplated use {0} for True values and {1} for the False ones'.format(
                true_options, false_options))
            print('/t Using False as default')
            return False

    def _parse_start_date(self, str_date):
        """
        Parse the date form string to datetime.
        It accepts several ways to introduce the date:
            YYYYMMDD, YYYY/MM/DD, YYYYMMDDhh, YYYYYMMDD.hh, YYYY/MM/DD_hh:mm:ss, YYYY-MM-DD_hh:mm:ss,
            YYYY/MM/DD hh:mm:ss, YYYY-MM-DD hh:mm:ss, YYYY/MM/DD_hh, YYYY-MM-DD_hh.

        :param str_date: Date to the day to simulate in string format.
        :type str_date: str, datetime

        :return: Date to the day to simulate in datetime format.
        :rtype: datetime.datetime
        """
        from datetime import datetime

        if self.new_date is not None:
            return self.new_date

        format_types = ['%Y%m%d', '%Y%m%d%H', '%Y%m%d.%H', '%Y/%m/%d_%H:%M:%S', '%Y-%m-%d_%H:%M:%S',
                        '%Y/%m/%d %H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y/%m/%d_%H', '%Y-%m-%d_%H', '%Y/%m/%d']
        date = None
        for date_format in format_types:
            try:
                date = datetime.strptime(str_date, date_format)
                break
            except ValueError as e:
                if str(e) == 'day is out of range for month':
                    raise ValueError(e)

        if date is None:
            raise ValueError("Date format '{0}' not contemplated. Use one of this: {1}".format(str_date, format_types))

        return date

    def _parse_end_date(self, end_date, start_date):
        """
        Parse the end date.
        If it's not defined it will be the same date that start_date (to do only one day).

        :param end_date: Date to the last day to simulate in string format.
        :type end_date: str, datetime

        :param start_date: Date to the first day to simulate.
        :type start_date: datetime.datetime

        :return: Date to the last day to simulate in datetime format.
        :rtype: datetime.datetime
        """
        if end_date is None:
            return start_date
        return self._parse_start_date(end_date)

    @staticmethod
    def _parse_list(str_list):
        import re
        try:
            return list(map(str, re.split(' , |, | ,|,| ; |; | ;|;| ', str_list)))
        except TypeError:
            return None
