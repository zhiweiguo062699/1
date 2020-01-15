#!/usr/bin/env python

from warnings import warn
from configargparse import ArgParser
import os
from mpi4py import MPI
from hermesv3_bu.tools.checker import error_exit


class Config(ArgParser):
    """
    Configuration arguments class.
    """
    def __init__(self, new_date=None, comm=None):
        """
        Read and parse all the arguments.

        :param new_date: Starting date for simulation loop day.
        :type new_date: datetime.datetime
        """
        if comm is None:
            comm = MPI.COMM_WORLD
        self.comm = comm

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
        p.add_argument('--first_time', required=False, default='False', type=str,
                       help='Indicates if you want to run it for first time (only create auxiliary files).')
        p.add_argument('--erase_auxiliary_files', required=False, default='False', type=str,
                       help='Indicates if you want to start from scratch removing the auxiliary files already created.')

        # ===== DOMAIN =====
        p.add_argument('--output_model', required=True, help='Name of the output model.',
                       choices=['MONARCH', 'CMAQ', 'WRF_CHEM', 'DEFAULT'])
        p.add_argument('--writing_processors', required=False, type=int,
                       help='Number of processors dedicated to write. ' +
                            'Maximum number accepted is the number of rows of the destiny grid.')
        p.add_argument('--output_attributes', required=False,
                       help='Path to the file that contains the global attributes.')

        p.add_argument('--vertical_description', required=True,
                       help='Path to the file that contains the vertical description of the desired output.')

        p.add_argument('--domain_type', required=True, help='Type of domain to simulate.',
                       choices=['lcc', 'rotated', 'mercator', 'regular', 'rotated_nested'])

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

        # Rotated_nested options
        p.add_argument('--parent_grid_path', required=False, type=str,
                       help='Path to the netCDF that contains the grid definition.')
        p.add_argument('--parent_ratio', required=False, type=int,
                       help='Ratio between the parent and the nested domain.')
        p.add_argument('--i_parent_start', required=False, type=int,
                       help='Location of the I to start the nested.')
        p.add_argument('--j_parent_start', required=False, type=int,
                       help='Location of the J to start the nested.')
        p.add_argument('--n_rlat', required=False, type=int,
                       help='Number of rotated latitude points.')
        p.add_argument('--n_rlon', required=False, type=int,
                       help='Number of rotated longitude points.')

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
        p.add_argument('--lat_ts', required=False, type=float, help='Latitude of true scale (degrees).')

        # Regular lat-lon options:
        p.add_argument('--lat_orig', required=False, type=float, help='Latitude of the corner of the first cell.')
        p.add_argument('--lon_orig', required=False, type=float, help='Longitude of the corner of the first cell.')
        p.add_argument('--n_lat', required=False, type=int, help='Number of latitude elements.')
        p.add_argument('--n_lon', required=False, type=int, help='Number of longitude elements.')
        p.add_argument('--inc_lat', required=False, type=float, help='Latitude grid resolution.')
        p.add_argument('--inc_lon', required=False, type=float, help='Longitude grid resolution.')

        # ===== SECTOR SELECTION =====
        p.add_argument('--traffic_processors', required=False, type=int, default=0,
                       help="Number of processors dedicated to simulate the road traffic sector " +
                            "(0 to deactivated the sector).")
        p.add_argument('--traffic_area_processors', required=False, type=int, default=0,
                       help="Number of processors dedicated to simulate the traffic area " +
                            "(evaporative and small cities) sector (0 to deactivated the sector).")
        p.add_argument('--aviation_processors', required=False, type=int, default=0,
                       help="Number of processors dedicated to simulate the livestock sector " +
                            "(0 to deactivated the sector).")
        p.add_argument('--point_sources_processors', required=False, type=int, default=0,
                       help="Number of processors dedicated to simulate the point sources sector " +
                            "(0 to deactivated the sector).")
        p.add_argument('--recreational_boats_processors', required=False, type=int, default=0,
                       help="Number of processors dedicated to simulate the recreational boats sector " +
                            "(0 to deactivated the sector).")
        p.add_argument('--shipping_port_processors', required=False, type=int, default=0,
                       help="Number of processors dedicated to simulate the shipping port sector " +
                            "(0 to deactivated the sector).")
        p.add_argument('--residential_processors', required=False, type=int, default=0,
                       help="Number of processors dedicated to simulate the residential combustion sector " +
                            "(0 to deactivated the sector).")
        p.add_argument('--livestock_processors', required=False, type=int, default=0,
                       help="Number of processors dedicated to simulate the livestock sector " +
                            "(0 to deactivated the sector).")
        p.add_argument('--crop_operations_processors', required=False, type=int, default=0,
                       help="Number of processors dedicated to simulate the agricultural crop operations sector " +
                            "(0 to deactivated the sector).")
        p.add_argument('--crop_fertilizers_processors', required=False, type=int, default=0,
                       help="Number of processors dedicated to simulate the agricultural crop fertilizers sector " +
                            "(0 to deactivated the sector).")
        p.add_argument('--agricultural_machinery_processors', required=False, type=int, default=0,
                       help="Number of processors dedicated to simulate the agricultural machinery sector " +
                            "(0 to deactivated the sector).")
        p.add_argument('--solvents_processors', required=False, type=int, default=0,
                       help="Number of processors dedicated to simulate the solvents sector " +
                            "(0 to deactivated the sector).")

        p.add_argument('--speciation_map', required=False,
                       help="Defines the path to the file that contains the mapping between input and output " +
                            "pollutant species")
        p.add_argument('--molecular_weights', required=True,
                       help='Path to the file that contains the molecular weights of the input pollutants.')

        # ===== SHAPEFILES =====
        p.add_argument('--nuts3_shapefile', required=False, type=str, default='True',
                       help="Defines the path to the shapefile with the NUTS2 administrative boundaries. Used in " +
                            "livestock, agricultural machinery, residential combustion and traffic area sector.")
        p.add_argument('--nuts2_shapefile', required=False, type=str, default='True',
                       help="Defines the path to the shapefile with the NUTS3 administrative boundaries. Used in " +
                            "agricultural crop operations, agricultural crop fertilizers, agricultural machinery, " +
                            "residential combustion and solvents sector.")
        p.add_argument('--population_density_map', required=False,
                       help="Defines the path to the GHS population density raster file. Used in residential " +
                            "combustion, traffic area and solvents sectors.")
        p.add_argument('--population_type_map', required=False,
                       help="Defines the path to the GHS population type raster file.")
        p.add_argument('--population_type_nuts2', required=False,
                       help="Defines the path to the CSV file that contains the total amount of urban and rural " +
                            "population registered at NUTS2 level (based on the GHS dataset).")
        p.add_argument('--population_type_nuts3', required=False,
                       help="Defines the path to the CSV file that contains the total amount of urban and rural " +
                            "population registered at NUTS3 level (based on the GHS dataset).")
        p.add_argument('--population_nuts2', required=False, type=str, default='True',
                       help="Defines the path to the CSV file that contains the total amount of population " +
                            "registered at NUTS2 level")
        p.add_argument('--land_uses_path', required=False,
                       help='Defines the path to the CORINE Land Cover land use raster file')
        p.add_argument('--land_uses_nuts2_path', required=False,
                       help="Defines the path to the CSV file that contains the total amount of each CLC " +
                            "land use area by NUTS2")

        p.add_argument('--clipping', required=False, type=str, default=None,
                       help="To clip the domain into an specific zone. It can be a shapefile path, a list of points " +
                            "to make a polygon or nothing to use the default clip: domain extension")

        # ===== METEO PATHS =====
        p.add_argument('--temperature_hourly_files_path', required=False, type=str, default='True',
                       help="Defines the path to the NetCDF files containing hourly mean 2m temperature data.")
        p.add_argument('--temperature_daily_files_path', required=False, type=str, default='True',
                       help="Defines the path to the NetCDF files containing daily mean 2m temperature data.")
        p.add_argument('--wind_speed_daily_files_path', required=False, type=str, default='True',
                       help="Defines the path to the NetCDF files containing daily mean 10m wind speed data.")
        p.add_argument('--precipitation_files_path', required=False, type=str, default='True',
                       help="Defines the path to the NetCDF files containing hourly mean precipitation data.")
        p.add_argument('--temperature_4d_dir', required=False, type=str, default='True',
                       help="Defines the path to the NetCDF files containing hourly mean 4D temperature data.")
        p.add_argument('--temperature_sfc_dir', required=False, type=str, default='True',
                       help="Defines the path to the NetCDF files containing hourly mean surface temperature data.")
        p.add_argument('--u_wind_speed_4d_dir', required=False, type=str, default='True',
                       help="Defines the path to the NetCDF files containing hourly mean 4D U wind component data.")
        p.add_argument('--v_wind_speed_4d_dir', required=False, type=str, default='True',
                       help="Defines the path to the NetCDF files containing hourly mean 4D V wind component data.")
        p.add_argument('--u10_wind_speed_dir', required=False, type=str, default='True',
                       help="Defines the path to the NetCDF files containing daily mean 10m U wind component data.")
        p.add_argument('--v10_wind_speed_dir', required=False, type=str, default='True',
                       help="Defines the path to the NetCDF files containing daily mean 10m V wind component data.")
        p.add_argument('--friction_velocity_dir', required=False, type=str, default='True',
                       help="Defines the path to the NetCDF files containing hourly mean 4D friction velocity data.")
        p.add_argument('--pblh_dir', required=False, type=str, default='True',
                       help="Defines the path to the NetCDF files containing hourly mean PBL height data.")
        p.add_argument('--obukhov_length_dir', required=False, type=str, default='True',
                       help="Defines the path to the NetCDF files containing hourly mean Obukhov length data.")
        p.add_argument('--layer_thickness_dir', required=False, type=str, default='True',
                       help="Defines the path to the NetCDF files containing hourly mean 4D layer thickness data.")

        # ***** AVIATION SECTOR *****
        p.add_argument('--aviation_source_pollutants', required=False,
                       help="List of pollutants considered for the calculation of the aviation sector.")
        p.add_argument('--airport_list', required=False,
                       help="Defines the list of airport codes to be considered for the calculation of the sector. " +
                            "By default, all the airports located within the working domain will be considered.")
        p.add_argument('--plane_list', required=False,
                       help="List of plane categories to be considered for the calculation of the sector. " +
                            "By default, all the plane categories are considered.")
        p.add_argument('--airport_shapefile_path', required=False,
                       help="Defines the path to the polygon shapefile with the airport infrastructure boundaries.")
        p.add_argument('--airport_runways_shapefile_path', required=False,
                       help="Defines the path to the polyline shapefile with the airport runways.")
        p.add_argument('--airport_runways_corners_shapefile_path', required=False,
                       help="Defines the path to the multipoint shapefile with the airport runway’s corners.")
        p.add_argument('--airport_trajectories_shapefile_path', required=False,
                       help="Defines the path to the polyline shapefile with the airport’s air trajectories.")
        p.add_argument('--airport_operations_path', required=False,
                       help="Defines the path to the CSV file that contains the number of monthly operations " +
                            "(arrival, departure) per airport and plane.")
        p.add_argument('--planes_path', required=False,
                       help="Defines the path to the CSV file that contains the description of the planes.")
        p.add_argument('--airport_times_path', required=False,
                       help="Defines the path to the CSV file that contains the times associates to each LTO phase " +
                            "per airport and plane.")
        p.add_argument('--airport_ef_dir', required=False,
                       help="Defines the path to the CSV files that contain the emission factors for each plane and " +
                            "LTO phase.")
        p.add_argument('--aviation_weekly_profiles', required=False,
                       help="Defines the path to the CSV file that contains the weekly temporal profiles per airport.")
        p.add_argument('--aviation_hourly_profiles', required=False,
                       help="Defines the path to the CSV file that contains the hourly temporal profiles per airport.")
        p.add_argument('--aviation_speciation_profiles', required=False,
                       help="Defines the path to the CSV file that contains the speciation profiles.")

        # ***** SHIPPING PORT SECTOR *****
        p.add_argument('--shipping_port_source_pollutants', required=False,
                       help="List of pollutants considered for the calculation of the shipping port sector.")
        p.add_argument('--vessel_list', required=False,
                       help="Defines the list of vessel categories to be considered for the emission calculation.")
        p.add_argument('--port_list', required=False,
                       help="Defines the list of ports to be considered for the emission calculation. ")
        p.add_argument('--hoteling_shapefile_path', required=False,
                       help="Defines the path to the multipolygon shapefile with the hotelling areas.")
        p.add_argument('--maneuvering_shapefile_path', required=False,
                       help="Defines the path to the multipolygon shapefile with the maneuvering areas.")
        p.add_argument('--shipping_port_ef_path', required=False,
                       help="Defines the path to the CSV file that contains the emission factors for each main and " +
                            "auxiliary engine class and fuel type.")
        p.add_argument('--shipping_port_engine_percent_path', required=False,
                       help="Defines the path to the CSV file that contains the engine class and fuel type split " +
                            "factors for each vessel category.")
        p.add_argument('--shipping_port_tonnage_path', required=False,
                       help="Defines the path to the CSV file that contains the number of annual operations and mean " +
                            "Gross Tonnage value per port and vessel category.")
        p.add_argument('--shipping_port_load_factor_path', required=False,
                       help="Defines the path to the CSV file that contains the average load factor and time spent " +
                            "for each vessel, engine and operation.")
        p.add_argument('--shipping_port_power_path', required=False,
                       help="Defines the path to the CSV file that contains the parameters for the main engine power " +
                            "calculation as a function of the Gross Tonnage and the average vessel's ratio of " +
                            "auxiliary engines/main engines.")
        p.add_argument('--shipping_port_monthly_profiles', required=False,
                       help="Defines the path to the CSV file that contains the monthly temporal profiles per port " +
                            "and vessel category.")
        p.add_argument('--shipping_port_weekly_profiles', required=False,
                       help="Defines the path to the CSV file that contains the weekly temporal profiles.")
        p.add_argument('--shipping_port_hourly_profiles', required=False,
                       help="Defines the path to the CSV file that contains the hourly temporal profiles per airport.")
        p.add_argument('--shipping_port_speciation_profiles', required=False,
                       help="Defines the path to the CSV file that contains the speciation profiles.")

        # ***** LIVESTOCK SECTOR *****
        p.add_argument('--livestock_source_pollutants', required=False,
                       help="List of pollutants considered for the calculation of the livestock sector.")
        p.add_argument('--animal_list', required=False,
                       help="Defines the list of livestock categories [cattle, chicken, goats, pigs or sheep] to be " +
                            "considered for the emission calculation.")
        p.add_argument('--gridded_livestock', required=False,
                       help="Defines the path to the GLWv3 livestock population density raster files. The string " +
                            "<animal> is automatically replaced by the different livestock categories considered " +
                            "for the calculation.")
        p.add_argument('--correction_split_factors', required=False,
                       help="Defines the path to the CSV file that contains the livestock subgroup split factors " +
                            "and adjusting factors to match the official statistics provided at the NUTS3 level. " +
                            "The string  is automatically replaced by the different livestock categories considered " +
                            "for the calculation.")
        p.add_argument('--denominator_yearly_factor_dir', required=False,
                       help="Define the path to the NetCDF file that contains the yearly average daily factor per " +
                            "grid cell.")
        p.add_argument('--livestock_ef_files_dir', required=False,
                       help="Defines the  path to the CSV files that contain the emission factors for each pollutant." +
                            " Each pollutant has its own emission factor file format.")
        p.add_argument('--livestock_monthly_profiles', required=False,
                       help="Defines the path to the CSV file that contains the monthly temporal profiles")
        p.add_argument('--livestock_weekly_profiles', required=False,
                       help="Defines the path to the CSV file that contains the weekly temporal profiles")
        p.add_argument('--livestock_hourly_profiles', required=False,
                       help="Defines the path to the CSV file that contains the hourly temporal profiles")
        p.add_argument('--livestock_speciation_profiles', required=False,
                       help="Defines the path to the CSV file that contains the speciation profiles")

        # ***** AGRICULTURAL SECTOR*****
        p.add_argument('--crop_by_nut_path', required=False,
                       help="Defines the path to the CSV file that contains the annual cultivated area of each crop " +
                            "by NUTS2")
        p.add_argument('--crop_from_landuse_path', required=False,
                       help="Defines the path to the CSV file that contains the mapping between CLC land use " +
                            "categories and crop categories")

        # ***** CROP OPERATIONS SECTOR
        p.add_argument('--crop_operations_source_pollutants', required=False,
                       help="List of pollutants considered for the calculation of the agricultural crop operations " +
                            "sector.")
        p.add_argument('--crop_operations_list', required=False,
                       help="List of crop categories considered for the calculation of the sector " +
                            "[wheat, rye, barley, oats].")
        p.add_argument('--crop_operations_ef_files_dir', required=False,
                       help="Defines the path to the CSV file that contains the emission factors for each crop " +
                            "operations and crop type.")
        p.add_argument('--crop_operations_monthly_profiles', required=False,
                       help="Defines the path to the CSV file that contains the monthly temporal profiles.")
        p.add_argument('--crop_operations_weekly_profiles', required=False,
                       help="Defines the path to the CSV file that contains the weekly temporal profiles.")
        p.add_argument('--crop_operations_hourly_profiles', required=False,
                       help="Defines the path to the CSV file that contains the hourly temporal profiles.")
        p.add_argument('--crop_operations_speciation_profiles', required=False,
                       help="Defines the path to the CSV file that contains the speciation profiles")

        # ***** CROP FERTILIZERS SECTOR *****
        p.add_argument('--crop_fertilizers_source_pollutants', required=False,
                       help="List of pollutants considered for the calculation of the agricultural crop fertilizers " +
                            "sector.")
        p.add_argument('--crop_fertilizers_list', required=False,
                       help="List of crop categories considered for the calculation of the sector [alfalfa, almond, " +
                            "apple, apricot, barley, cherry, cotton, fig, grape, lemonlime, maize, melonetc, oats, " +
                            "olive, orange, pea, peachetc, pear, potato, rice, rye, sunflower, tangetc, tomato, " +
                            "triticale, vetch, watermelon, wheat].")
        p.add_argument('--cultivated_ratio', required=False,
                       help="Defines the path to the CSV file that contains the ration of cultivated to fertilised " +
                            "area for crop category.")
        p.add_argument('--fertilizers_rate', required=False,
                       help="Defines the path to the CSV file that contains the fertilizer application rate for " +
                            "crop category and NUTS2.")
        p.add_argument('--crop_f_parameter', required=False,
                       help="Defines the path to the CSV file that contains: (i) the parameters for the calculation " +
                            "of the NH3 emission factor according to Bouwman and Boumans (2002) and (ii) the " +
                            "fraction of each fertilizer type used by NUTS2.")
        p.add_argument('--crop_f_fertilizers', required=False,
                       help="Defines the path to the CSV file that contains the fertilizer type-related parameters " +
                            "for the calculation of the NH3 emission factor according to Bouwman and Boumans (2002).")
        p.add_argument('--gridded_ph', required=False,
                       help="Defines the path to the ISRIC pH soil raster file.")
        p.add_argument('--gridded_cec', required=False,
                       help="Defines the path to the ISRIC CEC soil raster file.")
        p.add_argument('--fertilizers_denominator_yearly_factor_path', required=False,
                       help="Define the path to the NetCDF file that contains the yearly average daily factor per " +
                            "grid cell.")
        p.add_argument('--crop_calendar', required=False,
                       help="Defines the path to the CSV file that contains the parameters needed to define the " +
                            "timing of the fertilizer application per crop category.")
        p.add_argument('--crop_fertilizers_hourly_profiles', required=False,
                       help="Defines the path to the CSV file that contains the hourly temporal profiles.")
        p.add_argument('--crop_fertilizers_speciation_profiles', required=False,
                       help="Defines the path to the CSV file that contains the speciation profiles.")
        p.add_argument('--crop_growing_degree_day_path', required=False,
                       help="Define the path to the NetCDF file that contains the growing degree day valueper " +
                            "grid cell. The string <season> is automatically replaced for \"winter\" or \"spring\" " +
                            "as a function of the crop_calendar file. The string  <year> is automatically replaced " +
                            "for the year of simulation.")

        # ***** AGRICULTURAL MACHINERY SECTOR *****
        p.add_argument('--crop_machinery_source_pollutants', required=False,
                       help="List of pollutants considered for the calculation of the agricultural machinery sector.")
        p.add_argument('--crop_machinery_list', required=False,
                       help="List of crop categories considered for the calculation of the sector " +
                            "[barley, oats, rye, wheat].")
        p.add_argument('--machinery_list', required=False,
                       help="List of agricultural equipment categories considered for the calculation of the sector " +
                            "[tractors, harvesters, rotavators].")
        p.add_argument('--crop_machinery_deterioration_factor_path', required=False,
                       help="Defines the path to the CSV file that contains the deterioration factors per equipment " +
                            "category and pollutant.")
        p.add_argument('--crop_machinery_load_factor_path', required=False,
                       help="Defines the path to the CSV file that contains the load factors per equipment category.")
        p.add_argument('--crop_machinery_vehicle_ratio_path', required=False,
                       help="Defines the path to the CSV file that contains the equipment subgroup split factors by " +
                            "technology and NUTS3.")
        p.add_argument('--crop_machinery_vehicle_units_path', required=False,
                       help="Defines the path to the CSV file that contains the total amount of equipment by " +
                            "category and NUTS3.")
        p.add_argument('--crop_machinery_vehicle_workhours_path', required=False,
                       help="Defines the path to the CSV file that contains the number of hours that each equipment " +
                            "subgroup is used by NUTS3.")
        p.add_argument('--crop_machinery_vehicle_power_path', required=False,
                       help="Defines the path to the CSV file that contains the engine nominal power associated to " +
                            "each equipment category by NUTS3.")
        p.add_argument('--crop_machinery_ef_path', required=False,
                       help="Defines the path to the CSV file that contains the emission factors for each " +
                            "equipment subgroup.")
        p.add_argument('--crop_machinery_monthly_profiles', required=False,
                       help="Defines the path to the CSV file that contains the monthly temporal profiles.")
        p.add_argument('--crop_machinery_weekly_profiles', required=False,
                       help="Defines the path to the CSV file that contains the weekly temporal profiles.")
        p.add_argument('--crop_machinery_hourly_profiles', required=False,
                       help="Defines the path to the CSV file that contains the hourly temporal profiles.")
        p.add_argument('--crop_machinery_speciation_profiles', required=False,
                       help="Defines the path to the CSV file that contains the speciation profiles.")
        p.add_argument('--crop_machinery_nuts3', required=False,
                       help="Defines the path to the CSV file that contains the annual cultivated area of each crop " +
                            "by NUTS3")

        # ***** RESIDENTIAL SECTOR *****
        p.add_argument('--fuel_list', required=False,
                       help="List of fuel types considered for the calculation of the sector. (HD = heating diesel, " +
                            "LPG = liquefied petroleum gas, NG = natural gas; B = biomass, res = residential, " +
                            "com = commercial).")
        p.add_argument('--residential_source_pollutants', required=False,
                       help="List of pollutants considered for the calculation of the residential/commercial sector.")
        p.add_argument('--energy_consumption_nuts3', required=False,
                       help="Defines the path to the CSV file that contains the annual amount of energy consumed " +
                            "per fuel type and NUTS3.")
        p.add_argument('--energy_consumption_nuts2', required=False,
                       help="Defines the path to the CSV file that contains the annual amount of energy consumed " +
                            "per fuel type and NUTS2.")
        p.add_argument('--residential_spatial_proxies', required=False,
                       help="Defines the path to the CSV file that contains the type of population (urban, rural) " +
                            "assigned to each fuel for its spatial mapping.")
        p.add_argument('--residential_ef_files_path', required=False,
                       help="Defines the path to the CSV file that contains the emission factors for each fuel type.")
        p.add_argument('--residential_heating_degree_day_path', required=False,
                       help="Define the path to the NetCDF file that contains the yearly average HDD factor per " +
                            "grid cell.")
        p.add_argument('--residential_hourly_profiles', required=False,
                       help="Defines the path to the CSV file that contains the hourly temporal profiles per " +
                            "fuel type.")
        p.add_argument('--residential_speciation_profiles', required=False,
                       help="Defines the path to the CSV file that contains the speciation profiles.")

        # ***** RECREATIONAL BOATS SECTOR *****
        p.add_argument('--recreational_boats_source_pollutants', required=False,
                       help="List of pollutants considered for the calculation of the recreational boat sector.")
        p.add_argument('--recreational_boats_list', required=False,
                       help="List of recreational boat category codes considered for the calculation of the sector " +
                            "[YB_001,YB_002,SB_001,SB_002,SP_001,SP_002,OB_001,OB_002,WS_001,WS_002,YB_003,SB_003," +
                            "SP_004,SP_005,OB_002,WS_003,MB_001,MB_002,MB_003,MB_004,MB_005,MB_006,MS_001,MS_002," +
                            "SB_004,SB_005]. A description of each category code is available here.")
        p.add_argument('--recreational_boats_density_map', required=False,
                       help="Defines the path to the raster file used for performing the spatial distribution of " +
                            "the recreational boats.")
        p.add_argument('--recreational_boats_by_type', required=False,
                       help="Defines the path to the CSV file that contains the number of recreational boats per " +
                            "category and associated information (load factor, annual working hours, nominal engine " +
                            "power).")
        p.add_argument('--recreational_boats_ef_path', required=False,
                       help="Defines the path to the CSV file that contains the emission factors for each " +
                            "recreational boat category.")
        p.add_argument('--recreational_boats_monthly_profiles', required=False,
                       help="Defines the path to the CSV file that contains the monthly temporal profiles.")
        p.add_argument('--recreational_boats_weekly_profiles', required=False,
                       help="Defines the path to the CSV file that contains the weekly temporal profiles.")
        p.add_argument('--recreational_boats_hourly_profiles', required=False,
                       help="Defines the path to the CSV file that contains the hourly temporal profiles.")
        p.add_argument('--recreational_boats_speciation_profiles', required=False,
                       help="Defines the path to the CSV file that contains the speciation profiles.")

        # ***** POINT SOURCE SECTOR *****
        p.add_argument('--point_source_pollutants', required=False,
                       help="List of pollutants considered for the calculation of the point sources sector.")
        p.add_argument('--plume_rise', required=False,
                       help="Boolean that defines if the plume rise algorithm is activated or not.")
        p.add_argument('--point_source_snaps', required=False,
                       help="Defines the SNAP source categories considered during the emission calculation " +
                            "[01, 03, 04, 09].")
        p.add_argument('--point_source_catalog', required=False,
                       help="Defines the path to the CSV file that contains the description of each point source " +
                            "needed for the emission calculation (ID code, geographical location, activity and " +
                            "emission factors, physical stack parameters, temporal and speciation profile IDs)")
        p.add_argument('--point_source_monthly_profiles', required=False,
                       help="Defines the path to the CSV file that contains the monthly temporal profiles.")
        p.add_argument('--point_source_weekly_profiles', required=False,
                       help="Defines the path to the CSV file that contains the weekly temporal profiles.")
        p.add_argument('--point_source_hourly_profiles', required=False,
                       help="Defines the path to the CSV file that contains the hourly temporal profiles.")
        p.add_argument('--point_source_speciation_profiles', required=False,
                       help="Defines the path to the CSV file that contains the speciation profiles.")
        p.add_argument('--point_source_measured_emissions', required=False,
                       help="Defines the path to the CSV file that contains hourly measured emissions for a specific " +
                            "point source. The string <Code> is automatically replaced by the point source facility " +
                            "which activity factor in the “point_source_catalog” file is assigned with a “-1” value.")

        # ***** TRAFFIC SECTOR *****
        p.add_argument('--do_hot', required=False,
                       help="Boolean to define if the exhaust hot emissions are considered (1) or dismissed (0) " +
                            "during the calculation process.")
        p.add_argument('--do_cold', required=False,
                       help="Boolean to define if the exhaust cold-start emissions are considered (1) or dismissed " +
                            "(0) during the calculation process.")
        p.add_argument('--do_tyre_wear', required=False,
                       help="Boolean to define if the tyre wear emissions are considered (1) or dismissed (0) " +
                            "during the calculation process.")
        p.add_argument('--do_brake_wear', required=False,
                       help="Boolean to define if the brake wear emissions are considered (1) or dismissed (0) " +
                            "during the calculation process.")
        p.add_argument('--do_road_wear', required=False,
                       help="Boolean to define if the road wear emissions are considered (1) or dismissed (0) " +
                            "during the calculation process.")
        p.add_argument('--do_resuspension', required=False,
                       help="Boolean to define if the resuspension emissions are considered (1) or dismissed (0) " +
                            "during the calculation process.")
        p.add_argument('--resuspension_correction', required=False,
                       help="Boolean to define if the effect of precipitation on resuspension emissions is " +
                            "considered (1) or dismissed (0) during the calculation process.")
        p.add_argument('--write_rline', required=False,
                       help="Boolean to define if the emission output is written following the conventions and " +
                            "requirements of the R-LINE model. If the R-LINE option is activated, the user need to " +
                            "provide the R-LINE road link shapefile.")

        p.add_argument('--traffic_pollutants', required=False,
                       help="List of pollutants considered for the calculation of the traffic sector.")
        p.add_argument('--vehicle_types', required=False,
                       help="Defines the list of vehicle categories to be considered for the emission calculation.")
        p.add_argument('--load', type=float, required=False,
                       help="Defines the load percentage correction applicable to heavy duty vehicles and buses " +
                            "[0.0, 0.5 or 1.0].")
        p.add_argument('--road_link_path', required=False,
                       help="Defines the path to the shapefile with the road network and associated traffic flow " +
                            "information.")
        p.add_argument('--fleet_compo_path', required=False,
                       help="Defines the path to the CSV file that contains the vehicle fleet composition profiles.")
        p.add_argument('--traffic_ef_path', required=False,
                       help="Defines the path to the CSV files that contain the emission factors. Emission factor " +
                            "CSV files need to be provided separately for each source and pollutant.")
        p.add_argument('--traffic_speed_hourly_path', required=False,
                       help="Defines the path to the CSV files that contain the hourly temporal profiles for the " +
                            "average speed data.")
        p.add_argument('--traffic_monthly_profiles', required=False,
                       help="Defines the path to the CSV file that contains the monthly temporal profiles.")
        p.add_argument('--traffic_weekly_profiles', required=False,
                       help="Defines the path to the CSV file that contains the weekly temporal profiles.")
        p.add_argument('--traffic_hourly_profiles_mean', required=False,
                       help="Defines the path to the CSV file that contains the hourly profiles file.")
        p.add_argument('--traffic_hourly_profiles_weekday', required=False,
                       help="Defines the path to the CSV file that contains the weekday hourly temporal profiles.")
        p.add_argument('--traffic_hourly_profiles_saturday', required=False,
                       help="Defines the path to the CSV file that contains the Saturday-type hourly temporal " +
                            "profiles.")
        p.add_argument('--traffic_hourly_profiles_sunday', required=False,
                       help="Defines the path to the CSV file that contains the Sunday-type hourly temporal profiles.")
        p.add_argument('--traffic_speciation_profile_hot_cold', required=False,
                       help="Defines the path to the CSV file that contains the speciation profiles for the hot " +
                            "and cold-start emissions.")
        p.add_argument('--traffic_speciation_profile_tyre', required=False,
                       help="Defines the path to the CSV file that contains the speciation profiles for the tyre " +
                            "wear emissions.")
        p.add_argument('--traffic_speciation_profile_road', required=False,
                       help="Defines the path to the CSV file that contains the speciation profiles for the " +
                            "road wear emissions.")
        p.add_argument('--traffic_speciation_profile_brake', required=False,
                       help="Defines the path to the CSV file that contains the speciation profiles for the " +
                            "brake wear emissions.")
        p.add_argument('--traffic_speciation_profile_resuspension', required=False,
                       help="Defines the path to the CSV file that contains the speciation profiles for the " +
                            "resuspension emissions.")

        # ***** TRAFFIC AREA SECTOR *****
        p.add_argument('--traffic_area_pollutants', required=False,
                       help="List of pollutants considered for the calculation of the traffic area sector.")
        p.add_argument('--do_evaporative', required=False,
                       help="Boolean to define if the gasoline evaporative emissions are considered (1) or " +
                            "dismissed (0) during the calculation process.")
        p.add_argument('--traffic_area_gas_path', required=False,
                       help="Defines the path to the CSV file that contains the total amount of gasoline vehicles " +
                            "registered per vehicle category and NUTS3.")
        p.add_argument('--population_nuts3', required=False,
                       help="Defines the path to the CSV file that contains the total amount of urban and rural " +
                            "population registered at NUTS3 level.")
        p.add_argument('--traffic_area_speciation_profiles_evaporative', required=False,
                       help="Defines the path to the CSV file that contains the speciation profiles.")
        p.add_argument('--traffic_area_evaporative_ef_file', required=False,
                       help="Defines the path to the CSV file that contains the emission factors for each vehicle " +
                            "category and range of temperatures.")

        p.add_argument('--do_small_cities', required=False,
                       help="Boolean to define if the small city emissions are considered (1) or dismissed (0) " +
                            "during the calculation process.")
        p.add_argument('--traffic_area_small_cities_path', required=False,
                       help="Defines the path to the multipolygon shapefile with the small cities.")
        p.add_argument('--traffic_area_small_cities_ef_file', required=False,
                       help="Defines the path to the CSV file that contains the emission factors for the small cities.")
        p.add_argument('--small_cities_monthly_profile', required=False,
                       help="Defines the path to the CSV file that contains the monthly temporal profiles for the " +
                            "small cities.")
        p.add_argument('--small_cities_weekly_profile', required=False,
                       help="Defines the path to the CSV file that contains the weekly temporal profiles for the " +
                            "small cities.")
        p.add_argument('--small_cities_hourly_profile', required=False,
                       help="Defines the path to the CSV file that contains the hourly temporal profiles for the " +
                            "small cities.")
        p.add_argument('--traffic_area_speciation_profiles_small_cities', required=False,
                       help="Defines the path to the CSV file that contains the speciation profiles.")

        # ***** SOLVENTS SECTOR *****
        p.add_argument('--solvents_pollutants', required=False,
                       help="List of pollutants considered for the calculation of the solvents sector. " +
                            "Only 'nmvoc' is available.")
        # TODO add description for solvents sector
        p.add_argument('--solvents_proxies_path', required=False,
                       help="")
        p.add_argument('--solvents_yearly_emissions_by_nut2_path', required=False,
                       help="")
        p.add_argument('--solvents_point_sources_shapefile', required=False,
                       help="")
        p.add_argument('--solvents_point_sources_weight_by_nut2_path', required=False,
                       help="")
        p.add_argument('--solvents_monthly_profile', required=False,
                       help="Defines the path to the CSV file that contains the monthly temporal profiles.")
        p.add_argument('--solvents_weekly_profile', required=False,
                       help="Defines the path to the CSV file that contains the weekly temporal profiles.")
        p.add_argument('--solvents_hourly_profile', required=False,
                       help="Defines the path to the CSV file that contains the hourly temporal profiles.")
        p.add_argument('--solvents_speciation_profiles', required=False,
                       help="Defines the path to the CSV file that contains the speciation profiles.")

        arguments, unknown = p.parse_known_args()
        if len(unknown) > 0:
            warn("Unrecognized arguments: {0}".format(unknown))

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
                elif arguments.domain_type == 'rotated_nested':
                    arguments.__dict__[item] = arguments.__dict__[item].replace('<resolution>', '{1}_{2}'.format(
                        item, arguments.n_rlat, arguments.n_rlon))
                elif arguments.domain_type == 'lcc' or arguments.domain_type == 'mercator':
                    arguments.__dict__[item] = arguments.__dict__[item].replace('<resolution>', '{1}_{2}'.format(
                        item, arguments.inc_x, arguments.inc_y))

        arguments.emission_summary = self._parse_bool(arguments.emission_summary)
        arguments.start_date = self._parse_start_date(arguments.start_date, self.new_date)
        arguments.end_date = self._parse_end_date(arguments.end_date, arguments.start_date)
        arguments.output_name = self.get_output_name(arguments)

        arguments.first_time = self._parse_bool(arguments.first_time)
        arguments.erase_auxiliary_files = self._parse_bool(arguments.erase_auxiliary_files)
        self.create_dir(arguments.output_dir)

        if arguments.erase_auxiliary_files:
            if os.path.exists(arguments.auxiliary_files_path):
                if self.comm.Get_rank() == 0:
                    rmtree(arguments.auxiliary_files_path)
                self.comm.Barrier()
        self.create_dir(arguments.auxiliary_files_path)

        # Booleans
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
        arguments.do_solvents = arguments.solvents_processors > 0

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

        # Solvents lists
        arguments.solvents_pollutants = self._parse_list(arguments.solvents_pollutants)

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

    def create_dir(self, path):
        """
        Create the given folder if it is not created yet.

        :param path: Path to create.
        :type path: str
        """
        import os
        from mpi4py import MPI
        comm = self.comm.Split(color=0, key=0)
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

    def _parse_start_date(self, str_date, new_date=None):
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

        if new_date is not None:
            return new_date

        format_types = ['%Y%m%d', '%Y%m%d%H', '%Y%m%d.%H', '%Y/%m/%d_%H:%M:%S', '%Y-%m-%d_%H:%M:%S',
                        '%Y/%m/%d %H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y/%m/%d_%H', '%Y-%m-%d_%H', '%Y/%m/%d']
        date = None
        for date_format in format_types:
            try:
                date = datetime.strptime(str_date, date_format)
                break
            except ValueError as e:
                if str(e) == 'day is out of range for month':
                    error_exit(e)

        if date is None:
            error_exit("Date format '{0}' not contemplated. Use one of this: {1}".format(str_date, format_types))

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
