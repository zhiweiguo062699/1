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


class Config(ArgParser):
    """
    Initialization of the arguments that the parser can handle.
    """
    def __init__(self):
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
                       choices=['MONARCH', 'CMAQ', 'R-LINE', 'WRF_CHEM', 'DEFAULT'])
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
        p.add_argument('--do_traffic', required=False, type=str, default='True')
        p.add_argument('--do_traffic_area', required=False, type=str, default='True')
        p.add_argument('--do_aviation', required=False, type=str, default='True')
        p.add_argument('--do_point_sources', required=False, type=str, default='True')
        p.add_argument('--do_recreational_boats', required=False, type=str, default='True')
        p.add_argument('--do_shipping_port', required=False, type=str, default='True')
        p.add_argument('--do_residential', required=False, type=str, default='True')
        p.add_argument('--do_livestock', required=False, type=str, default='True')
        p.add_argument('--do_crop_operations', required=False, type=str, default='True')
        p.add_argument('--do_crop_fertilizers', required=False, type=str, default='True')
        p.add_argument('--do_agricultural_machinery', required=False, type=str, default='True')

        p.add_argument('--traffic_processors', required=False, type=int, default='True')
        p.add_argument('--traffic_area_processors', required=False, type=int, default='True')
        p.add_argument('--aviation_processors', required=False, type=int, default='True')
        p.add_argument('--point_sources_processors', required=False, type=int, default='True')
        p.add_argument('--recreational_boats_processors', required=False, type=int, default='True')
        p.add_argument('--shipping_port_processors', required=False, type=int, default='True')
        p.add_argument('--residential_processors', required=False, type=int, default='True')
        p.add_argument('--livestock_processors', required=False, type=int, default='True')
        p.add_argument('--crop_operations_processors', required=False, type=int, default='True')
        p.add_argument('--crop_fertilizers_processors', required=False, type=int, default='True')
        p.add_argument('--agricultural_machinery_processors', required=False, type=int, default='True')

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

        # ***** AVIATION SEECTOR *****
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

        arguments.start_date = self._parse_start_date(arguments.start_date)
        arguments.end_date = self._parse_end_date(arguments.end_date, arguments.start_date)
        arguments.output_name = self.get_output_name(arguments)

        arguments.erase_auxiliary_files = self._parse_bool(arguments.erase_auxiliary_files)
        self.create_dir(arguments.output_dir)

        if arguments.erase_auxiliary_files:
            if os.path.exists(arguments.auxiliary_files_path):
                rmtree(arguments.auxiliary_files_path)
        self.create_dir(arguments.auxiliary_files_path)

        arguments.do_traffic = self._parse_bool(arguments.do_traffic)
        arguments.do_traffic_area = self._parse_bool(arguments.do_traffic_area)
        arguments.do_aviation = self._parse_bool(arguments.do_aviation)
        arguments.do_point_sources = self._parse_bool(arguments.do_point_sources)
        arguments.do_recreational_boats = self._parse_bool(arguments.do_recreational_boats)
        arguments.do_shipping_port = self._parse_bool(arguments.do_shipping_port)
        arguments.do_residential = self._parse_bool(arguments.do_residential)
        arguments.do_livestock = self._parse_bool(arguments.do_livestock)
        arguments.do_crop_operations = self._parse_bool(arguments.do_crop_operations)
        arguments.do_crop_fertilizers = self._parse_bool(arguments.do_crop_fertilizers)
        arguments.do_agricultural_machinery = self._parse_bool(arguments.do_agricultural_machinery)

        arguments.airport_list = self._parse_list(arguments.airport_list)
        arguments.plane_list = self._parse_list(arguments.plane_list)
        arguments.aviation_source_pollutants = self._parse_list(arguments.aviation_source_pollutants)

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
            print 'WARNING: Boolean value not contemplated use {0} for True values and {1} for the False ones'.format(
                true_options, false_options
            )
            print '/t Using False as default'
            return False

    @staticmethod
    def _parse_start_date(str_date):
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
        format_types = ['%Y%m%d', '%Y%m%d%H', '%Y%m%d.%H', '%Y/%m/%d_%H:%M:%S', '%Y-%m-%d_%H:%M:%S',
                        '%Y/%m/%d %H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y/%m/%d_%H', '%Y-%m-%d_%H', '%Y/%m/%d']

        date = None
        for date_format in format_types:
            try:
                date = datetime.strptime(str_date, date_format)
                break
            except ValueError as e:
                if e.message == 'day is out of range for month':
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
