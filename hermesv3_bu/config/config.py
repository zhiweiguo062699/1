#!/usr/bin/env python

# Copyright 2018 Earth Sciences Department, BSC-CNS
#
# This file is part of HERMESv3_GR.
#
# HERMESv3_GR is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HERMESv3_GR is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HERMESv3_GR. If not, see <http://www.gnu.org/licenses/>.


from configargparse import ArgParser


class Config(ArgParser):
    """
    Initialization of the arguments that the parser can handle.
    """
    def __init__(self):
        super(Config, self).__init__()
        self.options = self.read_options()

    def read_options(self):
        """
        Reads all the options from command line or from the configuration file.
        The value of an argument given by command line has high priority that the one that appear in the
        configuration file.

        :return: Arguments already parsed.
        :rtype: Namespace
        """
        # p = ArgParser(default_config_files=['/home/Earth/mguevara/HERMES/HERMESv3/IN/conf/hermes.conf'])
        p = ArgParser()
        p.add_argument('-c', '--my-config', required=False, is_config_file=True, help='Path to the configuration file.')
        # TODO Detallar mas que significan 1, 2  y 3 los log_level
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

        p.add_argument('--output_timestep_type', required=True, help='Type of timestep.',
                       type=str, choices=['hourly', 'daily', 'monthly', 'yearly'])
        p.add_argument('--output_timestep_num', required=True, help='Number of timesteps to simulate.', type=int)
        p.add_argument('--output_timestep_freq', required=True, help='Frequency between timesteps.', type=int)

        p.add_argument('--output_model', required=True, help='Name of the output model.',
                       choices=['MONARCH', 'CMAQ', 'WRF_CHEM'])
        p.add_argument('--output_attributes', required=False,
                       help='Path to the file that contains the global attributes.')

        p.add_argument('--domain_type', required=True, help='Type of domain to simulate.',
                       choices=['global', 'lcc', 'rotated', 'mercator'])
        p.add_argument('--auxiliar_files_path', required=True,
                       help='Path to the directory where the necessary auxiliary files will be created if them are ' +
                            'not created yet.')

        p.add_argument('--vertical_description', required=True,
                       help='Path to the file that contains the vertical description of the desired output.')

        # Global options
        p.add_argument('--inc_lat', required=False, help='Latitude resolution for a global domain.', type=float)
        p.add_argument('--inc_lon', required=False, help='Longitude resolution for a global domain.', type=float)

        # Rotated options
        p.add_argument('--centre_lat', required=False,
                       help='Central geographic latitude of grid (non-rotated degrees). Corresponds to the TPH0D ' +
                            'parameter in NMMB-MONARCH.', type=float)
        p.add_argument('--centre_lon', required=False,
                       help='Central geographic longitude of grid (non-rotated degrees, positive east). Corresponds ' +
                            'to the TLM0D parameter in NMMB-MONARCH.', type=float)
        p.add_argument('--west_boundary', required=False,
                       help="Grid's western boundary from center point (rotated degrees). Corresponds to the WBD " +
                            "parameter in NMMB-MONARCH.", type=float)
        p.add_argument('--south_boundary', required=False,
                       help="Grid's southern boundary from center point (rotated degrees). Corresponds to the SBD " +
                            "parameter in NMMB-MONARCH.", type=float)
        p.add_argument('--inc_rlat', required=False,
                       help='Latitudinal grid resolution (rotated degrees). Corresponds to the DPHD parameter in ' +
                            'NMMB-MONARCH.', type=float)
        p.add_argument('--inc_rlon', required=False,
                       help='Longitudinal grid resolution (rotated degrees). Corresponds to the DLMD parameter  ' +
                            'in NMMB-MONARCH.', type=float)

        # Lambert conformal conic options
        p.add_argument('--lat_1', required=False,
                       help='Standard parallel 1 (in deg). Corresponds to the P_ALP parameter of the GRIDDESC file.',
                       type=float)
        p.add_argument('--lat_2', required=False,
                       help='Standard parallel 2 (in deg). Corresponds to the P_BET parameter of the GRIDDESC file.',
                       type=float)
        p.add_argument('--lon_0', required=False,
                       help='Longitude of the central meridian (degrees). Corresponds to the P_GAM parameter of ' +
                            'the GRIDDESC file.', type=float)
        p.add_argument('--lat_0', required=False,
                       help='Latitude of the origin of the projection (degrees). Corresponds to the Y_CENT  ' +
                            'parameter of the GRIDDESC file.', type=float)
        p.add_argument('--nx', required=False,
                       help='Number of grid columns. Corresponds to the NCOLS parameter of the GRIDDESC file.',
                       type=float)
        p.add_argument('--ny', required=False,
                       help='Number of grid rows. Corresponds to the NROWS parameter of the GRIDDESC file.',
                       type=float)
        p.add_argument('--inc_x', required=False,
                       help='X-coordinate cell dimension (meters). Corresponds to the XCELL parameter of the ' +
                            'GRIDDESC file.', type=float)
        p.add_argument('--inc_y', required=False,
                       help='Y-coordinate cell dimension (meters). Corresponds to the YCELL parameter of the ' +
                            'GRIDDESC file.', type=float)
        p.add_argument('--x_0', required=False,
                       help='X-coordinate origin of grid (meters). Corresponds to the XORIG parameter of the ' +
                            'GRIDDESC file.', type=float)
        p.add_argument('--y_0', required=False,
                       help='Y-coordinate origin of grid (meters). Corresponds to the YORIG parameter of the ' +
                            'GRIDDESC file.', type=float)

        # Mercator
        p.add_argument('--lat_ts', required=False, help='...', type=float)

        p.add_argument('--cross_table', required=True,
                       help='Path to the file that contains the information of the datasets to use.')
        p.add_argument('--p_vertical', required=True,
                       help='Path to the file that contains all the needed vertical profiles.')
        p.add_argument('--p_month', required=True,
                       help='Path to the file that contains all the needed monthly profiles.')
        p.add_argument('--p_day', required=True, help='Path to the file that contains all the needed daily profiles.')
        p.add_argument('--p_hour', required=True, help='Path to the file that contains all the needed hourly profiles.')
        p.add_argument('--p_speciation', required=True,
                       help='Path to the file that contains all the needed speciation profiles.')
        p.add_argument('--molecular_weights', required=True,
                       help='Path to the file that contains the molecular weights of the input pollutants.')
        p.add_argument('--world_info', required=True,
                       help='Path to the file that contains the world information like timezones, ISO codes, ...')

        options = p.parse_args()
        for item in vars(options):
            is_str = False
            exec ("is_str = str == type(options.{0})".format(item))
            if is_str:
                exec("options.{0} = options.{0}.replace('<input_dir>', options.input_dir)".format(item))
                exec("options.{0} = options.{0}.replace('<domain_type>', options.domain_type)".format(item))
                if options.domain_type == 'global':
                    exec("options.{0} = options.{0}.replace('<resolution>', '{1}_{2}')".format(
                        item, options.inc_lat, options.inc_lon))
                elif options.domain_type == 'rotated':
                    exec("options.{0} = options.{0}.replace('<resolution>', '{1}_{2}')".format(
                        item, options.inc_rlat, options.inc_rlon))
                elif options.domain_type == 'lcc' or options.domain_type == 'mercator':
                    exec("options.{0} = options.{0}.replace('<resolution>', '{1}_{2}')".format(
                        item, options.inc_x, options.inc_y))

        options.start_date = self._parse_start_date(options.start_date)
        options.end_date = self._parse_end_date(options.end_date, options.start_date)

        self.create_dir(options.output_dir)
        self.create_dir(options.auxiliar_files_path)

        return options

    def get_output_name(self, date):
        """
        Generates the full path of the output replacing <date> by YYYYMMDDHH, YYYYMMDD, YYYYMM or YYYY depending on the
        output_timestep_type.

        :param date: Date of the day to simulate.
        :type: datetime.datetime

        :return: Complete path to the output file.
        :rtype: str
        """
        import os
        if self.options.output_timestep_type == 'hourly':
            file_name = self.options.output_name.replace('<date>', date.strftime('%Y%m%d%H'))
        elif self.options.output_timestep_type == 'daily':
            file_name = self.options.output_name.replace('<date>', date.strftime('%Y%m%d'))
        elif self.options.output_timestep_type == 'monthly':
            file_name = self.options.output_name.replace('<date>', date.strftime('%Y%m'))
        elif self.options.output_timestep_type == 'yearly':
            file_name = self.options.output_name.replace('<date>', date.strftime('%Y'))
        else:
            file_name = self.options.output_name
        full_path = os.path.join(self.options.output_dir, file_name)
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
        :type str_date: str

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
        :type end_date: str

        :param start_date: Date to the first day to simulate.
        :type start_date: datetime.datetime

        :return: Date to the last day to simulate in datetime format.
        :rtype: datetime.datetime
        """
        if end_date is None:
            return start_date
        else:
            return self._parse_start_date(end_date)

    def set_log_level(self):
        """
        Defines the log_level using the common script settings.
        """
        import settings
        settings.define_global_vars(self.options.log_level)


if __name__ == '__main__':
    config = Config()
    print config.options
