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


import os
import timeit
from warnings import warn as warning
import hermesv3_gr.config.settings as settings


class Masking(object):
    """
    Masking object to apply simple mask or factor mask.

    :param world_info: Path to the file that contains the ISO Codes and other relevant information.
    :type world_info: str

    :param factors_mask_values: List of the factor mask values.
    :type factors_mask_values: list

    :param regrid_mask_values: List of the mask values.
    :type regrid_mask_values: list

    :param grid: Grid.
    :type grid: Grid

    :param world_mask_file:
    :type world_mask_file: str
    """

    def __init__(self, world_info, factors_mask_values, regrid_mask_values, grid, world_mask_file=None):
        from timezonefinder import TimezoneFinder

        st_time = timeit.default_timer()
        settings.write_log('\t\tCreating mask.', level=2)

        self.adding = None
        self.world_info = world_info
        self.country_codes = self.get_country_codes()
        self.world_mask_file = world_mask_file
        self.factors_mask_values = self.parse_factor_values(factors_mask_values)
        self.regrid_mask_values = self.parse_masking_values(regrid_mask_values)
        self.regrid_mask = None
        self.scale_mask = None
        self.timezonefinder = TimezoneFinder()

        self.grid = grid

        settings.write_time('Masking', 'Init', timeit.default_timer() - st_time, level=3)

    def get_country_codes(self):
        """
        Get the country code information.

        :return: Dictionary of country codes.
        :rtype: dict
        """
        import pandas as pd

        st_time = timeit.default_timer()

        dataframe = pd.read_csv(self.world_info, sep=';')
        del dataframe['time_zone'], dataframe['time_zone_code']
        dataframe = dataframe.drop_duplicates().dropna()
        dataframe = dataframe.set_index('country_code_alpha')
        countries_dict = dataframe.to_dict()
        countries_dict = countries_dict['country_code']

        settings.write_time('Masking', 'get_country_codes', timeit.default_timer() - st_time, level=3)
        return countries_dict

    @staticmethod
    def partlst(lst, num):
        """
        Split a Array in N balanced arrays.

        :param lst: Array to split
        :type lst: numpy.array

        :param num: Number of mini arrays.
        :type num: int

        :return: Array
        :type: numpy.array
        """
        import itertools
        # Partition @lst in @n balanced parts, in given order
        parts, rest = divmod(len(lst), num)
        lstiter = iter(lst)
        for j in xrange(num):
            plen = len(lst) / num + (1 if rest > 0 else 0)
            rest -= 1
            yield list(itertools.islice(lstiter, plen))

    def create_country_iso(self, in_nc):
        import numpy as np
        from hermesv3_gr.tools.netcdf_tools import extract_vars
        from hermesv3_gr.modules.writing.writer import Writer

        st_time = timeit.default_timer()
        settings.write_log('\t\t\tCreating {0} file.'.format(self.world_mask_file), level=2)
        # output_path = os.path.join(output_dir, 'iso.nc')

        lat_o, lon_o = extract_vars(in_nc, ['lat', 'lon'])
        lon = np.array([lon_o['data']] * len(lat_o['data']))
        lat = np.array([lat_o['data']] * len(lon_o['data'])).T

        dst_var = []
        num = 0
        points = np.array(zip(lat.flatten(), lon.flatten()))

        points_list = list(self.partlst(points, settings.size))

        for lat_aux, lon_aux in points_list[settings.rank]:
            num += 1

            settings.write_log("\t\t\t\tlat:{0}, lon:{1} ({2}/{3})".format(
                lat_aux, lon_aux, num, len(points_list[settings.rank])), level=3)

            tz = self.find_timezone(lat_aux, lon_aux)
            tz_id = self.get_iso_code_from_tz(tz)
            dst_var.append(tz_id)
        dst_var = np.array(dst_var)
        dst_var = settings.comm.gather(dst_var, root=0)

        if settings.rank == 0:
            dst_var = np.concatenate(dst_var)
            dst_var = dst_var.reshape((1,) + lat.shape)
            data = [{
                'name': 'timezone_id',
                'units': '',
                'data': dst_var,
            }]
            Writer.write_netcdf(self.world_mask_file, lat, lon, data, regular_latlon=True)
        settings.comm.Barrier()

        settings.write_time('Masking', 'create_country_iso', timeit.default_timer() - st_time, level=3)

        return True

    def find_timezone(self, latitude, longitude):

        st_time = timeit.default_timer()

        if longitude < -180:
            longitude += 360
        elif longitude > +180:
            longitude -= 360

        tz = self.timezonefinder.timezone_at(lng=longitude, lat=latitude)

        settings.write_time('Masking', 'find_timezone', timeit.default_timer() - st_time, level=3)

        return tz

    def get_iso_code_from_tz(self, tz):
        import pandas as pd

        st_time = timeit.default_timer()

        zero_values = [None, ]
        if tz in zero_values:
            return 0

        df = pd.read_csv(self.world_info, sep=';')
        code = df.country_code[df.time_zone == tz].values

        settings.write_time('Masking', 'get_iso_code_from_tz', timeit.default_timer() - st_time, level=3)

        return code[0]

    def parse_factor_values(self, values):
        """

        :param values:
        :return:
        :rtype: dict
        """
        import re

        st_time = timeit.default_timer()

        if type(values) != str:
            return None
        values = list(map(str, re.split(' , |, | ,|,', values)))
        scale_dict = {}
        for element in values:
            element = list(map(str, re.split("{0}{0}|{0}".format(' '), element)))
            scale_dict[int(self.country_codes[element[0]])] = element[1]

        settings.write_log('\t\t\tApplying scaling factors for {0}.'.format(values), level=3)
        settings.write_time('Masking', 'parse_factor_values', timeit.default_timer() - st_time, level=3)

        return scale_dict

    def parse_masking_values(self, values):
        """

        :param values:
        :return:
        :rtype: list
        """
        import re

        st_time = timeit.default_timer()

        if type(values) != str:
            return None
        values = list(map(str, re.split(' , |, | ,|,| ', values)))
        if values[0] == '+':
            self.adding = True
        elif values[0] == '-':
            self.adding = False
        else:
            if len(values) > 0:
                settings.write_log('WARNING: Check the .err file to get more info. Ignoring mask')
                if settings.rank == 0:
                    warning("WARNING: The list of masking does not start with '+' or '-'. Ignoring mask.")
            return None
        code_list = []
        for country in values[1:]:
            code_list.append(int(self.country_codes[country]))

        if self.adding:
            settings.write_log("\t\t\tCreating mask to do {0} countries.".format(values[1:]), level=3)
        else:
            settings.write_log("\t\t\tCreating mask to avoid {0} countries.".format(values[1:]), level=3)
        settings.write_time('Masking', 'parse_masking_values', timeit.default_timer() - st_time, level=3)

        return code_list

    def check_regrid_mask(self, input_file):

        if self.regrid_mask_values is not None:
            if not os.path.exists(self.world_mask_file):
                self.create_country_iso(input_file)
            self.regrid_mask = self.custom_regrid_mask()
        if self.factors_mask_values is not None:
            if not os.path.exists(self.world_mask_file):
                self.create_country_iso(input_file)
            self.scale_mask = self.custom_scale_mask()

    def custom_regrid_mask(self):
        import numpy as np
        from netCDF4 import Dataset

        st_time = timeit.default_timer()

        netcdf = Dataset(self.world_mask_file, mode='r')
        values = netcdf.variables['timezone_id'][:]
        netcdf.close()

        if self.adding:
            mask = np.zeros(values.shape)
            for code in self.regrid_mask_values:
                mask[values == code] = 1
        else:
            mask = np.ones(values.shape)
            for code in self.regrid_mask_values:
                mask[values == code] = 0

        settings.write_time('Masking', 'custom_regrid_mask', timeit.default_timer() - st_time, level=3)

        return mask

    def custom_scale_mask(self):
        import numpy as np
        from hermesv3_gr.tools.netcdf_tools import extract_vars

        st_time = timeit.default_timer()

        [values] = extract_vars(self.world_mask_file, ['timezone_id'])

        values = values['data']
        mask = np.ones(values.shape)
        for code, factor in self.factors_mask_values.iteritems():
            mask[values == code] = factor

        settings.write_time('Masking', 'custom_scale_mask', timeit.default_timer() - st_time, level=3)

        return mask


if __name__ == '__main__':
    pass
