#!/usr/bin/env python

"""
Copyright 2018 Earth Sciences Department, BSC-CNS

 This file is part of HERMESv3.

 HERMESv3 is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 HERMESv3 is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with HERMESv3. If not, see <http://www.gnu.org/licenses/>.
"""

__author__ = "Carles Tena"
__copyright__ = "Copyright 2018"
__email__ = "carles.tena@bsc.es"
__license__ = "GNU General Public License"
__maintainer__ = "Carles Tena"
__version__ = "3.3.1"

from memory_profiler import profile
import sys
import os
from mpi4py import MPI

parentPath = os.path.abspath(os.path.join('..', '..'))
if parentPath not in sys.path:
    sys.path.insert(0, parentPath)

from timeit import default_timer as gettime

from hermesv3_bu.config import settings
from hermesv3_bu.config.config import Config
from hermesv3_bu.modules.emision_inventories.emission_inventory import EmissionInventory
from hermesv3_bu.modules.vertical.vertical import VerticalDistribution
from hermesv3_bu.modules.temporal.temporal import TemporalDistribution
from hermesv3_bu.modules.bottomup.traffic.traffic import Traffic
from hermesv3_bu.modules.writing.writing_cmaq import WritingCmaq
from hermesv3_bu.modules.writing.writing import Writing
from hermesv3_bu.tools.netcdf_tools import *
from hermesv3_bu.modules.bottomup.point_source.point_source import PointSource
# import pyextrae.sequential as pyextrae


class Hermes(object):
    """
    Interface class for HERMESv3.
    """
    def __init__(self, config, new_date=None):
        from hermesv3_bu.modules.grids.grid import Grid
        from hermesv3_bu.modules.temporal.temporal import TemporalDistribution

        st_time = gettime()

        self.config = config
        self.options = config.options

        # updating starting date
        if new_date is not None:
            self.options.start_date = new_date

        config.set_log_level()

        self.grid = Grid.select_grid(self.options.domain_type, self.options.vertical_description, self.options.output_timestep_num, self.options.auxiliar_files_path, self.options.inc_lat,
                                     self.options.inc_lon, self.options.centre_lat, self.options.centre_lon,
                                     self.options.west_boundary, self.options.south_boundary, self.options.inc_rlat,
                                     self.options.inc_rlon,
                                     self.options.lat_1, self.options.lat_2, self.options.lon_0, self.options.lat_0,
                                     self.options.nx, self.options.ny, self.options.inc_x, self.options.inc_y,
                                     self.options.x_0, self.options.y_0)
        if not self.options.do_bottomup:
            self.emission_list = EmissionInventory.make_emission_list(self.options, self.grid, self.options.start_date)
        else:
            if self.options.do_traffic:
                self.traffic = Traffic(self.options.auxiliar_files_path, self.options.clipping,
                                       self.options.road_link_path, self.options.fleet_compo_path,
                                       self.options.speed_hourly_path, self.options.traffic_monthly_profiles,
                                       self.options.traffic_daily_profiles, self.options.traffic_hourly_profiles_mean,
                                       self.options.traffic_hourly_profiles_weekday,
                                       self.options.traffic_hourly_profiles_saturday,
                                       self.options.traffic_hourly_profiles_sunday, self.options.ef_path,
                                       self.options.traffic_pollutants, self.options.start_date, self.grid,
                                       vehicle_list=self.options.vehicle_types,
                                       load=self.options.load,
                                       timestep_type=self.options.output_timestep_type,
                                       timestep_num=self.options.output_timestep_num,
                                       timestep_freq=self.options.output_timestep_freq,
                                       speciation_map=self.options.traffic_speciation_map,
                                       hot_cold_speciation=self.options.traffic_speciation_profile_hot_cold,
                                       tyre_speciation=self.options.traffic_speciation_profile_tyre,
                                       road_speciation=self.options.traffic_speciation_profile_road,
                                       brake_speciation=self.options.traffic_speciation_profile_brake,
                                       resuspension_speciation=self.options.traffic_speciation_profile_resuspension,

                                       temp_common_path=self.options.temperature_files_path,
                                       output_type=self.options.output_type, output_dir=self.options.output_dir,
                                       molecular_weights_path=self.options.molecular_weights,)
            if self.options.do_point_sources:
                self.poin_source = PointSource(
                    self.grid, self.options.point_source_catalog, self.options.point_source_monthly_profiles,
                    self.options.point_source_daily_profiles, self.options.point_source_hourly_profiles,
                    self.options.point_source_speciation_map, self.options.point_source_speciation_profiles,
                    self.options.point_source_snaps, self.options.effective_stack_height,
                    self.options.point_source_pollutants, self.options.point_source_measured_emissions,
                    molecular_weights_path=self.options.molecular_weights)

        self.delta_hours = TemporalDistribution.calculate_delta_hours(self.options.start_date,
                                                                      self.options.output_timestep_type,
                                                                      self.options.output_timestep_num,
                                                                      self.options.output_timestep_freq)
        self.levels = VerticalDistribution.get_vertical_output_profile(self.options.vertical_description)

        print 'TIME -> HERMES.__init__: Rank {0} {1} s'.format(settings.rank, round(gettime() - st_time, 2))

    # @profile
    def main(self):
        """
        Main functionality of the model.
        """
        from multiprocessing import Process, Queue, cpu_count
        from threading import Thread
        import copy
        import gc
        import numpy as np
        from datetime import timedelta
        from cf_units import Unit

        if settings.log_level_1:
            print '===================================================='
            print '==================== HERMESv3.0 ===================='
            print '===================================================='

        if settings.log_level_3:
            st_time = gettime()
        else:
            st_time = None

        # date_aux = self.options.start_date

        # while date_aux <= self.options.end_date:
        if settings.log_level_1:
            print '\n\t================================================'
            print '\t\t STARTING emissions for {0}'.format(self.options.start_date.strftime('%Y/%m/%d %H:%M:%S'))
            print '\t================================================'
            st_time_1 = gettime()
        else:
            st_time_1 = None
        if not self.options.do_bottomup:
            for ei in self.emission_list:
                ei.do_regrid()
                if ei.vertical is not None:
                    vf_time = gettime()
                    ei.vertical_factors = ei.vertical.calculate_weights()
                    print "TIME -> Vertical_factors: {0} Rank {1} {2} s\n".format("{0}_{1}".format(ei.inventory_name, ei.sector), settings.rank, round(gettime() - vf_time, 4))
                if ei.temporal is not None:
                    tf_time = gettime()
                    ei.temporal_factors = ei.temporal.calculate_3d_temporal_factors()
                    print "TIME -> Temporal_factors: {0} Rank {1} {2} s\n".format("{0}_{1}".format(ei.inventory_name, ei.sector), settings.rank, round(gettime() - tf_time, 4))
                if ei.speciation is not None:
                    sp_time = gettime()
                    ei.emissions = ei.speciation.do_speciation(ei.emissions, self.grid.cell_area)
                    print "TIME -> Speciation: {0} Rank {1} {2} s\n".format("{0}_{1}".format(ei.inventory_name, ei.sector), settings.rank, round(gettime() - sp_time, 4))
        else:
            if self.options.do_traffic:
                e = self.traffic.calculate_traffic_line_emissions(
                    do_hot=self.options.do_hot,  do_cold=self.options.do_cold, do_tyre_wear=self.options.do_tyre_wear,
                    do_brake_wear=self.options.do_brake_wear, do_road_wear=self.options.do_road_wear,
                    do_resuspension=self.options.do_resuspension, do_evaporative=self.options.do_evaporative,
                    do_other_cities=self.options.do_other_cities)

                if self.options.output_type == 'R-LINE':
                    self.traffic.write_rline(e, self.options.output_dir, self.options.start_date)

                    if settings.log_level_1:
                        print '\t=========================================='
                        print '\t\t TIME {0} -> {1}'.format(self.options.start_date.strftime('%Y/%m/%d %H:%M:%S'),
                                                            round(gettime() - st_time_1, 2))
                        print '\t=========================================='

                    if self.options.start_date < self.options.end_date:
                        return self.options.start_date + timedelta(days=1)
                    return None

                self.emission_list = self.traffic.links_to_grid(e, self.grid.to_shapefile())

                if self.options.output_type == 'MONARCH':
                    pass
                    # TODO divide by cell/area
            if self.options.do_point_sources:
                e = self.poin_source.calculate_point_source_emissions(
                    self.options.start_date, self.delta_hours, self.levels)
                self.emission_list = self.poin_source.points_to_grid(
                    e, self.grid.to_shapefile(),  self.poin_source.speciation_map['dst'].values)

        writing_time = gettime()

        if self.options.output_type == 'CMAQ':
            writer = WritingCmaq
        elif self.options.output_type == 'MONARCH':
            writer = Writing

        if self.options.do_bottomup:
            if settings.rank == 0:
                writer.write_netcdf(self.config.get_output_name(self.options.start_date), self.grid,
                                    self.emission_list,
                                    levels=VerticalDistribution.get_vertical_output_profile(
                                        self.options.vertical_description),
                                    date=self.options.start_date, hours=self.delta_hours,
                                    point_source=self.options.do_point_sources)

        else:
            empty_dict = {}
            for ei in self.emission_list:
                for emi in ei.emissions:
                    if not emi['name'] in empty_dict:
                        dict_aux = emi.copy()
                        dict_aux['data'] = None
                        empty_dict[emi['name']] = dict_aux

            if settings.writing_serial:
                writer.write_serial_netcdf(self.config.get_output_name(self.options.start_date), self.grid, empty_dict.values(),
                                           self.emission_list,
                                           levels=VerticalDistribution.get_vertical_output_profile(
                                               self.options.vertical_description),
                                           date=self.options.start_date, hours=self.delta_hours)
            else:
                if settings.rank == 0:
                    print "TIME -> empty_list: {0} s\n".format(round(gettime() - writing_time, 2))
                    writer.create_parallel_netcdf(self.config.get_output_name(self.options.start_date), self.grid, empty_dict.values(),
                                                  levels=VerticalDistribution.get_vertical_output_profile(
                                                      self.options.vertical_description),
                                                  date=self.options.start_date, hours=self.delta_hours)
                    print 'NETCDF CREATED. Starting to write'
                settings.comm.Barrier()
                if settings.rank == 0:
                    print 'Starting to write'
                writer.write_parallel_netcdf(self.config.get_output_name(self.options.start_date), self.grid, empty_dict.keys(),
                                             self.emission_list)

        print "TIME -> Writing Rank {0} {1} s\n".format(settings.rank, round(gettime() - writing_time, 2))
        settings.comm.Barrier()
        if settings.log_level_2:
            print "TIME -> TOTAL Writing: {0} s\n".format(round(gettime() - writing_time, 2))
        if settings.log_level_1:
            print '\t=========================================='
            print '\t\t TIME {0} -> {1}'.format(self.options.start_date.strftime('%Y/%m/%d %H:%M:%S'),
                                                round(gettime() - st_time_1, 2))
            print '\t=========================================='

        if settings.log_level_3:
            print 'TIME -> HERMES.main: {0} s\n'.format(round(gettime() - st_time, 2))

        if self.options.start_date < self.options.end_date:
            return self.options.start_date + timedelta(days=1)

        return None


if __name__ == '__main__':
    date = Hermes(Config()).main()
    while date is not None:
        date = Hermes(Config(), new_date=date).main()
    sys.exit(0)
