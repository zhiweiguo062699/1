#!/usr/bin/env python

import timeit
from hermesv3_bu.logger.log import Log

SECTOR_LIST = ['traffic', 'traffic_area', 'aviation', 'point_sources', 'recreational_boats', 'shipping_port',
               'residential', 'livestock', 'crop_operations', 'crop_fertilizers', 'agricultural_machinery']


class SectorManager(object):
    def __init__(self, comm_world, logger, grid, clip, date_array, arguments):
        """

        :param comm_world: MPI Communicator

        :param logger: Logger
        :type logger: Log

        :param grid:
        :param clip:
        :param date_array:
        :type date_array: list

        :param arguments:
        :type arguments: NameSpace
        """
        spent_time = timeit.default_timer()
        self.logger = logger
        self.sector_list = self.make_sector_list(arguments, comm_world.Get_size())
        self.logger.write_log('Sector process distribution:')
        for sect, procs in self.sector_list.iteritems():
            self.logger.write_log('\t{0}: {1}'.format(sect, procs))

        color = 10
        for sector, sector_procs in self.sector_list.iteritems():
            if sector == 'aviation' and comm_world.Get_rank() in sector_procs:
                from hermesv3_bu.sectors.aviation_sector import AviationSector
                self.sector = AviationSector(
                    comm_world.Split(color, comm_world.Get_rank() - sector_procs[0]), self.logger,
                    arguments.auxiliary_files_path, grid.shapefile, clip, date_array,
                    arguments.aviation_source_pollutants, grid.vertical_desctiption, arguments.airport_list,
                    arguments.plane_list, arguments.airport_shapefile_path, arguments.airport_runways_shapefile_path,
                    arguments.airport_runways_corners_shapefile_path, arguments.airport_trajectories_shapefile_path,
                    arguments.airport_operations_path, arguments.planes_path, arguments.airport_times_path,
                    arguments.airport_ef_dir, arguments.aviation_weekly_profiles, arguments.aviation_hourly_profiles,
                    arguments.speciation_map, arguments.aviation_speciation_profiles, arguments.molecular_weights)
            elif sector == 'shipping_port' and comm_world.Get_rank() in sector_procs:
                from hermesv3_bu.sectors.shipping_port_sector import ShippingPortSector
                self.sector = ShippingPortSector(
                    comm_world.Split(color, comm_world.Get_rank() - sector_procs[0]), self.logger,
                    arguments.auxiliary_files_path, grid.shapefile, clip, date_array,
                    arguments.shipping_port_source_pollutants, grid.vertical_desctiption, arguments.vessel_list,
                    arguments.port_list, arguments.hoteling_shapefile_path, arguments.maneuvering_shapefile_path,
                    arguments.shipping_port_ef_path, arguments.shipping_port_engine_percent_path,
                    arguments.shipping_port_tonnage_path, arguments.shipping_port_load_factor_path,
                    arguments.shipping_port_power_path, arguments.shipping_port_monthly_profiles,
                    arguments.shipping_port_weekly_profiles, arguments.shipping_port_hourly_profiles,
                    arguments.speciation_map, arguments.shipping_port_speciation_profiles, arguments.molecular_weights)
            elif sector == 'livestock' and comm_world.Get_rank() in sector_procs:
                from hermesv3_bu.sectors.livestock_sector import LivestockSector
                self.sector = LivestockSector(
                    comm_world.Split(color, comm_world.Get_rank() - sector_procs[0]), self.logger,
                    arguments.auxiliary_files_path, grid.shapefile, clip, date_array,
                    arguments.livestock_source_pollutants, grid.vertical_desctiption, arguments.animal_list,
                    arguments.gridded_livestock, arguments.correction_split_factors,
                    arguments.temperature_daily_files_path, arguments.wind_speed_daily_files_path,
                    arguments.denominator_yearly_factor_dir, arguments.livestock_ef_files_dir,
                    arguments.livestock_monthly_profiles, arguments.livestock_weekly_profiles,
                    arguments.livestock_hourly_profiles, arguments.speciation_map,
                    arguments.livestock_speciation_profiles, arguments.molecular_weights, arguments.nut_shapefile_prov)
            color += 1

        self.logger.write_time_log('SectorManager', '__init__', timeit.default_timer() - spent_time)

    def run(self):
        spent_time = timeit.default_timer()
        emis = self.sector.calculate_emissions()
        self.logger.write_time_log('SectorManager', 'run', timeit.default_timer() - spent_time)
        return emis

    def make_sector_list(self, arguments, max_procs):
        spent_time = timeit.default_timer()
        sector_dict = {}
        accum = 0
        for sector in SECTOR_LIST:
            if vars(arguments)['do_{0}'.format(sector)]:
                n_procs = vars(arguments)['{0}_processors'.format(sector)]
                sector_dict[sector] = [accum + x for x in xrange(n_procs)]
                accum += n_procs
        if accum != max_procs:
            raise ValueError("The selected number of processors '{0}' does not fit ".format(max_procs) +
                             "with the sum of processors dedicated for all the sectors " +
                             "'{0}': {1}".format(accum, {sector: len(sector_procs)
                                                         for sector, sector_procs in sector_dict.iteritems()}))

        self.logger.write_time_log('SectorManager', 'make_sector_list', timeit.default_timer() - spent_time)
        return sector_dict
