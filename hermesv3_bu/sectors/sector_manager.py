#!/usr/bin/env python

SECTOR_LIST = ['traffic', 'traffic_area', 'aviation', 'point_sources', 'recreational_boats', 'shipping_port',
               'residential', 'livestock', 'crop_operations', 'crop_fertilizers', 'agricultural_machinery']


class SectorManager(object):
    def __init__(self, comm_world, grid, clip, date_array, arguments):
        self.sector_list = self.make_sector_list(arguments, comm_world.Get_size())

        color = 10
        for sector, sector_procs in self.sector_list.iteritems():
            if sector == 'aviation' and comm_world.Get_rank() in sector_procs:
                print '{0} -> {1}, {2}'.format(comm_world.Get_rank(), sector, sector_procs)
                from hermesv3_bu.sectors.aviation_sector import AviationSector
                self.sector = AviationSector(comm_world.Split(color, comm_world.Get_rank() - sector_procs[0]),
                                             arguments.auxiliary_files_path, grid.shapefile, clip,
                                             date_array, arguments.aviation_source_pollutants,
                                             grid.vertical_desctiption, arguments.airport_list, arguments.plane_list,
                                             arguments.airport_shapefile_path, arguments.airport_runways_shapefile_path,
                                             arguments.airport_runways_corners_shapefile_path,
                                             arguments.airport_trajectories_shapefile_path,
                                             arguments.airport_operations_path, arguments.planes_path,
                                             arguments.airport_times_path, arguments.airport_ef_dir,
                                             arguments.aviation_weekly_profiles, arguments.aviation_hourly_profiles,
                                             arguments.speciation_map, arguments.aviation_speciation_profiles,
                                             arguments.molecular_weights)
            elif sector == 'shipping_port' and comm_world.Get_rank() in sector_procs:
                print '{0} -> {1}, {2}'.format(comm_world.Get_rank(), sector, sector_procs)
                from hermesv3_bu.sectors.shipping_port_sector import ShippingPortSector
                self.sector = ShippingPortSector(comm_world.Split(color, comm_world.Get_rank() - sector_procs[0]),
                                                 arguments.auxiliary_files_path, grid.shapefile, clip, date_array,
                                                 arguments.shipping_port_source_pollutants,
                                                 grid.vertical_desctiption,
                                                 arguments.vessel_list,
                                                 arguments.port_list,
                                                 arguments.hoteling_shapefile_path,
                                                 arguments.maneuvering_shapefile_path,
                                                 arguments.shipping_port_ef_path,
                                                 arguments.shipping_port_engine_percent_path,
                                                 arguments.shipping_port_tonnage_path,
                                                 arguments.shipping_port_load_factor_path,
                                                 arguments.shipping_port_power_path,
                                                 arguments.shipping_port_monthly_profiles,
                                                 arguments.shipping_port_weekly_profiles,
                                                 arguments.shipping_port_hourly_profiles,
                                                 arguments.speciation_map,
                                                 arguments.shipping_port_speciation_profiles,
                                                 arguments.molecular_weights,)
            color += 1

    def run(self):
        return self.sector.calculate_emissions()

    @staticmethod
    def make_sector_list(arguments, max_procs):
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
        return sector_dict
