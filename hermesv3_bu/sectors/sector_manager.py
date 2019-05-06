#!/usr/bin/env python

SECTOR_LIST = ['traffic', 'traffic_area', 'aviation', 'point_sources', 'recreational_boats', 'shipping_port',
               'residential', 'livestock', 'crop_operations', 'crop_fertilizers', 'agricultural_machinery']


class SectorManager(object):
    def __init__(self, comm_world, grid, clip, date_array, arguments):
        self.sector_list = self.make_sector_list(arguments)

        if comm_world.Get_size() != sum(self.sector_list.values()):
            raise ValueError("The selected number of processors '{0}' does not fit ".format(comm_world.Get_size()) +
                             "with the sum of processors dedicated for all the sectors " +
                             "'{0}': {1}".format(sum(self.sector_list.values()), self.sector_list))
        from hermesv3_bu.sectors.aviation_sector import AviationSector
        self.sector = AviationSector(comm_world, arguments.auxiliary_files_path, grid.shapefile, clip, date_array,
                                     arguments.aviation_source_pollutants, grid.vertical_desctiption,
                                     arguments.airport_list, arguments.plane_list, arguments.airport_shapefile_path,
                                     arguments.airport_runways_shapefile_path,
                                     arguments.airport_runways_corners_shapefile_path,
                                     arguments.airport_trajectories_shapefile_path, arguments.airport_operations_path,
                                     arguments.planes_path, arguments.airport_times_path, arguments.airport_ef_dir,
                                     arguments.aviation_weekly_profiles, arguments.aviation_hourly_profiles,
                                     arguments.aviation_speciation_map, arguments.aviation_speciation_profiles,
                                     arguments.molecular_weights)

    def run(self):
        return self.sector.calculate_emissions()

    @staticmethod
    def make_sector_list(arguments):
        sector_dict = {}
        for sector in SECTOR_LIST:
            if vars(arguments)['do_{0}'.format(sector)]:
                sector_dict[sector] = vars(arguments)['{0}_processors'.format(sector)]
        return sector_dict
