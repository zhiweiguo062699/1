#!/usr/bin/env python

SECTOR_LIST = ['traffic', 'traffic_area', 'aviation', 'point_sources', 'recreational_boats', 'shipping_port',
               'residential', 'livestock', 'crop_operations', 'crop_fertilizers', 'agricultural_machinery']


class SectorManager(object):
    def __init__(self, comm_world, grid, clip, arguments):
        self.sector_list = self.make_sector_list(arguments)

        if comm_world.Get_size() != sum(self.sector_list.values()):
            raise ValueError("The selected number of processors '{0}' does not fit ".format(comm_world.Get_size()) +
                             "with the sum of processors dedicated for all the sectors " +
                             "'{0}': {1}".format(sum(self.sector_list.values()), self.sector_list))

        exit()

    @staticmethod
    def make_sector_list(arguments):
        sector_dict = {}
        for sector in SECTOR_LIST:
            if vars(arguments)['do_{0}'.format(sector)]:
                sector_dict[sector] = vars(arguments)['{0}_processors'.format(sector)]
        return sector_dict


