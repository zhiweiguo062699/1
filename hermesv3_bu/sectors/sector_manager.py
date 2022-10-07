#!/usr/bin/env python

import timeit
from hermesv3_bu.logger.log import Log
from hermesv3_bu.tools.checker import error_exit

SECTOR_LIST = ['aviation', 'point_sources', 'recreational_boats', 'shipping_port', 'residential', 'livestock',
               'crop_operations', 'crop_fertilizers', 'agricultural_machinery', 'solvents', 'traffic_area', 'traffic']


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
        self.__logger = logger
        self.sector_list = self.make_sector_list(arguments, comm_world.Get_size())
        self.__logger.write_log('Sector process distribution:')
        for sect, procs in self.sector_list.items():
            self.__logger.write_log('\t{0}: {1}'.format(sect, procs))

        color = 10
        agg_color = 99
        for sector, sector_procs in self.sector_list.items():
            if sector == 'aviation' and comm_world.Get_rank() in sector_procs:
                from hermesv3_bu.sectors.aviation_sector import AviationSector
                self.sector = AviationSector(
                    comm_world.Split(color, sector_procs.index(comm_world.Get_rank())), self.__logger,
                    arguments.auxiliary_files_path, grid, clip, date_array,
                    arguments.aviation_source_pollutants, grid.vertical_desctiption, arguments.airport_list,
                    arguments.plane_list, arguments.airport_shapefile_path, arguments.airport_runways_shapefile_path,
                    arguments.airport_runways_corners_shapefile_path, arguments.airport_trajectories_shapefile_path,
                    arguments.airport_operations_path, arguments.planes_path, arguments.airport_times_path,
                    arguments.airport_ef_dir, arguments.aviation_weekly_profiles, arguments.aviation_hourly_profiles,
                    arguments.speciation_map, arguments.aviation_speciation_profiles, arguments.molecular_weights)

            elif sector == 'shipping_port' and comm_world.Get_rank() in sector_procs:
                from hermesv3_bu.sectors.shipping_port_sector import ShippingPortSector
                self.sector = ShippingPortSector(
                    comm_world.Split(color, sector_procs.index(comm_world.Get_rank())), self.__logger,
                    arguments.auxiliary_files_path, grid, clip, date_array,
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
                    comm_world.Split(color, sector_procs.index(comm_world.Get_rank())), self.__logger,
                    arguments.auxiliary_files_path, grid, clip, date_array,
                    arguments.livestock_source_pollutants, grid.vertical_desctiption, arguments.animal_list,
                    arguments.gridded_livestock, arguments.correction_split_factors,
                    arguments.temperature_daily_files_path, arguments.wind_speed_daily_files_path,
                    arguments.denominator_yearly_factor_dir, arguments.livestock_ef_files_dir,
                    arguments.livestock_monthly_profiles, arguments.livestock_weekly_profiles,
                    arguments.livestock_hourly_profiles, arguments.speciation_map,
                    arguments.livestock_speciation_profiles, arguments.molecular_weights, arguments.nuts3_shapefile)

            elif sector == 'crop_operations' and comm_world.Get_rank() in sector_procs:
                from hermesv3_bu.sectors.agricultural_crop_operations_sector import AgriculturalCropOperationsSector
                agg_procs = AgriculturalCropOperationsSector.get_agricultural_processor_list(self.sector_list)
                comm_agr = comm_world.Split(agg_color, agg_procs.index(comm_world.Get_rank()))
                comm = comm_agr.Split(color, sector_procs.index(comm_world.Get_rank()))
                self.sector = AgriculturalCropOperationsSector(
                    comm_agr, comm, logger, arguments.auxiliary_files_path, grid, clip, date_array,
                    arguments.crop_operations_source_pollutants,
                    grid.vertical_desctiption, arguments.crop_operations_list, arguments.nuts2_shapefile,
                    arguments.land_uses_path, arguments.crop_operations_ef_files_dir,
                    arguments.crop_operations_monthly_profiles, arguments.crop_operations_weekly_profiles,
                    arguments.crop_operations_hourly_profiles, arguments.speciation_map,
                    arguments.crop_operations_speciation_profiles, arguments.molecular_weights,
                    arguments.land_uses_nuts2_path, arguments.crop_by_nut_path, arguments.crop_from_landuse_path)

            elif sector == 'crop_fertilizers' and comm_world.Get_rank() in sector_procs:
                from hermesv3_bu.sectors.agricultural_crop_fertilizers_sector import AgriculturalCropFertilizersSector
                agg_procs = AgriculturalCropFertilizersSector.get_agricultural_processor_list(self.sector_list)
                comm_agr = comm_world.Split(agg_color, agg_procs.index(comm_world.Get_rank()))
                comm = comm_agr.Split(color, sector_procs.index(comm_world.Get_rank()))
                self.sector = AgriculturalCropFertilizersSector(
                    comm_agr, comm, logger, arguments.auxiliary_files_path, grid, clip, date_array,
                    arguments.crop_fertilizers_source_pollutants, grid.vertical_desctiption,
                    arguments.crop_fertilizers_list, arguments.nuts2_shapefile, arguments.land_uses_path,
                    arguments.crop_fertilizers_hourly_profiles, arguments.speciation_map,
                    arguments.crop_fertilizers_speciation_profiles, arguments.molecular_weights,
                    arguments.land_uses_nuts2_path,  arguments.crop_by_nut_path, arguments.crop_from_landuse_path,
                    arguments.cultivated_ratio, arguments.fertilizers_rate, arguments.crop_f_parameter,
                    arguments.crop_f_fertilizers, arguments.gridded_ph, arguments.gridded_cec,
                    arguments.fertilizers_denominator_yearly_factor_path, arguments.crop_calendar,
                    arguments.temperature_daily_files_path, arguments.wind_speed_daily_files_path,
                    arguments.crop_growing_degree_day_path)

            elif sector == 'agricultural_machinery' and comm_world.Get_rank() in sector_procs:
                from hermesv3_bu.sectors.agricultural_machinery_sector import AgriculturalMachinerySector
                agg_procs = AgriculturalMachinerySector.get_agricultural_processor_list(self.sector_list)
                comm_agr = comm_world.Split(agg_color, agg_procs.index(comm_world.Get_rank()))
                comm = comm_agr.Split(color, sector_procs.index(comm_world.Get_rank()))
                self.sector = AgriculturalMachinerySector(
                    comm_agr, comm, logger, arguments.auxiliary_files_path, grid, clip, date_array,
                    arguments.crop_machinery_source_pollutants, grid.vertical_desctiption,
                    arguments.crop_machinery_list, arguments.nuts2_shapefile, arguments.machinery_list,
                    arguments.land_uses_path, arguments.crop_machinery_ef_path,
                    arguments.crop_machinery_monthly_profiles, arguments.crop_machinery_weekly_profiles,
                    arguments.crop_machinery_hourly_profiles,
                    arguments.speciation_map, arguments.crop_machinery_speciation_profiles, arguments.molecular_weights,
                    arguments.land_uses_nuts2_path, arguments.crop_by_nut_path, arguments.crop_from_landuse_path,
                    arguments.nuts3_shapefile, arguments.crop_machinery_deterioration_factor_path,
                    arguments.crop_machinery_load_factor_path, arguments.crop_machinery_vehicle_ratio_path,
                    arguments.crop_machinery_vehicle_units_path, arguments.crop_machinery_vehicle_workhours_path,
                    arguments.crop_machinery_vehicle_power_path, arguments.crop_machinery_nuts3)

            elif sector == 'residential' and comm_world.Get_rank() in sector_procs:
                from hermesv3_bu.sectors.residential_sector import ResidentialSector
                self.sector = ResidentialSector(
                    comm_world.Split(color, sector_procs.index(comm_world.Get_rank())), self.__logger,
                    arguments.auxiliary_files_path, grid, clip, date_array,
                    arguments.residential_source_pollutants, grid.vertical_desctiption, arguments.fuel_list,
                    arguments.nuts3_shapefile, arguments.nuts2_shapefile, arguments.population_density_map,
                    arguments.population_type_map, arguments.population_type_nuts2, arguments.population_type_nuts3,
                    arguments.energy_consumption_nuts3, arguments.energy_consumption_nuts2,
                    arguments.residential_spatial_proxies, arguments.residential_ef_files_path,
                    arguments.residential_heating_degree_day_path, arguments.temperature_daily_files_path,
                    arguments.residential_hourly_profiles, arguments.speciation_map,
                    arguments.residential_speciation_profiles, arguments.molecular_weights)

            elif sector == 'recreational_boats' and comm_world.Get_rank() in sector_procs:
                from hermesv3_bu.sectors.recreational_boats_sector import RecreationalBoatsSector
                self.sector = RecreationalBoatsSector(
                    comm_world.Split(color, sector_procs.index(comm_world.Get_rank())), self.__logger,
                    arguments.auxiliary_files_path, grid, clip, date_array,
                    arguments.recreational_boats_source_pollutants, grid.vertical_desctiption,
                    arguments.recreational_boats_list, arguments.recreational_boats_density_map,
                    arguments.recreational_boats_by_type, arguments.recreational_boats_ef_path,
                    arguments.recreational_boats_monthly_profiles, arguments.recreational_boats_weekly_profiles,
                    arguments.recreational_boats_hourly_profiles, arguments.speciation_map,
                    arguments.recreational_boats_speciation_profiles, arguments.molecular_weights)

            elif sector == 'point_sources' and comm_world.Get_rank() in sector_procs:
                from hermesv3_bu.sectors.point_source_sector import PointSourceSector
                if arguments.plume_rise_output and arguments.plume_rise:
                    plume_rise_out_file = arguments.output_name.replace('.nc', '_ps_plumerise.csv')
                else:
                    plume_rise_out_file = None
                self.sector = PointSourceSector(
                    comm_world.Split(color, sector_procs.index(comm_world.Get_rank())), self.__logger,
                    arguments.auxiliary_files_path, grid, clip, date_array,
                    arguments.point_source_pollutants, grid.vertical_desctiption,
                    arguments.point_source_catalog, arguments.point_source_monthly_profiles,
                    arguments.point_source_weekly_profiles, arguments.point_source_hourly_profiles,
                    arguments.speciation_map, arguments.point_source_speciation_profiles, arguments.point_source_snaps,
                    arguments.point_source_measured_emissions, arguments.molecular_weights,
                    plume_rise=arguments.plume_rise, plume_rise_filename=plume_rise_out_file, plume_rise_pahts={
                        'friction_velocity_path': arguments.friction_velocity_path,
                        'pblh_path': arguments.pblh_path,
                        'obukhov_length_path': arguments.obukhov_length_path,
                        'layer_thickness_path': arguments.layer_thickness_path,
                        'temperature_sfc_path': arguments.temperature_sfc_path,
                        'temperature_4d_path': arguments.temperature_4d_path,
                        'u10_wind_speed_path': arguments.u10_wind_speed_path,
                        'v10_wind_speed_path': arguments.v10_wind_speed_path,
                        'u_wind_speed_4d_path': arguments.u_wind_speed_4d_path,
                        'v_wind_speed_4d_path': arguments.v_wind_speed_4d_path})
            elif sector == 'traffic' and comm_world.Get_rank() in sector_procs:
                from hermesv3_bu.sectors.traffic_sector import TrafficSector
                self.sector = TrafficSector(
                    comm_world.Split(color, sector_procs.index(comm_world.Get_rank())), self.__logger,
                    arguments.auxiliary_files_path, grid, clip, date_array, arguments.traffic_pollutants,
                    grid.vertical_desctiption, arguments.road_link_path, arguments.fleet_compo_path,
                    arguments.traffic_speed_hourly_path, arguments.traffic_monthly_profiles,
                    arguments.traffic_weekly_profiles, arguments.traffic_hourly_profiles_mean,
                    arguments.traffic_hourly_profiles_weekday, arguments.traffic_hourly_profiles_saturday,
                    arguments.traffic_hourly_profiles_sunday, arguments.traffic_ef_path, arguments.vehicle_types,
                    arguments.load, arguments.speciation_map, arguments.traffic_speciation_profile_hot_cold,
                    arguments.traffic_speciation_profile_tyre, arguments.traffic_speciation_profile_road,
                    arguments.traffic_speciation_profile_brake, arguments.traffic_speciation_profile_resuspension,
                    arguments.temperature_hourly_files_path, arguments.output_dir, arguments.molecular_weights,
                    arguments.resuspension_correction, arguments.precipitation_files_path, arguments.do_hot,
                    arguments.do_cold, arguments.do_tyre_wear, arguments.do_brake_wear, arguments.do_road_wear,
                    arguments.do_resuspension, arguments.write_rline, arguments.traffic_scenario)

            elif sector == 'traffic_area' and comm_world.Get_rank() in sector_procs:
                from hermesv3_bu.sectors.traffic_area_sector import TrafficAreaSector
                self.sector = TrafficAreaSector(
                    comm_world.Split(color, sector_procs.index(comm_world.Get_rank())), self.__logger,
                    arguments.auxiliary_files_path, grid, clip, date_array, arguments.traffic_area_pollutants,
                    grid.vertical_desctiption, arguments.population_density_map, arguments.speciation_map,
                    arguments.molecular_weights, arguments.do_evaporative, arguments.traffic_area_gas_path,
                    arguments.population_nuts3, arguments.nuts3_shapefile,
                    arguments.traffic_area_speciation_profiles_evaporative, arguments.traffic_area_evaporative_ef_file,
                    arguments.temperature_hourly_files_path, arguments.do_small_cities,
                    arguments.traffic_area_small_cities_path, arguments.traffic_area_speciation_profiles_small_cities,
                    arguments.traffic_area_small_cities_ef_file, arguments.small_cities_monthly_profile,
                    arguments.small_cities_weekly_profile, arguments.small_cities_hourly_profile
                )
            elif sector == 'solvents' and comm_world.Get_rank() in sector_procs:
                from hermesv3_bu.sectors.solvents_sector import SolventsSector
                self.sector = SolventsSector(
                    comm_world.Split(color, sector_procs.index(comm_world.Get_rank())), self.__logger,
                    arguments.auxiliary_files_path, grid, clip, date_array, arguments.solvents_pollutants,
                    grid.vertical_desctiption, arguments.speciation_map, arguments.molecular_weights,
                    arguments.solvents_speciation_profiles, arguments.solvents_monthly_profile,
                    arguments.solvents_weekly_profile, arguments.solvents_hourly_profile,
                    arguments.solvents_proxies_path, arguments.solvents_yearly_emissions_by_nut2_path,
                    arguments.solvents_point_sources_shapefile, arguments.solvents_point_sources_weight_by_nut2_path,
                    arguments.population_density_map, arguments.population_nuts2, arguments.land_uses_path,
                    arguments.land_uses_nuts2_path, arguments.nuts2_shapefile)

            color += 1

        self.__logger.write_time_log('SectorManager', '__init__', timeit.default_timer() - spent_time)

    def run(self):
        spent_time = timeit.default_timer()
        emis = self.sector.calculate_emissions()
        self.__logger.write_time_log('SectorManager', 'run', timeit.default_timer() - spent_time)
        return emis

    def make_sector_list(self, arguments, max_procs):
        spent_time = timeit.default_timer()
        sector_dict = {}
        accum = 0
        for sector in SECTOR_LIST:
            if vars(arguments)['do_{0}'.format(sector)]:
                n_procs = vars(arguments)['{0}_processors'.format(sector)]
                sector_dict[sector] = [accum + x for x in range(n_procs)]
                accum += n_procs
        if accum != max_procs:
            error_exit("The selected number of processors '{0}' does not fit ".format(max_procs) +
                       "with the sum of processors dedicated for all the sectors " +
                       "'{0}': {1}".format(
                           accum, {sector: len(sector_procs) for sector, sector_procs in sector_dict.items()}))

        self.__logger.write_time_log('SectorManager', 'make_sector_list', timeit.default_timer() - spent_time)
        return sector_dict
