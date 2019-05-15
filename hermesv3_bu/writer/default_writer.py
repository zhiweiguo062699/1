#!/usr/bin/env python

import numpy as np
from netCDF4 import Dataset, date2num
from hermesv3_bu.writer.writer import Writer
from mpi4py import MPI


class DefaultWriter(Writer):
    def __init__(self, comm_wolrd, comm_write, netcdf_path, grid, date_array, pollutant_info, rank_distribution):
        """
        Initilise the Default writer that will write a NetCDF CF-1.6 complient.

        :param comm_wolrd: Global communicator for all the calculation process
        :type comm_wolrd: MPI.COMM

        :param comm_write: Sector communicator.
        :type comm_write: MPI.Intracomm

        :param netcdf_path: Path to the output NetCDF file-
        :type netcdf_path: str

        :param grid: Output grid definition.
        :type grid: hermesv3_bu.grids.grid.Grid

        :param date_array: Array with each time step to be calculated.
        :type date_array: list of datetime.datetime

        :param pollutant_info: Information related with the output pollutants, short description, units...
        :type pollutant_info: pandas.DataFrame

        :param rank_distribution: Information of the writing process. That argument is a dictionary with the writing
            process rank as key and another dictionary as value. That other dictionary contains:
            - shape: Shape to write
            - x_min: X minimum position to write on the full array.
            - x_max: X maximum position to write on the full array.
            - y_min: Y minimum position to write on the full array.
            - y_max: Y maximum position to write on the full array.
            - fid_min: Minimum cell ID of a flatten X Y domain.
            - fid_max: Maximum cell ID of a flatten X Y domain.

            e.g. 24 time steps. 48 vertical levels, 10 x 10
            {0: {'fid_min': 0, 'y_min': 0, 'y_max': 5, 'fid_max': 50, 'shape': (24, 48, 5, 10), 'x_max': 10,
                'x_min': 0},
            1: {'fid_min': 50, 'y_min': 5, 'y_max': 10, 'fid_max': 100, 'shape': (24, 48, 5, 10), 'x_max': 10,
                'x_min': 0}}
        :type rank_distribution: dict
        """
        super(DefaultWriter, self).__init__(comm_wolrd, comm_write, netcdf_path, grid, date_array, pollutant_info,
                                            rank_distribution)

    def unit_change(self, emissions):
        """
        No unit changes.

        :param emissions: Emissions on dataframe.
        :type emissions: pandas.DataFrame

        :return: Same emissions as input
        :rtype: pandas.DataFrame
        """

        return emissions

    def write_netcdf(self, emissions):
        """
        Create a NetCDF following the CF-1.6 conventions

        :param emissions: Emissions to write in the NetCDF with 'FID, level & time step as index and pollutant as
            columns.
        :type emissions: pandas.DataFrame
        """
        from cf_units import Unit

        netcdf = Dataset(self.netcdf_path, mode='w', parallel=True, comm=self.comm_write, info=MPI.Info())

        # ========== DIMENSIONS ==========
        if self.grid.grid_type == 'Regular Lat-Lon':
            netcdf.createDimension('lat', self.grid.center_latitudes.shape[0])
            netcdf.createDimension('lon', self.grid.center_longitudes.shape[0])
            var_dim = ('lat', 'lon',)
            lat_dim = ('lat',)
            lon_dim = ('lon',)

        elif self.grid.grid_type in ['Lambert Conformal Conic', 'Mercator']:
            netcdf.createDimension('y', len(self.grid.y))
            netcdf.createDimension('x', len(self.grid.x))
            var_dim = ('y', 'x',)
            lat_dim = lon_dim = var_dim

        elif self.grid.grid_type == 'Rotated':
            netcdf.createDimension('rlat', len(self.grid.rlat))
            netcdf.createDimension('rlon', len(self.grid.rlon))
            var_dim = ('rlat', 'rlon')
            lat_dim = lon_dim = var_dim
        else:
            var_dim = lat_dim = None

        netcdf.createDimension('nv', len(self.grid.boundary_latitudes[0, 0]))

        netcdf.createDimension('lev', len(self.grid.vertical_desctiption))
        netcdf.createDimension('time', len(self.date_array))

        # ========== VARIABLES ==========
        time = netcdf.createVariable('time', np.float64, ('time',))
        time.units = 'hours since {0}'.format(self.date_array[0].strftime("%Y-%m-%d %H:%M:%S"))
        time.standard_name = "time"
        time.calendar = "gregorian"
        time.long_name = "time"
        time[:] = date2num(self.date_array, time.units, calendar=time.calendar)

        lev = netcdf.createVariable('lev', np.float64, ('lev',))
        lev.units = Unit("m").symbol
        lev.positive = 'up'
        lev[:] = self.grid.vertical_desctiption

        lats = netcdf.createVariable('lat', np.float64, lat_dim)
        lats.units = "degrees_north"
        lats.axis = "Y"
        lats.long_name = "latitude coordinate"
        lats.standard_name = "latitude"
        lats[:] = self.grid.center_latitudes
        lats.bounds = "lat_bnds"
        lat_bnds = netcdf.createVariable('lat_bnds', np.float64, lat_dim + ('nv',))
        lat_bnds[:] = self.grid.boundary_latitudes

        lons = netcdf.createVariable('lon', np.float64, lon_dim)
        lons.units = "degrees_east"
        lons.axis = "X"
        lons.long_name = "longitude coordinate"
        lons.standard_name = "longitude"
        lons[:] = self.grid.center_longitudes
        lons.bounds = "lon_bnds"
        lon_bnds = netcdf.createVariable('lon_bnds', np.float64, lon_dim + ('nv',))
        lon_bnds[:] = self.grid.boundary_longitudes

        if self.grid.grid_type in ['Lambert Conformal Conic', 'Mercator']:
            x_var = netcdf.createVariable('x', np.float64, ('x',))
            x_var.units = Unit("km").symbol
            x_var.long_name = "x coordinate of projection"
            x_var.standard_name = "projection_x_coordinate"
            x_var[:] = self.grid.x

            y_var = netcdf.createVariable('y', np.float64, ('y',))
            y_var.units = Unit("km").symbol
            y_var.long_name = "y coordinate of projection"
            y_var.standard_name = "projection_y_coordinate"
            y_var[:] = self.grid.y

        elif self.grid.grid_type == 'Rotated':
            rlat = netcdf.createVariable('rlat', np.float64, ('rlat',))
            rlat.long_name = "latitude in rotated pole grid"
            rlat.units = Unit("degrees").symbol
            rlat.standard_name = "grid_latitude"
            rlat[:] = self.grid.rlat

            # Rotated Longitude
            rlon = netcdf.createVariable('rlon', np.float64, ('rlon',))
            rlon.long_name = "longitude in rotated pole grid"
            rlon.units = Unit("degrees").symbol
            rlon.standard_name = "grid_longitude"
            rlon[:] = self.grid.rlon

        # ========== POLLUTANTS ==========
        for var_name in emissions.columns.values:
            var_data = self.dataframe_to_array(emissions.loc[:, [var_name]])
            # var = netcdf.createVariable(var_name, np.float64, ('time', 'lev',) + var_dim,
            #                             chunksizes=self.rank_distribution[0]['shape'])
            var = netcdf.createVariable(var_name, np.float64, ('time', 'lev',) + var_dim)

            var[:, :,
                self.rank_distribution[self.comm_write.Get_rank()]['y_min']:
                self.rank_distribution[self.comm_write.Get_rank()]['y_max'],
                self.rank_distribution[self.comm_write.Get_rank()]['x_min']:
                self.rank_distribution[self.comm_write.Get_rank()]['x_max']] = var_data

            var.long_name = self.pollutant_info.loc[var_name, 'description']
            var.units = self.pollutant_info.loc[var_name, 'units']
            var.missing_value = -999.0
            var.coordinates = 'lat lon'
            if self.grid.grid_type == 'Regular Lat-Lon':
                var.grid_mapping = 'Latitude_Longitude'
            elif self.grid.grid_type == 'Lambert Conformal Conic':
                var.grid_mapping = 'Lambert_Conformal'
            elif self.grid.grid_type == 'Rotated':
                var.grid_mapping = 'rotated_pole'
            elif self.grid.grid_type == 'Mercator':
                var.grid_mapping = 'mercator'

        # ========== METADATA ==========
        if self.grid.grid_type == 'Regular Lat-Lon':

            mapping = netcdf.createVariable('Latitude_Longitude', 'i')
            mapping.grid_mapping_name = "latitude_longitude"
            mapping.semi_major_axis = 6371000.0
            mapping.inverse_flattening = 0

        elif self.grid.grid_type == 'Lambert Conformal Conic':
            mapping = netcdf.createVariable('Lambert_Conformal', 'i')
            mapping.grid_mapping_name = "lambert_conformal_conic"
            mapping.standard_parallel = "{0}, {1}".format(self.grid.attributes['lat_1'], self.grid.attributes['lat_2'])
            mapping.longitude_of_central_meridian = self.grid.attributes['lon_0']
            mapping.latitude_of_projection_origin = self.grid.attributes['lat_0']

        elif self.grid.grid_type == 'Rotated':
            mapping = netcdf.createVariable('rotated_pole', 'c')
            mapping.grid_mapping_name = 'rotated_latitude_longitude'
            mapping.grid_north_pole_latitude = 90 - self.grid.attributes['new_pole_latitude_degrees']
            mapping.grid_north_pole_longitude = self.grid.attributes['new_pole_longitude_degrees']

        elif self.grid.grid_type == 'Mercator':
            mapping = netcdf.createVariable('mercator', 'i')
            mapping.grid_mapping_name = "mercator"
            mapping.longitude_of_projection_origin = self.grid.attributes['lon_0']
            mapping.standard_parallel = self.grid.attributes['lat_ts']

        netcdf.setncattr('Conventions', 'CF-1.6')
        netcdf.close()
        return True
