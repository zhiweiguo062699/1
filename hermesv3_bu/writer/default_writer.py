#!/usr/bin/env python

import os
import numpy as np
from netCDF4 import Dataset, date2num
from hermesv3_bu.writer.writer import Writer
from mpi4py import MPI


class DefaultWriter(Writer):
    def __init__(self, comm_wolrd, comm_write, netcdf_path, grid, date_array, pollutant_info, rank_distribution):
        super(DefaultWriter, self).__init__(comm_wolrd, comm_write, netcdf_path, grid, date_array, pollutant_info,
                                            rank_distribution)

        print netcdf_path

    def unit_change(self, emissions):

        return emissions

    def write_netcdf(self, emissions):
        from cf_units import Unit

        netcdf = Dataset(self.netcdf_path, mode='w', parallel=True, comm=self.comm_write, info=MPI.Info())

        # ========== DIMENSIONS ==========
        print self.grid.grid_type
        if self.grid.grid_type == 'Regular Lat-Lon':
            netcdf.createDimension('lat', self.grid.center_latitudes.shape[0])
            netcdf.createDimension('lon', self.grid.center_longitudes.shape[0])
            var_dim = ('lat', 'lon')

        elif self.grid.grid_type in ['Lambert Conformal Conic', 'Mercator']:
            netcdf.createDimension('y', len(self.grid.y))
            netcdf.createDimension('x', len(self.grid.x))
            var_dim = ('y', 'x')
        elif self.grid.grid_type == 'Rotated':
            netcdf.createDimension('rlat', len(self.grid.rlat))
            netcdf.createDimension('rlon', len(self.grid.rlon))
            var_dim = ('rlat', 'rlon')

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

        lats = netcdf.createVariable('lat', np.float64, var_dim)
        lats.units = "degrees_north"
        lats.axis = "Y"
        lats.long_name = "latitude coordinate"
        lats.standard_name = "latitude"
        lats[:] = self.grid.center_latitudes
        lats.bounds = "lat_bnds"
        lat_bnds = netcdf.createVariable('lat_bnds', np.float64, var_dim + ('nv',))
        lat_bnds[:] = self.grid.boundary_latitudes

        lons = netcdf.createVariable('lon', np.float64, var_dim)
        lons.units = "degrees_east"
        lons.axis = "X"
        lons.long_name = "longitude coordinate"
        lons.standard_name = "longitude"
        lons[:] = self.grid.center_longitudes
        lons.bounds = "lon_bnds"
        lon_bnds = netcdf.createVariable('lon_bnds', np.float64, var_dim + ('nv',))
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
            var = netcdf.createVariable(var_name, np.float64, ('time', 'lev',) + var_dim, chunksizes=self.rank_distribution[0]['shape'])
            var[:, :,
            self.rank_distribution[self.comm_write.Get_rank()]['y_min']:
            self.rank_distribution[self.comm_write.Get_rank()]['y_max'],
            self.rank_distribution[self.comm_write.Get_rank()]['x_min']:
            self.rank_distribution[self.comm_write.Get_rank()]['x_max']] = var_data

        # ========== METADATA ==========
        if self.grid.grid_type == 'Regular Lat-Lon':
            mapping = netcdf.createVariable('crs', 'i')
            mapping.grid_mapping_name = "latitude_longitude"
            mapping.semi_major_axis = 6371000.0
            mapping.inverse_flattening = 0

        elif self.grid.grid_type == 'Lambert Conformal Conic':
            mapping = netcdf.createVariable('Lambert_conformal', 'i')
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

        netcdf.close()
        print self.netcdf_path


