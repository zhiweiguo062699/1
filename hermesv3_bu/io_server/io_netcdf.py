#!/usr/bin/env python

import sys
import os
from mpi4py import MPI
from datetime import date
import numpy as np
import geopandas as gpd
from netCDF4 import Dataset
from shapely.geometry import Point
from cf_units import num2date, CALENDAR_STANDARD
from geopandas import GeoDataFrame
from pandas import DataFrame

from hermesv3_bu.io_server.io_server import IoServer
from hermesv3_bu.tools.checker import check_files, error_exit


class IoNetcdf(IoServer):
    def __init__(self, comm):
        if comm is None:
            comm = MPI.COMM_WORLD
        super(IoNetcdf, self).__init__(comm)

    def get_data_from_netcdf(self, netcdf_path, var_name, date_type, date, geometry_df):
        """
        Read for extract a NetCDF variable in the desired points.

        :param netcdf_path: Path to the NetCDF that contains the data to extract.
        :type netcdf_path: str

        :param var_name: Name of the NetCDF variable to extract.
        :type var_name: str

        :param date_type: Option to set if we want to extract a 'daily' variable or a 'yearly' one.
        :type date_type: str

        :param date: Date of the day to extract.
        :type date: datetime.date

        :param geometry_df: GeoDataframe with the point where extract the variables.
        :type geometry_df: geopandas.GeoDataframe

        :return: GeoDataframe with the data in the desired points.
        :rtype: geopandas.GeoDataframe
        """
        check_files(netcdf_path)
        nc = Dataset(netcdf_path, mode='r')
        try:
            lat_o = nc.variables['latitude'][:]
            lon_o = nc.variables['longitude'][:]
        except KeyError as e:
            error_exit("{0} variable not found in {1} file.".format(str(e), netcdf_path))

        if date_type == 'daily':
            try:
                time = nc.variables['time']
            except KeyError as e:
                error_exit("{0} variable not found in {1} file.".format(str(e), netcdf_path))
            # From time array to list of dates.
            time_array = num2date(time[:], time.units, CALENDAR_STANDARD)
            time_array = np.array([aux.date() for aux in time_array])
            i_time = np.where(time_array == date)[0][0]
        elif date_type == 'yearly':
            i_time = 0

        # Find the index to read all the necessary information but avoiding to read as many unused data as we can
        i_min, i_max, j_min, j_max = self.find_lonlat_index(
            lon_o, lat_o, geometry_df['c_lon'].min(), geometry_df['c_lon'].max(),
            geometry_df['c_lat'].min(), geometry_df['c_lat'].max())

        # Clips the lat lons
        lon_o = lon_o[i_min:i_max]
        lat_o = lat_o[j_min:j_max]

        # From 1D to 2D
        lat = np.array([lat_o[:]] * len(lon_o[:])).T.flatten()
        lon = np.array([lon_o[:]] * len(lat_o[:])).flatten()
        del lat_o, lon_o

        # Reads the tas variable of the xone and the times needed.
        try:
            var = nc.variables[var_name][i_time, j_min:j_max, i_min:i_max]
        except KeyError as e:
            error_exit("{0} variable not found in {1} file.".format(str(e), netcdf_path))
        nc.close()

        var_df = gpd.GeoDataFrame(var.flatten().T, columns=[var_name], crs={'init': 'epsg:4326'},
                                  geometry=[Point(xy) for xy in zip(lon, lat)])
        var_df.loc[:, 'REC'] = var_df.index

        return var_df

    def get_hourly_data_from_netcdf(self, lon_min, lon_max, lat_min, lat_max, netcdf_dir, var_name, date_array):
        """
        Reads the temperature from the ERA5 var value.
        It will return only the involved cells of the NetCDF in DataFrame format.

        To clip the global NetCDF to the desired region it is needed the minimum and maximum value of the latitudes and
        longitudes of the centroids of all the road links.

        :param lon_min: Minimum longitude of the centroid of the road links.
        :type lon_min: float

        :param lon_max: Maximum longitude of the centroid of the road links.
        :type lon_max: float

        :param lat_min: Minimum latitude of the centroid of the road links.
        :type lat_min: float

        :param lat_max: Maximum latitude of the centroid of the road links.
        :type lat_max: float

        :return: Temperature, centroid of the cell and cell identificator (REC).
            Each time step is each column with the name t_<timestep>.
        :rtype: GeoDataFrame
        """
        path = os.path.join(netcdf_dir, '{0}_{1}{2}.nc'.format(var_name, date_array[0].year,
                                                               str(date_array[0].month).zfill(2)))
        # self.logger.write_log('Getting temperature from {0}'.format(path), message_level=2)
        check_files(path)
        nc = Dataset(path, mode='r')
        try:
            lat_o = nc.variables['latitude'][:]
            lon_o = nc.variables['longitude'][:]
            n_lat = len(lat_o)
            time = nc.variables['time']
        except KeyError as e:
            error_exit("{0} variable not found in {1} file.".format(str(e), path))
        # From time array to list of dates.
        time_array = num2date(time[:], time.units,  CALENDAR_STANDARD)
        i_time = np.where(time_array == date_array[0])[0][0]

        # Correction to set the longitudes from -180 to 180 instead of from 0 to 360.
        if lon_o.max() > 180:
            lon_o[lon_o > 180] -= 360

        # Finds the array positions for the clip.
        i_min, i_max, j_min, j_max = self.find_lonlat_index(lon_o, lat_o, lon_min, lon_max, lat_min, lat_max)

        # Clips the lat lons
        lon_o = lon_o[i_min:i_max]
        lat_o = lat_o[j_min:j_max]

        # From 1D to 2D
        lat = np.array([lat_o[:]] * len(lon_o[:])).T.flatten()
        lon = np.array([lon_o[:]] * len(lat_o[:])).flatten()
        # del lat_o, lon_o

        # Reads the var variable of the xone and the times needed.
        try:
            var = nc.variables[var_name][i_time:i_time + (len(date_array)), j_min:j_max, i_min:i_max]
        except KeyError as e:
            error_exit("{0} variable not found in {1} file.".format(str(e), path))

        nc.close()
        # That condition is fot the cases that the needed temperature is in a different NetCDF.
        while len(var) < len(date_array):
            aux_date = date_array[len(var) + 1]
            path = os.path.join(netcdf_dir, '{0}_{1}{2}.nc'.format(var_name, aux_date.year,
                                                                   str(aux_date.month).zfill(2)))
            # self.logger.write_log('Getting {0} from {1}'.format(var_name, path), message_level=2)
            check_files(path)
            nc = Dataset(path, mode='r')
            i_time = 0
            try:
                new_var = nc.variables[var_name][i_time:i_time + (len(date_array) - len(var)), j_min:j_max, i_min:i_max]
            except KeyError as e:
                error_exit("{0} variable not found in {1} file.".format(str(e), path))

            var = np.concatenate([var, new_var])

            nc.close()

        var = var.reshape((var.shape[0], var.shape[1] * var.shape[2]))
        df = gpd.GeoDataFrame(var.T, geometry=[Point(xy) for xy in zip(lon, lat)])
        # df.columns = ['t_{0}'.format(x) for x in df.columns.values[:-1]] + ['geometry']
        df.loc[:, 'REC'] = (((df.index // len(lon_o)) + j_min) * n_lat) + ((df.index % len(lon_o)) + i_min)

        return df

    @staticmethod
    def find_lonlat_index(lon, lat, lon_min, lon_max, lat_min, lat_max):
        """
        Find the NetCDF index to extract all the data avoiding the maximum of unused data.

        :param lon: Longitudes array from the NetCDF.
        :type lon: numpy.array

        :param lat: Latitude array from the NetCDF.
        :type lat: numpy.array

        :param lon_min: Minimum longitude of the point for the needed date.
        :type lon_min float

        :param lon_max: Maximum longitude of the point for the needed date.
        :type lon_max: float

        :param lat_min: Minimum latitude of the point for the needed date.
        :type lat_min: float

        :param lat_max: Maximum latitude of the point for the needed date.
        :type lat_max: float

        :return: Tuple with the four index of the NetCDF
        :rtype: tuple
        """
        import numpy as np

        aux = lon - lon_min
        aux[aux > 0] = np.nan
        i_min = np.where(aux == np.nanmax(aux))[0][0]

        aux = lon - lon_max
        aux[aux < 0] = np.nan
        i_max = np.where(aux == np.nanmin(aux))[0][0]

        aux = lat - lat_min
        aux[aux > 0] = np.nan
        j_max = np.where(aux == np.nanmax(aux))[0][0]

        aux = lat - lat_max
        aux[aux < 0] = np.nan
        j_min = np.where(aux == np.nanmin(aux))[0][0]

        return i_min, i_max + 1, j_min, j_max + 1

    @staticmethod
    def _parse_wrf_path(path, day):
        """
        Parse the path adding the date in the correct format.

        :param path: Path to the file.
        :type path: str

        :param day: Date of the day
        :return: date
        """
        path = path.replace('<YYYYMMDD>', day.strftime('%Y%m%d'))
        path = path.replace('<YYYYJJJ>', day.strftime('%Y%j'))

        return path

    def get_data_from_wrf(self, path, var_list, day, type, geometry):
        """

        :param path:
        :param var_list:
        :param day:
        :param type:
        :param geometry:
        :return:
        :rtype: DataFrame
        """
        path = self._parse_wrf_path(path, day)

        wrf_nc = Dataset(path, mode='r')
        for var_name in var_list:
            if type == 'daily':
                var = np.mean(wrf_nc.variables[var_name][:24, 0, :], axis=0)
                var = var.flatten()
            else:
                var = None

            geometry[var_name] = var[geometry.index]
        wrf_nc.close()

        return geometry


def write_coords_netcdf(netcdf_path, center_latitudes, center_longitudes, data_list, levels=None, date=None, hours=None,
                        boundary_latitudes=None, boundary_longitudes=None, cell_area=None, global_attributes=None,
                        regular_latlon=False,
                        rotated=False, rotated_lats=None, rotated_lons=None, north_pole_lat=None, north_pole_lon=None,
                        lcc=False, lcc_x=None, lcc_y=None, lat_1_2=None, lon_0=None, lat_0=None,
                        mercator=False, lat_ts=None):

    from netCDF4 import Dataset
    from cf_units import Unit, encode_time

    if not (regular_latlon or lcc or rotated or mercator):
        regular_latlon = True
    netcdf = Dataset(netcdf_path, mode='w', format="NETCDF4")

    # ===== Dimensions =====
    if regular_latlon:
        var_dim = ('lat', 'lon',)

        # Latitude
        if len(center_latitudes.shape) == 1:
            netcdf.createDimension('lat', center_latitudes.shape[0])
            lat_dim = ('lat',)
        elif len(center_latitudes.shape) == 2:
            netcdf.createDimension('lat', center_latitudes.shape[0])
            lat_dim = ('lon', 'lat', )
        else:
            print('ERROR: Latitudes must be on a 1D or 2D array instead of {0}'.format(len(center_latitudes.shape)))
            sys.exit(1)

        # Longitude
        if len(center_longitudes.shape) == 1:
            netcdf.createDimension('lon', center_longitudes.shape[0])
            lon_dim = ('lon',)
        elif len(center_longitudes.shape) == 2:
            netcdf.createDimension('lon', center_longitudes.shape[1])
            lon_dim = ('lon', 'lat', )
        else:
            print('ERROR: Longitudes must be on a 1D or 2D array instead of {0}'.format(len(center_longitudes.shape)))
            sys.exit(1)
    elif rotated:
        var_dim = ('rlat', 'rlon',)

        # Rotated Latitude
        if rotated_lats is None:
            print('ERROR: For rotated grids is needed the rotated latitudes.')
            sys.exit(1)
        netcdf.createDimension('rlat', len(rotated_lats))
        lat_dim = ('rlat', 'rlon',)

        # Rotated Longitude
        if rotated_lons is None:
            print('ERROR: For rotated grids is needed the rotated longitudes.')
            sys.exit(1)
        netcdf.createDimension('rlon', len(rotated_lons))
        lon_dim = ('rlat', 'rlon',)
    elif lcc or mercator:
        var_dim = ('y', 'x',)

        netcdf.createDimension('y', len(lcc_y))
        lat_dim = ('y', 'x', )

        netcdf.createDimension('x', len(lcc_x))
        lon_dim = ('y', 'x', )
    else:
        lat_dim = None
        lon_dim = None
        var_dim = None

    # Levels
    if levels is not None:
        netcdf.createDimension('lev', len(levels))

    # Bounds
    if boundary_latitudes is not None:
        try:
            netcdf.createDimension('nv', len(boundary_latitudes[0, 0]))
        except TypeError:
            netcdf.createDimension('nv', boundary_latitudes.shape[1])

    # Time
    netcdf.createDimension('time', None)

    # ===== Variables =====
    # Time
    if date is None:
        time = netcdf.createVariable('time', 'd', ('time',), zlib=True)
        time.units = "months since 2000-01-01 00:00:00"
        time.standard_name = "time"
        time.calendar = "gregorian"
        time.long_name = "time"
        time[:] = [0.]
    else:
        time = netcdf.createVariable('time', 'd', ('time',), zlib=True)
        u = Unit('hours')
        # Unit('hour since 1970-01-01 00:00:00.0000000 UTC')
        time.units = str(u.offset_by_time(encode_time(date.year, date.month, date.day, date.hour, date.minute,
                                                      date.second)))
        time.standard_name = "time"
        time.calendar = "gregorian"
        time.long_name = "time"
        time[:] = hours

    # Latitude
    lats = netcdf.createVariable('lat', 'f', lat_dim, zlib=True)
    lats.units = "degrees_north"
    lats.axis = "Y"
    lats.long_name = "latitude coordinate"
    lats.standard_name = "latitude"
    lats[:] = center_latitudes

    if boundary_latitudes is not None:
        lats.bounds = "lat_bnds"
        lat_bnds = netcdf.createVariable('lat_bnds', 'f', lat_dim + ('nv',), zlib=True)
        lat_bnds[:] = boundary_latitudes

    # Longitude
    lons = netcdf.createVariable('lon', 'f', lon_dim, zlib=True)

    lons.units = "degrees_east"
    lons.axis = "X"
    lons.long_name = "longitude coordinate"
    lons.standard_name = "longitude"
    lons[:] = center_longitudes
    if boundary_longitudes is not None:
        lons.bounds = "lon_bnds"
        lon_bnds = netcdf.createVariable('lon_bnds', 'f', lon_dim + ('nv',), zlib=True)
        lon_bnds[:] = boundary_longitudes

    if rotated:
        # Rotated Latitude
        rlat = netcdf.createVariable('rlat', 'f', ('rlat',), zlib=True)
        rlat.long_name = "latitude in rotated pole grid"
        rlat.units = Unit("degrees").symbol
        rlat.standard_name = "grid_latitude"
        rlat[:] = rotated_lats

        # Rotated Longitude
        rlon = netcdf.createVariable('rlon', 'f', ('rlon',), zlib=True)
        rlon.long_name = "longitude in rotated pole grid"
        rlon.units = Unit("degrees").symbol
        rlon.standard_name = "grid_longitude"
        rlon[:] = rotated_lons
    if lcc or mercator:
        x = netcdf.createVariable('x', 'd', ('x',), zlib=True)
        x.units = Unit("km").symbol
        x.long_name = "x coordinate of projection"
        x.standard_name = "projection_x_coordinate"
        x[:] = lcc_x

        y = netcdf.createVariable('y', 'd', ('y',), zlib=True)
        y.units = Unit("km").symbol
        y.long_name = "y coordinate of projection"
        y.standard_name = "projection_y_coordinate"
        y[:] = lcc_y

    cell_area_dim = var_dim
    # Levels
    if levels is not None:
        var_dim = ('lev',) + var_dim
        lev = netcdf.createVariable('lev', 'f', ('lev',), zlib=True)
        lev.units = Unit("m").symbol
        lev.positive = 'up'
        lev[:] = levels

    # All variables
    if len(data_list) is 0:
        var = netcdf.createVariable('aux_var', 'f', ('time',) + var_dim, zlib=True)
        var[:] = 0
    for variable in data_list:
        var = netcdf.createVariable(variable['name'], 'f', ('time',) + var_dim, zlib=True)
        var.units = Unit(variable['units']).symbol
        if 'long_name' in variable:
            var.long_name = str(variable['long_name'])
        if 'standard_name' in variable:
            var.standard_name = str(variable['standard_name'])
        if 'cell_method' in variable:
            var.cell_method = str(variable['cell_method'])
        var.coordinates = "lat lon"
        if cell_area is not None:
            var.cell_measures = 'area: cell_area'
        if regular_latlon:
            var.grid_mapping = 'crs'
        elif rotated:
            var.grid_mapping = 'rotated_pole'
        elif lcc:
            var.grid_mapping = 'Lambert_conformal'
        elif mercator:
            var.grid_mapping = 'mercator'
        try:
            var[:] = variable['data']
        except ValueError:
            print('VAR ERROR, netcdf shape: {0}, variable shape: {1}'.format(var[:].shape, variable['data'].shape))

    # Grid mapping
    if regular_latlon:
        # CRS
        mapping = netcdf.createVariable('crs', 'i')
        mapping.grid_mapping_name = "latitude_longitude"
        mapping.semi_major_axis = 6371000.0
        mapping.inverse_flattening = 0
    elif rotated:
        # Rotated pole
        mapping = netcdf.createVariable('rotated_pole', 'c')
        mapping.grid_mapping_name = 'rotated_latitude_longitude'
        mapping.grid_north_pole_latitude = 90 - north_pole_lat
        mapping.grid_north_pole_longitude = north_pole_lon
    elif lcc:
        # CRS
        mapping = netcdf.createVariable('Lambert_conformal', 'i')
        mapping.grid_mapping_name = "lambert_conformal_conic"
        mapping.standard_parallel = lat_1_2
        mapping.longitude_of_central_meridian = lon_0
        mapping.latitude_of_projection_origin = lat_0
    elif mercator:
        # Mercator
        mapping = netcdf.createVariable('mercator', 'i')
        mapping.grid_mapping_name = "mercator"
        mapping.longitude_of_projection_origin = lon_0
        mapping.standard_parallel = lat_ts

    # Cell area
    if cell_area is not None:
        c_area = netcdf.createVariable('cell_area', 'f', cell_area_dim)
        c_area.long_name = "area of the grid cell"
        c_area.standard_name = "cell_area"
        c_area.units = Unit("m2").symbol
        c_area[:] = cell_area

    if global_attributes is not None:
        netcdf.setncatts(global_attributes)

    netcdf.close()
