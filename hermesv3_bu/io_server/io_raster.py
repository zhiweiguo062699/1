#!/usr/bin/env python

import sys
import os
import timeit
from warnings import warn
from mpi4py import MPI
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import pandas as pd
import numpy as np


from hermesv3_bu.io_server.io_server import IoServer


class IoRaster(IoServer):
    def __init__(self, comm=None):
        if comm is None:
            comm = MPI.COMM_WORLD
        super(IoRaster, self).__init__(comm)

    def write_raster(self):
        pass

    def read_raster(self):
        pass

    def clip_raster_with_shapefile(self, raster_path, shape_path, clipped_raster_path, rank=0):
        """
        Clip a raster using given shapefile path.

        The clip is performed only by the selected rank process.

        :param raster_path: Path to the raster to clip.
        :type raster_path: str

        :param shape_path: Path to the shapefile with the polygons where clip the input raster.
        :type shape_path: str

        :param clipped_raster_path: Place to store the clipped raster.
        :type clipped_raster_path: str

        :param rank: Rank who have to do the work. Default 0
        :type rank: int

        :return: Path where is stored the clipped raster.
        :rtype: str
        """
        def getFeatures(gdf):
            """
            https://automating-gis-processes.github.io/CSC18/lessons/L6/clipping-raster.html
            Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
            import json
            return [json.loads(gdf.to_json())['features'][0]['geometry']]

        if self.comm.Get_rank() == rank:
            data = rasterio.open(raster_path)
            geo = gpd.read_file(shape_path)
            if len(geo) > 1:
                geo = gpd.GeoDataFrame(geometry=[geo.geometry.unary_union], crs=geo.crs)
            geo = geo.to_crs(crs=data.crs.data)
            coords = getFeatures(geo)

            out_img, out_transform = mask(data, shapes=coords, crop=True)
            out_meta = data.meta.copy()

            out_meta.update({
                "driver": "GTiff",
                "height": out_img.shape[1],
                "width": out_img.shape[2],
                "transform": out_transform,
                "crs": data.crs})
            if not os.path.exists(os.path.dirname(clipped_raster_path)):
                os.makedirs(os.path.dirname(clipped_raster_path))
            dst = rasterio.open(clipped_raster_path, "w", **out_meta)
            dst.write(out_img)
        self.comm.Barrier()

        return clipped_raster_path

    def clip_raster_with_shapefile_poly(self, raster_path, geo, clipped_raster_path, rank=0, nodata=0):
        """
        Clip a raster using given shapefile.

        The clip is performed only by the master (rank 0) process.

        :param raster_path: Path to the raster to clip.
        :type raster_path: str

        :param geo: Shapefile with the polygons where clip the input raster.
        :type geo: geopandas.GeoDataFrame

        :param clipped_raster_path: Place to store the clipped raster.
        :type clipped_raster_path: str

        :param rank: Rank who have to do the work. Default 0
        :type rank: int

        :param nodata: Value for the no data elements. Default 0
        :type nodata: float

        :return: Path where is stored the clipped raster.
        :rtype: str
        """
        def get_features(gdf):
            """
            https://automating-gis-processes.github.io/CSC18/lessons/L6/clipping-raster.html
            Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
            import json
            return [json.loads(gdf.to_json())['features'][0]['geometry']]

        if self.comm.Get_rank() == rank:
            data = rasterio.open(raster_path)

            if len(geo) > 1:
                geo = gpd.GeoDataFrame(geometry=[geo.geometry.unary_union], crs=geo.crs)
            geo = geo.to_crs(crs=data.crs.data)
            coords = get_features(geo)

            out_img, out_transform = mask(data, shapes=coords, crop=True, all_touched=True, nodata=nodata)
            out_meta = data.meta.copy()

            out_meta.update(
                {
                    "driver": "GTiff",
                    "height": out_img.shape[1],
                    "width": out_img.shape[2],
                    "transform": out_transform,
                    "crs": data.crs
                })
            if not os.path.exists(os.path.dirname(clipped_raster_path)):
                os.makedirs(os.path.dirname(clipped_raster_path))
            dst = rasterio.open(clipped_raster_path, "w", **out_meta)
            dst.write(out_img)
        self.comm.Barrier()

        return clipped_raster_path

    def create_bounds(self, coordinates, inc, number_vertices=2, inverse=False):
        """
        Calculate the vertices coordinates.

        :param coordinates: Coordinates in degrees (latitude or longitude)
        :type coordinates: numpy.array

        :param inc: Increment between center values.
        :type inc: float

        :param number_vertices: Non mandatory parameter that informs the number of vertices that must have the
                boundaries (by default 2).
        :type number_vertices: int

        :param inverse: For some grid latitudes.
        :type inverse: bool

        :return: Array with as many elements as vertices for each value of coords.
        :rtype: numpy.array
        """
        spent_time = timeit.default_timer()
        # Create new arrays moving the centers half increment less and more.
        coords_left = coordinates - inc / 2
        coords_right = coordinates + inc / 2

        # Defining the number of corners needed. 2 to regular grids and 4 for irregular ones.
        if number_vertices == 2:
            # Create an array of N arrays of 2 elements to store the floor and the ceil values for each cell
            bound_coords = np.dstack((coords_left, coords_right))
            bound_coords = bound_coords.reshape((len(coordinates), number_vertices))
        elif number_vertices == 4:
            # Create an array of N arrays of 4 elements to store the corner values for each cell
            # It can be stored in clockwise starting form the left-top element, or in inverse mode.
            if inverse:
                bound_coords = np.dstack((coords_left, coords_left, coords_right, coords_right))
            else:
                bound_coords = np.dstack((coords_left, coords_right, coords_right, coords_left))
        else:
            raise ValueError('ERROR: The number of vertices of the boundaries must be 2 or 4.')
        #self.logger.write_time_log('IoRaster', 'create_bounds', timeit.default_timer() - spent_time, 3)
        return bound_coords

    def to_shapefile(self, raster_path, out_path=None, write=False, crs=None, rank=0, nodata=0):
        """

        :param raster_path:
        :param out_path:
        :param write:
        :param crs:
        :param rank:
        :param nodata:
        :return:
        """
        from shapely.geometry import Polygon
        from hermesv3_bu.grids.grid import Grid

        if self.comm.Get_rank() == rank:
            ds = rasterio.open(raster_path)

            grid_info = ds.transform

            lons = np.arange(ds.width) * grid_info[0] + grid_info[2]
            lats = np.arange(ds.height) * grid_info[4] + grid_info[5]

            # 1D to 2D
            c_lats = np.array([lats] * len(lons)).T.flatten()
            c_lons = np.array([lons] * len(lats)).flatten()
            del lons, lats

            b_lons = self.create_bounds(c_lons, grid_info[0], number_vertices=4) + grid_info[0]/2
            b_lats = self.create_bounds(c_lats, grid_info[4], number_vertices=4, inverse=True) + grid_info[4]/2

            df_lats = pd.DataFrame(b_lats[0], columns=['b_lat_1', 'b_lat_2', 'b_lat_3', 'b_lat_4'])
            df_lons = pd.DataFrame(b_lons[0], columns=['b_lon_1', 'b_lon_2', 'b_lon_3', 'b_lon_4'])
            df = pd.concat([df_lats, df_lons], axis=1)

            del df_lats, df_lons, b_lats, b_lons

            df['p1'] = zip(df.b_lon_1, df.b_lat_1)
            del df['b_lat_1'], df['b_lon_1']
            df['p2'] = zip(df.b_lon_2, df.b_lat_2)
            del df['b_lat_2'], df['b_lon_2']
            df['p3'] = zip(df.b_lon_3, df.b_lat_3)
            del df['b_lat_3'], df['b_lon_3']
            df['p4'] = zip(df.b_lon_4, df.b_lat_4)
            del df['b_lat_4'], df['b_lon_4']

            list_points = df.values

            del df['p1'], df['p2'], df['p3'], df['p4']

            data = ds.read(1).flatten()

            geometry = [Polygon(list(points)) for points in list_points]
            gdf = gpd.GeoDataFrame(data, columns=['data'], crs=ds.crs, geometry=geometry)
            gdf.loc[:, 'CELL_ID'] = xrange(len(gdf))
            gdf = gdf[gdf['data'] != nodata]

            if crs is not None:
                gdf = gdf.to_crs(crs)

            if write:
                if not os.path.exists(os.path.dirname(out_path)):
                    os.makedirs(os.path.dirname(out_path))
                gdf.to_file(out_path)
        else:
            gdf = None

        if self.comm.Get_size() > 1:
            gdf = self.comm.bcast(gdf, root=0)

        return gdf

    def value_to_shapefile(self, raster_path, value, out_path=None, crs=None, rank=0):
        from hermesv3_bu.grids.grid import Grid
        from shapely.geometry import Polygon

        if self.comm.Get_rank() == rank:
            ds = rasterio.open(raster_path)
            data = ds.read(1).flatten()

            grid_info = ds.transform

            lons = np.arange(ds.width) * grid_info[1] + grid_info[0]
            lats = np.arange(ds.height) * grid_info[5] + grid_info[3]

            # 1D to 2D
            c_lats = np.array([lats] * len(lons)).T.flatten()
            c_lons = np.array([lons] * len(lats)).flatten()
            del lons, lats

            b_lons = Grid.create_bounds(c_lons, grid_info[1], number_vertices=4) + grid_info[1]/2
            b_lats = Grid.create_bounds(c_lats, grid_info[1], number_vertices=4, inverse=True) + grid_info[5]/2

            df_lats = pd.DataFrame(b_lats[0], columns=['b_lat_1', 'b_lat_2', 'b_lat_3', 'b_lat_4'])
            df_lons = pd.DataFrame(b_lons[0], columns=['b_lon_1', 'b_lon_2', 'b_lon_3', 'b_lon_4'])
            df = pd.concat([df_lats, df_lons], axis=1)
            del df_lats, df_lons, b_lats, b_lons

            index = np.where(data == value)[0]
            data = data[index]

            df = df[~df.index.isin(index)]

            df['p1'] = zip(df.b_lon_1, df.b_lat_1)
            del df['b_lat_1'], df['b_lon_1']
            df['p2'] = zip(df.b_lon_2, df.b_lat_2)
            del df['b_lat_2'], df['b_lon_2']
            df['p3'] = zip(df.b_lon_3, df.b_lat_3)
            del df['b_lat_3'], df['b_lon_3']
            df['p4'] = zip(df.b_lon_4, df.b_lat_4)
            del df['b_lat_4'], df['b_lon_4']

            list_points = df.as_matrix()
            del df['p1'], df['p2'], df['p3'], df['p4']

            gdf = gpd.GeoDataFrame(data, columns=['data'], crs=ds.crs,
                                   geometry=[Polygon(list(points)) for points in list_points])
            gdf.loc[:, 'CELL_ID'] = index

            gdf = gdf[gdf['data'] > 0]
            if crs is not None:
                gdf = gdf.to_crs(crs)

            if out_path is not None:
                if not os.path.exists(os.path.dirname(out_path)):
                    os.makedirs(os.path.dirname(out_path))
                gdf.to_file(out_path)
            # gdf = np.array_split(gdf, self.size)
        else:
            gdf = None

        gdf = self.comm.bcast(gdf, root=0)

        return gdf
