#!/usr/bin/env python

import os
import timeit
from mpi4py import MPI
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

from hermesv3_bu.io_server.io_server import IoServer
from hermesv3_bu.io_server.io_shapefile import IoShapefile
from hermesv3_bu.tools.checker import check_files, error_exit


class IoRaster(IoServer):
    def __init__(self, comm=None):
        if comm is None:
            comm = MPI.COMM_WORLD
        super(IoRaster, self).__init__(comm)

    def clip_raster_with_shapefile(self, raster_path, shape_path, clipped_raster_path, values=None, nodata=0):
        """
        Clip a raster using given shapefile path.

        The clip is performed only by the selected rank process.

        :param raster_path: Path to the raster to clip.
        :type raster_path: str

        :param shape_path: Path to the shapefile with the polygons where clip the input raster.
        :type shape_path: str

        :param clipped_raster_path: Place to store the clipped raster.
        :type clipped_raster_path: str

        :param values: List of data values to clip.
        :type values: list

        :return: Path where is stored the clipped raster.
        :rtype: str
        """
        def getFeatures(gdf):
            """
            https://automating-gis-processes.github.io/CSC18/lessons/L6/clipping-raster.html
            Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
            import json
            return [json.loads(gdf.to_json())['features'][0]['geometry']]
        check_files([raster_path, shape_path])
        data = rasterio.open(raster_path)
        geo = gpd.read_file(shape_path)
        if len(geo) > 1:
            geo = gpd.GeoDataFrame(geometry=[geo.geometry.unary_union], crs=geo.crs)
        geo = geo.to_crs(crs=data.crs.data)
        coords = getFeatures(geo)

        out_img, out_transform = mask(data, shapes=coords, crop=True, all_touched=True, nodata=nodata)
        if values is not None:
            out_img[~np.isin(out_img, values)] = nodata

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

        return clipped_raster_path

    def clip_raster_with_shapefile_poly(self, raster_path, geo, clipped_raster_path, values=None, nodata=0):
        """
        Clip a raster using given shapefile.

        The clip is performed only by the master (rank 0) process.

        :param raster_path: Path to the raster to clip.
        :type raster_path: str

        :param geo: Shapefile with the polygons where clip the input raster.
        :type geo: GeoDataFrame

        :param clipped_raster_path: Place to store the clipped raster.
        :type clipped_raster_path: str

        :param values: List of data values to clip.
        :type values: list

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
        check_files(raster_path)
        data = rasterio.open(raster_path)

        if len(geo) > 1:
            geo = gpd.GeoDataFrame(geometry=[geo.geometry.unary_union], crs=geo.crs)
        geo = geo.to_crs(crs=data.crs.data)
        coords = get_features(geo)

        out_img, out_transform = mask(data, shapes=coords, crop=True, all_touched=True, nodata=nodata)
        if values is not None:
            out_img[~np.isin(out_img, values)] = nodata
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
            error_exit('ERROR: The number of vertices of the boundaries must be 2 or 4.')
        # self.logger.write_time_log('IoRaster', 'create_bounds', timeit.default_timer() - spent_time, 3)
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

        if self.comm.Get_rank() == rank:
            gdf = self.to_shapefile_serie(raster_path, out_path=out_path, write=write, crs=crs, nodata=nodata)
        else:
            gdf = None

        if self.comm.Get_size() > 1:
            gdf = self.comm.bcast(gdf, root=0)

        return gdf

    def to_shapefile_serie_by_cell(self, raster_path, out_path=None, write=False, crs=None, nodata=0):
        """

        :param raster_path:
        :param out_path:
        :param write:
        :param crs:
        :param nodata:
        :return:
        """

        if out_path is None or not os.path.exists(out_path):
            ds = rasterio.open(raster_path)

            grid_info = ds.transform
            # TODO remove when new version will be installed
            if rasterio.__version__ == '0.36.0':
                lons = np.arange(ds.width) * grid_info[1] + grid_info[0]
                lats = np.arange(ds.height) * grid_info[5] + grid_info[3]
            elif rasterio.__version__ == '1.0.21':
                lons = np.arange(ds.width) * grid_info[0] + grid_info[2]
                lats = np.arange(ds.height) * grid_info[4] + grid_info[5]
            else:
                lons = np.arange(ds.width) * grid_info[0] + grid_info[2]
                lats = np.arange(ds.height) * grid_info[4] + grid_info[5]

            # 1D to 2D
            c_lats = np.array([lats] * len(lons)).T.flatten()
            c_lons = np.array([lons] * len(lats)).flatten()

            del lons, lats
            if rasterio.__version__ == '0.36.0':
                b_lons = self.create_bounds(c_lons, grid_info[1], number_vertices=4) + grid_info[1] / 2
                b_lats = self.create_bounds(c_lats, grid_info[1], number_vertices=4, inverse=True) + grid_info[5] / 2
            elif rasterio.__version__ == '1.0.21':
                b_lons = self.create_bounds(c_lons, grid_info[0], number_vertices=4) + grid_info[0] / 2
                b_lats = self.create_bounds(c_lats, grid_info[4], number_vertices=4, inverse=True) + grid_info[4] / 2
            else:
                b_lons = self.create_bounds(c_lons, grid_info[0], number_vertices=4) + grid_info[0] / 2
                b_lats = self.create_bounds(c_lats, grid_info[4], number_vertices=4, inverse=True) + grid_info[4] / 2

            b_lats = b_lats.reshape((b_lats.shape[1], b_lats.shape[2]))
            b_lons = b_lons.reshape((b_lons.shape[1], b_lons.shape[2]))

            gdf = gpd.GeoDataFrame(ds.read(1).flatten(), columns=['data'], index=range(b_lons.shape[0]), crs=ds.crs)
            gdf['geometry'] = None

            for i in range(b_lons.shape[0]):
                gdf.loc[i, 'geometry'] = Polygon([(b_lons[i, 0], b_lats[i, 0]),
                                                  (b_lons[i, 1], b_lats[i, 1]),
                                                  (b_lons[i, 2], b_lats[i, 2]),
                                                  (b_lons[i, 3], b_lats[i, 3]),
                                                  (b_lons[i, 0], b_lats[i, 0])])

            gdf['CELL_ID'] = gdf.index

            gdf = gdf[gdf['data'] != nodata]

            if crs is not None:
                gdf = gdf.to_crs(crs)

            if write:
                if not os.path.exists(os.path.dirname(out_path)):
                    os.makedirs(os.path.dirname(out_path))
                gdf.to_file(out_path)

        else:
            gdf = gpd.read_file(out_path)

        return gdf

    def to_shapefile_serie(self, raster_path, out_path=None, write=False, crs=None, nodata=0):
        """

        :param raster_path:
        :param out_path:
        :param write:
        :param crs:
        :param nodata:
        :return:
        """

        if out_path is None or not os.path.exists(out_path):
            import rasterio
            from rasterio.features import shapes
            mask = None
            src = rasterio.open(raster_path)

            image = src.read(1)  # first band
            image = image.astype(np.float32)
            geoms = (
                {'properties': {'data': v}, 'geometry': s}
                for i, (s, v) in enumerate(shapes(image, mask=mask, transform=src.transform)))

            gdf = gpd.GeoDataFrame.from_features(geoms)

            gdf.loc[:, 'CELL_ID'] = range(len(gdf))
            gdf = gdf[gdf['data'] != nodata]

            # Error on to_crs function of geopandas that flip lat with lon in the non dict form
            if src.crs == 'EPSG:4326':
                gdf.crs = {'init': 'epsg:4326'}
            else:
                gdf.crs = src.crs

            if crs is not None:
                gdf = gdf.to_crs(crs)

            if write:
                if not os.path.exists(os.path.dirname(out_path)):
                    os.makedirs(os.path.dirname(out_path))
                gdf.to_file(out_path)

        else:
            gdf = gpd.read_file(out_path)
        gdf.set_index('CELL_ID', inplace=True)
        return gdf

    def to_shapefile_parallel(self, raster_path, gather=False, bcast=False, crs=None, nodata=0):
        spent_time = timeit.default_timer()
        if self.comm.Get_rank() == 0:
            ds = rasterio.open(raster_path)
            grid_info = ds.transform

            # TODO remove when new version will be installed
            if rasterio.__version__ == '0.36.0':
                lons = np.arange(ds.width) * grid_info[1] + grid_info[0]
                lats = np.arange(ds.height) * grid_info[5] + grid_info[3]
            elif rasterio.__version__ == '1.0.21':
                lons = np.arange(ds.width) * grid_info[0] + grid_info[2]
                lats = np.arange(ds.height) * grid_info[4] + grid_info[5]
            else:
                lons = np.arange(ds.width) * grid_info[0] + grid_info[2]
                lats = np.arange(ds.height) * grid_info[4] + grid_info[5]

            # 1D to 2D
            c_lats = np.array([lats] * len(lons)).T.flatten()
            c_lons = np.array([lons] * len(lats)).flatten()
            del lons, lats
            if rasterio.__version__ == '0.36.0':
                b_lons = self.create_bounds(c_lons, grid_info[1], number_vertices=4) + grid_info[1] / 2
                b_lats = self.create_bounds(c_lats, grid_info[1], number_vertices=4, inverse=True) + grid_info[5] / 2
            elif rasterio.__version__ == '1.0.21':
                b_lons = self.create_bounds(c_lons, grid_info[0], number_vertices=4) + grid_info[0] / 2
                b_lats = self.create_bounds(c_lats, grid_info[4], number_vertices=4, inverse=True) + grid_info[4] / 2
            else:
                b_lons = self.create_bounds(c_lons, grid_info[0], number_vertices=4) + grid_info[0] / 2
                b_lats = self.create_bounds(c_lats, grid_info[4], number_vertices=4, inverse=True) + grid_info[4] / 2

            b_lats = b_lats.reshape((b_lats.shape[1], b_lats.shape[2]))
            b_lons = b_lons.reshape((b_lons.shape[1], b_lons.shape[2]))

            gdf = gpd.GeoDataFrame(ds.read(1).flatten(), columns=['data'], index=range(b_lons.shape[0]), crs=ds.crs)
            gdf['geometry'] = None
        else:
            gdf = None
            b_lons = None
            b_lats = None
        self.comm.Barrier()
        gdf = IoShapefile(self.comm).split_shapefile(gdf)

        b_lons = IoShapefile(self.comm).split_shapefile(b_lons)
        b_lats = IoShapefile(self.comm).split_shapefile(b_lats)

        i = 0
        for j, df_aux in gdf.iterrows():
            gdf.loc[j, 'geometry'] = Polygon([(b_lons[i, 0], b_lats[i, 0]),
                                              (b_lons[i, 1], b_lats[i, 1]),
                                              (b_lons[i, 2], b_lats[i, 2]),
                                              (b_lons[i, 3], b_lats[i, 3]),
                                              (b_lons[i, 0], b_lats[i, 0])])
            i += 1

        gdf['CELL_ID'] = gdf.index
        gdf = gdf[gdf['data'] != nodata]
        if crs is not None:
            gdf = gdf.to_crs(crs)

        if gather and not bcast:
            gdf = IoShapefile(self.comm).gather_shapefile(gdf)
        elif gather and bcast:
            gdf = IoShapefile(self.comm).gather_bcast_shapefile(gdf)
        return gdf
