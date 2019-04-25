#!/usr/bin/env python

import sys
import os
from timeit import default_timer as gettime
from warnings import warn
import rasterio

import IN.src.config.settings as settings
from IN.src.modules.io.io import Io


class IoRaster(Io):
    def __init__(self):
        super(IoRaster, self).__init__()

    def write_raster(self):
        pass

    def read_raster(self):
        pass

    def clip_raster_with_shapefile(self, raster_path, shape_path, clipped_raster_path):
        if settings.log_level_3:
            st_time = gettime()
        else:
            st_time = None

        def getFeatures(gdf):
            """
            https://automating-gis-processes.github.io/CSC18/lessons/L6/clipping-raster.html
            Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
            import json
            return [json.loads(gdf.to_json())['features'][0]['geometry']]

        import geopandas as gpd
        from rasterio.mask import mask
        if self.rank == 0:
            data = rasterio.open(raster_path)
            geo = gpd.read_file(shape_path)
            if len(geo) > 1:
                geo = gpd.GeoDataFrame(geometry=[geo.geometry.unary_union], crs=geo.crs)
            geo = geo.to_crs(crs=data.crs.data)
            coords = getFeatures(geo)

            out_img, out_transform = mask(raster=data, shapes=coords, crop=True)
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
        print 'TIME -> IoRaster.clip_raster_with_shapefile: Rank {0} {1} s'.format(settings.rank, round(gettime() - st_time, 2))

        return clipped_raster_path

    def clip_raster_with_shapefile_poly(self, raster_path, geo, clipped_raster_path, nodata=0):
        if settings.log_level_3:
            st_time = gettime()
        else:
            st_time = None
        def getFeatures(gdf):
            """
            https://automating-gis-processes.github.io/CSC18/lessons/L6/clipping-raster.html
            Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
            import json
            return [json.loads(gdf.to_json())['features'][0]['geometry']]

        import geopandas as gpd
        from rasterio.mask import mask
        if self.rank == 0:
            data = rasterio.open(raster_path)

            if len(geo) > 1:
                geo = gpd.GeoDataFrame(geometry=[geo.geometry.unary_union], crs=geo.crs)
            geo = geo.to_crs(crs=data.crs.data)
            coords = getFeatures(geo)

            out_img, out_transform = mask(raster=data, shapes=coords, crop=True, all_touched=True, nodata=nodata)
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
        print 'TIME -> IoRaster.clip_raster_with_shapefile_poly Rank {0} {1} s'.format(settings.rank, round(gettime() - st_time, 2))

        return clipped_raster_path

    def to_shapefile(self, raster_path, out_path=None, write=False, crs=None, nodata=0):
        import geopandas as gpd
        import pandas as pd
        import numpy as np
        from shapely.geometry import Polygon
        from IN.src.modules.grids.grid import Grid
        if settings.log_level_3:
            st_time = gettime()
        else:
            st_time = None
        if self.rank == 0:
            ds = rasterio.open(raster_path)

            grid_info = ds.transform
            # grid_info = ds.affine.Affine

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

            df['p1'] = zip(df.b_lon_1, df.b_lat_1)
            del df['b_lat_1'], df['b_lon_1']
            df['p2'] = zip(df.b_lon_2, df.b_lat_2)
            del df['b_lat_2'], df['b_lon_2']
            df['p3'] = zip(df.b_lon_3, df.b_lat_3)
            del df['b_lat_3'], df['b_lon_3']
            df['p4'] = zip(df.b_lon_4, df.b_lat_4)
            del df['b_lat_4'], df['b_lon_4']

            # list_points = df.as_matrix()
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

        print 'TIME -> IoRaster.to_shapefile Rank {0} {1} s'.format(settings.rank, round(gettime() - st_time, 2))
        if self.size > 1:
            gdf = self.comm.bcast(gdf, root=0)

        return gdf

    def clip_raster_with_shapefile_serie(self, raster_path, shape_path, clipped_raster_path):
        if settings.log_level_3:
            st_time = gettime()
        else:
            st_time = None

        def getFeatures(gdf):
            """
            https://automating-gis-processes.github.io/CSC18/lessons/L6/clipping-raster.html
            Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
            import json
            return [json.loads(gdf.to_json())['features'][0]['geometry']]

        import geopandas as gpd
        from rasterio.mask import mask
        data = rasterio.open(raster_path)
        geo = gpd.read_file(shape_path)
        if len(geo) > 1:
            geo = gpd.GeoDataFrame(geometry=[geo.geometry.unary_union], crs=geo.crs)
        geo = geo.to_crs(crs=data.crs.data)
        coords = getFeatures(geo)

        out_img, out_transform = mask(raster=data, shapes=coords, crop=True)
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
        print 'TIME -> IoRaster.clip_raster_with_shapefile: Rank {0} {1} s'.format(settings.rank, round(gettime() - st_time, 2))

        return clipped_raster_path

    def clip_raster_with_shapefile_poly_serie(self, raster_path, geo, clipped_raster_path, nodata=0):
        if settings.log_level_3:
            st_time = gettime()
        else:
            st_time = None
        def getFeatures(gdf):
            """
            https://automating-gis-processes.github.io/CSC18/lessons/L6/clipping-raster.html
            Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
            import json
            return [json.loads(gdf.to_json())['features'][0]['geometry']]

        import geopandas as gpd
        from rasterio.mask import mask
        data = rasterio.open(raster_path)

        if len(geo) > 1:
            geo = gpd.GeoDataFrame(geometry=[geo.geometry.unary_union], crs=geo.crs)
        geo = geo.to_crs(crs=data.crs.data)
        coords = getFeatures(geo)

        out_img, out_transform = mask(raster=data, shapes=coords, crop=True, all_touched=True, nodata=nodata)
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
        print 'TIME -> IoRaster.clip_raster_with_shapefile_poly Rank {0} {1} s'.format(settings.rank, round(gettime() - st_time, 2))

        return clipped_raster_path

    def to_shapefile_serie(self, raster_path, out_path=None, write=False, crs=None, nodata=0):
        import geopandas as gpd
        import pandas as pd
        import numpy as np
        from shapely.geometry import Polygon
        from IN.src.modules.grids.grid import Grid
        if settings.log_level_3:
            st_time = gettime()
        else:
            st_time = None

        ds = rasterio.open(raster_path)

        # if nodata is None:
        #     nodata = ds.nodata

        grid_info = ds.transform
        # grid_info = ds.affine.Affine

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

        df['p1'] = zip(df.b_lon_1, df.b_lat_1)
        del df['b_lat_1'], df['b_lon_1']
        df['p2'] = zip(df.b_lon_2, df.b_lat_2)
        del df['b_lat_2'], df['b_lon_2']
        df['p3'] = zip(df.b_lon_3, df.b_lat_3)
        del df['b_lat_3'], df['b_lon_3']
        df['p4'] = zip(df.b_lon_4, df.b_lat_4)
        del df['b_lat_4'], df['b_lon_4']

        # list_points = df.as_matrix()
        list_points = df.values

        del df['p1'], df['p2'], df['p3'], df['p4']

        data = ds.read(1)
        mask = ds.read_masks(1) == 0
        # print data
        # print mask
        # print type(mask)
        # sys.exit(1)
        data[mask] = 0
        # print data.sum()
        # sys.exit(1)

        geometry = [Polygon(list(points)) for points in list_points]

        gdf = gpd.GeoDataFrame(data.flatten(), columns=['data'], crs=ds.crs, geometry=geometry)

        gdf.loc[:, 'CELL_ID'] = xrange(len(gdf))

        gdf = gdf[gdf['data'] != nodata]

        if crs is not None:
            gdf = gdf.to_crs(crs)

        if write:
            if not os.path.exists(os.path.dirname(out_path)):
                os.makedirs(os.path.dirname(out_path))
            gdf.to_file(out_path)

        print 'TIME -> IoRaster.to_shapefile Rank {0} {1} s'.format(settings.rank, round(gettime() - st_time, 2))

        return gdf

    def value_to_shapefile(self, raster_path, value, out_path=None, crs=None):
        import geopandas as gpd
        import pandas as pd
        import numpy as np
        from shapely.geometry import Polygon
        from IN.src.modules.grids.grid import Grid
        if settings.log_level_3:
            st_time = gettime()
        else:
            st_time = None
        if self.rank == 0:
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
            # print df
            # sys.exit()
            #
            # df = df.loc[index, :]

            df['p1'] = zip(df.b_lon_1, df.b_lat_1)
            del df['b_lat_1'], df['b_lon_1']
            df['p2'] = zip(df.b_lon_2, df.b_lat_2)
            del df['b_lat_2'], df['b_lon_2']
            df['p3'] = zip(df.b_lon_3, df.b_lat_3)
            del df['b_lat_3'], df['b_lon_3']
            df['p4'] = zip(df.b_lon_4, df.b_lat_4)
            del df['b_lat_4'], df['b_lon_4']
            sys.exit()
            list_points = df.as_matrix()
            del df['p1'], df['p2'], df['p3'], df['p4']

            gdf = gpd.GeoDataFrame(data, columns=['data'], crs=ds.crs, geometry=[Polygon(list(points)) for points in list_points])
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
        sys.exit()
        print 'TIME -> IoRaster.value_to_shapefile Rank {0} {1} s'.format(settings.rank, round(gettime() - st_time, 2))

        return gdf