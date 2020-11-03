"""
Utility functions for training a Wellpad model
"""
# flake8: noqa
import tensorflow as tf
import descarteslabs as dl
import numpy as np
import json
import os
import datetime

from shapely.geometry import shape
from osgeo import gdal, ogr
from typing import Sequence
from PIL import Image

numpy_dtype_to_gdal = {
    np.dtype("bool"): gdal.GDT_Byte,
    np.dtype("byte"): gdal.GDT_Byte,
    np.dtype("uint8"): gdal.GDT_Byte,
    np.dtype("uint16"): gdal.GDT_UInt16,
    np.dtype("int16"): gdal.GDT_Int16,
    np.dtype("uint32"): gdal.GDT_UInt32,
    np.dtype("int32"): gdal.GDT_Int32,
    np.dtype("float32"): gdal.GDT_Float32,
    np.dtype("float64"): gdal.GDT_Float64,
    "bool": gdal.GDT_Byte,
    "byte": gdal.GDT_Byte,
    "uint8": gdal.GDT_Byte,
    "uint16": gdal.GDT_UInt16,
    "int16": gdal.GDT_Int16,
    "uint32": gdal.GDT_UInt32,
    "int32": gdal.GDT_Int32,
    "float32": gdal.GDT_Float32,
    "float64": gdal.GDT_Float64,
    "uint": gdal.GDT_UInt16,
    "int": gdal.GDT_Int32,
    "float": gdal.GDT_Float64,
}


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if isinstance(value, np.ndarray):
        value = value.flatten().tolist()
    elif not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float64_feature(value):
    """Wrapper for inserting float64 features into Example proto."""
    if isinstance(value, np.ndarray):
        value = value.flatten().tolist()
    elif not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_example(img_data, target_data, img_shape, target_shape, dltile):
    """ Converts image and target data into TFRecords example.
    
    Parameters
    ----------
    img_data: ndarray
        Image data
    target_data: ndarray
        Target data
    img_shape: tuple
        Shape of the image data (h, w, c)
    target_shape: tuple
        Shape of the target data (h, w, c)
    dltile: str
        DLTile key
    
    Returns
    -------
    Example: TFRecords example
        TFRecords example
    """
    if len(target_shape) == 2:
        target_shape = (*target_shape, 1)

    features = {
        "image/image_data": _float64_feature(img_data),
        "image/height": _int64_feature(img_shape[0]),
        "image/width": _int64_feature(img_shape[1]),
        "image/channels": _int64_feature(img_shape[2]),
        "target/target_data": _float64_feature(target_data),
        "target/height": _int64_feature(target_shape[0]),
        "target/width": _int64_feature(target_shape[1]),
        "target/channels": _int64_feature(target_shape[2]),
        "dltile": _bytes_feature(tf.compat.as_bytes(dltile)),
    }

    return tf.train.Example(features=tf.train.Features(feature=features))


def get_center_location(polygon):
    """ Determines the center coordinates given the coordinates of a polygon
    
    Parameters
    ----------
    polygon: dict
        Dictionary containing the polygon coordinates
    
    Returns
    -------
    lon, lat: tuple
        Coordinates of the polygon center
    """
    coords_lonlat = shape(polygon).centroid.coords.xy

    return coords_lonlat[0][0], coords_lonlat[1][0]


def make_ground_dataset_from_polygons(
    polygons,
    products,
    bands,
    resolution,
    tilesize,
    pad,
    start_datetime,
    end_datetime,
    out_file,
    seed=0,
):
    """ Generates ground truth dataset from polygons.
    
    Parameters
    ----------
    polygons: str or dict
        Either a GeoJSON file or dictionary of feature collection
    products: str
        Product identifier to pull the imagery from
    bands: list
        List of bands to select
    resolution: float
        Image resolution
    tilesize: int
        Tilesize in pixels
    pad: int
        Tile padding
    start_datetime: str
        Start time to search for scenes for
    end_datetime: str
        End time to search for scenes for
    out_file: str
        Name of the output file
    seed: int
        Random seed
    
    Returns
    -------
    n_samples: int
        Number of samples written to TFRecords
    """
    # load polygons
    if isinstance(polygons, str):
        fname = polygons
        with open(fname) as f:
            poly = json.load(f)
        polygons = {"type": "FeatureCollection", "features": poly["features"]}
        ogr_ds = ogr.Open(fname)
    elif isinstance(polygons, dict):
        assert polygons["type"] == "FeatureCollection"
        ogr_ds = ogr.Open(json.dumps(polygons))
    else:
        print("Wrong datatype for POLYGONS.")
        return 0
    n_features = len(polygons["features"])

    layer = ogr_ds.GetLayer()

    progress = ProgressBar(
        n_features - 1, 20, display_eta=True, display_steps_per_second=True
    )

    # Get dltiles for all polygons and convert to TFRecords example
    writer = tf.io.TFRecordWriter(out_file)

    exception_list = []
    num_attempts = 3
    count = 0
    for i, p in enumerate(polygons["features"]):
        attempts = 0
        while attempts < num_attempts:
            try:
                lon, lat = get_center_location(p["geometry"])
                dltile = dl.scenes.DLTile.from_latlon(
                    lat=lat, lon=lon, tilesize=tilesize, resolution=resolution, pad=pad
                )

                # Search for scenes for the given dltile
                scenes, ctx = dl.scenes.search(
                    aoi=dltile,
                    products=products,
                    start_datetime=start_datetime,
                    end_datetime=end_datetime,
                )
                scene_ids = [sc.properties.id for sc in scenes]

                if len(scene_ids) == 0:
                    break

                # Get raster image
                arr = scenes.mosaic(bands="red green blue", ctx=ctx, bands_axis=-1)

                # Using the metadata get the target
                ds_target = gdal_dataset_from_geocontext(
                    ctx,
                    1,
                    driver_name="GTiff",
                    savename="tmp.tiff",
                    dtype="byte",
                    options=["COMPRESS=LZW"],
                )

                gdal.RasterizeLayer(
                    ds_target,
                    [1],
                    layer,
                    burn_values=[1],
                    options=["ALL_TOUCHED=TRUE", "COMPRESS=LZW"],
                )

                del ds_target

                img_target = np.array(Image.open("tmp.tiff"))

                os.remove("tmp.tiff")

                # Get TFRecord example
                example = convert_to_example(
                    arr, img_target, arr.shape, img_target.shape, dltile.key
                )
                writer.write(example.SerializeToString())

                count += 1
                break
            except Exception as e:
                exception_list.append(
                    "Caught exception for lat: {} and lon: {} - {}".format(lat, lon, e)
                )
                attempts += 1

        progress(
            "{0}/{1} ({2:.0f}%)".format(i + 1, n_features, (i + 1) / n_features * 100),
            "",
            i + 1,
        )

    writer.close()

    if len(exception_list) > 0:
        print("WARNING: Some exceptions were caught while getting dltiles")
        for e in exception_list:
            print(e)

    return count


def gdal_dataset_from_geocontext(
    ctx: dict,
    n_bands: int,
    driver_name: str = "MEM",
    savename: str = "",
    dtype: str = "byte",
    options: Sequence = None,
):
    """Get a GDAL dataset using geocontext returned by dl.scenes.search.
    The output GDAL dataset will have the proper geo metdata, but
    won't contain raster data.  To do that, use gdal_dataset_from_narray.
    Parameters
    ----------
    ctx: dict
        Geocontext as returned by dl.scenes.search(...)
    n_bands: int
        The number of raster bands for the output dataset.
        You must specify manually, because the data product you're trying to
        save might have more or fewer bands than the original image.
    driver_name: str (optional)
        gdal driver name. Eg: MEM or GTiff
    savename: str (optional)
        Path to save dataset, if saving is desired.
    dtype: str (optional)
        Numpy style datatype for the dataset
    options: list (optional)
        A list of gdal dataset options like ['COMPRESS=LZW']

    Returns
    -------
    ds: gdal.Dataset
        The output dataset
    """
    options = options or []

    n_rows = ctx.tilesize
    n_cols = ctx.tilesize
    gdal_dtype = numpy_dtype_to_gdal[dtype]
    driver = gdal.GetDriverByName(driver_name)
    ds = driver.Create(savename, n_rows, n_cols, n_bands, gdal_dtype, options=options)
    # Grab projection and geotransform from metadata.
    proj_wkt = ctx.wkt
    ds.SetProjection(proj_wkt)
    ds.SetGeoTransform(ctx.geotrans)

    return ds


class ProgressBar(object):
    """ Class for displaying a progress bar on the command line """

    def __init__(self, full, length, display_eta=False, display_steps_per_second=False):
        self.full = full
        self.length = length
        self.display_eta = display_eta
        self.display_steps_per_second = display_steps_per_second
        self.last_step_time = (0, datetime.datetime.now().timestamp())

    def __call__(self, pre_text, text, status):
        frac = int(self.length / self.full * status)
        s = ["{:<10s} \t|".format(pre_text)]
        s.extend(["#" for i in range(frac)])
        s.extend(["-" for i in range(self.length - frac)])
        s.append("| ")

        current_step_time = datetime.datetime.now().timestamp()
        if self.display_steps_per_second:
            steps_per_second = (status - self.last_step_time[0]) / (
                current_step_time - self.last_step_time[1]
            )
            s.append("{0:.1f} steps/s ".format(steps_per_second))
        if self.display_eta:
            eta = (
                (self.full - status)
                * (current_step_time - self.last_step_time[1])
                / (status - self.last_step_time[0])
            )
            s.append("ETA: {0:.0f}s".format(eta))
        s.append(text)
        s.append("        ")

        self.last_step_time = (status, current_step_time)

        print("".join(s), end="\r")

    def reset(self):
        print("\n")
