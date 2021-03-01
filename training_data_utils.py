import tensorflow as tf
import descarteslabs as dl
import numpy as np
from osgeo import gdal, ogr
import os
from typing import Sequence

### Functions from Descartes "wellpads" sample utils.py



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
    "uint": gdal.GDT_UInt16,###
    "int": gdal.GDT_Int32,
    "float": gdal.GDT_Float64,
}


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
    
    # HSG: NB - the descartes sample code didn't allow for padding here
    n_rows = ctx.tilesize + ctx.pad * 2
    n_cols = ctx.tilesize + ctx.pad * 2
    gdal_dtype = numpy_dtype_to_gdal[dtype]
    driver = gdal.GetDriverByName(driver_name)
    ds = driver.Create(savename, n_rows, n_cols, n_bands, gdal_dtype, options=options)
    # Grab projection and geotransform from metadata.
    proj_wkt = ctx.wkt
    ds.SetProjection(proj_wkt)
    ds.SetGeoTransform(ctx.geotrans)

    return ds


# End Descartes labs sample functions

class TileJobConfig:
    """Simple data class to hold the necessary info for creating one training sample.
    Primary purpose is to provide a hashable means for passing all the data needed to extract 
    one sample, so it can be easily pickled and used by joblib etc, including in particular 
    the dltile object."""
    def __init__(self, dltile, out_folder_base, dl_product, ref_date, labels_data,
                 min_date = None, max_date = None, max_cloud_fraction=None,
                 label_attr=None, label_lyr_num=0, bands="red green blue"):
        self.DLTILE = dltile
        self.OUTFOLDER = out_folder_base
        self.PRODUCT=dl_product
        self.TARGETDATE=ref_date
        self.MIN_DATE=min_date
        self.MAX_DATE=max_date
        self.MAX_CLOUD_FRACTION=max_cloud_fraction
        self.LABEL_DS=labels_data
        self.LABEL_BURN_ATTR=label_attr
        self.LABEL_LYR_NUM=label_lyr_num
        self.BANDS=bands
        
    @classmethod
    def from_run_config(cls, run_config, dltile, out_folder_base, ref_date, min_date, max_date, max_cloud_fraction):
        """factory method to create a TileJobConfig where most of the parameters are replaced by a SampleCreationConfig 
        object"""
        lbl_data = run_config.LABEL_DATA()
        return cls(dltile = dltile,
                  out_folder_base=out_folder_base,
                  dl_product=run_config.PRODUCT(),
                  ref_date=ref_date, min_date=min_date, max_date=max_date, max_cloud_fraction=max_cloud_fraction,
                  labels_data=lbl_data.OGR_DATASET,
                  label_attr=lbl_data.BURN_ATTRIB,
                  label_lyr_num=lbl_data.get_layer_index(),
                  bands=run_config.BANDS())


def get_scene_date_diff_mapper(reference_date):
    """Returns a function that will compare the date of a scene to the 
    originally-specified reference date"""
    # (note use of closure)
    def get_date_diff(scene):
        scene_date = scene.properties['date'].date()
        offset = abs(scene_date - reference_date)
        return offset
    return get_date_diff


def create_img_array(ctx, product, reference_date, min_date=None, max_date=None, 
                     bands='red green blue', max_cloud_fraction=None):
    """Creates a mosaic of scenes matching the specified geocontext (dltile).
    The mosaic will be created by selecting all scenes intersecting the tile; filtering 
    to those after min_date and/or before max_date if specified; filtering to those with cloud 
    fraction < max_cloud_fraction if specified; and then prioritising the scenes closest in 
    time to the specified reference date.
    
    To create a mosaic of "latest available" data, just pass today's date or some arbitrary 
    future date for `reference_date` and None for `min_date` and `max_date`.
    
    The RGB image data (or other bands as specified) are returned as a 3D array
    whose shape will be equal to the geocontext's shape + 2*padding, * n bands. 
    The data will be processed to "surface" (surface reflectance) where available. 
    
    For datasets that support it, the "Surface Reflectance" processed data will be returned 
    (as opposed to e.g. TOA).
    
    If an error occurs then None will be returned. This generally suggests an error emanating 
    from the Descartes Labs API. This can suggest that no data are available for the specified 
    product / date range / cloud specification. However it can also occur randomly. So it's 
    always worth re-trying a few times, and or checking any error tiles in the DL Viewer to see 
    if data really are unavailable."""
    
    searchparams = {
        "aoi" : ctx,
        "products" : product
    }
    if  min_date is not None:
        searchparams['start_datetime'] = min_date.isoformat()
    if max_date is not None:
        searchparams['end_datetime'] = max_date.isoformat()
    if max_cloud_fraction is not None:
        #searchparams['cloud_fraction'] = max_cloud_fraction
        # or 
        searchparams['query'] = (dl.properties.cloud_fraction < max_cloud_fraction)
    #scenes, newctx = dl.scenes.search(ctx, products=product)
    scenes, newctx = dl.scenes.search(**searchparams)
    
    if len(scenes) == 0:
        return None
    
    # Sort the candidate scenes by the absolute difference between their date and the 
    # reference date. Sort in reverse order so the closest in time is last in the iterable 
    # SceneCollection. From the DL SceneCollection.mosaic() docs:
    #     `Where multiple scenes overlap, only data from the scene that comes last in 
    #     the SceneCollection is used.`
    date_diff_mapper = get_scene_date_diff_mapper(reference_date)
    sorted_scenes = scenes.sorted(date_diff_mapper, reverse=True)
    
    try:
        arr = sorted_scenes.mosaic(bands=bands, ctx=ctx, bands_axis=-1, processing_level="surface")
        return arr
    except:
        return None


def create_label_array(ctx, label_data, attrib_to_burn=None, layer_idx=0):
    """Rasterises the label data (path to OGR datasource) within the specified geocontext.
    If attrib_to_burn is not specified then all features will be rasterised as 1 and other 
    areas as 0. If attrib_to_burn is specified then it must contain values 0-255. Returns a
    2D array with shape equal to geocontext's shape + 2*padding"""
    drv = gdal.GetDriverByName('MEM')
    img_size = ctx.tilesize + ctx.pad*2
    mem_ds = drv.Create('tmp', img_size, img_size, 1, gdal.GDT_Byte)
    mem_ds.SetProjection(ctx.wkt)
    mem_ds.SetGeoTransform(ctx.geotrans)
    label_ogr_ds = ogr.Open(label_data)
    label_lyr = label_ogr_ds.GetLayerByIndex(layer_idx)
    if attrib_to_burn:
        gdal.RasterizeLayer(mem_ds, [1], label_lyr, options=['ALL_TOUCHED=TRUE',f'ATTRIBUTE={attrib_to_burn}'])
    else:
        gdal.RasterizeLayer(mem_ds, [1], label_lyr, burn_values=[1], options=['ALL_TOUCHED=TRUE'])
    arr = mem_ds.ReadAsArray()
    mem_ds = None
    return arr


def create_chips_for_tile(job_details: TileJobConfig) -> tuple:
    """Creates image chips (geotiff training samples) for the specified  TileJobConfig.
    
    The image and label data files will be placed into /images and /labels subfolders below 
    the specified output folder location, and their name will be the DLTile's key with ':' 
    replaced by '#'.
    
    Returns a 3-tuple of (job_details, path_to_image, path_to_label). If the request to the 
    Descartes Labs API failed to generate an image, then returns (job_details, None, None). 
    In this case, you should re-try several times before concluding that data are not 
    available (or check in the DL Viewer), as the API is prone to transient failures.
    """
    dltile = job_details.DLTILE
    out_base = job_details.OUTFOLDER
    product = job_details.PRODUCT
    target_date = job_details.TARGETDATE
    label_data = job_details.LABEL_DS
    label_lyr = job_details.LABEL_LYR_NUM
    label_attrib = job_details.LABEL_BURN_ATTR
    bands = job_details.BANDS
    
    min_date = job_details.MIN_DATE
    max_date = job_details.MAX_DATE
    max_cloud_fraction = job_details.MAX_CLOUD_FRACTION
    
    out_img_folder = os.path.join(out_base, 'images')
    out_lbl_folder = os.path.join(out_base, 'labels')
    if not os.path.exists(out_img_folder):
        os.makedirs(out_img_folder)
    if not os.path.exists(out_lbl_folder):
        os.makedirs(out_lbl_folder)
        
    dltile_key = dltile.key
    # Store geotiffs using the dltile key encoded into the filename so we don't have to 
    # later use any fancy logic to re-parse it from the geotransform.
    fn = dltile_key.replace(':','#')
    # By wrapping the dltile in a TileJobConfig we are able to pass it directly within 
    # joblib. Otherwise we'd have had to pass in the string key, and reconstruct the DLTile
    # here. That is unnecessarily expensive as it involves API calls.
    #dltile = dl.scenes.DLTile.from_key(dltile_key)
    
    # get the image data from descartes labs
    img_arr = create_img_array(ctx=dltile, product=product, 
                               reference_date=target_date, min_date=min_date, max_date=max_date,
                               max_cloud_fraction=max_cloud_fraction,
                               bands=bands)
    if img_arr is None:
        return (job_details, None, None)
    # rasterise the label data
    lbl_arr = create_label_array(ctx=dltile, label_data=label_data, 
                                 attrib_to_burn=label_attrib,
                                layer_idx=label_lyr)
    img_file = os.path.join(out_img_folder, fn) + ".tif"
    lbl_file = os.path.join(out_lbl_folder, fn) + ".tif"
    # save the data to compressed geotiffs
    n_img_bands = img_arr.shape[-1]
    img_ds = gdal_dataset_from_geocontext(ctx=dltile, n_bands=n_img_bands, driver_name="GTiff", 
                                          savename=img_file, dtype=img_arr.dtype, 
                                          options=['COMPRESS=LZW', 'TILED=TRUE', 'NUM_THREADS=4'])
    for b in range(n_img_bands):
        bnd = img_ds.GetRasterBand(b+1)
        bnd.WriteArray(img_arr[:,:,b])
    img_ds.FlushCache()
    img_ds=None
    lbl_ds = gdal_dataset_from_geocontext(ctx=dltile, n_bands=1, driver_name="GTiff", 
                                          savename=lbl_file, dtype=lbl_arr.dtype, 
                                          options=['COMPRESS=LZW', 'TILED=TRUE', 'NUM_THREADS=4'])
    lbl_ds.GetRasterBand(1).WriteArray(lbl_arr)
    lbl_ds=None
    # return the paths
    return (job_details, img_file,lbl_file)

