import geopandas as gpd
import descarteslabs as dl
from osgeo import ogr, gdal
import os
import numpy as np
from typing import Sequence

__all__ = ['DLTileJobConfig', 'OGRLabelDataDesc', 'DLSampleCreationConfig', 
'create_chips_for_tile',  'create_img_array_for_tile', 'create_label_array_for_tile']

class DLTileJobConfig:
    """Data class to hold the necessary info for creating one training sample, being one exported 
    image and one rasterised label data image corresponding to the extent and resolution of the 
    given DLTile.

    Primary purpose of this class is to provide a hashable means for passing all the data needed 
    to extract one sample, so it can be easily pickled and used by joblib etc for exporting image 
    chips in parallel, including in particular the dltile object which is not directly pickleable."""
    def __init__(self, dltile, out_folder_base, dl_product, ref_date, labels_data,
                 min_date = None, max_date = None, max_cloud_fraction=None,
                 label_attr=None, label_lyr_num=0, bands="red green blue", 
                 label_nodata_value=255):
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
        self.LABEL_NODATA_VALUE=label_nodata_value
        
    @classmethod
    def from_run_config(cls, run_config, dltile, out_folder_base, 
                        ref_date, min_date=None, max_date=None, 
                        max_cloud_fraction=None):
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
                  bands=run_config.BANDS(),
                  label_nodata_value=run_config.GET_LABEL_NODATA_VALUE())



class OGRLabelDataDesc:
    
    def __init__(self, ogr_dataset, ogr_layer_name_or_idx=0, attrib_to_burn=None):
        """Params:
        ogr_dataset: the path to an OGR-supported dataset e.g. shapefile, GeoJSON file, or file GDB folder
        ogr_layer_name_or_idx: the layername OR index of the layer within the dataset. For any data type that only 
            contains one layer (e.g. GeoJSON or shapefile), use 0 (or the filename without extension). Otherwise 
            use the name (=feature class name) to save having to find the index
        attrib_to_burn: an integer-type feature attribute giving the values to burn to the label images. If None, 
            then all features will be burned with a constant value of 1
        """
        self.OGR_DATASET = ogr_dataset
        self.OGR_LAYER_REF = ogr_layer_name_or_idx
        self.BURN_ATTRIB = attrib_to_burn
        self._cached_layer_idx = None

    def get_layer_index(self):
        """Finds and returns the index of the configured layer within the dataset.
        
        Returns -1 if the layer is not found in the dataset."""
        if isinstance(self.OGR_LAYER_REF, int):
            return self.OGR_LAYER_REF
        else:
            if self._cached_layer_idx is not None:
                return self._cached_layer_idx
            layernames = []
            label_ogr_ds = ogr.Open(self.OGR_DATASET)
            for layer_idx in range(label_ogr_ds.GetLayerCount()):
                lyr = label_ogr_ds.GetLayerByIndex(layer_idx)
                layernames.append(lyr.GetName()) 
            label_ogr_ds = None
            if self.OGR_LAYER_REF in layernames:
                self._cached_layer_idx = layernames.index(self.OGR_LAYER_REF)
            else:
                self._cached_layer_idx = -1
            return self._cached_layer_idx
        #label_lyr = label_ogr_ds.GetLayerByIndex(label_lyr_idx)



class  DLSampleCreationConfig:
    """Holds the parameters needed for the overall configuration of  an extraction of image 
    chips and accompanying label data chips from the Descartes Labs (DL) API 
    
    Image chip locations, extent, and naming are based on the DL tiling scheme. An 
    instance of this class is configured with the desired image chip size, padding, 
    and resolution, and an OGR-compatible spatial dataset giving the area to be 
    covered. 
    
    The necessary DLTiles to cover this area will be calculated and then 
    these, along with the other necessary parameters such as product ID, bands, and 
    output folder can be used to create TileJobConfig objects for each tile that 
    needs exporting using create_tile_job_configs().
    """
    
    def __init__(self,  tile_size, tile_padding, tile_res_m, 
                  dl_product, bands, 
                 sample_folder_root, source_tag,
                 label_data_config,
                max_cloud_fraction=None,
                label_nodata_value=255):
        """Params:
        tile_size: The size of the (square) tiles in pixels, INCLUDING padding.
            i.e. each tile will have tile_size - (2*tile_padding) pixels that are 
            unique to it and not shared with adjacent ones, in each dimension
        tile_padding: The number of pixels of padding (overlap) between the tiles. 
        tile_res_m: The pixel resolution in metres that images will be created at.
            The spatial extent of each tile is thus tile_size * tile_res_m.
        dl_product: The Descartes Labs product ID (search in catalog.descarteslabs.com)
        bands: A space-separatd string giving the band names to export e.g. "red green blue" 
            Check the available names in the DL catalog
        sample_folder_root: The folder below which to create a folder for this export session.
        source_tag: A tag to identify this export in exported folder and file names 
            E.g. "sentinel"
        label_data_config: An OGRLabelDataDesc object which specifies the vector dataset 
            to rasterise to make the labels data corresponding to the imagery data
        max_cloud_fraction: A decimal value between 0 and 1 specifying the maximum allowable 
            cloud fraction in images selected for mosaicking - where supported (e.g. works 
            with Sentinel-2 data but not with Pleiades). If None then no cloud filtering will 
            be done.
        """
        # readonly items, if these change then we need to recalculate tiles so handle 
        # via getter/setter or totally readonly
        self._TILE_SIZE = tile_size - 2 * tile_padding
        self._TILE_PAD = tile_padding
        self._TILE_RES = tile_res_m
        self._LABEL_DATA = label_data_config 
        # no sense in being alterable
        self._root = sample_folder_root
        self._tag = source_tag
        
        self._PRODUCT = dl_product
        self._BANDS = bands
        self._MAX_CLOUD_FRACTION = max_cloud_fraction
        self._LABEL_NDV = label_nodata_value
        # expensive-to-create things that we'll cache
        self._dl_tiles = None
        self._dl_tile_ids = None
        self._gdf_wgs84 = None

        
    def TILE_SIZE_PAD_RES(self, size_pad_res=None):
        if size_pad_res is None:
            return (self._TILE_SIZE, self._TILE_PAD, self._TILE_RES)
        s, p, r = size_pad_res
        s = s  - (2 * p)
        if s != self._TILE_SIZE or p != self._TILE_PAD or r != self._TILE_RES:
            print("Updating tile  configuration: tiles will be re-populated on next request")
            self._TILE_SIZE = s
            self._TILE_PAD = p
            self._TILE_RES = r
            self._invalidate_tiles()
        return (self._TILE_SIZE, self._TILE_PAD, self._TILE_RES)
        

    def LABEL_DATA(self):
        return self._LABEL_DATA # readonly
    
    
    def PRODUCT(self):
        return self._PRODUCT
    
    
    def BANDS(self):
        return self._BANDS
    
    
    def GET_MAX_CLOUD_FRACTION(self):
        return self._MAX_CLOUD_FRACTION
    
    
    def SET_MAX_CLOUD_FRACTION(self, new_cf):
        self._MAX_CLOUD_FRACTION = new_cf
        
        
    def GET_LABEL_NODATA_VALUE(self):
        return self._LABEL_NDV
    
    
    def SET_LABEL_NODATA_VALUE(self, value):
        self._LABEL_NDV=value
    
    
    def _invalidate_tiles(self):
        self._dl_tiles = None
        self._dl_tile_ids = None
    
    
    def _tag_with_cf(self):
        if self._MAX_CLOUD_FRACTION is None:
            return self._tag
        return f"{self._tag}-cf{str(self._MAX_CLOUD_FRACTION).replace('.','p')}"
    
    
    def _total_tile_size(self):
        return self._TILE_SIZE + (2 * self._TILE_PAD)
    
    
    def images_dir_name(self, loc, year):
        """Suggest the name of the subfolder for this export session based on the 
         pre-configured and passed variables:
            
            `tag-cloudfraction_resolution_padding_tilesize_loc_year`
            
            Images and labels will then go into images/ and labels/ subfolders of this."""
        return os.path.join(self._root,
                            f"{self._tag_with_cf()}_{self._TILE_RES}m_{self._TILE_PAD}pad_{self._total_tile_size()}_{loc}_{year}")
    
    
    def dataset_name(self, loc, year, tfrecord_type="arr"):
        """Suggest the name of the tfrecords dataset to be created from this export session:
        
        `tag-cloudfraction_tfrecordtype_loc_year"""
        return f"{self._tag_with_cf()}_{tfrecord_type}_{loc}_{year}"        
        
    
    def get_tiles(self):
        """Find DLTiles of the configured size/resolution that intersect the features in the 
        configured labels dataset. 
        
        Will be populated on first call or after updating tile size 
        details - this can take a minute or two."""
        if self._dl_tiles is None:
            self._populate_DLTiles()
        return self._dl_tiles
    
    
    def get_tile_ids(self):
        """Find DLTiles of the configured size/resolution that intersect the features in the 
        configured labels dataset, and return their IDs (keys). 
        
        Will be populated on first call or after updating tile size details - this can take
        a minute or two."""
        if self._dl_tile_ids is None:
            self._populate_DLTiles()
        return self._dl_tile_ids
    
    
    def get_labeldata_wgs84_df(self):
        """Get a pandas geodataframe of the configured OGR label dataset, 
        reprojected if necessary to EPSG:4326"""
        if self._gdf_wgs84 is not None:
            return self._gdf_wgs84
        gdf_labels = gpd.read_file(self._LABEL_DATA.OGR_DATASET, 
        layer = self._LABEL_DATA.get_layer_index())
        # our labels data are in UTM (or could be anything else) but DLTile.from_shape needs WGS84
        self._gdf_wgs84 = gdf_labels.to_crs('EPSG:4326')
        return self._gdf_wgs84
        
        
    def create_tile_job_configs(self, loc_label, year_label, 
                                ref_date, min_date=None, max_date=None):
        """Creates a list of DLTileJobConfigs, one for each tile required to cover the 
        label dataset at the specified tilesize / resolution, each with the specified 
        location and year labels (used to generate output file paths), dates, and 
        cloud fractions (for filtering the source data when creating the requested mosaics).
        
        The returned list can be used to map `create_chips_for_tile` over, either sequentially 
        or in parallel."""
        tile_export_jobs = [DLTileJobConfig.from_run_config(self, 
                                    dltile=t,
                                    out_folder_base=self.images_dir_name(loc_label, year_label),
                                    ref_date=ref_date,
                                    min_date=min_date,
                                    max_date=max_date,
                                    max_cloud_fraction=self._MAX_CLOUD_FRACTION
                                    ) for t in self.get_tiles()]
        return tile_export_jobs
        
        
    def _populate_DLTiles(self):
        
        gdf_wgs84 = self.get_labeldata_wgs84_df()
        
        # We need to get all DLTiles of the specified size and resolution that intersect the extent polygons.
        # We'll dissolve them first, and split back by parts. This will reduce the complexity of the geometries 
        # because all our the slum polygons sit within the non-slum ones and so can be merged into them for 
        # the purposes of finding intersecting tiles. This will help discourage DLTile.from_shape from breaking, 
        # which it likes to do sometimes.
        dissolved = gpd.geoseries.GeoSeries([geom for geom in gdf_wgs84.unary_union.geoms])
        
        # We'll then find all the intersecting tiles for each remaining geometry by iterating through one at a time. 
            # i.e. a panda-ised version of this
            #all_tiles = []
            #for _,f in gdf_wgs84.iterrows():
            #    these_tiles = dl.scenes.DLTile.from_shape(f['geometry'], TILE_RES, TILE_SIZE, TILE_PAD)
            #    print(str(len(these_tiles))+', ',end='')

            #    all_tiles.extend(these_tiles)
        # TODO use pandarallel or something to speed this up
        tiles = dissolved.apply(dl.scenes.DLTile.from_shape, args=(
            self._TILE_RES, 
            self._TILE_SIZE, 
            self._TILE_PAD
        ))  
        
        # Each row is now a list of tiles that intersected that polygon. Flatten back to a single list.
        all_tiles = [ x for i, y in tiles.apply(list).iteritems() for x in y]
        
        # Because multiple input polygons may intersect the same DLTile, there are likely to be duplicates 
        # as we have searched for covering tiles separately for each polygon each turn. The easiest way to 
        # avoid this happening would probably be to concatenate all shapes in to a single multipolygon and 
        # make a single call to DLTile.from_shape, but that's not very flexible, would be slow,  and there 
        # seems to be some limit on the complexity of geometry that DLTile.from_shape can cope with.

        # Instead, get a unique set of the tiles.
        # DLTile is not hashable so we can't just do set(all_tiles), instead we will use the string keys of 
        # the tiles as a comparator. That would suggest this approach to creating the unique list of tiles:

        # all_tile_ids = [t.key for t in all_tiles]
        # unique_tile_ids = set(all_tile_ids)
        # unique_tiles = [dl.scenes.DLTile.from_key(k) for k in unique_tile_ids]
                
        # However, DON'T do this: it's VERY slow. I think that creating a DLTile.from_key must be server-side 
        # so there are lots of round trips. Instead we'll compare the existing tile objects on their key and 
        # use it to build the list of unique tiles. 
        # This is the Decorate-Sort-Undecorate pattern, i.e. roll-your-own Set.
        unique_tile_ids = set()
        unique_tiles = []
        for tile in all_tiles:
            k = tile.key
            if k not in unique_tile_ids:
                unique_tile_ids.add(k)
                unique_tiles.append(tile)
                
        self._dl_tiles = unique_tiles
        self._dl_tile_ids = unique_tile_ids
                
            
            
def _get_scene_date_diff_mapper(reference_date):
    """Returns a function that will compare the date of a scene to the 
    originally-specified reference date"""
    # (note use of closure)
    def get_date_diff(scene):
        scene_date = scene.properties['date'].date()
        offset = abs(scene_date - reference_date)
        return offset
    return get_date_diff


def create_cloudmasked_s2_array(ctx, min_date=None, max_date=None,
                                bands="red green blue"):
    """Create a cloudfree mosaic of Sentinel-2 scenes matching the specified geocontext (dltile).
    
    The mosaic will be created by selecting all scenes intersecting the file; filtering to those after 
    min_date and/or before max_date if specified; applying the cloud mask from the separate 
    Descartes Labs S2 cloud mask product on a pixel-wise basis, and then returning the median of the 
    unmasked values at each pixel.
    
    If an error occurs then None will be returned. This generally suggests an error emanating 
    from the Descartes Labs API. This can suggest that no data are available for the specified 
    product / date range / cloud specification. However it can also occur randomly. So it's 
    always worth re-trying a few times, and or checking any error tiles in the DL Viewer to see 
    if data really are unavailable.
    
    See also create_img_array_for_tile which filters scenes by overall cloud fraction (if known) 
    but does not mask for remaining cloud within selected scenes.
    """
    # https://gis.stackexchange.com/a/367013
    s2_product = "sentinel-2:L1C"
    s2_cloud_product = "sentinel-2:L1C:dlcloud:v1"
    
    searchparams = {
        "aoi" : ctx,
        "products" : s2_product #product
    }
    if  min_date is not None:
        searchparams['start_datetime'] = min_date.isoformat()
    if max_date is not None:
        searchparams['end_datetime'] = max_date.isoformat()
    #scenes, newctx = dl.scenes.search(ctx, products=product)
    s2_scenes, s2_ctx = dl.scenes.search(**searchparams)
      
    if len(s2_scenes) == 0:
        return None
    
    s2_bands_stack = s2_scenes.stack(bands, s2_ctx, processing_level='surface', bands_axis=-1)
    
    searchparams["products"] = s2_cloud_product
    cloud_scenes, _ = dl.scenes.search(**searchparams)
    valid_cloudfree_mask_stack = cloud_scenes.stack('valid_cloudfree', s2_ctx, data_type='Byte', bands_axis=-1)
    n_bands = s2_bands_stack.shape[-1]
    valid_cloudfree_mask_stack = np.repeat(valid_cloudfree_mask_stack, repeats=n_bands, axis=-1)
    
    cloudfree_data_stack = np.ma.masked_where(valid_cloudfree_mask_stack==0, s2_bands_stack)
    
    cloudfree_mosaic = np.ma.median(cloudfree_data_stack, axis=0)
    return cloudfree_mosaic


def create_img_array_for_tile(ctx, product, reference_date, min_date=None, max_date=None, 
                     bands='red green blue', max_cloud_fraction=None):
    """Creates a mosaic of scenes matching the specified geocontext (dltile).
    The mosaic will be created by selecting all scenes intersecting the tile; filtering 
    to those after min_date and/or before max_date if specified; filtering to those with cloud 
    fraction < max_cloud_fraction if specified; and then prioritising the scenes closest in 
    time to the specified reference date.
    
    To create a mosaic of "latest available" data, just pass today's date or some arbitrary 
    future date for `reference_date` and None for `min_date` and `max_date`.
    
    The RGB image data (or other bands as specified) are returned as a 3D array
    whose shape will be equal to the geocontext's shape + 2*padding, * n bands i.e. 
    (height, width, bands).
    
    For datasets that support it, the "Surface Reflectance" processed data will be returned 
    (as opposed to e.g. TOA).
    
    If an error occurs then None will be returned. This generally suggests an error emanating 
    from the Descartes Labs API. This can suggest that no data are available for the specified 
    product / date range / cloud specification. However it can also occur randomly. So it's 
    always worth re-trying a few times, and or checking any error tiles in the DL Viewer to see 
    if data really are unavailable."""
    
    # TODO: use the scenecollection.stack method to mosaic rather than scenecollection.mosaic 
    # and thus allow median (or mean) mosaic rather than first. Then make reference_date an optional 
    # parameter, return closest-in-time pixel if it's not None as currently, and return median of 
    # all matching pixels otherwise. See the S2-specific create_cloudmasked_s2_array function.
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
    date_diff_mapper = _get_scene_date_diff_mapper(reference_date)
    sorted_scenes = scenes.sorted(date_diff_mapper, reverse=True)
    
    try:
        arr = sorted_scenes.mosaic(bands=bands, ctx=ctx, bands_axis=-1, processing_level="surface")
        return arr
    except:
        return None



def create_label_array_for_tile(ctx, label_data, attrib_to_burn=None, layer_idx=0, background_value=255):
    """Rasterises the label data (path to OGR datasource) within the specified geocontext.
    
    Parameters
    ----------
    ctx: DLTile (or dict)
        Geocontext as returned by dl.scenes.search(...). For example a DLTile object
    label_data: string
        Path to an OGR dataset (not layer) e.g. a file geodatabase folder, or a shapefile 
        or GeoJSON file
    attrib_to_burn: string
        Name of an attribute on the features giving the value that intersecting pixels should have.
        Must contain values in range 0 <= value <= 255. 
        If None, then all polygons will be burnt with value 1 and all other areas as nodata_value.
    layer_idx: int
        Index of the OGR layer within the OGR dataset. For shapefiles, GeoJSON, etc, these datasets 
        contain a single layer so this should be left at the default 0. For datasets that can contain 
        multiple layers (e.g. file geodatabase) this specifies which layer to use.
    background_value:
        What value should be given to any pixels that fall within the tile (ctx) but are not covered by 
        any polygon? 
        
        In a binary classification scheme (e.g. of-interest and non-interest polygons with
        values 1 and 0) you may wish to assume that areas not covered by any polygon  
        are the same as non-interest polygons (in which case provide 0 in this scenario), or you may wish 
        to record a separate "don't know" value, in which case provide something suitable such as the 
        default 255.
    
    Returns
    -------
    arr: ndarray
        A 2D uint8 array containing the burned pixel values with shape equal to geocontext's 
        shape + 2*padding
    """
    drv = gdal.GetDriverByName('MEM')
    img_size = ctx.tilesize + ctx.pad*2
    mem_ds = drv.Create('tmp', img_size, img_size, 1, gdal.GDT_Byte)
    mem_ds.SetProjection(ctx.wkt)
    mem_ds.SetGeoTransform(ctx.geotrans)
    _ = np.full([img_size,img_size], background_value, np.uint8)
    mem_ds.GetRasterBand(1).WriteArray(_)
    label_ogr_ds = ogr.Open(label_data)
    label_lyr = label_ogr_ds.GetLayerByIndex(layer_idx)
    # NB ALL_TOUCHED means all pixels intersecting a polygon at all are selected. This is as opposed to 
    # only pixels whose centre point falls within the polygon. The potential difficulty with setting this 
    # is that pixels on the boundary between two polygons will be selected by both, and thus will end up 
    # with the value of whichever polygon the OGR reader happens to emit last. So long as our pixels are 
    # small relative to the polygon sizes, we don't really need to worry, but if this wasn't the case it may 
    # be better not to set ALL_TOUCHED=TRUE.
    if attrib_to_burn:
        gdal.RasterizeLayer(mem_ds, [1], label_lyr, options=['ALL_TOUCHED=TRUE',f'ATTRIBUTE={attrib_to_burn}'])
    else:
        gdal.RasterizeLayer(mem_ds, [1], label_lyr, burn_values=[1], options=['ALL_TOUCHED=TRUE'])
    
    arr = mem_ds.ReadAsArray()
    mem_ds = None
    return arr



def create_chips_for_tile(job_details: DLTileJobConfig) -> tuple:
    """Creates image chips (geotiff training samples) for the specified  TileJobConfig.
    
    The image and label data files will be placed into /images and /labels subfolders below 
    the specified output folder location, and their name will be the DLTile's key with ':' 
    replaced by '#'.
    
    Returns a 3-tuple of (job_details, path_to_image, path_to_label). If the request to the 
    Descartes Labs API failed to generate an image, then returns (job_details, None, None). 
    In this case, you should re-try several times before concluding that data are not 
    available (or check in the DL Viewer), as the API is prone to transient failures.

    Function is pickleable so can be used as the task in multiprocessing via joblib etc
    """
    dltile = job_details.DLTILE
    out_base = job_details.OUTFOLDER
    product = job_details.PRODUCT
    target_date = job_details.TARGETDATE
    label_data = job_details.LABEL_DS
    label_lyr = job_details.LABEL_LYR_NUM
    label_attrib = job_details.LABEL_BURN_ATTR
    label_ndv = job_details.LABEL_NODATA_VALUE
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
    
    if max_cloud_fraction == 0 and product == "sentinel-2:L1C":
        # Sentinel 2 data with no max-cloud-fraction specified, use the pixelwise cloudfree median
        # composite method. NB reference date will be ignored
        img_arr = create_cloudmasked_s2_array(ctx=dltile, min_date=min_date, max_date=max_date, bands=bands)
    else:
        # Other data products or S2 data with max-cloud-fraction None or >0, use the scenewise-cloud-cover-only 
        # filtering method
        img_arr = create_img_array_for_tile(ctx=dltile, product=product, 
                                   reference_date=target_date, min_date=min_date, max_date=max_date,
                                   max_cloud_fraction=max_cloud_fraction,
                                   bands=bands)
        
    if img_arr is None:
        return (job_details, None, None)
    # rasterise the label data
    lbl_arr = create_label_array_for_tile(ctx=dltile, label_data=label_data, 
                                 attrib_to_burn=label_attrib,
                                layer_idx=label_lyr, background_value=label_ndv)
    img_file = os.path.join(out_img_folder, fn) + ".tif"
    lbl_file = os.path.join(out_lbl_folder, fn) + ".tif"
    # save the data to compressed geotiffs
    n_img_bands = img_arr.shape[-1]
    img_ds = _gdal_dataset_from_geocontext(ctx=dltile, n_bands=n_img_bands, driver_name="GTiff", 
                                          savename=img_file, dtype=img_arr.dtype, 
                                          options=['COMPRESS=LZW', 'TILED=TRUE', 'NUM_THREADS=4'])
    for b in range(n_img_bands):
        bnd = img_ds.GetRasterBand(b+1)
        bnd.WriteArray(img_arr[:,:,b])
    img_ds.FlushCache()
    img_ds=None
    lbl_ds = _gdal_dataset_from_geocontext(ctx=dltile, n_bands=1, driver_name="GTiff", 
                                          savename=lbl_file, dtype=lbl_arr.dtype, 
                                          options=['COMPRESS=LZW', 'TILED=TRUE', 'NUM_THREADS=4'])
    lbl_band = lbl_ds.GetRasterBand(1)
    if label_ndv is not None:
        lbl_band.SetNoDataValue(label_ndv)
    lbl_band.WriteArray(lbl_arr)
    lbl_band = None
    lbl_ds = None
    # return the paths
    return (job_details, img_file,lbl_file)


# Function from Descartes "wellpads" sample utils.py
def _gdal_dataset_from_geocontext(
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
    gdal_dtype = _numpy_dtype_to_gdal[dtype]
    driver = gdal.GetDriverByName(driver_name)
    ds = driver.Create(savename, n_rows, n_cols, n_bands, gdal_dtype, options=options)
    # Grab projection and geotransform from metadata.
    proj_wkt = ctx.wkt
    ds.SetProjection(proj_wkt)
    ds.SetGeoTransform(ctx.geotrans)
    return ds


# From Descartes "wellpads" sample utils.py
_numpy_dtype_to_gdal = {
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
