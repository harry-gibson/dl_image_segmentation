import tensorflow as tf
import numpy as np
from rasterio import MemoryFile
from rasterio.plot import reshape_as_image


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example prototype with 
    tf.train.Int64List type
    
    Value can be a numpy ndarray (of int datatype!) or a python list"""
    if isinstance(value, np.ndarray):
        value = value.flatten().tolist()
    elif not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float64_feature(value):
    """Wrapper for inserting float64 features into Example prototype with 
    tf.train.FloatList type
    
    Value can be a numpy ndarray (of float datatype!) or a python list 
    or a tf constant"""
    if isinstance(value, np.ndarray):
        #print("float64 feature: data are ndarray")
        value = value.flatten()#.tolist()
    elif isinstance(value, type(tf.constant(0))):
        value = value.numpy().flatten()#.tolist()
    elif not isinstance(value, list):
        #print("float64 feature: data are not list")
        value = [value]
    #else:
        #print("float64 feature: data are list")
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example prototype with 
    tf.train.BytesList type."""
    if isinstance(value, np.ndarray):
        #print("bytes feature: received array, storing as bytes string")
        value = [value.tobytes()]
    elif isinstance(value, type(tf.constant(0))):
        #print("bytes feature: received tensor, storing as bytes string")
        value = [value.numpy().tobytes()]
    elif not isinstance(value, list):
        #print("bytes feature: data are not list")
        value=[value]
    #else:
        #print("bytes feature: data are list")
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
        
    
def convert_to_example(img_data, target_data, img_h, img_w, img_b, target_h, target_w, identifier):
    """ Converts image and target data into TFRecords example.
    
    Derived from the Descartes Labs wellpads sample but extensively modified to handle different 
    types of imagery and different options for storage.
    
    Parameters
    ----------
    img_data: ndarray OR bytes object
        Image data
    target_data: ndarray OR bytes object
        Target data
    img_h, img_w, img_b: Shape of the image data (y,x,z)
    target_h, target_w: Shape of the target data (must be 1 band; this function does not enforce 
    that it must be same size as image but it generally should be.)
    identifier: str
        Identifier which can later be used to lookup / reconstruct the georeferencing information 
        of this tile. For example a DLTile key, in the case of Descartes Labs based tiles. If not 
        working with DLTiles, you would need to concoct a string representation of the geotransform 
        and projection and use that.

    If img_data and target_data are both bytes objects, or are ndarrays or tensors with UInt8 
    datatype, then they will both be stored as BytesList features. Otherwise they will both be
    stored as FloatList features. 
    
    Thus PNG/JPG images, as well as 8-bit images in other formats (e.g. GeoTIFF) can be passed as the
    raw encoded content (as bytes), OR decoded first to an 8-bit ndarray and the ndarray passed instead.
    In either such case the tfrecords will contain BytesList features. Parsing the tfrecords needs different 
    functions depending on what was passed and how it was stored (see below) because TF-native code can only
    decode PNG/JPG image formats (which are always 8 bit and 1 or 3 bands).
    
    Images with >8 bit depth and/or more than 3 bands can be passed as the raw GeoTIFF (etc) content, as bytes, 
    OR decoded first to a higher-bit-depth ndarray and the ndarray passed instead. In the first case the tfrecords 
    will contain BytesList features as for PNG etc and must be decoded on read. In the second case the tfrecords 
    will contain FloatList features and can be parsed directly to arrays. To keep parsing simpler we only store as 
    bytes if both img and target are compatible.

    In general this function should be called within a parallel processing pipeline using either 
    process_dataset_multithreaded or process_dataset_mp. Each of these functions can be called with 
    store_as_array=False to store the raw encoded image bytes in the TFRecords for decoding later, or 
    store_as_array=True to decode the images and store the array data in the TFRecords. 
    
    A summary of how to pass and subsequently decode different data types is below. If in doubt: 
    use process_dataset_mp(store_as_array=True) for the most flexible pipeline.

    * 8-bit 3-band (or 1 band) JPG/PNG images ONLY: either 
      - Pass raw file bytes (PNG/JPG encoded data) using 
          `process_dataset_multithreaded(store_as_array=False)`
            - tfrecords contain PNG/JPG-compressed data and are therefore much smaller on disk
            - creation of tfrecords from images uses TF code so can be multithreaded
            - optimally, read the tfrecords using `parse_encoded_rgb_img_example`; reading and 
              decoding of tfrecords uses TF code so can be multithreaded
            - alternatively read using `parse_gdal_example_py`
      - Decode to 8-bit ndarray, pass 8-bit ndarray using 
          `process_dataset_multithreaded(store_as_array=True)`
            - tfrecords contain uncompressed 8-bit array data and are therefore larger on disk
            - creation of tfrecords from images uses TF code so can be multithreaded
            - optimally, read the tfrecords using parse_8bit_array_example; reading and 
              decoding of tfrecords uses TF code so can be multithreaded
    * Any GDAL-compatible raster format (includes JPG/PNG and all other formats): either
      - Pass raw file bytes (e.g. PNG/JPG/GeoTIFF/BMP etc data) using 
          `process_dataset_mp(store_as_array=False)` 
            - tfrecords contain raw encoded file data and therefore may be smaller on disk, depending 
              on the file data (e.g. GeoTIFFs may or may not be compressed)
            - creation of tfrecords from images uses rasterio/GDAL code even though the images are not 
              being decoded, so cannot fully benefit from multithreading, however disk I/O is done via 
              TF so it is still quite fast
            - read the tfrecords using either `parse_gdal_example_tf` or `parse_gdal_example_py`;
              the former is slightly faster but always returns 32-bit arrays whereas the latter 
              returns arrays according to the original images datatype. Disk I/O uses TF methods 
              so is efficient but decoding image data from stored bytes uses rasterio/GDAL code; 
              thus may be a bottleneck in training pipeline (GIL not released).
      - Decode to n-bit ndarray, pass n-bit ndarray using `process_dataset_mp(store_as_array=True)`
            - tfrecords contain uncompressed image array data as FloatList features (or BytesList if 
              images were 8-bit) and are therefore larger on disk
            - creation of tfrecords from images uses rasterio/GDAL code so cannot fully benefit
              from multithreading, however disk I/O is done via TF so it is still quite fast
            - read the tfrecords using `parse_higher_example` (or `parse_8bit_array_example` if 
              you are sure the images were 8-bit.  Reading / decoding the tfrecords uses TF code only
              therefore is efficient in training pipeline.

    
    Returns
    -------
    Example: TFRecords example
        TFRecords example with the following feature template:
            
            feature = {
                "image/image_data": <BytesList OR FloatList feature, depending on bit depth of passed data>,
                "image/height": <Int64 feature specifying Y dimension of image>,
                "image/width": <Int64 feature specifying X dimension of image>,
                "image/channels": <Int64 feature specifying Y dimension of image (number of bands)>,
                "target/target_data": <BytesList OR FloatList feature, depending on bit depth of target 
                    burn attribute and of image data>,
                "target/height": <Int64 feature specifying Y dimension of image, will be same as image/height>,
                "target/width": <Int64 feature specifying X dimension of image, wil lbe same as image/width>,
                "identifier": <BytesList feature giving the passed identifier e.g. DLTile key>
            }
        
        In the case of 8-bit imagery, the example can be parsed using features_8bit_image. 
        In the case of higher bit-depth imagery, the example can be parsed using features_image_arrays
        Various other functions are provided to parse these examples using these templates into ndarrays for use in 
        training.

    """
    image_is_bytes=False
    target_is_bytes=False
    if isinstance(img_data, bytes):
        # a non-decoded image (png or jpeg on-disk actual file content)
        #wrapped_img_data = _bytes_feature(img_data)
        image_is_bytes=True
    elif isinstance(img_data, np.ndarray) or isinstance(img_data, type(tf.constant(0))):
        if img_data.dtype == 'uint8':
            # E.g. a decoded PNG/JPG/other 8-bit imagery
            #wrapped_img_data = _bytes_feature(img_data)
            image_is_bytes = True
            #wrapped_img_data = _float64_feature(img_data)
        #else:
            #print("img_data are other array")
            #wrapped_img_data = _float64_feature(img_data)
    #else:
    #    wrapped_img_data = _float64_feature(img_data)
    
    
    if isinstance(target_data, bytes):
        #wrapped_target_data = _bytes_feature(target_data)
        target_is_bytes=True
        
    elif isinstance(target_data,  np.ndarray) or isinstance(target_data, type(tf.constant(0))):
        if target_data.dtype == 'uint8' and image_is_bytes:
            wrapped_target_data = _bytes_feature(target_data)
            target_is_bytes=True
            #wrapped_target_data = _float64_feature(target_data)
        #else:
        #    wrapped_target_data = _float64_feature(target_data)
    #else:
    #    wrapped_target_data = _float64_feature(target_data)
    if image_is_bytes and target_is_bytes:
        wrapped_img_data = _bytes_feature(img_data)
        wrapped_target_data = _bytes_feature(target_data)
    else:
        wrapped_img_data = _float64_feature(img_data)
        wrapped_target_data = _float64_feature(target_data)
        
    features = {
        "image/image_data": wrapped_img_data,
        "image/height": _int64_feature(img_h),
        "image/width": _int64_feature(img_w),
        "image/channels": _int64_feature(img_b),
        "target/target_data": wrapped_target_data,
        "target/height": _int64_feature(target_h),
        "target/width": _int64_feature(target_w),
        #"target/channels": _int64_feature(target _shape[2]),
        "identifier": _bytes_feature(tf.compat.as_bytes(identifier))
    }

    return tf.train.Example(features=tf.train.Features(feature=features))


# feature template for image / target data that are stored as bytes strings
# This could be encoded png / jpeg data, or a decoded 8-bit array encoded as bytes
featuretemplate_bytestring_imagechip = {
    'image/image_data': tf.io.FixedLenFeature([], tf.string),
    'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/width': tf.io.FixedLenFeature([], tf.int64),
    'image/channels': tf.io.FixedLenFeature([], tf.int64),
    'target/target_data': tf.io.FixedLenFeature([], tf.string),
    'target/height': tf.io.FixedLenFeature([], tf.int64),
    'target/width': tf.io.FixedLenFeature([], tf.int64),
    'identifier': tf.io.FixedLenFeature([], tf.string)
}


# feature template for image / target data that are stored as array data
# This could be a decoded 8-bit image with >3 bands (e.g. geotiff), or one 
# with higher datatpye and any numebr of bands
featuretemplate_ndarray_imagechip = {
    'image/image_data': tf.io.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True),
    'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/width': tf.io.FixedLenFeature([], tf.int64),
    'image/channels': tf.io.FixedLenFeature([], tf.int64),
    #'target/target_data': tf.io.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True),
    'target/target_data': tf.io.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True),
    'target/height': tf.io.FixedLenFeature([], tf.int64),
    'target/width': tf.io.FixedLenFeature([], tf.int64),
    'identifier': tf.io.FixedLenFeature([], tf.string)
}


def _parse_byteslist_proto(example_proto):
    '''parses an example protobuf in which image/image_data and target/target_data 
    stored as a BytesList. They are returned as a raw bytes string, 
    it is up to caller to know whether this is a representation of an 8-bit 
    numpy array or an encoded png/jpg image'''
    image_features = tf.io.parse_single_example(example_proto, featuretemplate_bytestring_imagechip)
    
    img_height = tf.cast(image_features['image/height'], tf.int32)
    img_width = tf.cast(image_features['image/width'], tf.int32)
    img_channels = tf.cast(image_features['image/channels'], tf.int32)
    img_recorded_shape = (img_height, img_width, img_channels)
    
    target_height = tf.cast(image_features['target/height'], tf.int32)
    target_width = tf.cast(image_features['target/width'], tf.int32)
    target_recorded_shape = (target_height, target_width)
    
    img_bytes = image_features['image/image_data']
    target_bytes = image_features['target/target_data']
    identifier = tf.cast(image_features['identifier'], tf.string)
    
    return (img_bytes, img_recorded_shape, 
            target_bytes, target_recorded_shape, 
            identifier)


def parse_encoded_rgb_img_proto(example_proto):
    """parses an example protobuf in which image/image_data and target/target_data 
    are encoded PNG or JPG image with 3 and 1 bands respectively. Image data are 
    decoded to array using `tf.io.decode_image` thus only TF-supported image data 
    can be parsed (PNG, JPG, BMP).
    
    Returns 3-tuple of (img_array, label_array, identifier (DLTile key etc))"""
    
    img_bytes, im_rec_shp, target_bytes, tgt_rec_shp, identifier = ( 
        _parse_byteslist_proto(example_proto))
    
    # Use the tensorflow library to decode; for supported imagetypes this should 
    # be faster than rasterio/gdal. This function works for PNG, JPG, GIF, BMP
    # expand_animations=False in case it's a GIF
    img_arr = tf.io.decode_image(img_bytes, expand_animations=False)
    # as the image is stored in full its shape is implicit. Just check that it was 
    # recorded correctly in the feature template though
    #for i in range(len(img_arr.shape)):
    #    assert img_arr.shape[i] == im_rec_shp[i], f"{img_arr.shape[i]} is not same as {im_rec_shp[i]}"
        
    target_arr = tf.io.decode_image(target_bytes, expand_animations=False)
    #assert target_arr.shape[0] == tgt_rec_shp[0]
    #assert target_arr.shape[1] == tgt_rec_shp[1]
    
    return img_arr, target_arr, identifier


def parse_8bit_array_proto(example_proto):
    """Parses an example protobuf in which image/image_data and target/target_data 
    are 8-bit numpy arrays, stored as bytes strings
    
    Returns 3-tuple of (img_array, label_array, identifier (DLTile key etc))"""
    
    # use the same function for reading as for rgb encoded images
    img_bytes, im_rec_shp, target_bytes, tgt_rec_shp, identifier = ( 
        _parse_byteslist_proto(example_proto))
    
    img_arr_1d = tf.io.decode_raw(img_bytes, out_type='uint8')
    assert img_arr_1d.shape[0] == im_rec_shp[0] * im_rec_shp[1] * im_rec_shp[2], \
        "Decoded shape is %r - does not match" % img_arr_1d.shape
    # we have to reconstruct the correct shape from the recorded info
    img_arr = tf.reshape(tf.squeeze(img_arr_1d), tf.stack(im_rec_shp))
    
    target_arr_1d = tf.io.decode_raw(target_bytes, out_type='uint8')
    assert target_arr_1d.shape[0] == tgt_rec_shp[0] * tgt_rec_shp[1]
    target_arr = tf.reshape(tf.squeeze(target_arr_1d), tf.stack(tgt_rec_shp))
    
    return img_arr, target_arr, identifier


def _parse_bytes_gdal_numpyfunc(img_bytes_np, tgt_bytes_np):
    with MemoryFile(img_bytes_np) as memfile:
        with memfile.open() as src:
            img_arr = src.read()
    
    with MemoryFile(tgt_bytes_np) as memfile:
        with memfile.open() as src:
            target_arr = src.read()
    
    return (reshape_as_image(img_arr).astype(np.float32), 
            reshape_as_image(target_arr).astype(np.float32))


def parse_encoded_gdal_proto_wrapped(example_proto):
    """Parses an example protobuf in which image/image_data and target/target_data
    are encoded GDAL/rasterio-compatible image data. Wraps the underlying gdal function 
    which actually decodes the data into a typed numpy_function which means this parser 
    can be run in a pipeline, but pre-defines the array return type as float32.
    
    See also parse_encoded_gdal_example_eager which is not wrapped and so can only be 
    run in eager mode but allows different return types
    
    Returns 3-tuple of (img_array, label_array, identifier (DLTile key etc))"""
    # use the same function for reading as for rgb encoded images, in order to 
    # benefit from speed of tf.io.gfile
    (img_bytes, im_rec_shp, target_bytes, tgt_rec_shp, identifier) = _parse_byteslist_proto(example_proto)
    img_arr, target_arr = tf.numpy_function(_parse_bytes_gdal_numpyfunc, [img_bytes, target_bytes], [tf.float32, tf.float32])
    return img_arr, target_arr, identifier
    
    
def parse_encoded_gdal_proto_eager(example_proto):
    """ parses an example protobuf in which image/image_data and target/target_data 
    are encoded GDAL/rasterio-compatible image data. Arrays are returned with whatever 
    datatype they have on the input images. Needs access to the .numpy() 
    attribute of the tensors and so must be run in eager mode or else wrapped within a 
    tf.py_function, which would need to know the datatype that will be returned.
    
    See also parse_encoded_gdal_proto_wrapped which provides a wrapped version, which 
    can be run in a pipeline and returns float32 arrays in all cases.
    
    Returns 3-tuple of (img_array, label_array, identifier (DLTile key etc))
    """

    
    # use the same function for reading as for rgb encoded images, in order to 
    # benefit from speed of tf.io.gfile
    img_bytes, im_rec_shp, target_bytes, tgt_rec_shp, identifier = ( 
        _parse_byteslist_proto(example_proto))
    
    # decode the image bytes using rasterio, to parse any gdal-supported image format 
    with MemoryFile(img_bytes.numpy()) as memfile:
        with memfile.open() as src:
            img_arr = src.read()
    # swap axis order to that which tensorflow world expects i.e. height,width,bands 
    # rather than the normal (for GIS) bands,height,width
    img_arr = reshape_as_image(img_arr)
    # as the image is stored in full its shape is implicit. Just check that it was 
    # recorded correctly in the feature template though
    assert img_arr.shape == im_rec_shp
    
    with MemoryFile(target_bytes.numpy()) as memfile:
        with memfile.open() as src:
            target_arr = src.read()
    target_arr = reshape_as_image(target_arr)
    assert target_arr.shape[0] == tgt_rec_shp[0]
    assert target_arr.shape[1] == tgt_rec_shp[1]
        
    return img_arr, target_arr, identifier

    
def parse_higher_dtype_array_proto(example_proto):
    """Parses an example protobuf in which image/image_data and target/target_data 
    are numpy arrays, stored as floatlists
    
    Returns 3-tuple of (img_array, label_array, identifier (DLTile key etc))"""
    image_features = tf.io.parse_single_example(example_proto, featuretemplate_ndarray_imagechip)
    
    img_height = tf.cast(image_features['image/height'], tf.int32)
    img_width = tf.cast(image_features['image/width'], tf.int32)
    img_channels = tf.cast(image_features['image/channels'], tf.int32)
    
    target_height = tf.cast(image_features['target/height'], tf.int32)
    target_width = tf.cast(image_features['target/width'], tf.int32)
    
    img_raw = tf.reshape(tf.squeeze(image_features['image/image_data']),
                        tf.stack([img_height, img_width, img_channels]))
    
    target_raw = tf.reshape(tf.squeeze(image_features['target/target_data']),
                        tf.stack([target_height, target_width]))
    # todo: maybe always store target as bytes?
    # target_bytes = image_features['target/target_data']
    # target_raw = tf.reshape(tf.squeeze(tf.io.decode_raw(target_bytes, 'uint8')),
    #                    tf.stack([target_height, target_width]))
     
    identifier = tf.cast(image_features['identifier'], tf.string)
    
    return img_raw, target_raw, identifier
