import  tensorflow as tf
import numpy as np
from rasterio import MemoryFile
from rasterio.plot import reshape_as_image

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
    """Wrapper for inserting bytes features into Example proto."""
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
    Derived from the Descartes Labs wellpads sample but extensively modified.
    
    Parameters
    ----------
    img_data: ndarray OR bytes object
        Image data
    target_data: ndarray OR bytes object
        Target data
    img_h, img_w, img_b: Shape of the image data
    target_h, target_w: Shape of the target data (must be 1 band; this function does not enforce 
    that it must be same size as image but it generally should be.)
    identifier: str
        Identifier which can later be used to lookip/reconstruct the georeferencing information 
        of this tile. For example a DLTile key, in the case of Descartes Labs based tiles. Or 
        concoct a string representation of the geotransform and projection and use that.

    If img_data and target_data are both bytes objects, or are ndarrays or tensors with UInt8 
    datatype, then they will both be stored as BytesList features. Otherwise they will both be
    stored as FloatList features. (Note that this implies that we aren't supporting Int16 / 32 / 64 arrays yet)
    To keep parsing simpler we only store as bytes if both img and target are compatible.
    Returns
    -------
    Example: TFRecords example
        TFRecords example
    """
    image_is_bytes=False
    target_is_bytes=False
    if isinstance(img_data, bytes):
        # a non-decoded image (png or jpeg file content)
        #wrapped_img_data = _bytes_feature(img_data)
        image_is_bytes=True
    elif isinstance(img_data, np.ndarray) or isinstance(img_data, type(tf.constant(0))):
        if img_data.dtype == 'uint8':
            #print("img_data are 8 bit array")
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
features_8bit_image = {
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
features_image_arrays = {
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


def _parse_8bit_img_example(example_proto):
    '''parses an example protobuf in which image/image_data and target/target_data 
    stored as a BytesList. They are returned as a raw bytes string, 
    it is up to caller to know whether this is a representation of an 8-bit 
    numpy array or an encoded png/jpg image'''
    image_features = tf.io.parse_single_example(example_proto, features_8bit_image)
    
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


def parse_encoded_rgb_img_example(example_proto):
    '''parses an example protobuf in which image/image_data and target/target_data 
    are encoded PNG or JPG image with 3 and 1 bands respectively'''
    
    img_bytes, im_rec_shp, target_bytes, tgt_rec_shp, identifier = ( 
        _parse_8bit_img_example(example_proto))
    
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


def parse_8bit_array_example(example_proto):
    '''parses an example protobuf in which image/image_data and target/target_data 
    are bytes-encoded 8-bit numpy arrays'''
    
    # use the same function for reading as for rgb encoded images
    img_bytes, im_rec_shp, target_bytes, tgt_rec_shp, identifier = ( 
        _parse_8bit_img_example(example_proto))
    
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


def parse_gdal_example_tf(example_proto):
    ''' parses an example protobuf in which image/image_data and target/target_data
    are bytes-encoded GDAL/rasterio-compatible image data. Needs access to the .numpy() 
    attribute of the tensors and so must be run in eager mode or within a tf.py_function.
    
    See also parse_gdal_example_tf which hides this detail'''
    # use the same function for reading as for rgb encoded images, in order to 
    # benefit from speed of tf.io.gfile
    (img_bytes, im_rec_shp, target_bytes, tgt_rec_shp, identifier) = _parse_8bit_img_example(example_proto)
    img_arr, target_arr = tf.numpy_function(_parse_bytes_gdal_numpyfunc, [img_bytes, target_bytes], [tf.float32, tf.float32])
    return img_arr, target_arr, identifier
    
    
def parse_gdal_example_py(example_proto):
    ''' parses an example protobuf in which image/image_data and target/target_data 
    are bytes-encoded GDAL/rasterio-compatible image data. Arrays are returned with whatever 
    datatype they have on the input images. Needs access to the .numpy() 
    attribute of the tensors and so must be run in eager mode or within a tf.py_function, which 
    would need to know the datatype that will be returned.
    
    See also parse_gdal_example_tf which hides this detail and returns float32 arrays in all cases'''
    
    # use the same function for reading as for rgb encoded images, in order to 
    # benefit from speed of tf.io.gfile
    img_bytes, im_rec_shp, target_bytes, tgt_rec_shp, identifier = ( 
        _parse_8bit_img_example(example_proto))
    
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

    
def parse_higher_example(example_proto):
    ''' parses an example protobuf in which image/image_data and target/target_data 
    are numpy arrays, stored as floatlists'''
    image_features = tf.io.parse_single_example(example_proto, features_image_arrays)
    
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
