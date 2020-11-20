import  tensorflow as tf
import numpy as np

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
    if isinstance(value, np.ndarray):
        value = [value.tobytes()]
    elif not isinstance(value, list):
        value=[value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
        
    
def convert_to_example(img_data, target_data, img_h, img_w, img_b, target_h, target_w, identifier):
    """ Converts image and target data into TFRecords example.
    From the Descartes Labs wellpads sample.
    
    Parameters
    ----------
    img_data: ndarray or bytes object
        Image data
    target_data: ndarray
        Target data
    img_shape: tuple
        Shape of the image data (h, w, c)
    target_shape: tuple
        Shape of the target data (h, w, c)
    identifier: str
        Identifier which can later be used to lookip/reconstruct the georeferencing information 
        of this tile. For example a DLTile key, in the case of Descartes Labs based tiles. Or 
        concoct a string representation of the geotransform and projection and use that.
    
    Returns
    -------
    Example: TFRecords example
        TFRecords example
    """
    #if len(target_shape) == 2:
    #    target_shape = (*target_shape, 1)

    if isinstance(img_data, np.ndarray):
        if img_data.dtype == 'uint8':
            wrapped_img_data = _bytes_feature(img_data)
        else:
            wrapped_img_data = _float64_feature(img_data)
    elif isinstance(img_data, bytes):
        wrapped_img_data = _bytes_feature(img_data)
    else:
        wrapped_img_data = _float64_feature(img_data)
    if ((isinstance(target_data,  np.ndarray) and target_data.dtype == 'uint8')
     or isinstance(target_data, bytes)):
        wrapped_target_data = _bytes_feature(target_data)
    else:
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
        "identifier": _bytes_feature(tf.compat.as_bytes(identifier)),
    }

    return tf.train.Example(features=tf.train.Features(feature=features))

