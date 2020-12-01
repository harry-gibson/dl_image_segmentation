# Based  on
#https://github.com/tensorflow/models/blob/f87a58cd96d45de73c9a8330a06b2ab56749a7fa/research/inception/inception/data/build_image_data.py#L291
# modified for parsing imagery other than 8-bit 3-band PNG/JPGs.
# (See img_to_tf_threaded for an optimised version for using on those filetypes)

import random
from datetime import datetime
import tensorflow as tf
import os
import sys
import numpy as np
from joblib import Parallel, delayed
import multiprocessing

import rasterio
from rasterio.plot import reshape_as_raster, reshape_as_image
from rasterio.io import MemoryFile

from tf_example_creation import convert_to_example

       
def load_image_rasterio(img_path, parse_dltile_filename=True, decode=True):
    """Process a single image file, for any GDAL-compatible image type"""
    # we use rasterio to parse the actual image data into an array
    # so that we can handle multi bands, different filetypes, dtypes
    # But we still use tf api to handle the actual file reading as it is
    # so much faster.
    img_arr = None
    with tf.io.gfile.GFile(img_path, 'rb') as f:
        image_data = f.read()
        with MemoryFile(image_data) as memfile:
            with memfile.open() as src:
                if decode:
                    img_arr = src.read()
                gt_str = str(src.get_transform())
                crs_str = str(src.read_crs())
                height = src.height
                width = src.width
                bands = src.count

    #with rasterio.open(img_path) as src:
    #    img_arr = src.read() # reads all bands to 3d array bands,rows,cols
    #    gt_str = str(src.get_transform())
    #    crs_str = str(src.read_crs())

    
    if parse_dltile_filename:
        tile_key = '.'.join(os.path.basename(img_path).split(os.extsep)[:-1]).replace('#',':')
    else:
        if not (gt_str is None or crs_str is None):
            tile_key = '|'.join((os.path.basename(img_path), 
                                 gt_str, crs_str))
        else:
            tile_key = os.path.basename(img_path)
    if decode:
        img_arr = reshape_as_image(img_arr) #  converts dim order to rows,cols,bands
        # just in case we later extend to read a window/part of the dataset, put this trap 
        # in to make sure we don't trip over returning the wrong shape
        assert (height,width,bands)==img_arr.shape
        return img_arr, height, width, bands, tile_key
    else:
        return image_data, height, width, bands, tile_key


def _process_image_files_mp_worker(proc_index, ranges, 
                                   name, img_filenames, lbl_filenames, output_directory, 
                                   num_shards,
                                   dltile_from_filename,
                                  store_as_array):
    """Processes and saves part of a list of images as TFRecord in 1 process.
    Args:
      proc_index: integer, unique batch to run index is within [0, len(ranges)).
      ranges: list of pairs of integers specifying ranges of each batches to
        analyze in parallel.
      name: string, unique identifier specifying the data set
      img_filenames: list of strings; each string is a path to an image file
      lbl_filenames: list of strings; each string is a path to a label image file
      output_directory: folder to write the tfrecords to
      num_shards: integer number of shards for this data set
      dltile_from_filename: If this option is set, the filename will have any "#" converted 
      to ":" and be used as the identifier. Otherwise the filename will be concatenated 
      with the image GT and CRS strings for the identifier.
      store_as_array: If False then the image data will be stored directly (as whatever format 
      it was in originally), if True then it will be decoded and stored as UInt8 or Float64 array
    """
    # Each proc produces N shards where N = int(num_shards / num_threads).
    # For instance, if num_shards = 128, and the num_threads = 2, then the first
    # process would produce shards [0, 64).
    num_proc = len(ranges)
    assert not num_shards % num_proc
    num_shards_per_batch = int(num_shards / num_proc)

    shard_ranges = np.linspace(ranges[proc_index][0],
                             ranges[proc_index][1],
                             num_shards_per_batch + 1).astype(int)
    num_files_in_proc = ranges[proc_index][1] - ranges[proc_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = proc_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(output_directory, output_filename)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        writer = tf.io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = img_filenames[i]
            label = lbl_filenames[i]
            
            try:
                image_buffer, iheight, iwidth, ibands, itile_key = load_image_rasterio(
                    filename, dltile_from_filename, store_as_array)
                lbl_buffer, lheight, lwidth, lbands, ltile_key = load_image_rasterio(
                    label, dltile_from_filename, store_as_array)
                assert itile_key == ltile_key
            except Exception as e:
                print(e)
                print('SKIPPED: Unexpected eror while decoding %s.' % filename)
                continue
            
            example = convert_to_example(image_buffer, lbl_buffer, 
                                         iheight, iwidth, ibands, 
                                         lheight, lwidth, itile_key)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1

            if not counter % 100:
                print('%s [process %d]: Processed %d of %d images in process batch.' %
                  (datetime.now(), proc_index, counter, num_files_in_proc))
                sys.stdout.flush()

        writer.close()
        print('%s [process %d]: Wrote %d images to %s' %
          (datetime.now(), proc_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print('%s [process %d]: Wrote %d images to %d shards.' %
        (datetime.now(), proc_index, counter, num_shards_per_batch))
    sys.stdout.flush()
    
    
def _process_image_files_mp(name, img_files, lbl_files, out_folder,
                            num_shards, num_proc,
                            dltile_from_filename,
                           store_as_array):
    assert len(img_files) == len(lbl_files)

    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(img_files), num_proc + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])
    proc_args = []
    num_proc = len(ranges)
    for proc_idx in range(len(ranges)):
        args=(proc_idx, ranges, 
              name, img_files, lbl_files, out_folder, 
              num_shards, 
              dltile_from_filename,
             store_as_array)
        proc_args.append(args)
    res = Parallel(n_jobs=num_proc)(delayed(_process_image_files_mp_worker)(*a) for a in proc_args)
    return res


def _find_image_files(data_dir, file_ext):
    """Build a list of all images files and labels in the data set.
    Args:
    data_dir: string, path to the root directory of images.
      Assumes that the image data set resides in JPEG files located in
      the following directory structure.
        data_dir/dog/another-image.JPEG
        data_dir/dog/my-image.jpg
      where 'dog' is the label associated with these images.
    labels_file: string, path to the labels file.
      The list of valid labels are held in this file. Assumes that the file
      contains entries as such:
        dog
        cat
        flower
      where each line corresponds to a label. We map each label contained in
      the file to an integer starting with the integer 0 corresponding to the
      label contained in the first line.
    Returns:
    filenames: list of strings; each string is a path to an image file.
    texts: list of strings; each string is the class, e.g. 'dog'
    labels: list of integer; each integer identifies the ground truth.
    """
    print('Determining list of input files and labels from %s.' % data_dir)


    # Leave label index 0 empty as a background class.

    # Construct the list of JPEG files and labels.
    img_file_path = '%s/images/*.%s' % (data_dir, file_ext)
    lbl_file_path = '%s/labels/*.%s' % (data_dir, file_ext)
    filenames = tf.io.gfile.glob(img_file_path)
    labels = tf.io.gfile.glob(lbl_file_path)

    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to label in the
    # saved TFRecord files. Make the randomization repeatable.
    shuffled_index = list(range(len(filenames)))
    random.seed(12345)
    random.shuffle(shuffled_index)

    filenames = [filenames[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]

    print('Found %d %s image files and %d label files inside %s.' %
        (len(filenames), file_ext, len(labels), data_dir))
    return filenames, labels


def process_dataset_mp(name, directory, out_directory,
                       num_shards, num_proc=None,
                       dltile_from_filename=True,
                       file_ext='tif',
                      store_as_array=True):
    """Process a folder of images and label images and save it as TFRecords. Processes any 
    GDAL-compatible image format. 
    (See also Use process_dataset_threaded for a faster implementation compatible with RGB 
    or single-band images only, in PNG or JPG format.)
    Args:
      name: string, unique identifier specifying the data set, used to name the output files.
      directory: string, root path to the data set. Must have subfolders 'images' and 'labels'
      out_directory: folder where the tfrecords will be saved
      num_shards: integer number of shards (split tfrecords files) for this data set. Must be 
      a multiple of num_proc.
      num_proc: number of proc to use for parallel processing. Generally ok to use at least 1
      per core.
      dltile_from_filename: Defines how the georeferencing information should be stored in the record. 
      
      The TFRecord examples will have an "identifier" feature which will be needed to match the record back to the source image, and so recover the georeferencing information for the data if it's stored as arrays.
      If this argument is True then the filename is assumed to be a DLTile key, with ':' replaced by '#', so the identifier will be set to the filename with '#' replaced ':' in the identifier. 
      If this is False, then the identifier will be set to the '{IMG_FILENAME}|{IMG_GEOTRANSFORM}|{IMG_CRS}'.
      file_ext: The extension of the image files to search for in the directory. (Labels and images 
      must be in same format).
      store_as_array: If False then binary encoded image data will be stored in the TFRecords, which in the case of a compressed image format will give smaller files. However the images will then need 
      to be decoded in the training pipeline which may be slower. If True then the images will be decoded and the data arrays will be stored, either as UInt8 or Float64 types.
    """
    if not num_proc:
        num_proc = num_shards
    filenames, labels = _find_image_files(directory, file_ext)
    _process_image_files_mp(name, filenames, labels, out_directory,
                            num_shards, num_proc,
                            dltile_from_filename,
                           store_as_array)
