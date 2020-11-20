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


def _process_image_and_lbl(img_path, lbl_path):
    img_arr, shp, dlkey = _process_image(img_path)
    lbl_arr, _, _ = _process_image(lbl_path)
    return img_arr, lbl_arr, shp, dlkey


def load_image_rasterio(img_path, parse_dltile_filename=True):

    # we use rasterio to parse the actual image data into an array
    # so that we can handle multi bands, different filetypes, dtypes
    # But we still use tf api to handle the actual file reading as it is
    # so much faster.
    with tf.gfile.FastGFile(img_path, 'rb') as f:
        image_data = f.read()
        with MemoryFile(image_data) as memfile:
            with memfile.open() as src:
                img_arr = src.read()
                gt_str = str(src.get_transform())
                crs_str = str(src.read_crs())

    #with rasterio.open(img_path) as src:
    #    img_arr = src.read() # reads all bands to 3d array bands,rows,cols
    #    gt_str = str(src.get_transform())
    #    crs_str = str(src.read_crs())

    img_arr = reshape_as_image(img_arr) #  converts dim order to rows,cols,bands
    if parse_dltile_filename:
        tile_key = '.'.join(os.path.basename(img_path).split(os.extsep)[:-1]).replace('#',':')
    else:
        if not (gt_str is None or crs_str is None):
            tile_key = '|'.join((gt_str, crs_str))
        else:
            tile_key = os.path.basename(img_path)
    return img_arr, img_arr.shape, tile_key


def _process_image_files_mp_worker(proc_index, ranges, 
                                   name, img_filenames, lbl_filenames, output_directory, 
                                   num_shards,
                                   dltile_from_filename):
    """Processes and saves part of a list of images as TFRecord in 1 process.
  Args:
    proc_index: integer, unique batch to run index is within [0, len(ranges)).
    ranges: list of pairs of integers specifying ranges of each batch to
      analyze in parallel.
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    texts: list of strings; each string is human readable, e.g. 'dog'
    labels: list of integer; each integer identifies the ground truth
    num_shards: integer number of shards for this data set.
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
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = img_filenames[i]
            label = lbl_filenames[i]
            
            try:
                image_buffer, image_shape, tile_key = load_image_rasterio(filename,
                                                                          dltile_from_filename)
                lbl_buffer, label_shape, _ = load_image_rasterio(label,
                                                                 dltile_from_filename)
            except Exception as e:
                print(e)
                print('SKIPPED: Unexpected eror while decoding %s.' % filename)
                continue
            
            example = convert_to_example(image_buffer, lbl_buffer, 
                                         image_shape[0], image_shape[1], image_shape[2], 
                                         label_shape[0], label_shape[1],
                                         tile_key)
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
        (datetime.now(), proc_index, counter, num_files_in_proc))
    sys.stdout.flush()
    
    
def _process_image_files_mp(name, img_files, lbl_files, out_folder,
                            num_shards, num_proc,
                            dltile_from_filename):
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
              dltile_from_filename)
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
    filenames = tf.gfile.Glob(img_file_path)
    labels = tf.gfile.Glob(lbl_file_path)

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
                       file_ext='tif'):
    """Process a complete data set and save it as a TFRecord.
    Args:
    name: string, unique identifier specifying the data set.
    directory: string, root path to the data set.
    num_shards: integer number of shards for this data set.
    """
    if not num_proc:
        num_proc = num_shards
    filenames, labels = _find_image_files(directory, file_ext)
    _process_image_files_mp(name, filenames, labels, out_directory,
                            num_shards, num_proc,
                            dltile_from_filename)
