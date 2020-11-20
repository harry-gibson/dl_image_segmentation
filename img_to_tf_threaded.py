# Based on 
#https://github.com/tensorflow/models/blob/f87a58cd96d45de73c9a8330a06b2ab56749a7fa/research/inception/inception/data/build_image_data.py
# modified to read labels from images in a parallel folder
# i.e. expects data_dir/images/image0.png and data_dir/labels/image0.png

import random
import threading
from datetime import datetime
import tensorflow as tf
import os
import sys
import numpy as np

from tf_example_creation import convert_to_example

class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

        self._decode_png_data = tf.placeholder(dtype=tf.string)
        self._decode_png = tf.image.decode_png(self._decode_png_data, channels=3)

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] <= 3
        return image

    def decode_png(self, image_data):
        image = self._sess.run(self._decode_png,
                               feed_dict={self._decode_png_data: image_data})
        assert(len(image.shape)==3)
        assert(image.shape[2] <= 3)
        return image

def _is_png(filename):
    """Determine if a file contains a PNG format image.
    Args:
      filename: string, path of the image file.
    Returns:
      boolean indicating if the image is a PNG.
    """
    return '.png' in filename

def _process_image(filename, coder, parse_dltile_filename=True, png_to_jpg=False):
    """Process a single image file.
    Args:
      filename: string, path to an image file e.g., '/path/to/example.png'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the image file.
    with tf.gfile.FastGFile(filename, 'rb') as f:
        image_data = f.read()

    # Convert any PNG to JPEG's for consistency.
    # Modified to not do this unless requested: why would we want to?
    #  It makes the data smaller but that's because it throws info away. If we cared
    # about smaller files we wouldn't be using tfrecords!! Would need to compare
    # model results to see if it affects accuracy. Modified coder to read png instead
    if _is_png(filename):
        if not png_to_jpg:
            image = coder.decode_png(image_data)
        else:
            print('Converting PNG to JPEG for %s' % filename)
            image_data = coder.png_to_jpeg(image_data)
            image = coder.decode_jpeg(image_data)

    # Decode the RGB JPEG.
    else:
        image = coder.decode_jpeg(image_data)

    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] <= 3
    if parse_dltile_filename:
        tile_key = '.'.join(os.path.basename(filename).split(os.extsep)[:-1]).replace('#',':')
    else:
        tile_key = os.path.basename(filename)

    return image_data, height, width, image.shape[2], tile_key


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _process_image_files_worker(coder, thread_index, ranges,
                                name, filenames, labels, out_folder,
                                num_shards,
                                dltile_from_filename,
                                png_to_jpg):
    """Processes and saves list of images as TFRecord in 1 thread.
    Args:
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
      thread_index: integer, unique batch to run index is within [0, len(ranges)).
      ranges: list of pairs of integers specifying ranges of each batches to
        analyze in parallel.
      name: string, unique identifier specifying the data set
      filenames: list of strings; each string is a path to an image file
      texts: list of strings; each string is human readable, e.g. 'dog'
      labels: list of integer; each integer identifies the ground truth
      num_shards: integer number of shards for this data set.
    """
    # Each thread produces N shards where N = int(num_shards / num_threads).
    # For instance, if num_shards = 128, and the num_threads = 2, then the first
    # thread would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(out_folder, output_filename)
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]
            label = labels[i]

            try:
                image_buffer, iheight, iwidth, ibands, idlkey = _process_image(filename, coder,
                                                                               dltile_from_filename, png_to_jpg)
                lbl_buffer, lheight, lwidth, lbands, ldlkey = _process_image(filename, coder,
                                                                             dltile_from_filename, png_to_jpg)
                assert idlkey == ldlkey
            except Exception as e:
                print(e)
                print('SKIPPED: Unexpected eror while decoding %s.' % filename)
                continue

            example = convert_to_example(image_buffer, lbl_buffer, iheight, iwidth, ibands,
                                         lheight, lwidth, idlkey)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1

            if not counter % 1000:
                print('%s [thread %d]: Processed %d of %d images in thread batch.' %
                      (datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()

        writer.close()
        print('%s [thread %d]: Wrote %d images to %s' %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print('%s [thread %d]: Wrote %d images to %d shards.' %
          (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()


def _process_image_files(name, img_files, lbl_files, out_folder,
                         num_shards, num_threads,
                         dltile_from_filename, png_to_jpg):
    """Process and save list of images as TFRecord of Example protos.
    Args:
      name: string, unique identifier specifying the data set
      filenames: list of strings; each string is a path to an image file
      texts: list of strings; each string is human readable, e.g. 'dog'
      labels: list of integer; each integer identifies the ground truth
      num_shards: integer number of shards for this data set.
    """
    assert len(img_files) == len(lbl_files)

    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(img_files), num_threads + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (num_threads, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a generic TensorFlow-based utility for converting all image codings.
    coder = ImageCoder()

    threads = []
    for thread_index in range(len(ranges)):
        args = (coder, thread_index, ranges,
                name, img_files, lbl_files,  out_folder,
                num_shards,
                dltile_from_filename, png_to_jpg)
        t = threading.Thread(target=_process_image_files_worker, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %
          (datetime.now(), len(img_files)))
    sys.stdout.flush()


def _find_image_files(data_dir):
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

    # Construct the list of PNG files and labels.
    img_file_path = '%s/images/*.png' % (data_dir)
    lbl_file_path = '%s/labels/*.png' % (data_dir)
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

    print('Found %d png files across %d labels inside %s.' %
          (len(filenames), len(labels), data_dir))
    return filenames, labels


def process_dataset_multithreaded(name, directory, out_directory, num_shards, num_threads=None,
                                  dltile_from_filename=True,
                                  convert_png_to_jpg=False):
    """Process a complete data set and save it as a TFRecord.
    Args:
      name: string, unique identifier specifying the data set.
      directory: string, root path to the data set.
      num_shards: integer number of shards for this data set.
    """
    if not num_threads:
        num_threads=num_shards
    assert not num_shards%num_threads, ("Num shards must be a multiple of num threads (incl 1*)")
    filenames, labels = _find_image_files(directory)
    _process_image_files(name, filenames, labels, out_directory,
                         num_shards, num_threads,
                         dltile_from_filename, convert_png_to_jpg)