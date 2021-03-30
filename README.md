# dl_image_segmentation

Contains code for retrieving and preparing data for running image segmentation deep learning models, with a focus on the Descartes Labs API.

Repository consists of a python package [dl_segmentation_utils](./dl_segmentation_utils/), and three Jupyter notebooks  demonstrating use of the package functionality, as described below: 

## Creation of training data

The notebook [create_training_samples.ipynb](./create_training_samples.ipynb) contains code for generating training data, retrieving imagery from the [Descartes Labs catalog](https://www.descarteslabs.com/#dataRefinery) and creating corresponding label data from a provided spatial dataset. The data retrieval is based on the Descartes Labs API - it uses their functionality to divide the AOI into tiles, and retrieves imagery from their catalog. As such, registration / authentication with Descartes Labs is needed before this can be used.

The training data are created in the form of two parallel folders named `/images` and `/labels` with the following properties:
* Each folder contains an identically-named set of files. 
* Each pair of files (one from the `/images` folder and one from the `/labels` folder) comprises a training sample. 
* Each file is an LZW-compressed GeoTIFF covering the extent of one [DLTile](https://docs.descarteslabs.com/descarteslabs/scenes/docs/geocontext.html#descarteslabs.scenes.geocontext.DLTile) (authentication needed) and the name of the file is the DLTile key, with the ':' character replaced by '#'. 
* The images have the size, spatial resolution, and padding (overlap) specified in the notebook, and this information is embedded in the filename (each unique set of values for these variables generates a unique set of DLTiles). 
* The files in the `/images` folder are multi-band, where the number of bands and the datatype of the images depends on the source imagery product selected for retrieval. 
* The files in the `/labels` folder are single-band 8-bit images, representing a rasterised version of the provided OGR-compatible vector ground truth dataset for the area of that tile. Based on a choice in the notebook, the pixel values can be given by an attribute in the dataset, or all features can be given a value of 1.

This format is equivalent to that created by the ArcGIS [Export Training Data For Deep Learning](https://pro.arcgis.com/en/pro-app/latest/tool-reference/image-analyst/export-training-data-for-deep-learning.htm) tool in the "Classified Tiles" format. Following the ESRI convention we refer to the files as "image chips".

The images retrieved comprise a mosaic of available imagery, after optionally filtering by date range and cloud cover. At each pixel the output value is selected by a choice of methods: either taken from the image that is closest in time to a specified reference date  (whilst also optionally falling between a min/max date and having cloud cover not greater than that specified); or a median of pixels remaining after cloud masking and optional date filtering.

Image retrieval can be run in parallel on a single machine; speed varies based on the response time of the DL API and depends on the size of images and the size of the catalog. As a guide, retrieving VHR RGB data from the Airbus Pleiades dataset to 256x256 pixel tiles averages about 4 images per second when appropriately parallelised.

## Translation of image chips to TFRecords

The notebook [translate_chips_to_tfrecords.ipynb](./translate_chips_to_tfrecords.ipynb) contains code for translating image chips in the format described above into sharded TFRecord files. This code is not specific to Descartes Labs and can be used on image chip datasets created by other means such as the ESRI toolset.

Various options are provided for storing the data in the TFRecords, depending on the requirements for minimising file size vs ease of use. These boil down to a choice between storing the raw bytes of the (compressed) JPG/PNG/GeoTIFF-encoded imagery in the TFRecord, vs decoding the images to numerical arrays and storing those. Where the images are being decoded, there is also a choice between using the TF I/O image decoders to do this vs using more flexible python libraries (GDAL/Rasterio). In the former case the decoding is highly optimised and can be multithreaded, but only RGB 8-bit images in PNG/JPG/BMP format can be used. In the latter case, any [GDAL-compatible format](https://gdal.org/drivers/raster/index.html) can be used but the decoding is slightly slower and parallelisation must be by multiprocessing. In practice, the penalty is not large and a folder of ~6000 256x256 pixel RGB images can be converted in a few seconds by either approach.

## Parsing TFRecords

The notebook [parse_tfrecords.ipynb](./parse_tfrecords.py) contains sample code for parsing TFRecord datasets created by the above means. This notebook does not provide any end-to-end workflows but rather demonstration code for how to parse the TFRecords for use within a model development and training pipeline.

