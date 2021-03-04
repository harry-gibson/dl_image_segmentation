# dl_image_segmentation

Contains code for retrieving and preparing data for running image segmentation deep learning models.

The code for data retrieval is based on the Descartes Labs API - it uses their functionality to divide the AOI into tiles, and retrieves imagery from their catalog. 

The code for converting locally-stored image chips into TFRecords is not specific to Descartes Labs and can be used on image chip datasets created by other means.
