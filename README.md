# Classification of histological images
This project aims to create a prediction model of the tissue type based on histological whole slide images. The data is part of the GTEx project.
One goal is to obtain correlation of image features with gene expression patterns.

## Data 
Data is obtained from the Tissues& Histology section from the [GTEx portal] (https://www.gtexportal.org/home/).
Files are downloaded as Aperio Images in SVS format.


## Data Preparation
The [PyHist package] (https://pyhist.readthedocs.io/en/latest/) is used to segment the image and extract the patches or tiles. As command-line tool it can be easily configured. Example parameters

1. PyHist
    * patch-size 128 
    * content-threshold 0.5 
    * output-downsample 16 
    * tilecross-downsample 32


## Models


1. Scratch CNN
    * 4 X Conv2D (32, 64, 128, 18) + BatchNorm + MaxPool2D
    * Flatten layer
    * DropOut 0.5
    * BatchNorm
    * Dense Layer (#tissues) - sigmoid

2. VGG16
    * Pre-trained layers
    * Flatten layer
    * Dense layer (50) - relu
    * Dense Layer (#tissues) – sigmoid

3. Xception (imagenet)
    * Pre-trained layers - imagenet
    * Flatten layer
    * Dense Layer (#tissues) - sigmoid

