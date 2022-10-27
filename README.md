## Classification of histological images


# Data Preparation
1. Data download from GTEx Portal
2. PyHist
    * patch-size 128 
    * content-threshold 0.5 
    * output-downsample 16 
    * tilecross-downsample 32


# Models
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

