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
To create the predictive machine learning models the Python Data Science Ecosystem modules were used. In particular, Keras was used to train de CNN models. We trained one model from scratch and use two-pretrained architectures. The last layers were trained, followed by a fine-tunning step. The following architures were evaluated.

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
    * Dense Layer (correspond to #tissues) – sigmoid

3. Xception (imagenet)
    * Pre-trained layers - imagenet
    * Flatten layer
    * Dense Layer (correspond to #tissues) - sigmoid


### Training Process
![image](https://user-images.githubusercontent.com/22194539/198322041-f1fc7332-a681-4074-93fa-c5ec8738dd5e.png)


1. Batch Training
    * Batch size = 10, 15 or 20
2. Loss 
    * Cross entropy Binary or Categorical
3. Learn rate
    * Adam - Higher in fit and lower in tuning (1.e-5)
4. Evaluation on Test and validation dataset
    * At tile and sample level. A majority voting scheme was used, i.e. the most frequent label in the tiles from a sample was used to label the sample.

## Dataset

* Samples: 5469 (different tissues)
* Tiles: 483018
* Average tiles per sample: 88

* Number of samples in train set: 3500
* Number of samples in test set: 875
* Number of samples in validation set: 1094

Distribution of samples and tiles per tissue

![image](https://user-images.githubusercontent.com/22194539/198324091-be40ffc4-d1ee-42d1-aeb8-d9675342b3a2.png)


## Results
Some results on the test set. This is still a work in progress project and the results are constantly being updated.
Results are based on the VGG16 architecture.

* Tiles

![image](https://user-images.githubusercontent.com/22194539/198324286-268d8a78-20f5-4c7a-a889-a05b9e0f0a83.png)

* Samples

![image](https://user-images.githubusercontent.com/22194539/198324351-d4c81357-d79f-4207-ab04-3a3456a442b9.png)


![image](https://user-images.githubusercontent.com/22194539/198324440-485ec93c-5173-4065-9925-54764d13423b.png)



Clearly, histopathological images have enough signal that allows a classification per tissue types. The poorer performance for some tissues is mostly due to an unbalance representation. 


![image](https://user-images.githubusercontent.com/22194539/198322165-a2613515-3931-4d47-9357-0d1bf7229cf0.png)

