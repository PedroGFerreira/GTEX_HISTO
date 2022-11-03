
The **img2tiles.py** script is used to download the whole slide images from the GTEx portal and through the **PyHist** tool, divides the image into tiles, selects the tiles with minimum area covered with tissue and other filters. 
It requires the installation of **PyHist**, see:
https://pyhist.readthedocs.io/en/latest



The **cnn_tissue_classifier.py** script performs the classification of the images per tissue. Use the command line to set the parameters for train and fine tune. There are three architectures based on: i) built from scratch CNN; ii) CNN pre-trained with VGG16 and CNN pre-trained with xception.

The scripts runs better in a GPU. You will need to configure the GPU in your system.  To configure the use of your system set th following line:

# 0 for first GPU; 1 for second GPU and -1 for CPU
os.environ["CUDA_VISIBLE_DEVICES"]="0" # second gpu

The variable TF_CPP_MIN_LOG_LEVEL controls the level of log. Check here the different levels:
https://deepreg.readthedocs.io/en/latest/docs/logging.html


TF_CPP_MIN_LOG_LEVEL

"0” Log all messages.
“1” Log all messages except INFO. 
“2” Log all messages except INFO and WARNING. (default)
“3” Log all messages except INFO, WARNING, and ERROR.
