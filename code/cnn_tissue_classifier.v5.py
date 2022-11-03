#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 22:21:45 2022

@author: Pedro G. Ferreira
Classifier for tissue images


"""
import os, logging, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import matplotlib.image as img
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, applications
import random
import argparse


# 0 for first GPU; 1 for second GPU and -1 for CPU
os.environ["CUDA_VISIBLE_DEVICES"]="1" # second gpu

# global vars
tissue_list = []
tissue_encoding = {}
# initialize the total number of training and testing samples
NUM_TRAIN_SAMPLES = 0
NUM_VALID_SAMPLES = 0
NUM_TEST_SAMPLES = 0
tissue_encoding = {}
ADAM_LEARN_RATE = 0.01

def scan_tiles_bylist(sample_list, folder, metadata, include_data = 1):
    '''loads a dataset with the sample tiles from the samples indicated in sample-list'''
    # existing folders
    samples_subfolders = [ f.name for f in os.scandir(folder) if f.is_dir() ]
    # intersect with sample list to make sure they are present
    samples_subfolders = list(set(samples_subfolders).intersection( set(sample_list)))
    data_lst = []
    cnt = 0
    df = []
    res = []
    
    for sample in samples_subfolders:
        sample_tiles_folder = "%s/%s/%s_tiles" %(folder, sample, sample)
        if os.path.isdir(sample_tiles_folder):
            tiles = [ f.name for f in os.scandir(sample_tiles_folder) if f.is_file() ]
            for t in tiles:
                fig = sample_tiles_folder + "/" + t		
                data = img.imread(fig)
                tissue = metadata.loc[sample].Tissue
                if data.shape != (128, 128, 3):
                    continue
                    print(data.shape)
                    print(sample + " " + tissue + " "  +  " " + t + " " + fig)
                #plt.imshow(data)
                data_lst.append(data)
                tissue = metadata.loc[sample].Tissue
                res.append((t, sample, tissue))
        cnt += 1
    if include_data:
        df = np.stack(data_lst, axis = 0) # create a 4D array from a list of 3D arrays
    else:
        df = []
    res_df = pd.DataFrame(res, columns = ["Tile","Sample","Tissue"])
    return (df, res_df)


def plot_cf_matrix(cf_matrix, model_type, file_name, names, image_type):
    '''  Plots the confusion matrix of the model '''
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    
    labels = [f"{v1}\n{v2}\n" for v1, v2 in zip(group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(int(cf_matrix.shape[0]), int(cf_matrix.shape[0]))

    plt.figure(figsize=(15,8))
    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')

    ax.set_title(('Tissue Confusion Matrix - %s - %s \n\n') % (image_type, model_type));
    ax.set_xlabel('\nPredicted Tissue')
    ax.set_ylabel('Actual Tissue');
    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(names)
    ax.yaxis.set_ticklabels(names)
    ## Display the visualization of the Confusion Matrix.
    #plt.show()
    plt.savefig(file_name)

def plot_metric(history, metric):
    ''' plots the figure with metrics '''
    train_metrics = history.history[metric]
    val_metrics = history.history['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics)
    plt.plot(epochs, val_metrics)
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    #plt.show()
    file_name = metric + ".png"
    plt.savefig(file_name)


def load_gtex_metadata(metadata_file):
    '''
    Input: file with GTEx metadata retrieved from GTEx portal 
    https://gtexportal.org/home/histologyPage --> All histology Samples (csv)
    Returns pandas object with table
    loads the info in the table to a Pandas dataframe
    '''
    # metadata file
    metadata = pd.read_csv(metadata_file, sep = ",")
    # remove spaces from column labels
    metadata.columns = [c.replace(" ","") for c in list(metadata.columns)]
    #  modifies the DataFrame in place (do not create a new object).
    metadata.set_index("TissueSampleID", inplace = True)
    return metadata
  
  
def most_frequent(List):
    ''' returns the most frequent event in a list '''
    return max(set(List), key= List.count)


def vote_prediction(df):
    ''' receives a pandas df with tiles predictions and sample value
    infers the sample class based on the predictions using a voting scheme like majority'''
    sample_lst = list(np.unique(df.Sample))
    res = []
    for sample in sample_lst:
        tile_predictions = list(df.loc[df.Sample == sample, "Predicted"])
        most_predicted = most_frequent(tile_predictions)
        res.append((sample, most_predicted))
    pred_df = pd.DataFrame(res, columns = ["Sample", "Predicted"])
    return pred_df
 
def evaluate_model(df, tile_metadata, model, label, df_generator, df_steps, model_type):
    ''' returns evaluation statistics at tile and sample level. '''
    tiles_tissue_lst = list(tile_metadata.Tissue)
    # Results for tiles
    df_labels = get_tissue_encodings(tiles_tissue_lst)
    score = model.evaluate(df, df_labels, verbose=0)
    print(label +" Loss:", score[0])
    print(label + " Accuracy:", score[1])
    
    # predictions
    y_pred_proba = model.predict(df) # prediction probabilities
    y_pred_classes = y_pred_proba.argmax(axis = -1) # predicted classes : tiles
    tile_metadata["Predicted"] = y_pred_classes
    y_classes = df_labels.argmax(axis = -1) # true classes : tiles
    tile_metadata["Actual"] = y_classes
    print("Tile Report\n")
    names = list(map(str, np.unique(tiles_tissue_lst)))
    file_name = "TileReport_cf_matrix_" + model_type + ".png"
    cf_matrix = confusion_matrix(y_classes, y_pred_classes)
    print(cf_matrix)
    plot_cf_matrix(cf_matrix, model_type, file_name, names, "Tile")
    print(classification_report(y_classes, y_pred_classes, target_names = names))
    
    # Results per Sample
    # true values
    sample_true_classes = tile_metadata.loc[:,["Sample","Tissue","Actual"]].drop_duplicates()
    sample_pred_classes = vote_prediction(tile_metadata)
    sample_true_classes = sample_true_classes.merge(sample_pred_classes, left_on='Sample', right_on="Sample")
    print("Sample Report\n")   
    file_name = "SampleReport_cf_matrix_" + model_type + ".png"
    names = list(map(str, np.unique(sample_true_classes.Tissue)))
    cf_matrix = confusion_matrix(sample_true_classes.Actual, sample_true_classes.Predicted)
    print(cf_matrix)
    plot_cf_matrix(cf_matrix, model_type, file_name, names, "Sample")
    print(classification_report(sample_true_classes.Actual, sample_true_classes.Predicted,target_names = names))
       

def prepare_model(batch_generator, train_steps, model, batch_size, epochs, validation_steps, validation_data):
    ''' Compiles the model and fits using a batch generator object  '''
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy", tf.keras.metrics.CategoricalAccuracy()])
    history = model.fit(batch_generator, 
                        epochs=epochs, 
                        steps_per_epoch = train_steps, 
                        validation_steps= validation_steps,
                        validation_data = validation_data, 
                        verbose = 1,
			            workers = 1)
                        #use_multiprocessing=True)
    return history

def encode_all_tissues(tissue_lst):
    '''returns the encoding object for all tissues'''
    le = preprocessing.LabelEncoder()
    le.fit(tissue_lst)
    # transform categorical to numerical
    return le


def setup_model(input_shape, num_classes):
    '''
    defines the architecture of the scratch model 
    '''
    print("Number of labels %d" %(num_classes))
    model = keras.Sequential([
        keras.Input(shape=input_shape),

    layers.Conv2D(16, kernel_size=(5, 5), activation="relu"),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
	
	layers.Conv2D(32, kernel_size=(5, 5), activation="relu"),
    layers.Dropout(0.3),
	layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
	
	layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.Dropout(0.3),
	layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
        
	layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
    layers.Dropout(0.3),
	layers.BatchNormalization(),
	layers.MaxPooling2D(pool_size=(2, 2)),
	
    layers.Flatten(),
    layers.Dropout(0.5),
	layers.BatchNormalization(),
        
	layers.Dense(num_classes, activation="softmax"),
    ])
    return model

def setup_model_pretrained_Xception(input_shape, num_classes):
    '''
    defines the model based on transfer learning of the Xception model pre-trainedwith imagenet
    pre-trained layers are at this point set as non-trainable
    We add additional layers at the end of the model: flatten layer and 2 dense layers 
    softmax is used for multi-class classification
    '''
    
    # base model with pre-trained weights
    model = applications.Xception(include_top=False, input_shape=input_shape, weights="imagenet")
    # freeze the model; mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
    # create a new model on top
    inputs = layers.Input(shape=input_shape)
    x = model(inputs, training = False)
    # convert features of shape model.output_shape[1:] to vector
    # add new classifier layers
    x1 = layers.GlobalAveragePooling2D()(x)
    # a dense classifier with n classes
    output = layers.Dense(num_classes, activation='softmax')(x1)
    # define new model
    model = models.Model(inputs=[model.inputs], outputs=output)
    return model

def setup_model_pretrained_VGG16(input_shape, num_classes):
    '''
    defines the model based on transfer learning of the pre-trained VGG16 model with imagenet
    pre-trained layers are at this point set as non-trainable
    We add additional layers at the end of the model: flatten layer and 2 dense layers 
   '''
    # load model without classifier layers
    model = applications.VGG16(include_top=False, input_shape=input_shape)
    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
    # add new classifier layers
    flat1 = layers.Flatten()(model.layers[-1].output)
    class1 = layers.Dense(50, activation='relu')(flat1)
    output = layers.Dense(num_classes, activation='softmax')(class1)
    # define new model
    model = models.Model(inputs=model.inputs, outputs=output)
    return model



def finetune_model(fine_tune_at, batch_generator, train_steps, model, batch_size, epochs, validation_steps, validation_data):
    '''
    After training, models can be fine tuned to try to improve performance.
    Set all layers of the model as trainable.
    Use a very low learning rate    '''
    # Unfreeze the base model
    model.trainable = False
    print("start to use at layer: %d" %( fine_tune_at))
    for layer in model.layers[fine_tune_at:]:
    	layer.trainable = True
	
    for layer in model.layers:
        print("{}: {}".format(layer, layer.trainable))

    model.compile(loss="categorical_crossentropy", 
                  optimizer=keras.optimizers.Adam(ADAM_LEARN_RATE),  # Very low learning rate
                  metrics=["accuracy", tf.keras.metrics.CategoricalAccuracy()])
		  
    print("fit for fine tuning")
    model.fit(batch_generator, 
                        epochs=epochs, 
                        steps_per_epoch = train_steps, 
                        validation_steps= validation_steps,
                        validation_data = validation_data, 
                        verbose = 2)

def scan_tiles(folder, metadata, top_n):
    '''
    Scan tiles and keep there labels; traverse the folder structure
    '''
    samples_subfolders = [ f.name for f in os.scandir(folder) if f.is_dir() ]
    data_lst = []
    #labels_lst = []
    tissue_lst = []
    sample_lst = []
    cnt = 0
    if top_n == -1:
        top_n = 1000000000
    for sample in samples_subfolders:
        print(str(cnt) + " " + sample)
        if cnt > top_n:
            break
        else:
            sample_tiles_folder = "%s/%s/%s_tiles" %(folder, sample, sample)
            sample_lst.append(sample)
            if os.path.isdir(sample_tiles_folder):
                tiles = [ f.name for f in os.scandir(sample_tiles_folder) if f.is_file() ]
                for t in tiles:
                    fig = sample_tiles_folder + "/" + t
                    tissue = metadata.loc[sample].Tissue
                    #tissue_enc = enc.transform(np.array(tissue).reshape(-1,1)).toarray()
                    #a = print(tissue_enc[0])
                    #print(sample + " " + tissue + " "  +  " " + t + " " + fig)
                    data = img.imread(fig)
                    if data.shape != (128, 128, 3):
                        continue
                        print(data.shape)
                        print(sample + " " + tissue + " "  +  " " + t + " " + fig)
                    #plt.imshow(data)
                    data_lst.append(data)
                    #labels_lst.append(tissue_enc)
                    #tissue_lst.append(tissue)
            cnt += 1
    df = np.stack(data_lst, axis = 0) # create a 4D array from a list of 3D arrays
    #labels_lst = np.array(labels_lst)
    return (df, tissue_lst, sample_lst)

def batch_generator(sample_ids, sample_labels, batch_size, steps, folder, metadata):
    '''
    Given that the dataset may easily reach several Gigabyte and do not fit in CPU/GPU
    Model will be trained in batches. This function receives a list of sample identifiers
    a batch size and through various iterations creates sucessive batches of images
    '''
    idx=1
    while True: 
        yield load_data(sample_ids, sample_labels, idx-1, batch_size, folder, metadata)## Yields data
        if idx<steps:
            idx+=1
        else:
            idx=1



def load_data(sample_ids, sample_labels, idx, batch_size, folder, metadata):
    '''
    Loads the data in batches 
    '''
    pos = idx * batch_size
    sample_list = sample_ids[pos: pos+batch_size]
    # scan tiles and generate  a dataframe (df) a list of tissues per tile (tissue_lst) and list of samples
    df, tile_metadata = scan_tiles_bylist(sample_list, folder, metadata, 1)
    df_labels = get_tissue_encodings(tile_metadata.Tissue)
    return (df, df_labels)

    
        
def tissue_OHE(tissues_lst):
    '''     One hot encoding of the tissue labels.  '''
    enc = preprocessing.OneHotEncoder()
    enc.fit(np.array(tissues_lst).reshape(-1,1))
    return enc
        
def get_tissue_list(folder, metadata):
    '''
    Parameters folder : folder path for the location of the tile images
    Description: Scans all the folders in this path and get the list of samples.
    From the list of samples retrieve the corresponding list of tissues.
    This will be useful to perform the Encoding of the output label
    Returns: unique list of tissues in the folder
    '''
    samples = [ f.name for f in os.scandir(folder) if f.is_dir() ]
    #tissues = list(set(list(metadata[samples,:].Tissue)))
    #tissues = list(set(list(metadata.loc[metadata.index.isin(samples)].Tissue)))   
    samples2tissues = pd.DataFrame(metadata[metadata.index.isin(samples)]["Tissue"]) 
    samples2tissues.reset_index(inplace = True)
    samples2tissues.columns = ["Sampleid","Tissue"]
    return samples2tissues



def get_tissue_encodings(tissue_ids):
    '''
    Function to get the encoded label of a tissue identifier.
    '''
    res = []
    for tid in tissue_ids:
        res.append(tissue_encoding[tid])
    res = np.array(res)
    return res


def select_all_samples(folder, metadata):
    ''' Function that loads all samples from the database '''
    res = []
    tis = []
    samples = [ f.name for f in os.scandir(folder) if f.is_dir() ]
    samples2tissues = pd.DataFrame(metadata[metadata.index.isin(samples)]["Tissue"]) 
    samples2tissues.reset_index(inplace = True)
    samples2tissues.columns = ["Sampleid", "Tissue"]
    res.extend(list(samples2tissues.Sampleid))
    tis.extend(list(samples2tissues.Tissue))
    return res, tis

   
def select_samples(sel_tissues, k, folder, metadata):
    '''
    Function that select k samples to obtain a smaller dataset for testing and debugging
    '''
    res = []
    tis = []
    samples = [ f.name for f in os.scandir(folder) if f.is_dir() ]
    samples2tissues = pd.DataFrame(metadata[metadata.index.isin(samples)]["Tissue"]) 
    samples2tissues.reset_index(inplace = True)
    samples2tissues.columns = ["Sampleid", "Tissue"]
    for st in sel_tissues:
        res.extend(list(samples2tissues.loc[samples2tissues.Tissue == st,].head(k).Sampleid))
        tis.extend(list(samples2tissues.loc[samples2tissues.Tissue == st,].head(k).Tissue))
    return res, tis

        
def get_stats(folder, metadata):
    '''
    Obtain a table with statistics wit the available images
    Number of samples and tiles per tissue
    '''
    num_samples = 0
    num_tiles = 0
    tissue_cnt = {}
    tissue_tile_cnt = {}
    tile_per_sample = 0
    samples_subfolders = [ f.name for f in os.scandir(folder) if f.is_dir() ]
    for sample in samples_subfolders:
        num_samples += 1
        sample_tiles_folder = "%s/%s/%s_tiles" %(folder, sample, sample)
        tissue = metadata.loc[sample].Tissue
        if tissue in tissue_cnt.keys():
            tissue_cnt[tissue] +=1
        else:
            tissue_cnt[tissue] = 1

        if os.path.isdir(sample_tiles_folder):
            tiles = [ f.name for f in os.scandir(sample_tiles_folder) if f.is_file() ]
            tile_per_sample = len(tiles)
            for t in tiles:
                num_tiles += 1
        if tissue in tissue_tile_cnt.keys():
            tissue_tile_cnt[tissue] += tile_per_sample
        else:
            tissue_tile_cnt[tissue] = tile_per_sample

    avg_tile_sample = num_tiles // num_samples
    print ("-----------------------------------------------")
    print("Samples %d  Tiles %d\n" % (num_samples, num_tiles))
    print("Avg Tiles per samples %d\n" % (avg_tile_sample))
    print ("%55s %3s %20s" % ("Tissue","Samples","Tiles"))
    for (k, v) in tissue_cnt.items():
        tc = tissue_tile_cnt[k]
        print ("%55s : %3d% 15d" % (k, v, tc))
    print ("-----------------------------------------------")
	

def read_commandline_args():
    '''
    read the parameters from the command line
    '''
    # default values; easier fo testing
    fit_model = 1
    fine_tune = 0
    model_evaluate = 1
    batch_size = 10
    model_batch_size = 10
    epochs = 5
    debug_mode = 0 # debug mode
    # for fine tunning
    fine_tune_epochs = 25
    fine_tune_model_batch_size = 5
    
    parser = argparse.ArgumentParser(description="Run CNN for Histo images",  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-mt", "--model_type", type=str, default = "vgg16", help="possible architectures: scratch_cnn, vgg16, xception")
    parser.add_argument("-fitm", "--fit_model", type=int,  default = fit_model, help="fit the model 0 or 1")
    parser.add_argument("-ft", "--fine_tune", type=int,  default = fine_tune, help="fine tune: 0 or 1")
    parser.add_argument("-me", "--model_evaluate", type=int,  default = model_evaluate, help="print statistics for model evaluation")
    # hyper parameters
    parser.add_argument("-bt", "--batch_size", type=int,  default = batch_size, help="batch size")
    parser.add_argument("-mbt", "--model_batch_size", type=int,  default = model_batch_size, help="model batch size")
    parser.add_argument("-epc", "--epochs", type=int,  default = epochs, help="trainign epochs")
    parser.add_argument("-dbg", "--debug_mode", type=int,  default = debug_mode, help="create a dataset for debugging (1) or use full dataset for training (0)")
    
    parser.add_argument("-ft_model_batch_size", "--fine_tune_model_batch_size", type = int,  default = fine_tune_model_batch_size, help="fine tune model batch size")
    parser.add_argument("-ft_tune_epochs", "--fine_tune_epochs", type = int,  default = fine_tune_epochs, help="fine tune epochs")

    args = parser.parse_args()
    config = vars(args)
    return config
  

def main():
    # define here the main architectures and the modes to run
    # the architectures can be vgg16, xception or scratch
    # you can either fit/train the model or read already existing model
    # fine tuning is a slow process and is optional
    # define if model evaluation is to be performed
    config = read_commandline_args()
    
    model_type = config["model_type"]
    fit_model = config["fit_model"]
    fine_tune = config["fine_tune"]
    model_evaluate = config["model_evaluate"]
    debug_mode = config["debug_mode"]

    # local and server dirs
    images_folder = "output"
    #os.chdir('/Users/test/Dropbox/Mac (2)/Desktop/Keras/images/')
    os.chdir('/home/pferreira/gtex_histo/')
    
    # load metadata
    # metadata_file = "/Users/test/Dropbox/Mac (2)/Desktop/Keras/images/GTExPortal.csv"
    metadata_file = "/home/pferreira/gtex_histo/GTExPortal.csv"
    metadata = load_gtex_metadata(metadata_file)
    get_stats(images_folder, metadata)
    
    # get sample ids and tissues
    s2t = get_tissue_list(images_folder, metadata)
   
    
    if debug_mode:
        # select samples to create a much smaller dataset for testing debugging
        sel_tissues = ["Liver","Lung","Uterus", "Skin - Sun Exposed (Lower leg)",
        "Adipose - Subcutaneous","Stomach","Muscle - Skeletal",
        "Stomach","Muscle - Skeletal", "Nerve - Tibial","Adipose - Visceral (Omentum)",
        "Artery - Aorta", "Artery - Coronary", "Artery - Tibial", "Spleen","Heart - Left Ventricle",
        "Esophagus - Gastroesophageal Junction", 
        "Esophagus - Mucosa"]
        k  = 25 # sets thte number of samples to load
        res, tis = select_samples(sel_tissues, k, images_folder, metadata)
        s2t = pd.DataFrame(tis, res)
        s2t.reset_index(inplace=True)
        s2t.columns = ["Sampleid", "Tissue"]
        print(s2t.Tissue.value_counts())
    else:
        # read all available samples from the database
        k = 0
        res, tis = select_all_samples(images_folder, metadata)
        s2t = pd.DataFrame(tis, res)
        s2t.reset_index(inplace=True)
        s2t.columns = ["Sampleid", "Tissue"]
        print(s2t.Tissue.value_counts())
		

    # get unique tissuesprint
    tissue_list = list(set(s2t.Tissue))
    # tissue encoding
    tissue_enc = encode_all_tissues(tissue_list)
    tissue_ohe = keras.utils.to_categorical(tissue_enc.transform(tissue_list))
    for (t, ohe) in zip(tissue_list, tissue_ohe):
        tissue_encoding[t] = ohe
    #print(tissue_encoding)
    
    # split the samples and respetive labels into train and test set
    # test / remainder
    X_ids, X_test_ids, y_ids, y_test_ids = train_test_split(s2t.Sampleid, s2t.Tissue, test_size=0.2, stratify = s2t.Tissue, random_state=42)   
    # train / validation
    X_train_ids, X_valid_ids, y_train_ids, y_valid_ids = train_test_split(X_ids, y_ids, test_size=0.2, stratify = y_ids, random_state=42)   
    
    
    NUM_TRAIN_SAMPLES = len(X_train_ids)
    NUM_VALID_SAMPLES = len(X_valid_ids)
    NUM_TEST_SAMPLES = len(X_test_ids)
    
    batch_size = config["batch_size"]
    
    train_steps = np.ceil(NUM_TRAIN_SAMPLES/batch_size)
    valid_steps = np.ceil(NUM_VALID_SAMPLES/batch_size)
    test_steps = np.ceil(NUM_TEST_SAMPLES/batch_size)
    print("Training steps %d; validation steps %d; test steps %d" %(train_steps, valid_steps, test_steps))
    print("Num. Samples Train %d; Validation %d; Test %d" %(NUM_TRAIN_SAMPLES, NUM_VALID_SAMPLES, NUM_TEST_SAMPLES))
    
    # create batch generators to load teh dataset into batches
    my_training_batch_generator = batch_generator(X_train_ids, y_train_ids, batch_size, train_steps, images_folder, metadata)
    my_valid_batch_generator = batch_generator(X_valid_ids, y_valid_ids, batch_size, valid_steps, images_folder, metadata)
    my_test_batch_generator = batch_generator(X_test_ids, y_test_ids, batch_size, test_steps, images_folder, metadata)
    
    # dataset
    input_shape = (128, 128, 3)
    num_classes = len(np.unique(tissue_list))
    model_batch_size = config["model_batch_size"]
    epochs = config["epochs"]
   	
	
    if model_type == "scratch_cnn":
        print("Scratch CNN")
        model = setup_model(input_shape, num_classes)
        model_file_name = "scratch_cnn.h5"
        last_layers = 1 # offset of the layers for fine tunning
        model_type = "Scratch_CNN"

    if model_type == "vgg16":
        model = setup_model_pretrained_VGG16(input_shape, num_classes)
        print("Model: Pretrained VGG16\n")
        model_file_name = "pretrained_vgg16.h5"
        last_layers = 4 # offset of the layers for fine tunning
        model_type = "VGG16"
    
    if model_type == "xception":
        model = setup_model_pretrained_Xception(input_shape, num_classes)
        print("Model: Pretrained Xception \n")
        model_file_name = "pretrained_xception.h5"
        last_layers = 1 # offset of the layers for fine tunning
        model_type = "Xception_CNN"
    
    print("Saving model to ", model_file_name)
    tf.keras.models.save_model(model, model_file_name)
	
	# fit the model
    if fit_model:
        print(model.summary())
        print("Epochs %d; Model Batch size %d; K=%d; Train Batch size %d" %(epochs, model_batch_size, k, batch_size))
        history = prepare_model(my_training_batch_generator, train_steps, model, model_batch_size, epochs, valid_steps, my_valid_batch_generator)
        tf.keras.models.save_model(model, model_file_name)
        plot_metric(history, 'accuracy')
        plot_metric(history, 'loss')
		
    print("Loading model from ", model_file_name)
    model = tf.keras.models.load_model(model_file_name)
	
    if model_evaluate:
        
        # evaluate the model in the train dataset
        X_train, tile_metadata = scan_tiles_bylist(X_train_ids, images_folder, metadata, 1)
        print("Train set results\n")
        evaluate_model(X_train, tile_metadata, model, "Train", my_training_batch_generator, train_steps, model_type)
        del X_train
        print("\n-------------------------------------------------------------\n")
    
	    # evaluate the model in the validation dataset
        X_valid, tile_metadata = scan_tiles_bylist(X_valid_ids, images_folder, metadata, 1)
        print("Validation set results\n")
        evaluate_model(X_valid, tile_metadata, model, "Validation", my_valid_batch_generator, valid_steps, model_type)
        del X_valid
        print("\n-------------------------------------------------------------\n")
    
        # evaluate the model in the test dataset
        X_test, tile_metadata = scan_tiles_bylist(X_test_ids, images_folder, metadata, 1)
        print("Test Results")
        evaluate_model(X_test, tile_metadata, model, "Test", my_test_batch_generator, test_steps, model_type)
        del X_test
        print("\n-------------------------------------------------------------\n")
    

    if fine_tune:
        # parameters for fine tuning
        model_batch_size =  config["fine_tune_model_batch_size"] # decrease model batch size
        epochs = config["fine_tune_epochs"]  
        model_layers = len(model.layers)
        start_trainable_layer = model_layers - 1 - last_layers
        
        print("Number of layers in base model: %d; Num. Trainable layers:%d" % (model_layers, start_trainable_layer))
        my_training_batch_generator = batch_generator(X_train_ids, y_train_ids, batch_size, train_steps, images_folder, metadata)
        my_valid_batch_generator = batch_generator(X_valid_ids, y_valid_ids, batch_size, valid_steps, images_folder, metadata)
        finetune_model(start_trainable_layer, my_training_batch_generator, train_steps, model, model_batch_size, epochs, valid_steps, my_valid_batch_generator)
        print("Test Results --> After Fine Tune")
        X_test, tile_metadata = scan_tiles_bylist(X_test_ids, images_folder, metadata, 1)
        evaluate_model(X_test, tile_metadata, model, "Test", my_test_batch_generator, test_steps, model_type)
        del X_test
        print("\n-------------------------------------------------------------\n")
    
main()





	


    