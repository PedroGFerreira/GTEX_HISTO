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
#import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
#import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, applications
#import random

# global vars
tissue_list = []
smoking_encoding = {}
donor_smk_nonsmk_status = pd.DataFrame()
# initialize the total number of training and testing samples
NUM_TRAIN_SAMPLES = 0
NUM_VALID_SAMPLES = 0
NUM_TEST_SAMPLES = 0
tissue_encoding = {}
ADAM_LEARN_RATE = 0.0001

##### ########## ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
def load_gtex_metadata(metadata_file):
    '''
    Input: file with GTEx metadata retrieved from GTEx portal 
    https://gtexportal.org/home/histologyPage --> All histology Samples (csv)
    Returns pandas object with table
    loads the info in the table to a Pandas dataframe
    '''
    # metadata file
    metadata = pd.read_csv(metadata_file, sep=",")
    # remove spaces from column labels
    metadata.columns = [c.replace(" ","") for c in list(metadata.columns)]
    #  modifies the DataFrame in place (do not create a new object).
    metadata.set_index("TissueSampleID", inplace=True)
    return metadata
  
  
def most_frequent(List):
    return max(set(List), key= List.count)


def vote_prediction(df):
    '''receives a pandas df with tiles predictions and sample value
    infers the sample class based on the predictions using a voting scheme like majority'''
    sample_lst = list(np.unique(df.Sample))
    res = []
    for sample in sample_lst:
        tile_predictions = list(df.loc[df.Sample == sample,"Predicted"])
        most_predicted = most_frequent(tile_predictions)
        res.append((sample, most_predicted))
    pred_df = pd.DataFrame(res, columns = ["Sample","Predicted"])
    return pred_df


def load_smoking_data(smk_annot_file, metadata):
    smk = pd.read_csv(smk_annot_file, sep=";")
    lung_samples = metadata.loc[metadata["Tissue"]=="Lung"]
    smk.set_index("Donor", inplace=True)
    lung_samples_annot = list(set(smk.index).intersection(list(lung_samples.SubjectID)))    
    donor_status = pd.DataFrame(smk.loc[lung_samples_annot,"SmokerStatus"]).reset_index()
    lung_samples = lung_samples.reset_index()
    donor_status = donor_status.merge(lung_samples, left_on='Donor', right_on="SubjectID")
    return donor_status

##### ########## ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####


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
		
                if data.shape != (128, 128, 3):
                    continue
                    #print(data.shape)
                    #print(sample + " " + tissue + " "  +  " " + t + " " + fig)
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




 
def evaluate_model(df, tile_metadata, model, label, df_generator, df_steps):
    smoking_status_lst = list(donor_smk_nonsmk_status.loc[tile_metadata.Sample,"SmokerStatus"])
    # Results for tiles
    df_labels = get_smoking_encodings((smoking_status_lst))
    score = model.evaluate(df, df_labels.argmax(axis = -1), verbose=0)
    print(label +" Loss:", score[0])
    print(label + " Accuracy:", score[1])
    
    # predictions
    y_pred_proba = model.predict(df) # prediction probabilities
    y_pred_classes = y_pred_proba.argmax(axis = -1) # predicted classes : tiles
    tile_metadata["Predicted"] = y_pred_classes
    y_classes = df_labels.argmax(axis = -1) # true classes : tiles
    tile_metadata["Actual"] = y_classes
    print("Tile Report\n")
    print(confusion_matrix(y_classes, y_pred_classes))
    print(classification_report(y_classes, y_pred_classes, target_names = list(map(str, np.unique(smoking_status_lst)))))
    
    # Results per Sample
    # true values
    sample_true_classes = tile_metadata.loc[:,["Sample","Tissue","Actual"]].drop_duplicates()
    sample_pred_classes = vote_prediction(tile_metadata)
    sample_true_classes = sample_true_classes.merge(sample_pred_classes, left_on='Sample', right_on="Sample")
    #print("sample:true:classes")
    #print(sample_true_classes)
    print("Sample Report\n")   
    print(confusion_matrix(sample_true_classes.Actual, sample_true_classes.Predicted))
    print(classification_report(sample_true_classes.Actual, sample_true_classes.Predicted,target_names = list(map(str, np.unique(smoking_status_lst)))))
       

def prepare_model(batch_generator, train_steps, model, batch_size, epochs, validation_steps, validation_data):
    # check optimizer as rmsprop
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy", tf.keras.metrics.CategoricalAccuracy()])
    #model.fit(df, tissue_lst_ohe, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    model.fit(batch_generator, 
                        epochs=epochs, 
                        steps_per_epoch = train_steps, 
                        validation_steps= validation_steps,
                        validation_data = validation_data, 
                        verbose = 1)
			            #workers = 1)
                        #use_multiprocessing=True)





def setup_model(input_shape, num_classes):
    print("Number of classes %d" %(num_classes))
    if num_classes == 2:
        num_classes = 1 # for binary classification
        
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        
	layers.Conv2D(32, kernel_size=(5, 5), activation="relu"),
	layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
	
	layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
	layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
        
	layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
	layers.BatchNormalization(),
	layers.MaxPooling2D(pool_size=(2, 2)),
	
    layers.Conv2D(128, kernel_size=(2, 2), activation="relu"),
	layers.BatchNormalization(),
	layers.MaxPooling2D(pool_size=(2, 2)),
	
    layers.Flatten(),
    layers.Dropout(0.5),
	layers.BatchNormalization(),
        
	layers.Dense(1, activation="softmax"),
    ])
    return model

#def setup_model_pretrained_Xception(input_shape, num_classes):
#    # base model with pre-trained weights
#    if num_classes == 2:
#        num_classes = 1 # for binary classification
#    model = applications.Xception(include_top=False, input_shape=input_shape, weights="imagenet")
#    # freeze the model; mark loaded layers as not trainable
#    for layer in model.layers:
#        layer.trainable = False
#    # create a new model on top
#    inputs = layers.Input(shape=input_shape)
#    x = model(inputs, training = False)
#    # convert features of shape model.output_shape[1:] to vector
#    # add new classifier layers
#    x1 = layers.GlobalAveragePooling2D()(x)
#    # a dense classifier with n classes
#    output = layers.Dense(num_classes, activation='softmax')(x1)
#    # define new model
#    model = models.Model(inputs=[model.inputs], outputs=output)
#    return model

def setup_model_pretrained_VGG16(input_shape, num_classes):
    if num_classes == 2:
        num_classes = 1 # for binary classification
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

def setup_model_pretrained_Xception(input_shape, num_classes):
    if num_classes == 2:
        num_classes = 1 # for binary classification
    # base model with pre-trained weights
    model = applications.Xception(include_top=False, input_shape=input_shape, weights="imagenet")
    # freeze the model; mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
    # create a new model on top
    inputs = keras.Input(shape=input_shape)
    x = model(inputs, training = False)
    fx = layers.Flatten()(x)
    dense = layers.Dense(num_classes, activation='softmax')
    dx = dense(fx)
    model = models.Model(inputs=inputs, outputs=dx)
    return model



def finetune_model(fine_tune_at, batch_generator, train_steps, model, batch_size, epochs, validation_steps, validation_data):
    # Unfreeze the base model
    model.trainable = True
    for layer in model.layers[fine_tune_at:]:
    	layer.trainable = True
	
    for layer in model.layers:
        print("{}: {}".format(layer, layer.trainable))

    model.compile(loss="binary_crossentropy", 
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
    idx=1
    while True: 
        yield load_data(sample_ids, sample_labels, idx-1, batch_size, folder, metadata)## Yields data
        if idx<steps:
            idx+=1
        else:
            idx=1



def load_data(sample_ids, sample_labels, idx, batch_size, folder, metadata):
    pos = idx * batch_size
    sample_list = sample_ids[pos: pos+batch_size]
    # scan tiles and generate  a dataframe (df) a list of tissues per tile (tissue_lst)
    # and list of samples
    df, tile_metadata = scan_tiles_bylist(sample_list, folder, metadata, 1)
    df_labels = get_smoking_encodings(list(donor_smk_nonsmk_status.loc[tile_metadata.Sample,"SmokerStatus"]))
    df_labels = df_labels.argmax(axis = -1)
    return (df, df_labels)

def get_smoking_encodings(smoking_labels):
    res = []
    #print("Tissue encoding...")
    #print(tissue_encoding)
    for sl in smoking_labels:
        res.append(smoking_encoding[sl])
    res = np.array(res)
    return res

def set_smoking_encoding():
    status_types = list(np.unique(donor_smk_nonsmk_status.SmokerStatus))
    smk_status_enc = encode_smoking_status(status_types)
    # to return hot encoded
    smk_status_ohe = keras.utils.to_categorical(smk_status_enc.transform(status_types))
    # to return binary
    #smk_status_ohe = smk_status_enc.transform(status_types)
    for (s, ohe) in zip(status_types, smk_status_ohe):
        smoking_encoding[s] = ohe
    
def encode_smoking_status(status_types):
    '''returns the encoding object for all tissues'''
    le = preprocessing.LabelEncoder()
    le.fit(status_types)
    # transform categorical to numerical
    return le
     
def tissue_OHE(tissues_lst):
    enc = preprocessing.OneHotEncoder()
    enc.fit(np.array(tissues_lst).reshape(-1,1))
    #enc.transform(np.array(tissues_lst).reshape(-1,1)).toarray()
    #enc.transform(np.array(tissues_lst[0]).reshape(-1,1)).toarray()
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




    
def select_samples(sel_tissues, k, folder, metadata):
    res = []
    tis = []
    
    samples = [ f.name for f in os.scandir(folder) if f.is_dir() ]
    samples2tissues = pd.DataFrame(metadata[metadata.index.isin(samples)]["Tissue"]) 
    samples2tissues.reset_index(inplace = True)
    samples2tissues.columns = ["Sampleid","Tissue"]
    
    for st in sel_tissues:
        res.extend(list(samples2tissues.loc[samples2tissues.Tissue == st,].head(k).Sampleid))
        tis.extend(list(samples2tissues.loc[samples2tissues.Tissue == st,].head(k).Tissue))
    return res, tis
        
def get_stats_lung(folder, metadata, donor_smk_nonsmk_status):
    num_samples = 0
    num_tiles = 0
    tissue_cnt = {}
    tissue_tile_cnt = {}
    tile_per_sample = 0
    lung_sample_list = []
    samples_subfolders = [ f.name for f in os.scandir(folder) if f.is_dir() ]
    
    for sample in samples_subfolders:
        num_samples += 1
        sample_tiles_folder = "%s/%s/%s_tiles" %(folder, sample, sample)
        tissue = metadata.loc[sample].Tissue
        if tissue in tissue_cnt.keys():
            tissue_cnt[tissue] +=1
        else:
            tissue_cnt[tissue] = 1

        if tissue == "Lung":
            lung_sample_list.append(sample)


        if os.path.isdir(sample_tiles_folder):
            tiles = [ f.name for f in os.scandir(sample_tiles_folder) if f.is_file() ]
            tile_per_sample = len(tiles)
            for t in tiles:
                num_tiles += 1
        if tissue in tissue_tile_cnt.keys():
            tissue_tile_cnt[tissue] += tile_per_sample
        else:
            tissue_tile_cnt[tissue] = tile_per_sample

    tissue = "Lung"
    avg_tile_sample = tissue_tile_cnt[tissue] // tissue_cnt[tissue]
    # samples with images and smoking annotation
    lung_sample_list = list(set(lung_sample_list) & set(donor_smk_nonsmk_status.index))
    donor_smk_nonsmk_status = donor_smk_nonsmk_status.loc[lung_sample_list,:]
    
    
    print ("-----------------------------------------------")
    print("Lung Samples %d  Tiles %d\n" % (tissue_cnt[tissue], tissue_tile_cnt[tissue]))
    print("Avg Tiles per samples %d\n" % (avg_tile_sample))
    print("Smoking Status:")
    print(donor_smk_nonsmk_status.SmokerStatus.value_counts().reset_index())
    print ("-----------------------------------------------")
    

def main():
    ### MAIN PARAMETERS
    model_type = "scratch_cnn" # "vgg16"   "xception" "scratch_cnn
    fit_model = 1 # 1 fit model ; 0 skip fitting
    fine_tune = 1 # activate fine tuning
    model_evaluate = 1 # model evaluation or skip evaluation 


    images_folder = "output"
    #os.chdir('/Users/test/Dropbox/Mac (2)/Desktop/Keras/images/')
    os.chdir('/home/pferreira/gtex_histo/')
    
    # load metadata
    #metadata_file = "/Users/test/Dropbox/Mac (2)/Desktop/Keras/images/GTExPortal.csv"
    metadata_file = "/home/pferreira/gtex_histo/GTExPortal.csv"
    metadata = load_gtex_metadata(metadata_file)
    get_stats(images_folder, metadata)
    #sys.exit(1)
    
    # get sample ids and tissues
    s2t = get_tissue_list(images_folder, metadata)
    #s2t = s2t.loc[random.sample(range(0,s2t.shape[0]), 200),]
    #tc = pd.DataFrame(s2t.Tissue.value_counts())
    #s2t = s2t.loc[s2t["Tissue"].isin(list(tc[tc.Tissue > 1].index))]
    #s2t = s2t.loc[~s2t["Tissue"].isin(["Thyroid","Esophagus - Muscularis"])]
    
    # select samples
    if 1:
        	sel_tissues = ["Liver","Lung","Uterus", "Skin - Sun Exposed (Lower leg)",
    	"Adipose - Subcutaneous","Stomach","Muscle - Skeletal",
    	"Stomach","Muscle - Skeletal", "Nerve - Tibial","Adipose - Visceral (Omentum)",
    	"Artery - Aorta", "Artery - Coronary", "Artery - Tibial", "Spleen","Heart - Left Ventricle",
    	"Esophagus - Gastroesophageal Junction", 
    	"Esophagus - Mucosa"]
        	k  = 150
        	res, tis = select_samples(sel_tissues, k, images_folder, metadata)
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
    batch_size = 10
    train_steps = np.ceil(NUM_TRAIN_SAMPLES/batch_size)
    valid_steps = np.ceil(NUM_VALID_SAMPLES/batch_size)
    test_steps = np.ceil(NUM_TEST_SAMPLES/batch_size)
    print("Training steps %d; validation steps %d; test steps %d" %(train_steps, valid_steps, test_steps))
    print("Num. Samples Train %d; Validation %d; Test %d" %(NUM_TRAIN_SAMPLES, NUM_VALID_SAMPLES, NUM_TEST_SAMPLES))
    
    my_training_batch_generator = batch_generator(X_train_ids, y_train_ids, batch_size, train_steps, images_folder, metadata)
    my_valid_batch_generator = batch_generator(X_valid_ids, y_valid_ids, batch_size, valid_steps, images_folder, metadata)
    my_test_batch_generator = batch_generator(X_test_ids, y_test_ids, batch_size, test_steps, images_folder, metadata)
    
    # dataset
    #df, tissue_lst, sample_lst = scan_tiles(images_folder, metadata, 15)
    #df, tissue_lst, sample_lst = scan_tiles_bylist(list(X_train_ids), images_folder, metadata)
    #print(pd.DataFrame(tissue_lst).value_counts())
    #print(metadata.loc[sample_lst,].Tissue.value_counts())
    
    input_shape = (128, 128, 3)
    num_classes = len(np.unique(tissue_list))
    #num_classes = len(np.unique(y_train_ids))
    model_batch_size = 10
    epochs = 55
   	
	
    if model_type == "scratch_cnn":
        print("Scratch CNN")
        model = setup_model(input_shape, num_classes)
        model_file_name = "scratch_cnn.h5"
        last_layers = 1
	
    if model_type == "vgg16":
        model = setup_model_pretrained_VGG16(input_shape, num_classes)
        print("Model: Pretrained VGG16\n")
        model_file_name = "pretrained_vgg16.h5"
        last_layers = 6
    
    if model_type == "xception":
        model = setup_model_pretrained_Xception(input_shape, num_classes)
        print("Model: Pretrained Xception \n")
        model_file_name = "pretrained_xception.h5"
        last_layers = 1
	
	# fit the model
    if fit_model:
        model_layers = len(model.layers)
        start_trainable_layer = model_layers - 1 - last_layers
        print("Number of layers in base model: ", model_layers)
        print(model.summary())
        print("Epochs %d; Model Batch size %d; K=%d; Train Batch size %d" %(epochs, model_batch_size, k, batch_size))
        prepare_model(my_training_batch_generator, train_steps, model, model_batch_size, epochs, valid_steps, my_valid_batch_generator)
        tf.keras.models.save_model(model_file_name)
	
    model = tf.keras.models.load_model(model_file_name)
	
    if model_evaluate:
        # evaluate the model in the train dataset
        X_train, tile_metadata = scan_tiles_bylist(X_train_ids, images_folder, metadata, 1)
        print("Train set results\n")
        evaluate_model(X_train, tile_metadata, model, "Train", my_training_batch_generator, train_steps)
        del X_train
        print("\n-------------------------------------------------------------\n")
    
	    # evaluate the model in the validation dataset
        X_valid, tile_metadata = scan_tiles_bylist(X_valid_ids, images_folder, metadata, 1)
        print("Validation set results\n")
        evaluate_model(X_valid, tile_metadata, model, "Validation", my_valid_batch_generator, valid_steps)
        del X_valid
        print("\n-------------------------------------------------------------\n")
    
        # evaluate the model in the test dataset
        X_test, tile_metadata = scan_tiles_bylist(X_test_ids, images_folder, metadata, 1)
        print("Test Results")
        evaluate_model(X_test, tile_metadata, model, "Test", my_test_batch_generator, test_steps)
        del X_test
        print("\n-------------------------------------------------------------\n")
    

    if fine_tune:
        	model_batch_size = 10 # decrease model batch size
        	epochs = 55  # decrease epochs
        	#print("Train steps %d" %(train_steps))
        	my_training_batch_generator = batch_generator(X_train_ids, y_train_ids, batch_size, train_steps, images_folder, metadata)
        	my_valid_batch_generator = batch_generator(X_valid_ids, y_valid_ids, batch_size, valid_steps, images_folder, metadata)
        	finetune_model(start_trainable_layer, my_training_batch_generator, train_steps, model, model_batch_size, epochs, valid_steps, my_valid_batch_generator)
        	print("Test Results --> After Fine Tune")
        	X_test, tile_metadata = scan_tiles_bylist(X_test_ids, images_folder, metadata, 1)
        	evaluate_model(X_test, tile_metadata, model, "Test", my_test_batch_generator, test_steps)
        	del X_test
        	print("\n-------------------------------------------------------------\n")
    
main()


def smoker_vs_nonsmoker_classifier():
    ### MAIN PARAMETERS
    model_type = "scratch_cnn" # "vgg16"   "xception" "scratch_cnn
    fit_model = 1 # 1 fit model ; 0 skip fitting
    fine_tune = 1 # activate fine tuning
    model_evaluate = 1 # model evaluation or skip evaluation 
    
    # images folder
    folder = "output"
    os.chdir('/Users/test/Dropbox/Mac (2)/Desktop/Keras/images/')
    #os.chdir('/home/pferreira/gtex_histo/')
    
    # load metadata
    metadata_file = "/Users/test/Dropbox/Mac (2)/Desktop/Keras/images/GTExPortal.csv"
    #metadata_file = "/home/pferreira/gtex_histo/GTExPortal.csv"
    metadata = load_gtex_metadata(metadata_file)
    
    # list of available samples
    s2t = get_tissue_list(folder, metadata)
    # lung samples with images
    lung_sample_list = list(s2t.loc[s2t.Tissue =="Lung",].Sampleid)
    
    
    # smoking annotation file
    smk_annot_file = "/Users/test/Dropbox/Mac (2)/Desktop/Keras/images/smoking_annotation.csv"
    donor_smk_status = load_smoking_data(smk_annot_file, metadata)
    # select only samples from smoker and non-smokers
    donor_smk_nonsmk_status = donor_smk_status.loc[(donor_smk_status.SmokerStatus == "Smoker" )| (donor_smk_status.SmokerStatus == "Non Smoker") ,:]
    donor_smk_nonsmk_status.set_index("TissueSampleID", inplace = True, drop = False)
    
    # samples with images and smoking annotation
    lung_sample_list = list(set(lung_sample_list) & set(donor_smk_nonsmk_status.index))
    donor_smk_nonsmk_status = donor_smk_nonsmk_status.loc[lung_sample_list,:]
    get_stats_lung(folder, metadata, donor_smk_nonsmk_status) 
    
    # split the samples and respetive labels into train and test set
    # test / remainder
    X_ids, X_test_ids, y_ids, y_test_ids = train_test_split(donor_smk_nonsmk_status.TissueSampleID, donor_smk_nonsmk_status.SmokerStatus, test_size=0.2, stratify = donor_smk_nonsmk_status.SmokerStatus, random_state=42)   
    # train / validation
    X_train_ids, X_valid_ids, y_train_ids, y_valid_ids = train_test_split(X_ids, y_ids, test_size=0.2, stratify = y_ids, random_state=42)   
    
    
    NUM_TRAIN_SAMPLES = len(X_train_ids)
    NUM_VALID_SAMPLES = len(X_valid_ids)
    NUM_TEST_SAMPLES = len(X_test_ids)
    batch_size = 10
    train_steps = np.ceil(NUM_TRAIN_SAMPLES/batch_size)
    valid_steps = np.ceil(NUM_VALID_SAMPLES/batch_size)
    test_steps = np.ceil(NUM_TEST_SAMPLES/batch_size)
    print("Training steps %d; validation steps %d; test steps %d" %(train_steps, valid_steps, test_steps))
    print("Num. Samples Train %d; Validation %d; Test %d" %(NUM_TRAIN_SAMPLES, NUM_VALID_SAMPLES, NUM_TEST_SAMPLES))
    
    my_training_batch_generator = batch_generator(X_train_ids, y_train_ids, batch_size, train_steps, folder, metadata)
    my_valid_batch_generator = batch_generator(X_valid_ids, y_valid_ids, batch_size, valid_steps, folder, metadata)
    my_test_batch_generator = batch_generator(X_test_ids, y_test_ids, batch_size, test_steps, folder, metadata)
  
    input_shape = (128, 128, 3)
    num_classes = 2
    model_batch_size = 10
    epochs = 1
    
    if model_type == "scratch_cnn":
        print("Scratch CNN")
        model = setup_model(input_shape, num_classes)
        model_file_name = "scratch_cnn.h5"
        last_layers = 1
	
    if model_type == "vgg16":
        model = setup_model_pretrained_VGG16(input_shape, num_classes)
        print("Model: Pretrained VGG16\n")
        model_file_name = "pretrained_vgg16.h5"
        last_layers = 6
    
    if model_type == "xception":
        model = setup_model_pretrained_Xception(input_shape, num_classes)
        print("Model: Pretrained Xception \n")
        model_file_name = "pretrained_xception.h5"
        last_layers = 1
	
	# fit the model
    if fit_model:
        model_layers = len(model.layers)
        start_trainable_layer = model_layers - 1 - last_layers
        print("Number of layers in base model: ", model_layers)
        print(model.summary())
        print("Epochs %d; Model Batch size %d; Train Batch size %d" %(epochs, model_batch_size, batch_size))
        prepare_model(my_training_batch_generator, train_steps, model, model_batch_size, epochs, valid_steps, my_valid_batch_generator)
        tf.keras.models.save_model(model, model_file_name)
	
    model = tf.keras.models.load_model(model_file_name)
    
    if model_evaluate:
        # evaluate the model in the train dataset
        X_train, tile_metadata = scan_tiles_bylist(X_train_ids, folder, metadata, 1)
        print("Train set results\n")
        evaluate_model(X_train, tile_metadata, model, "Train", my_training_batch_generator, train_steps)
        del X_train
        print("\n-------------------------------------------------------------\n")
    
	    # evaluate the model in the validation dataset
        X_valid, tile_metadata = scan_tiles_bylist(X_valid_ids, folder, metadata, 1)
        print("Validation set results\n")
        evaluate_model(X_valid, tile_metadata, model, "Validation", my_valid_batch_generator, valid_steps)
        del X_valid
        print("\n-------------------------------------------------------------\n")
    
        # evaluate the model in the test dataset
        X_test, tile_metadata = scan_tiles_bylist(X_test_ids, folder, metadata, 1)
        print("Test Results")
        evaluate_model(X_test, tile_metadata, model, "Test", my_test_batch_generator, test_steps)
        del X_test
        print("\n-------------------------------------------------------------\n")
    

    if fine_tune:
        	model_batch_size = 10 # decrease model batch size
        	epochs = 2  # decrease epochs
        	#print("Train steps %d" %(train_steps))
        	my_training_batch_generator = batch_generator(X_train_ids, y_train_ids, batch_size, train_steps, folder, metadata)
        	my_valid_batch_generator = batch_generator(X_valid_ids, y_valid_ids, batch_size, valid_steps, folder, metadata)
        	finetune_model(start_trainable_layer, my_training_batch_generator, train_steps, model, model_batch_size, epochs, valid_steps, my_valid_batch_generator)
        	print("Test Results --> After Fine Tune")
        	X_test, tile_metadata = scan_tiles_bylist(X_test_ids, folder, metadata, 1)
        	evaluate_model(X_test, tile_metadata, model, "Test", my_test_batch_generator, test_steps)
        	del X_test
        	print("\n-------------------------------------------------------------\n")