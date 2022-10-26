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

# global vars
tissue_list = []
tissue_encoding = {}
# initialize the total number of training and testing samples
NUM_TRAIN_SAMPLES = 0
NUM_VALID_SAMPLES = 0
NUM_TEST_SAMPLES = 0
tissue_encoding = {}





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
                res.append((t, sample, tissue))
        cnt += 1
    if include_data:
        df = np.stack(data_lst, axis = 0) # create a 4D array from a list of 3D arrays
    else:
        df = []
    res_df = pd.DataFrame(res, columns = ["Tile","Sample","Tissue"])
    return (df, res_df)




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
    #print("pred_df")
    #print(pred_df)
    return pred_df
 
def evaluate_model(df, tile_metadata, model, label, df_generator, df_steps, model_type):
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
    cf_matrix = confusion_matrix(y_classes, y_pred_classes)
    print(cf_matrix)
    names = list(map(str, np.unique(tiles_tissue_lst)))
    file_name = "TileReport_cf_matrix_" + model_type + ".png"
    plot_cf_matrix(cf_matrix, model_type, file_name, names, "Tile")
    print(classification_report(y_classes, y_pred_classes, target_names = names))
    
    # predictions with generator
    #y_pred_proba = model.predict(df_generator, steps=df_steps) # prediction probabilities
    #y_pred_classes = y_pred_proba.argmax(axis = -1) # predicted classes : tiles
    #tile_metadata["Predicted"] = y_pred_classes
    #y_classes = df_labels.argmax(axis = -1) # true classes : tiles
    #tile_metadata["Actual"] = y_classes
    #print("Tile Report ---- GENERATOR \n")
    #print(confusion_matrix(y_classes, y_pred_classes))
    #print(classification_report(y_classes, y_pred_classes, target_names = list(map(str, np.unique(tiles_tissue_lst)))))
    
    
    # Results per Sample
    # true values
    sample_true_classes = tile_metadata.loc[:,["Sample","Tissue","Actual"]].drop_duplicates()
    sample_pred_classes = vote_prediction(tile_metadata)
    sample_true_classes = sample_true_classes.merge(sample_pred_classes, left_on='Sample', right_on="Sample")
    #print("sample:true:classes")
    #print(sample_true_classes)
    print("Sample Report\n")
    file_name = "SampleReport_cf_matrix_" + model_type + ".png"
    cf_matrix = confusion_matrix(sample_true_classes.Actual, sample_true_classes.Predicted)
    names = list(map(str, np.unique(sample_true_classes.Tissue)))
    plot_cf_matrix(cf_matrix, model_type, file_name, names, "Sample")
    print(classification_report(sample_true_classes.Actual, sample_true_classes.Predicted,target_names = names))
       

def prepare_model(batch_generator, train_steps, model, batch_size, epochs, validation_steps, validation_data):
    # encode tissue labels to on hot encoding
    #le = preprocessing.LabelEncoder()
    #le.fit(tissue_lst)
    # transform categorical to numerical
    #tissue_lst_enc = le.transform(tissue_lst)
    # one hot encoding
    #tissue_lst_ohe = keras.utils.to_categorical(tissue_lst_enc)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy", tf.keras.metrics.CategoricalAccuracy()])
    #model.fit(df, tissue_lst_ohe, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    model.fit(batch_generator, 
                        epochs=epochs, 
                        steps_per_epoch = train_steps, 
                        validation_steps= validation_steps,
                        validation_data = validation_data, 
                        verbose = 1)
                        #use_multiprocessing=True)


def encode_all_tissues(tissue_lst):
    '''returns the encoding object for all tissues'''
    le = preprocessing.LabelEncoder()
    le.fit(tissue_lst)
    # transform categorical to numerical
    return le


def setup_model(input_shape, num_classes):
    print("Number of labels %d" %(num_classes))
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
	layers.BatchNormalization(),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
	layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
	layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
	layers.BatchNormalization(),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model


def setup_model_pretrained_VGG16(input_shape, num_classes):
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

def finetune_xception_model(batch_generator, train_steps, model, batch_size, epochs, validation_steps, validation_data):
    
  # Unfreeze the base model
    model.trainable = True

    model.compile(loss="categorical_crossentropy", 
                  optimizer=keras.optimizers.Adam(1e-5),  # Very low learning rate
                  metrics=["accuracy", tf.keras.metrics.CategoricalAccuracy()])

    model.fit(batch_generator, 
                        epochs=epochs, 
                        steps_per_epoch = train_steps, 
                        validation_steps= validation_steps,
                        validation_data = validation_data, 
                        verbose = 1)


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
    #df = pd.read_csv(Train_df, skiprows=idx*batch_size, nrows=batch_size)
    #x = df.iloc[:,1:]
    # y = df.iloc[:,0]
    #print("\n idx %i" % (idx))
    pos = idx * batch_size
    sample_list = sample_ids[pos: pos+batch_size]
    #sample_list_ids = sample_labels[pos: pos+batch_size]
    #print(sample_list)
    # scan tiles and generate  a dataframe (df) a list of tissues per tile (tissue_lst)
    # and list of samples
    df, tile_metadata = scan_tiles_bylist(sample_list, folder, metadata, 1)
    #df_labels = keras.utils.to_categorical(label_enc.transform(sample_list_ids))
    df_labels = get_tissue_encodings(tile_metadata.Tissue)
    #print(df_labels)
    #print("unique %d " % (len(np.unique(df_labels))))
    return (df, df_labels)

    
    
        
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



def get_tissue_encodings(tissue_ids):
    res = []
    #print("Tissue encoding...")
    #print(tissue_encoding)
    
    for tid in tissue_ids:
        res.append(tissue_encoding[tid])
    res = np.array(res)
    return res
    
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
        
def get_stats(folder, metadata):
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
    

def main():

    images_folder = "output"
    os.chdir('/Users/test/Dropbox/Mac (2)/Desktop/Keras/images/')
    #os.chdir('/home/pferreira/gtex_histo/')
    
    # load metadata
    metadata_file = "/Users/test/Dropbox/Mac (2)/Desktop/Keras/images/GTExPortal.csv"
    #metadata_file = "/home/pferreira/gtex_histo/GTExPortal.csv"
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
    	"Adipose - Subcutaneous","Stomach","Muscle - Skeletal"]
        	#"Stomach","Muscle - Skeletal", "Nerve - Tibial","Adipose - Visceral (Omentum)", 
        	#"Artery - Aorta", "Artery - Coronary", "Artery - Tibial", "Spleen","Heart - Left Ventricle",
        	#"Esophagus - Gastroesophageal Junction", 
        	#"Esophagus - Mucosa"]
        	k  = 6
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
    X_ids, X_test_ids, y_ids, y_test_ids = train_test_split(s2t.Sampleid, s2t.Tissue, test_size=0.2, stratify = s2t.Tissue)   
    # train / validation
    X_train_ids, X_valid_ids, y_train_ids, y_valid_ids = train_test_split(X_ids, y_ids, test_size=0.2, stratify = y_ids)   
    
    
    NUM_TRAIN_SAMPLES = len(X_train_ids)
    NUM_VALID_SAMPLES = len(X_valid_ids)
    NUM_TEST_SAMPLES = len(X_test_ids)
    batch_size = 1
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
    epochs = 2
    model_type = "VGG16"
    
    #model = setup_model(input_shape, num_classes)
    model = setup_model_pretrained_VGG16(input_shape, num_classes)
    #model = setup_model_pretrained_Xception(input_shape, num_classes)
    prepare_model(my_training_batch_generator, train_steps, model, model_batch_size, epochs, valid_steps, my_valid_batch_generator)
    
    #finetune_xception_model(my_training_batch_generator, train_steps, model, model_batch_size, epochs, valid_steps, my_valid_batch_generator)
    
    # re-initialize our testing data generator, this time for evaluating
    #my_test_batch_generator = batch_generator(X_test_ids, y_test_ids, batch_size, test_steps, images_folder, metadata, tissue_enc)
    # make predictions on the testing images, finding the index of the
    # label with the corresponding largest predicted probability
    # preds = model.predict(x=my_test_batch_generator, steps=(NUM_TEST_SAMPLES // batch_size))
    # predIdxs = np.argmax(predIdxs, axis=1)
    
    X_train, tile_metadata = scan_tiles_bylist(X_train_ids, images_folder, metadata, 1)
    print("Train set results\n")
    evaluate_model(X_train, tile_metadata, model, "Train", my_training_batch_generator, train_steps, model_type)
    del X_train
    print("\n-------------------------------------------------------------\n")
    
    X_valid, tile_metadata = scan_tiles_bylist(X_valid_ids, images_folder, metadata, 1)
    print("Validation set results\n")
    evaluate_model(X_valid, tile_metadata, model, "Validation", my_valid_batch_generator, valid_steps, model_type)
    del X_valid
    print("\n-------------------------------------------------------------\n")
    
    print("Test Results")
    X_test, tile_metadata = scan_tiles_bylist(X_test_ids, images_folder, metadata, 1)
    evaluate_model(X_test, tile_metadata, model, "Test", my_test_batch_generator, test_steps, model_type)
    del X_test
    print("\n-------------------------------------------------------------\n")

main()


def smoker_vs_nonsmoker_classifier():
    # images folder
    folder = "output"
    os.chdir('/Users/test/Dropbox/Mac (2)/Desktop/Keras/images/')
    #os.chdir('/home/pferreira/gtex_histo/')
    
    # load metadata
    metadata_file = "/Users/test/Dropbox/Mac (2)/Desktop/Keras/images/GTExPortal.csv"
    #metadata_file = "/home/pferreira/gtex_histo/GTExPortal.csv"
    metadata = load_gtex_metadata(metadata_file)
    
    # smoking annotation file
    smk_annot_file = "/Users/test/Dropbox/Mac (2)/Desktop/Keras/images/smoking_annotation.csv"
    donor_smk_status = load_smoking_data(smk_annot_file, metadata)
    donor_smk_status.loc[(donor_smk_status.SmokerStatus == "Smoker" )| (donor_smk_status.SmokerStatus == "Non Smoker") ,:]
    
    


def load_smoking_data(smk_annot_file, metadata):
    smk = pd.read_csv(smk_annot_file, sep=";")
    lung_samples = metadata.loc[metadata["Tissue"]=="Lung"]
    smk.set_index("Donor", inplace=True)
    lung_samples_annot = list(set(smk.index).intersection(list(lung_samples.SubjectID)))    
    donor_status = pd.DataFrame(smk.loc[lung_samples_annot,"SmokerStatus"]).reset_index()
    lung_samples = lung_samples.reset_index()
    donor_status = donor_status.merge(lung_samples, left_on='Donor', right_on="SubjectID")
    return donor_status



def plot_cf_matrix(cf_matrix, model_type, file_name, names, image_type):
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    
    labels = [f"{v1}\n{v2}\n" for v1, v2 in zip(group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(3,3)

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
    
   
# read pre-trained model
model = tf.keras.models.load_model("pretrained_vgg16.h5")
X_test, tile_metadata = scan_tiles_bylist(X_test_ids, images_folder, metadata, 1)
df = X_test
 
y_pred_proba = model.predict(df) # prediction probabilities
y_pred_classes = y_pred_proba.argmax(axis = -1) # predicted classes : tiles

idx = 245
actual_class = 16
explainer = lime_image.LimeImageExplainer(random_state=42)
explanation = explainer.explain_instance(X_test[idx], model.predict)
image, mask = explanation.get_image_and_mask(actual_class, positive_only=True, hide_rest=False)
plt.imshow(mark_boundaries(image, mask))
