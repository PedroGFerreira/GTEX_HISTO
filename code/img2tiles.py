import os
import pandas as pd
import subprocess
import sys, getopt
import argparse

os.chdir('/Users/test/Dropbox/Mac (2)/Desktop/Keras/images/')
# ssh pferreira@192.168.40.85
#Vilar2022


# DEFAULT VALUES FOR SCRIPT 
pyhist_folder = "/home/pferreira/PyHIST/"
# first n samples from tissue




parser = argparse.ArgumentParser(description="Just an example",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-ns", "--num_samples", type=int, default = 10, help="number of samples to extract")
parser.add_argument("-tis", "--tissue", type=str,  default="Lung", help="tissue data")
parser.add_argument("-ps", "--patch-size", type=int,  default = 128, help="patch size 64, 128, ...")
parser.add_argument("-ct", "--content-threshold", type=float,  default = 0.05, help="area of tile covered by tissue images")
args = parser.parse_args()
config = vars(args)
print(config)
# set values to specific variables
num_samples = config["num_samples"]
tissue = config["tissue"]
patch_size = config["patch_size"]
content_threshold = config["content_threshold"]
download_sample = 16


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

# metadata file
metadata = load_gtex_metadata("GTExPortal.csv")

def already_processed(folder):
    # get samples already processed
    samples = [ f.name for f in os.scandir(folder) if f.is_dir() ]
    tissues = list(set(list(metadata.loc[samples].Tissue)))
    return (samples, tissues)

folder = "output"
samples_tiled, tissues = already_processed(folder)

# samples to process
sample_ids = list(metadata[metadata["Tissue"]== tissue].iloc[0:num_samples].index)



# example 
# curl 'https://brd.nci.nih.gov/brd/imagedownload/GTEX-1117F-0126' --output 'GTEX-1117F-0126.svs'
# python /home/pferreira/PyHIST/pyhist.py --patch-size 64 --output-downsample 16 --save-patches --save-tilecrossed-image --info "verbose" GTEX-1117F-0126.svs
for sid in sample_ids:
    if sid not in samples_tiled:
        # retrieve file
        curl_cmd = "curl \'https://brd.nci.nih.gov/brd/imagedownload/"
        curl_cmd = curl_cmd + sid + "\' --output \'" +  sid + ".svs\'"
        #print (curl_cmd)
        print("Curl for file {}".format(sid) )
        os.system(curl_cmd)
        
        # process to tiling with PyHIST
        pyhist = "python3 %spyhist.py --method \"otsu\" --patch-size %d  --content-threshold %f --output-downsample %d --save-patches --save-tilecrossed-image --info \"verbose\" %s.svs" % (pyhist_folder, patch_size, content_threshold, download_sample, sid)
        print("Call PyHist on file {}.svs".format(sid) )
        print(pyhist)
        os.system(pyhist)
        
        # remove svs file due to size
        rm_svs = "rm %s.svs" % (sid)
        print("remove file {}".format(sid) )
        os.system(rm_svs)
        print("\n\n")
    else:
        print ("samples %s already processed" % (sid))


#python /home/pferreira/PyHIST/pyhist.py --method "otsu" --patch-size 128 --content-threshold 0.5 --output-downsample 16  --tilecross-downsample 32 --save-patches --save-mask --save-tilecrossed-image --info "verbose" --output 'GTEX-1117F-0126.svs'  GTEX-1117F-0126.svs
