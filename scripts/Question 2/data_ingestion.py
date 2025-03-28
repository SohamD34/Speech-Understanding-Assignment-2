import os 
import gdown
import zipfile
import kagglehub
from utils import *
os.chdir('../../')



def download_data(dataset_name):
    
    log_text('logs/question2.txt', 'Accessing the Kaggle Hub API...')

    path = kagglehub.dataset_download(dataset_name)

    log_text('logs/question2.txt', f'Dataset file is downloaded at {path}')



def extract_data(data_zip_loc):
    
    log_text('logs/question1.txt', f'Extracting the dataset...')
    with zipfile.ZipFile(data_zip_loc, 'r') as zip_ref:
        zip_ref.extractall('data')

    os.remove(data_zip_loc)
    log_text('logs/question1.txt', 'Extraction completed!')




if __name__ == '__main__':

    log_text('logs/question2.txt', 'Ingesting Languages Audio Dataset...\n')
    dataset_name = 'hbchaitanyabharadwaj/audio-dataset-with-10-indian-languages'

    download_data(dataset_name)
    # extract_data('data/voxceleb1.zip')
    # rename_dir('data/wav', 'data/voxceleb1')