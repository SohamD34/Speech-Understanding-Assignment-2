import os 
import gdown
import zipfile
from utils import *
import pandas as pd
import requests
import csv
os.chdir('../../')


def download_data(url, dataset_name):
    
    file_id = url.split('/')[-2]
    download_url = f'https://drive.google.com/uc?id={file_id}'
    output_loc = f'data/{dataset_name}.zip'

    if not os.path.exists('data'):
        os.makedirs('data')

    log_text('logs/question1.txt', 'Accessing the public URL...')
    gdown.download(download_url, output_loc, quiet=False)
    
    log_text('logs/question1.txt', f'Dataset ZIP file is downloaded at {output_loc}/file.zip')



def extract_data(data_zip_loc):
    
    log_text('logs/question1.txt', f'Extracting the dataset...')
    with zipfile.ZipFile(data_zip_loc, 'r') as zip_ref:
        zip_ref.extractall('data')

    os.remove(data_zip_loc)
    log_text('logs/question1.txt', 'Extraction completed!')



def rename_dir(orig_dir_path, new_dir_name):
    
    log_text('logs/question1.txt', 'Renaming the directory...')
    os.rename(orig_dir_path, new_dir_name)
    log_text('logs/question1.txt', f'Directory renamed to {new_dir_name}\n')




def get_trial_pairs(txt_file_path, output_path):
    
    log_text('logs/question1.txt', 'Fetching trial pairs...')
    with open(txt_file_path, 'r') as file:
        lines = file.readlines()

    data = [line.strip().split(' ') for line in lines]
    df = pd.DataFrame(data, columns=['Column1', 'Column2', 'Column3'])

    df.to_csv(output_path, index=False)
    log_text('logs/question1.txt', f'Trial pairs saved to {output_path}')
    


def load_trial_pairs(csv_path):
    """
    Load the trial pairs from the VoxCeleb1 trial pairs CSV file.
    """
    log_text('logs/question1.txt', "Loading trial pairs from CSV...")
    
    pairs = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 3:
                # Format: label, audio_path1, audio_path2
                label = int(row[0])
                audio_path1 = row[1]
                audio_path2 = row[2]
                pairs.append((label, audio_path1, audio_path2))
    
    log_text('logs/question1.txt', f"Loaded {len(pairs)} trial pairs.")
    return pairs



if __name__ == '__main__':

    log_text('logs/question1.txt', 'Ingesting Vox Celeb 1 Dataset...\n')
    url = 'https://drive.google.com/file/d/1AC-Q8dEw1LTPdpEi5ofS04ZmZWdl8BBg/view?usp=drive_link'
    dataset_name = 'voxceleb1'

    download_data(url, dataset_name)
    extract_data('data/voxceleb1.zip')
    rename_dir('data/wav', 'data/voxceleb1')

    log_text('logs/question1.txt', 'Ingesting Vox Celeb 2 Dataset...\n')
    url = 'https://drive.google.com/file/d/1Onu4jzcyasrxTR1rRT9rl9AfkUrueM6o/view?usp=drive_link'
    dataset_name = 'voxceleb2'

    download_data(url, dataset_name)
    extract_data('data/voxceleb2.zip')
    rename_dir('data/aac', 'data/voxceleb2')
    

    get_trial_pairs('data/voxceleb1_trial_pairs.txt', 'data/voxceleb1_trial_pairs.csv')
