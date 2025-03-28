import os 
import gdown
import zipfile
from utils import *

print(os.getcwd())
os.chdir('../../')
print(os.getcwd())



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
    