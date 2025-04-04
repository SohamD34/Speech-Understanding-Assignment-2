import logging
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from functools import partial
import re
import random
import os
import gdown



def setup_logger(log_file_path):
    '''
    A helper function to setup the logger - allowing it to access files, read & write them.
    '''
    logger = logging.getLogger('custom_logger')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        fh = logging.FileHandler(log_file_path)
        formatter = logging.Formatter('%(asctime)s - %(message)s')

        fh.setLevel(logging.INFO)    
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger



def log_text(log_file_path, text):
    '''
    A helper function to log the text to the file path specified. 
    '''
    logger = setup_logger(log_file_path)
    logger.info(text)





def download_checkpoint():
    """
    Download the wavlm_base_plus checkpoint from Google Drive.
    https://drive.usercontent.google.com/download?id=1OMdkp5Vv8A9WnHSTSoDwA8hxQWsEAu85&export=download&authuser=0&confirm=t&uuid=a64ed218-1916-4492-8fd8-06d8fa3dbcc5&at=AEz70l5_VmQCkAgiR4OoSRUPC7In:1743420047651    """
    file_ids = [
        "1OMdkp5Vv8A9WnHSTSoDwA8hxQWsEAu85",  # Corresponds to wavlm_base_plus
    ]
    
    file_names = ["wavlm_base_plus_nofinetune.pth"]
    
    base_url = "https://drive.google.com/uc?export=download&id="
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    download_dir = os.path.join(current_dir, "checkpoints")
    
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    
    for file_id, file_name in zip(file_ids, file_names):
        url = base_url + file_id
        output_path = os.path.join(download_dir, file_name)
        print(f"Downloading {file_name} from {url}...")
        gdown.download(url, output_path, quiet=False)
        print(f"Downloaded {file_name} to {output_path}")