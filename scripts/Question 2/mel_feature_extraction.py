import librosa as lb
import numpy as np
import pandas as pd
import os
import kagglehub as khub
from utils import *
os.chdir('../../')


def mel_feature_extraction(file_path, n_mels=13):
    
    y, sr = lb.load(file_path)
    
    mel_spectrogram = lb.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    
    mel_spectrogram_db = lb.power_to_db(mel_spectrogram, ref=np.max)
    
    mfccs = lb.feature.mfcc(S=mel_spectrogram_db, sr=sr, n_mfcc=n_mels)
    
    return mfccs.tolist()



def get_dataset(dataset_name, download_path):
    
    os.makedirs(download_path, exist_ok=True)
    
    dataset_pth = khub.dataset_download(dataset_name, path=download_path)
    log_text('logs/question2.txt', f'Dataset file is downloaded at {dataset_pth}')
    
    return dataset_pth


if __name__ == '__main__':

    dataset_path = get_dataset('hbchaitanyabharadwaj/audio-dataset-with-10-indian-languages/', 'data/audio_dataset/')
    # print(dataset_path)

    print(dir(khub))