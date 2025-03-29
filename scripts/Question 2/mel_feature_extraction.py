import librosa as lb
import numpy as np
import pandas as pd
import os
import zipfile 
import matplotlib.pyplot as plt
from utils import *
import warnings
warnings.filterwarnings("ignore")
os.chdir('../../')



def unzip_dataset(zip_path):

    unzip_path = 'data/audio_dataset/'
    
    if not os.path.exists(unzip_path):
        os.makedirs(unzip_path)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(unzip_path)
    
    os.remove(zip_path)
    os.rmdir('scripts/Question 2/data')
    log_text("logs/question2.txt",f"Dataset unzipped at {unzip_path}!")



def mel_feature_extraction(y, sr, n_mels=13):
    
    mel_spectrogram = lb.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spectrogram_db = lb.power_to_db(mel_spectrogram, ref=np.max)
    mfccs = lb.feature.mfcc(S=mel_spectrogram_db, sr=sr, n_mfcc=n_mels)
    
    return mfccs[:n_mels].tolist()




def get_mfcc_features(data_path):

    all_mfccs = []
    all_labels = []


    for language in os.listdir(data_path):
        print('Converting language:', language) 

        # count = 0
        language_path = os.path.join(data_path, language)

        for file in os.listdir(language_path):
            if file.endswith('.mp3'):
                file_path = os.path.join(language_path, file)

                try:
                    y, sr = lb.load(file_path)
                    if len(y) != 0:
                        mfccs = mel_feature_extraction(y, sr, 13)

                        row_means = np.mean(mfccs, axis=1)
                        row_stds = np.std(mfccs, axis=1)
                        mfccs = np.concatenate((row_means, row_stds)).tolist()

                        all_mfccs.append(mfccs)
                        all_labels.append(language)
                except:
                    continue

        log_text("logs/question2.txt",f"MFCC features extracted for {language} language")

    all_mfccs = pd.DataFrame(all_mfccs)
    all_mfccs['label'] = all_labels
    output_path = 'data/audio_dataset/mel_features.csv'
    all_mfccs.to_csv(output_path, index=False)

    log_text("logs/question2.txt",f"MFCC features extracted and saved to {output_path}")
    return output_path




def generate_spectrograms(language_path):

    for language in os.listdir(language_path):
        count = 0
        language_spectrogram_path = f'scripts/Question 2/spectrograms/{language}/'
        os.makedirs(language_spectrogram_path, exist_ok=True)

        for file in os.listdir(os.path.join(language_path, language)):
            if file.endswith('.mp3') and count < 5:
                file_path = os.path.join(language_path, language, file)
                y, sr = lb.load(file_path)

                if len(y) > 0:
                    mel_spectrogram = lb.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                    mel_spectrogram_db = lb.power_to_db(mel_spectrogram, ref=np.max)

                    plt.figure(figsize=(10, 4))
                    lb.display.specshow(mel_spectrogram_db, sr=sr, x_axis='time', y_axis='mel', cmap='viridis')
                    plt.colorbar(format='%+2.0f dB')
                    plt.title(f'Mel Spectrogram - {file}')
                    plt.tight_layout()

                    output_file = os.path.join(language_spectrogram_path, f"{os.path.splitext(file)[0]}.png")
                    plt.savefig(output_file)
                    plt.close()
                    count += 1
            else:
                break

    log_text("logs/question2.txt", f"Spectrograms generated and saved in 'scripts/Question 2/spectrograms/'")





if __name__ == '__main__':

    # Part 1 - Importing the Dataset

    if not os.path.exists('data/audio_dataset/Language Detection Dataset'):
        unzip_dataset('scripts/Question 2/data/archive.zip')
    

    # Part 2 - Extracting Mel Cepstral Coefficients (MFCCs) 

    data_path = 'data/audio_dataset/Language Detection Dataset'
    data_path = get_mfcc_features(data_path)
    print(f"MFCC features saved to {data_path}")


    # Part 3 - Generating and Visualising Spectrograms

    generate_spectrograms('data/audio_dataset/Language Detection Dataset/')
    print("Spectrograms generated and saved in 'scripts/Question 2/spectrograms/'")

