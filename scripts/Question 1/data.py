import os
import torch
from torchaudio.transforms import Resample
import soundfile as sf
import pandas as pd
from torch.utils.data import Dataset
import subprocess



class VoxCeleb1(Dataset):
    def __init__(self, data_dir, transform=None, sample_freq=16000):
        self.data_dir = data_dir
        self.transform = transform
        self.audio_files = [os.path.join(dirpath, f) for dirpath, _, filenames in os.walk(data_dir) for f in filenames if f.endswith('.wav')]
        
        self.sample_freq = sample_freq

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.data_dir, self.audio_files[idx])
        
        waveform, sr1 = sf.read(audio_path)

        waveform = torch.from_numpy(waveform).unsqueeze(0).float()
        resample1 = Resample(orig_freq=sr1, new_freq=self.sample_freq)
        waveform = resample1(waveform)

        if self.transform:
            waveform = self.transform(waveform)

        return waveform
    
    def __getitem__(self, path):
        audio_path = os.path.join(self.data_dir, path)
        
        waveform, sr1 = sf.read(audio_path)

        waveform = torch.from_numpy(waveform).unsqueeze(0).float()
        resample1 = Resample(orig_freq=sr1, new_freq=self.sample_freq)
        waveform = resample1(waveform)

        if self.transform:
            waveform = self.transform(waveform)

        return waveform




class VoxCeleb2(Dataset):
    def __init__(self, data_dir, transform=None, sample_freq=16000):
        
        to_wav = True
        self.data_dir = data_dir
        for dirpath, _, filenames in os.walk(data_dir):
            for filename in filenames:
                if filename.endswith('.wav'):
                    to_wav = False
                    break
                
        if to_wav:
            print("wav files not found. Converting to wav format...")
            m4a_to_wav(data_dir)
            print("Converted m4a files to wav format.")
           
         
        self.dirs = os.listdir(data_dir)
        self.dirs.sort()
        
        self.train_dirs = self.dirs[:100]
        self.test_dirs = self.dirs[100:]
        
        self.train_file_paths = [os.path.join(data_dir, i, j, k) for i in self.train_dirs for j in os.listdir(os.path.join(data_dir, i)) for k in os.listdir(os.path.join(data_dir, i, j)) if k.endswith('.wav')]
        self.train_ids = [i.split('/')[-3] for i in self.train_file_paths]
        train_ids = list(set(self.train_ids))
        self.train_dct = {train_ids[i] : i for i in range(len(train_ids))}  
        
        self.test_file_paths = [os.path.join(data_dir, i, j, k) for i in self.test_dirs for j in os.listdir(os.path.join(data_dir, i)) for k in os.listdir(os.path.join(data_dir, i, j)) if k.endswith('.wav')]
        self.test_ids = [i.split('/')[-3] for i in self.test_file_paths]
        test_ids = list(set(self.test_ids))
        self.test_dct = {test_ids[i] : i for i in range(len(test_ids))}  
        
        self.train = True
        
        self.sample_freq = sample_freq
        
        self.transform = None
        
    def __len__(self):
        if self.train:
            return len(self.train_file_paths)
        else:
            return len(self.test_file_paths)

    def __getitem__(self, idx: int):
        audio_path = None
        id_ = None
        if self.train:
            audio_path = self.train_file_paths[idx]
            identity = self.train_ids[idx]
            id_ = self.train_dct[identity]
        else:
            audio_path = self.test_file_paths[idx]
            identity = self.test_ids[idx]
            id_ = self.test_dct[identity]
            
        waveform, sr1 = sf.read(audio_path)

        waveform = torch.from_numpy(waveform).unsqueeze(0).float()
        resample1 = Resample(orig_freq=sr1, new_freq=self.sample_freq)
        waveform = resample1(waveform)

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, id_
    
    def __getitem__(self, path: str):
        print(path)
        audio_path = os.path.join(self.data_dir, path)
        
        waveform, sr1 = sf.read(audio_path)

        waveform = torch.from_numpy(waveform).unsqueeze(0).float()
        resample1 = Resample(orig_freq=sr1, new_freq=self.sample_freq)
        waveform = resample1(waveform)

        if self.transform:
            waveform = self.transform(waveform)

        return waveform
        


        
class VoxCelebmix(Dataset):
    def __init__(self, data_dir, transform=None, sample_freq=16000):   
        
        self.train_metadata = pd.read_csv(os.path.join(data_dir, 'train','metadata.csv'))
        self.test_metadata = pd.read_csv(os.path.join(data_dir, 'test','metadata.csv'))
        
        self.train = True
        
        self.sample_freq = sample_freq
        
        self.transform = None
        
    def __len__(self):
        if self.train:
            return self.train_metadata.shape[0]
        else:
            return self.test_metadata.shape[0]

    def __getitem__(self, idx: int):
        mix_path = None
        src1_path = None
        src2_path = None
        
        sp_1 = None
        sp_2 = None
        
        mix_len = None
        if self.train:
            mix_path = self.train_metadata['mix_path'][idx]
            src1_path = self.train_metadata['src1_path'][idx]
            src2_path = self.train_metadata['src2_path'][idx]
            
            sp_1 = self.train_metadata['speaker1'][idx]
            sp_2 = self.train_metadata['speaker2'][idx]
            
            mix_len = self.train_metadata['mixture_length'][idx]
            
            
        else:
            mix_path = self.train_metadata['mix_path'][idx]
            src1_path = self.train_metadata['src1_path'][idx]
            src2_path = self.train_metadata['src2_path'][idx]
            
            sp_1 = self.train_metadata['speaker1'][idx]
            sp_2 = self.train_metadata['speaker2'][idx]
            
            mix_len = self.train_metadata['mixture_length'][idx]
            
        waveform_mix, sr1_mix = sf.read(mix_path)

        waveform_mix = torch.from_numpy(waveform_mix).unsqueeze(0).float()
        resample1 = Resample(orig_freq=sr1_mix, new_freq=self.sample_freq)
        waveform_mix = resample1(waveform_mix)

        waveform_s1, sr1_s1 = sf.read(src1_path)

        waveform_s1 = torch.from_numpy(waveform_s1).unsqueeze(0).float()
        resample1 = Resample(orig_freq=sr1_s1, new_freq=self.sample_freq)
        waveform_s1 = resample1(waveform_s1)
        
        waveform_s2, sr1_s2 = sf.read(src2_path)
        
        waveform_s2 = torch.from_numpy(waveform_s2).unsqueeze(0).float()
        resample1 = Resample(orig_freq=sr1_s2, new_freq=self.sample_freq)
        waveform_s2 = resample1(waveform_s2)

        return waveform_mix, waveform_s1, waveform_s2, sp_1, sp_2, mix_len




def m4a_to_wav(data_dir):
    with open(os.path.join(data_dir, 'convert.sh'), 'w') as f:
        content = '''# copy this to root directory of data and 
# chmod a+x convert.sh
# ./convert.sh
# https://unix.stackexchange.com/questions/103920/parallelize-a-bash-for-loop

open_sem(){
    mkfifo pipe-$$
    exec 3<>pipe-$$
    rm pipe-$$
    local i=$1
    for((;i>0;i--)); do
    printf %s 000 >&3
    done
}
run_with_lock(){
    local x
    read -u 3 -n 3 x && ((0==x)) || exit $x
    (
    ( "$@"; )
    printf '%.3d' $? >&3
    )&
}

N=32 # number of vCPU
open_sem $N
for f in $(find . -name "*.m4a"); do
    run_with_lock ffmpeg -loglevel panic -i "$f" -ar 16000 "${f%.*}.wav"
done
'''
        f.write(content)
        print("convert.sh script created successfully.")
        
    subprocess.run(['chmod', 'a+x', os.path.join(data_dir, 'convert.sh')])
    print("convert.sh script made executable.")
        
    subprocess.run(['bash', 'convert.sh'], cwd=data_dir)
    print("convert.sh script executed successfully.")
        
    subprocess.run(['rm', 'convert.sh'], cwd=data_dir)
    print("convert.sh script removed successfully.")