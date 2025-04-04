import os
import argparse
import glob
import random
import soundfile as sf
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import *
os.chdir('../..')




def list_identity_dirs(vox2_dir):
    """
    List identity directories under vox2_dir sorted in ascending order.
    """
    identities = [
        os.path.join(vox2_dir, d) for d in os.listdir(vox2_dir)
        if os.path.isdir(os.path.join(vox2_dir, d))
    ]
    return sorted(identities)




def build_metadata(identity_dirs):
    """
    Build a dictionary mapping speaker ID (directory name) to list of utterance file paths.
    Searches recursively for .wav files.
    """
    metadata = {}
    for id_dir in identity_dirs:
        spk_id = os.path.basename(id_dir)
        utt_files = glob.glob(os.path.join(id_dir, '**', '*.wav'), recursive=True)
        if len(utt_files) > 0:
            metadata[spk_id] = utt_files
            
    return metadata



def create_mixtures(metadata, output_subdir, sample_rate, mix_ratio, min_duration, num_mixtures):
    """
    For a given metadata dictionary (speaker -> list of utterance paths),
    randomly pick two different speakers and mix one utterance from each.
    The overlapping is done by summing the two signals where the second signal is placed
    at a random offset in the first signal.
    Outputs mixtures, source files, and writes a CSV metadata file.
    """
    speakers = list(metadata.keys())
    mixtures_info = []

    mix_out_dir = os.path.join(output_subdir, "mixtures")
    src_out_dir = os.path.join(output_subdir, "sources")
    os.makedirs(mix_out_dir, exist_ok=True)
    
    # log_text('logs/question1.txt', speakers)
    
    for i in tqdm(range(num_mixtures), desc="Creating mixtures"):
        spk1, spk2 = random.sample(speakers, 2)
        utt1 = random.choice(metadata[spk1])
        utt2 = random.choice(metadata[spk2])

        try:
            audio1, sr1 = sf.read(utt1, dtype='float32')
            audio2, sr2 = sf.read(utt2, dtype='float32')
        except Exception as e:
            log_text('logs/question1.txt', f"Error reading files {utt1} or {utt2}: {e}")
            continue

        if sr1 != sample_rate or sr2 != sample_rate:
            continue

        if len(audio1) < sample_rate * min_duration or len(audio2) < sample_rate * min_duration:
            continue

        overlap_length = int(min(len(audio1), len(audio2)) * mix_ratio)
        if overlap_length <= 0:
            continue

        max_offset = len(audio1) - overlap_length
        offset = random.randint(0, max_offset) if max_offset > 0 else 0

        mixture = np.copy(audio1)
        available_length = len(audio1) - offset
        length_to_mix = min(available_length, len(audio2), overlap_length)
        mixture[offset:offset+length_to_mix] += audio2[:length_to_mix]


        mix_id = f"mix_{i:05d}"
        mix_filename = mix_id + ".wav"
        mix_path = os.path.join(mix_out_dir, mix_filename)

        sf.write(mix_path, mixture, sample_rate)

        src1_dir = os.path.join(src_out_dir, spk1)
        src2_dir = os.path.join(src_out_dir, spk2)
        os.makedirs(src1_dir, exist_ok=True)
        os.makedirs(src2_dir, exist_ok=True)
        src1_path = os.path.join(src1_dir, mix_filename)
        src2_path = os.path.join(src2_dir, mix_filename)
        sf.write(src1_path, audio1, sample_rate)
        sf.write(src2_path, audio2, sample_rate)

        mixtures_info.append({
            "mix_id": mix_id,
            "speaker1": spk1,
            "speaker2": spk2,
            "utt1": utt1,
            "utt2": utt2,
            "mix_path": os.path.abspath(mix_path),
            "src1_path": os.path.abspath(src1_path),
            "src2_path": os.path.abspath(src2_path),
            "offset": offset,
            "overlap_length": length_to_mix,
            "mixture_length": len(mixture)
        })

    meta_df = pd.DataFrame(mixtures_info)
    meta_csv_path = os.path.join(output_subdir, "metadata.csv")
    meta_df.to_csv(meta_csv_path, index=False)
    log_text('logs/question1.txt', f"Saved mixture metadata CSV at: {meta_csv_path}")




    
if __name__ == '__main__':
    

    vox2_dir = 'data/voxceleb2'
    output_dir = 'data/mixture'
    sample_rate = 16000
    mix_ratio = 0.5
    min_duration = 1.0
    num_mixtures = 1000

    identities = list_identity_dirs(vox2_dir)
    if len(identities) < 100:
        raise ValueError("At least 100 identity directories are needed in the VoxCeleb2 dataset.")

    log_text('logs/question1.txt', f"Found {len(identities)} identities in {vox2_dir}")

    train_identity_dirs = identities[:50]
    test_identity_dirs = identities[50:100]

    log_text('logs/question1.txt', "Building training metadata...")
    train_metadata = build_metadata(train_identity_dirs)

    log_text('logs/question1.txt', "Building testing metadata...")
    test_metadata = build_metadata(test_identity_dirs)

    train_output_dir = os.path.join(output_dir, "vox2_mix", "train")
    test_output_dir = os.path.join(output_dir, "vox2_mix", "test")
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    log_text('logs/question1.txt', "Creating training mixtures...")
    create_mixtures(train_metadata, train_output_dir, sample_rate, mix_ratio, min_duration, num_mixtures)
    
    log_text('logs/question1.txt', "Creating testing mixtures...")
    create_mixtures(test_metadata, test_output_dir, sample_rate, mix_ratio, min_duration, num_mixtures)

    log_text('logs/question1.txt', 'Finished creating mixtures at data/mixtures.\n')
