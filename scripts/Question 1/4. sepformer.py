import os
import pandas as pd
from sepformer import *
from finetuning import *
from evaluation import *
from data import VoxCelebmix
import random
from tqdm import tqdm
from utils import *
import warnings
warnings.filterwarnings("ignore")

os.chdir('../..')

DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
VOX_MIX_DIR = 'data/vox'



def create_metrics_dir(sepformer, vox_mix, n_samples=1000, sample_rate=16000):
        
    metrics_dir = {}

    for i in tqdm(range(n_samples)):
        sample_idx = random.randint(0, len(vox_mix) - 1)
        wav_mix, wav_s1, wav_s2, sp1, sp2, _ = vox_mix[sample_idx]
        
        min_sz = min(wav_s1.shape[-1], wav_s2.shape[-1])
        
        if wav_s1.shape[-1] > min_sz:
            wav_s1 = wav_s1[:, :min_sz]
        if wav_s2.shape[-1] > min_sz:
            wav_s2 = wav_s2[:, :min_sz]

        pred_s1, pred_s2 = speaker_separation(sepformer, wav_mix)

        
        gt = torch.cat([wav_s1, wav_s2], dim=0).numpy()
        pred = torch.cat([pred_s1, pred_s2], dim=0).numpy()
        
        min_sz = min(gt.shape[-1], pred.shape[-1])
        
        if gt.shape[-1] > min_sz:
            gt = gt[:, :min_sz]
        if pred.shape[-1] > min_sz:
            pred = pred[:, :min_sz]

        sdr, sir, sar, pesq_scores = compute_separation_metrics(gt, pred, sample_rate=sample_rate)
        
        if sp1 not in metrics_dir.keys():
            metrics_dir[sp1] = [[sdr[0]], [sir[0]], [sar[0]], [pesq_scores[0]]]
        else:
            metrics_dir[sp1][0].append(sdr[0])
            metrics_dir[sp1][1].append(sir[0])
            metrics_dir[sp1][2].append(sar[0])
            metrics_dir[sp1][3].append(pesq_scores[0])
            
        if sp2 not in metrics_dir.keys():
            metrics_dir[sp2] = [[sdr[1]], [sir[1]], [sar[1]], [pesq_scores[1]]]
        else:
            metrics_dir[sp2][0].append(sdr[1])
            metrics_dir[sp2][1].append(sir[1])
            metrics_dir[sp2][2].append(sar[1])
            metrics_dir[sp2][3].append(pesq_scores[1])

    return metrics_dir



def get_speaker_ids(data):

    speaker_dict = {}
    for i in tqdm(range(len(data))):
        _, wav_1, wav_2, sp1, sp2, _ = data[i]
        
        if sp1 not in speaker_dict.keys():
            speaker_dict[sp1] = wav_1

        if sp2 not in speaker_dict.keys():
            speaker_dict[sp2] = wav_2

    return speaker_dict



def test(sepformer, model, vox_mix, embedding, speaker_id, n_samples):

    labels = []
    pred_labels = []

    for i in tqdm(range(n_samples)):
        sample_idx = random.randint(0, len(vox_mix) - 1)
        wav_mix, wav_s1, wav_s2, sp1, sp2, _ = vox_mix[sample_idx]
        
        min_sz = min(wav_s1.shape[-1], wav_s2.shape[-1])
        
        if wav_s1.shape[-1] > min_sz:
            wav_s1 = wav_s1[:, :min_sz]
        if wav_s2.shape[-1] > min_sz:
            wav_s2 = wav_s2[:, :min_sz]

        pred_s1, pred_s2 = speaker_separation(sepformer, wav_mix)

        
        gt = torch.cat([wav_s1, wav_s2], dim=0).numpy()
        pred = torch.cat([pred_s1, pred_s2], dim=0).numpy()
        
        min_sz = min(gt.shape[-1], pred.shape[-1])
        
        if gt.shape[-1] > min_sz:
            gt = gt[:, :min_sz]
        if pred.shape[-1] > min_sz:
            pred = pred[:, :min_sz]
            
        
        embeds = model(pred)
        cos_sim = F.cosine_similarity(embeds, embedding, dim=1)
        cos_sim = cos_sim.cpu().numpy()        
        idx = cos_sim.argmax()
        
        labels.append(speaker_id[idx])
        pred_labels.append(sp1)
    
    return labels, pred_labels
    


if __name__ == '__main__':

    vox_mix = VoxCelebmix(data_dir=VOX_MIX_DIR)
    sepformer = get_sepformer_model(device=DEVICE)
    log_text('logs/question1.txt', 'Loaded Sepformer model...')

    n_samples = 1000
    metrics_dir = create_metrics_dir(sepformer, vox_mix=vox_mix, n_samples=1000)

    df_data = []

    for k, v in metrics_dir.items():
        row = {
            "Key": k,
            "SIR_MEAN": np.mean(v[1]), "SIR_VARIANCE": np.var(v[1]),
            "SAR_MEAN": np.mean(v[2]), "SAR_VARIANCE": np.var(v[2]),
            "SDR_MEAN": np.mean(v[0]), "SDR_VARIANCE": np.var(v[0]),
            "PESQ_MEAN": np.mean(v[3]), "PESQ_VARIANCE": np.var(v[3]),
        }
        df_data.append(row)

    df = pd.DataFrame(df_data)
    df.set_index("Key", inplace=True)
    log_text('logs/question1.txt', 'Craeted metrics dataset.')


    # PART 1 - CHECKING ON NON-FINETUNED BASE MODEL

    log_text('logs/question1.txt', 'Testing on non-finetuned base model.')
    
    model = init_model('scripts/Question 1/models/wavlm_base/wavlm_large_finetune.pth')
    model.to(DEVICE)
    log_text('logs/question1.txt', 'Initialised WavLM base model.')


    speaker_dict = get_speaker_ids(vox_mix)
    speaker_list = list(speaker_dict.keys())
    speaker_id = {}

    embedding = torch.zeros((len(speaker_dict)), 256)


    for idx, kv in enumerate(speaker_dict.items()):
        k, wav = kv
        
        wav = torch.stack([wav, wav], dim=0)
        wav = wav.to(DEVICE)
        wav = wav.squeeze_(1)
        embed = model(wav)[0, ...]

        embedding[idx, :] = embed
        speaker_id[idx] = speaker_list.index(k)

    log_text('logs/question1.txt', 'Created input embeddings')

    labels, pred_labels = test(sepformer, model, vox_mix, embedding, speaker_id, n_samples=n_samples)
    log_text('logs/question1.txt', f'Accuracy = {acc(model, [labels, pred_labels])}')



    # PART 2 - TESTING ON FINETUNED MODEL

    replace_linear_with_lora(model, DEVICE)
    model.load_state_dict(torch.load('scripts/Question 1/models/wavlm_base/wavlm_base_lora_finetune.pth'), strict=False)

    log_text('logs/question1.txt', 'Initialised finetuned model...')

    for idx, kv in enumerate(speaker_dict.items()):
        k, wav = kv
        
        wav = torch.stack([wav, wav], dim=0)
        wav = wav.to(DEVICE)
        wav = wav.squeeze_(1)
        embed = model(wav)[0, ...]

        embedding[idx, :] = embed
        speaker_id[idx] = speaker_list.index(k)

    log_text('logs/question1.txt', 'Created input embeddings')

    labels, pred_labels = test(sepformer, model, vox_mix, embedding, n_samples=n_samples)
    log_text('logs/question1.txt', f'Accuracy = {acc(model, [labels, pred_labels])}')

