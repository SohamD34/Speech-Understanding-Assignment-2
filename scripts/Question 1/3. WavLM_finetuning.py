import os
from finetuning import train_lora, replace_linear_with_lora
from data import VoxCeleb1, VoxCeleb2
from evaluation import init_model, evaluate
import torch
import warnings
warnings.filterwarnings('ignore')
from utils import *

os.chdir('../..')

if __name__=="__main__":

    vox1_dat = VoxCeleb1('data/voxceleb1')
    log_text('logs/question1.txt', 'Loading Vox 1 dataset...')

    vox2_dat = VoxCeleb2('data/voxceleb2')
    log_text('logs/question1.txt', 'Loading Vox2 dataset...')
    
    CHECKPOINT = 'scripts/Question 1/models/wavlm_base/wavlm_base_lora_finetune.pth'
    model = init_model(CHECKPOINT)
    log_text('logs/question1.txt','Loaded base model from checkpoint')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    replace_linear_with_lora(model, device)
    log_text('logs/question1.txt','Converted model to LoRA')
    
    log_text('logs/question1.txt', 'Finetuning the model...')
    train_loss, test_acc, test_auc, test_f1, test_precision, test_recall = train_lora(model, vox2_dat, epochs=10, batch_size=128, device=device, model_save=True, save_path='models/wavlm_base_lora_finetune.pth')
    
    log_text('logs/question1.txt', 'Evaluating finetuned model...')
    auc_score, acc_score, f1, precision, recall, eer, tar_1_far, speaker_id_acc = evaluate(model, vox1_dat, 'data/vox1_trial_pairs.txt', batch_size=128, device=device)
    metrics = f"""Metrics:
    AUC: {auc_score}
    Accuracy: {acc_score}
    F1_score; {f1}
    Precision: {precision}
    Recall: {recall}
    EER: {eer}
    TAR @ 1% FAR: {tar_1_far}
    
    Speaker Identification Accuracy: {speaker_id_acc}
"""
    log_text('logs/question1.txt', metrics)