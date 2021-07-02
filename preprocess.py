import pandas as pd
import numpy as np
import os, librosa, glob, shutil
from tqdm import tqdm
from hparams import *


text_dir = './archive/transcript.v.1.4.txt'

metadata = pd.read_csv(text_dir, dtype='object', sep='|', header=None)
wav_dir = metadata[0].values

out_dir = './data'
os.makedirs(out_dir, exist_ok=True)
os.makedirs(out_dir + '/mel', exist_ok=True)
os.makedirs(out_dir + '/audio', exist_ok=True)

for idx, fn in enumerate(tqdm(wav_dir)):
    file_dir = './archive/kss/'+ fn
    wav, _ = librosa.load(file_dir, sr=sample_rate)
    spec = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    spec = np.abs(spec)
    mel_filter = librosa.filters.mel(sample_rate, n_fft, mel_dim)
    mel_spec = np.dot(mel_filter, spec)
    mel_spec = mel_spec.astype(np.float32)

    mel_name = 'kss-mel-%05d.npy' % idx
    np.save(os.path.join(out_dir + '/mel', mel_name), mel_spec, allow_pickle=False)

    audio_name = 'kss-audio-%05d.npy' % idx
    np.save(os.path.join(out_dir + '/audio', audio_name), wav, allow_pickle=False)

os.makedirs(valid_dir, exist_ok=True)
os.makedirs(valid_dir + '/mel', exist_ok=True)
os.makedirs(valid_dir + '/audio', exist_ok=True)
mel_list = sorted(glob.glob(os.path.join(out_dir + '/mel', '*.npy')))
wav_list = sorted(glob.glob(os.path.join(out_dir + '/audio', '*.npy')))
for i in range(valid_n):
    shutil.move(mel_list[i], valid_dir + '/mel')
    shutil.move(wav_list[i], valid_dir + '/audio')
