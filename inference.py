import os, argparse, glob, librosa, librosa.display, torch
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm
from gan import Generator
from matplotlib import cm
from hparams import *


def main(args):
    vocoder = Generator()
    vocoder = vocoder.cuda()
    ckpt = torch.load(args.checkpoint)
    vocoder.load_state_dict(ckpt['G'])
    testset = glob.glob(os.path.join(args.test_dir, '*.wav'))
    for i, test_path in enumerate(tqdm(testset)):
        mel, spectrogram = process_audio(test_path)
        g_audio = vocoder(mel.cuda())
        g_audio = g_audio.squeeze().cpu()
        audio = g_audio.detach().numpy() * 32768
        g_spec = librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        sf.write(os.path.join(args.save_dir, 'generated-{}.wav'.format(i)),
                 audio.astype('int16'),
                 sample_rate)
        plot_stft(spectrogram, g_spec, i)


def process_audio(wav_path):
    wav, sr = librosa.core.load(wav_path, sr=sample_rate)
    mel_basis = librosa.filters.mel(sr, n_fft=n_fft, n_mels=mel_dim)
    spectrogram = librosa.stft(y=wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mel_spectrogram = np.dot(mel_basis, np.abs(spectrogram)).astype(np.float32)
    mel_spectrogram = torch.from_numpy(mel_spectrogram)
    return mel_spectrogram.unsqueeze(0), spectrogram


def plot_stft(spectrogram, g_spec, idx):
    plt.figure(figsize=(12, 8))

    spectrogram = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max)
    plt.subplot(2, 1, 1)
    librosa.display.specshow(spectrogram, x_axis='time', y_axis='log', hop_length=256)
    plt.title('original audio spectrogram')

    g_spec = librosa.amplitude_to_db(np.abs(g_spec), ref=np.max)
    plt.subplot(2, 1, 2)
    librosa.display.specshow(g_spec, x_axis='time', y_axis='log', hop_length=256)
    plt.title('generated audio spectrogram')

    plt.tight_layout()
    fn = 'spectrogram-%d.png' % idx
    plt.savefig(args.save_dir + '/' + fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', '-t', default='./test')
    parser.add_argument('--checkpoint', '-p', required=True)
    parser.add_argument('--save_dir', '-s', default='./output')
    args = parser.parse_args()
    save_dir = os.path.join(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    main(args)
