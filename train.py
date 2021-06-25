import os, argparse, traceback, glob, librosa, random, itertools, time, torch
import numpy as np
import soundfile as sf
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from gan import Generator, MultiScale
from hparams import *


class MelDataset(Dataset):
    def __init__(self, mel_list, audio_list):
        self.mel_list = mel_list
        self.audio_list = audio_list

    def __len__(self):
        return len(self.mel_list)

    def __getitem__(self, idx):
        mel = np.load(self.mel_list[idx])
        mel = torch.from_numpy(mel).float()
        start = random.randint(0, mel.size(1) - seq_len - 1)
        mel = mel[:, start : start + seq_len]

        wav = np.load(self.audio_list[idx])
        wav = torch.from_numpy(wav).float()
        start *= hop_length
        wav = wav[start : start + seq_len * hop_length]

        return mel, wav.unsqueeze(0)


def train(args):
    base_dir = 'data'
    mel_list = sorted(glob.glob(os.path.join(base_dir + '/mel', '*.npy')))
    audio_list = sorted(glob.glob(os.path.join(base_dir + '/audio', '*.npy')))
    trainset = MelDataset(mel_list, audio_list)
    train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=0, shuffle=True, drop_last=True)

    test_mel = sorted(glob.glob(os.path.join(valid_dir + '/mel', '*.npy')))
    testset = []
    for d in test_mel:
        mel = np.load(d)
        mel = torch.from_numpy(mel).float()
        mel = mel.unsqueeze(0)
        testset.append(mel)

    G = Generator().cuda()
    D = MultiScale().cuda()

    g_optimizer = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
    d_optimizer = optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.9))

    step, epochs = 0, 0
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint)
        G.load_state_dict(ckpt['G'])
        g_optimizer.load_state_dict(ckpt['g_optimizer'])
        D.load_state_dict(ckpt['D'])
        d_optimizer.load_state_dict(ckpt['d_optimizer'])
        step = ckpt['step'],
        epochs = ckpt['epoch']
        print('Load Status: Epochs %d, Step %d' % (epochs, step))

    torch.backends.cudnn.benchmark = True

    start = time.time()
    try:
        for epoch in itertools.count(epochs):
            for (mel, audio) in train_loader:
                mel = mel.cuda()
                audio = audio.cuda()

                # Discriminator
                d_real = D(audio)
                d_loss_real = 0
                for scale in d_real:
                    d_loss_real += F.relu(1 - scale[-1]).mean()

                fake_audio = G(mel)
                d_fake = D(fake_audio.cuda().detach())
                d_loss_fake = 0
                for scale in d_fake:
                    d_loss_fake += F.relu(1 + scale[-1]).mean()

                d_loss = d_loss_real + d_loss_fake

                D.zero_grad()
                d_loss.backward()
                d_optimizer.step()

                # Generator
                d_fake = D(fake_audio.cuda())
                g_loss = 0
                for scale in d_fake:
                    g_loss += -scale[-1].mean()

                # Feature Matching
                feature_loss = 0
                for i in range(3):
                    for j in range(len(d_fake[i]) - 1):
                        feature_loss += F.l1_loss(d_fake[i][j], d_real[i][j].detach())

                g_loss += lambda_feat * feature_loss

                G.zero_grad()
                g_loss.backward()
                g_optimizer.step()

                step += 1
                if step % log_step == 0:
                    print('step: {}, D_loss: {:.3f}, G_loss: {:.3f}, {:.3f} sec/step'.format(
                        step, d_loss, g_loss, (time.time() - start) / log_step))
                    start = time.time()

                if step % checkpoint_step == 0:
                    save_dir = './ckpt/' + args.name
                    with torch.no_grad():
                        for i, mel_test in enumerate(testset):
                            g_audio = G(mel_test.cuda())
                            g_audio = g_audio.squeeze().cpu()
                            audio = (g_audio.numpy() * 32768)
                            sf.write(os.path.join(save_dir, 'generated-{}-{}.wav'.format(step, i)),
                                     audio.astype('int16'),
                                     sample_rate)

                    print("Saving checkpoint")
                    torch.save({
                        'G': G.state_dict(),
                        'g_optimizer': g_optimizer.state_dict(),
                        'D': D.state_dict(),
                        'd_optimizer': d_optimizer.state_dict(),
                        'step': step,
                        'epoch': epoch
                    }, os.path.join(save_dir, 'ckpt-{}.pt'.format(step)))

    except Exception as e:
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-p', default=None)
    parser.add_argument('--name', '-n', required=True)
    args = parser.parse_args()
    save_dir = os.path.join('./ckpt', args.name)
    os.makedirs(save_dir, exist_ok=True)
    train(args)
