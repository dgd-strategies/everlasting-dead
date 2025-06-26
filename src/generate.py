import argparse
import glob
import os

import torch
import torchaudio
from torch import nn


def build_model(n_mels):
    return nn.GRU(
        input_size=n_mels, hidden_size=256, num_layers=2, batch_first=True
    )


def load_checkpoint(model_dir):
    ckpts = sorted(glob.glob(os.path.join(model_dir, "model.pt")))
    if not ckpts:
        raise FileNotFoundError("No checkpoint found")
    ckpt_path = ckpts[-1]
    state = torch.load(ckpt_path, map_location="cpu")
    model = build_model(n_mels=80)
    model.load_state_dict(state)
    model.eval()
    return model


def generate(model, length=16000, sample_rate=16000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    hidden = None
    n_mels = 80
    output = torch.zeros(1, 1, n_mels).to(device)
    mel_frames = []
    for _ in range(length):
        out, hidden = model(output, hidden)
        output = out[:, -1:, :]
        mel_frames.append(output.squeeze(0).squeeze(0).cpu())
    mel = torch.stack(mel_frames).transpose(0, 1)
    inv_mel = torchaudio.transforms.InverseMelScale(n_stft=201, n_mels=n_mels)
    griffin_lim = torchaudio.transforms.GriffinLim(n_fft=400, hop_length=160)
    spec = inv_mel(mel)
    waveform = griffin_lim(spec)
    return waveform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="models")
    parser.add_argument("--output_file", default="generated.wav")
    args = parser.parse_args()

    model = load_checkpoint(args.model_dir)
    waveform = generate(model)
    torchaudio.save(args.output_file, waveform, 16000)
    print(f"Saved audio to {args.output_file}")


if __name__ == "__main__":
    main()
