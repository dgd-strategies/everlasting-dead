import argparse
import glob
import os

import torch
import torchaudio
from torch import nn
from torch.utils.data import Dataset, DataLoader


class AudioDataset(Dataset):
    """Loads audio files and converts them to mel spectrogram tensors."""

    def __init__(
        self, data_dir, sample_rate=16000, n_fft=400, hop_length=160, n_mels=80
    ):
        self.files = sorted(
            [
                f
                for f in glob.glob(os.path.join(data_dir, "*"))
                if f.lower().endswith((".wav", ".flac"))
            ]
        )
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.files[idx])
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sr, self.sample_rate
            )
        mel_spec = (
            self.mel(waveform).squeeze(0).transpose(0, 1)
        )  # (time, n_mels)
        return mel_spec


def collate_fn(batch):
    # Pad sequences to the same length
    lengths = [b.size(0) for b in batch]
    max_len = max(lengths)
    n_mels = batch[0].size(1)
    padded = torch.zeros(len(batch), max_len, n_mels)
    for i, b in enumerate(batch):
        padded[i, : b.size(0)] = b
    return padded, torch.tensor(lengths)


def build_model(n_mels):
    return nn.GRU(
        input_size=n_mels, hidden_size=256, num_layers=2, batch_first=True
    )


def train(model, dataloader, epochs, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        for batch, lengths in dataloader:
            batch = batch.to(device)
            output, _ = model(batch)
            loss = criterion(output, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--model_dir", default="models")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    dataset = AudioDataset(args.data_dir)
    dataloader = DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(n_mels=80).to(device)

    train(model, dataloader, args.epochs, device)

    ckpt_path = os.path.join(args.model_dir, "model.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved model to {ckpt_path}")


if __name__ == "__main__":
    main()
