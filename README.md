# Everlasting Dead

This repository contains a minimal example of how you could train a
machine learning model on a collection of Grateful Dead live
recordings and then generate new audio in a similar style. The
repository does not include any copyrighted audio files. You must
provide your own dataset of Grateful Dead live recordings and place the
files inside the `data/` directory (or adjust the path in the training
script).

**Disclaimer**: Training a high‑quality generative model on audio data
is computationally expensive. The code here is intended as a starting
point only. It uses a simple recurrent neural network and
`torchaudio` for audio processing. Feel free to adapt the code to more
advanced models.

## Setup

1. Install Python 3.10 or higher.
2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Place your Grateful Dead live recordings (e.g. FLAC or WAV files)
   under the `data/` directory. You can obtain many of these legally
   from [archive.org](https://archive.org/details/GratefulDead).

## Training

Run the training script with:

```bash
python src/train.py --data_dir data --epochs 10 --model_dir models
```

This will preprocess the audio into Mel spectrograms and train a simple
sequence model. The trained weights will be saved in `models/`.

## Generating Audio

After training, you can generate a new audio clip with:

```bash
python src/generate.py --model_dir models --output_file generated.wav
```

The script loads the latest checkpoint in `models/` and produces a
short audio sample saved to `generated.wav`.

## Notes

- The provided scripts are intentionally lightweight. For serious
  results, consider more sophisticated architectures like WaveNet,
  DiffWave, or fine‑tuned transformers.
- Always respect copyright laws when downloading or distributing
  recordings.
