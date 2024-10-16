# Optimized Whisper

## Required packages

See `docker/cuda-12.3/Dockerfile` file to understand what packages your system should have.

## Installation

```
uv venv --python 3.12

source .venv/bin/activate

uv pip install --upgrade pip

uv pip install --upgrade --pre --index-url https://download.pytorch.org/whl/nightly/cu121 torch

uv pip install --upgrade transformers yt-dlp sphn
uv pip install git+https://github.com/mobiusml/hqq.git

git clone https://github.com/pytorch/ao
cd ao
python setup.py install
```

```
yt-dlp --extract-audio --audio-format wav -o "audio.wav" https://www.youtube.com/watch?v=u4dc1axRwE4

ffmpeg -y -i audio.wav -f segment -segment_time 30 -ac 1 -ar 16000 audio-chunk-%03d.wav
```

## Run

```
python run_batch.py
```

## Benchmarks with Whisper

```
Quantized Whisper Turbo:

  All Duration: 1174.8053 seconds (19.58 minutes)
  All RTF: 0.0031
  All elapsed: 3.6684 seconds

Quantized Whisper Large V3:

  All Duration: 1174.6987 seconds (19.58 minutes)
  All RTF: 0.0096
  All elapsed: 11.2452 seconds
```

## Development

```
uv pip install ruff

ruff check
ruff format
```

