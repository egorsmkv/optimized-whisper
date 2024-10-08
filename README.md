# Optimized Whisper

## Required software

```
apt update && apt install -y ffmpeg build-essential cmake clang nvtop
```

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

## Run with Flash-Attention 2

```
export FLASH_ATTENTION_SKIP_CUDA_BUILD=true

uv pip install flash-attn --no-build-isolation
```

```
python run_batch_fa2.py
```

Benchmarks with Whisper Large V3:

```
Quantized Turbo:

  All Duration: 1174.8053
  All RTF: 0.0031
  All elapsed: 3.6684

Quantized:

  All Duration: 1174.6987
  All RTF: 0.0096
  All elapsed: 11.2452

FA2:

  All Duration: 1174.6987
  All RTF: 0.0194
  All elapsed: 22.8016
```

### Development

```
uv pip install ruff

ruff check
ruff format
```

