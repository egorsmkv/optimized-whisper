# Optimized Whisper

## Required software

```
apt update && apt install -y rustc cargo ffmpeg build-essential cmake clang nvtop
```

## Installation

```
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv --python 3.12

source .venv/bin/activate

uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
uv pip install -U transformers yt-dlp setuptools maturin patchelf
uv pip install git+https://github.com/mobiusml/hqq.git

git clone https://github.com/pytorch/ao
cd ao
python setup.py install

git clone https://github.com/LaurentMazare/sphn sphn-code
cd sphn-code
maturin build --release

uv pip install --force-reinstall /root/sphn-code/target/wheels/sphn-0.1.2-cp312-cp312-manylinux_2_34_x86_64.whl
```

```
git clone https://github.com/egorsmkv/optimized-whisper

cd optimized-whisper

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

Some benchmarks:

```
Quant:

  All Duration: 1174.6987
  All RTF: 0.0111
  All elapsed: 12.9838

FA2 + Quant:

  All Duration: 1174.6987
  All RTF: 0.0209
  All elapsed: 24.5375

FA2:

  All Duration: 1174.6987
  All RTF: 0.0262
  All elapsed: 30.7904
```

### Development

```
uv pip install ruff

ruff check
ruff format
```

