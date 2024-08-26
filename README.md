```
apt update
apt install -y ffmpeg build-essential
```

```
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv --python 3.10

source .venv/bin/activate

uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu118
uv pip install torchaudio
uv pip install -U transformers yt-dlp setuptools
uv pip install git+https://github.com/mobiusml/hqq.git

git clone https://github.com/pytorch/ao
cd ao
python setup.py install
```

```
yt-dlp --extract-audio --audio-format wav -o "audio.wav" https://www.youtube.com/watch?v=u4dc1axRwE4

ffmpeg -y -i audio.wav -f segment -segment_time 30 -ac 1 -ar 16000 audio-chunk-%03d.wav
```

### dev

```
uv pip install ruff

ruff check
ruff format
```
