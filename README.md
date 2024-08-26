```
apt update

apt install -y rustc cargo 
apt install -y ffmpeg build-essential cmake
```

```
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv --python 3.10

source .venv/bin/activate

uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu118
uv pip install librosa
uv pip install -U transformers yt-dlp setuptools
uv pip install git+https://github.com/mobiusml/hqq.git

git clone https://github.com/pytorch/ao
cd ao
python setup.py install

uv pip install maturin

git clone https://github.com/LaurentMazare/sphn sphn-code
cd sphn-code
maturin build --release

uv pip install --force-reinstall /root/x/sphn-code/target/wheels/sphn-0.1.2-cp310-cp310-manylinux_2_34_x86_64.whl
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
