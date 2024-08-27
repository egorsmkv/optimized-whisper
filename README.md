```
apt update

apt install -y rustc cargo ffmpeg build-essential cmake clang
```

```
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv --python 3.12

source .venv/bin/activate

uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124
uv pip install -U transformers yt-dlp setuptools maturin
uv pip install git+https://github.com/mobiusml/hqq.git

git clone https://github.com/pytorch/ao
cd ao
python setup.py install

git clone https://github.com/LaurentMazare/sphn sphn-code
cd sphn-code
maturin build --release

uv pip install --force-reinstall /root/sphn-code/target/wheels/sphn-0.1.2-cp310-cp310-manylinux_2_34_x86_64.whl
```

```
yt-dlp --extract-audio --audio-format wav -o "audio.wav" https://www.youtube.com/watch?v=u4dc1axRwE4

ffmpeg -y -i audio.wav -f segment -segment_time 30 -ac 1 -ar 16000 audio-chunk-%03d.wav
```

```
wget -O run_batch.py "https://raw.githubusercontent.com/egorsmkv/optimized-whisper/main/run_batch.py?token=GHSAT0AAAAAABZRL3ITAFJVN767FLZ4J7YCZWMUABQ"

python run_batch.py
```

### dev

```
uv pip install ruff

ruff check
ruff format
```
