import time

import torch
import sphn
import torchao
import hqq

import transformers

from glob import glob

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from hqq.models.hf.base import AutoHQQHFModel
from hqq.utils.patching import prepare_for_inference
from hqq.core.quantize import HQQBackend, HQQLinear, BaseQuantizeConfig

# logging
torch._logging.set_logs(graph_breaks=True, recompiles=True)

device = "cuda:0"
model_id = "openai/whisper-large-v3"
compute_dtype = torch.bfloat16
attn_implementation = "sdpa"
language = 'ukrainian'
bs = 24

print("torch:", torch.__version__)
print("torchao:", torchao.__version__)
print("hqq:", hqq.__version__)
print("transformers:", transformers.__version__)

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,
    use_safetensors=True,
    attn_implementation=attn_implementation,
    torch_dtype=compute_dtype,
)

processor = AutoProcessor.from_pretrained(model_id)

quant_config = BaseQuantizeConfig(
    nbits=4,
    group_size=64,
    quant_scale=False,
    quant_zero=False,
    axis=1,
    offload_meta=False,
)
HQQLinear.set_backend(HQQBackend.PYTORCH)

AutoHQQHFModel.quantize_model(
    model.model.decoder,
    quant_config=quant_config,
    compute_dtype=compute_dtype,
    device=device,
)

prepare_for_inference(model.model.decoder, backend="torchao_int4")

model.generation_config.cache_implementation = "static"
model.model.decoder.forward = torch.compile(
    model.model.decoder.forward, mode="reduce-overhead", fullgraph=True
)

model.model.encoder.to(device)

files = glob("audio-chunk-*.wav")
print("Audios files:", len(files))


def make_batches(iterable, n=1):
    ln = len(iterable)
    for ndx in range(0, ln, n):
        yield iterable[ndx : min(ndx + n, ln)]


def load_features(filename):
    data, _ = sphn.read(filename)

    input_features = processor(
        torch.tensor(data[0]), sampling_rate=16_000, return_tensors="pt"
    ).input_features

    input_features = input_features.to(compute_dtype).to(device)

    return input_features


def load_features_warmup(filename):
    data, _ = sphn.read(filename, sample_rate=16_000, duration_sec=1.0)  # load only 1 second

    input_features = processor(
        torch.tensor(data[0]), sampling_rate=16_000, return_tensors="pt"
    ).input_features

    input_features = input_features.to(compute_dtype).to(device)

    return input_features


warmup_input_features_batch = []
for f in files:
    warmup_input_features_batch.append(load_features_warmup(f))

print("W Files=", len(warmup_input_features_batch))

input_features_batch = []
for f in files:
    durations = sphn.durations([f])
    input_features_batch.append({
        'features': load_features(f),
        'duration': durations[0],
    })

print("Files=", len(input_features_batch))


concatenated_warmup = torch.cat(warmup_input_features_batch[:bs], dim=0)


t0 = time.time()

for it in range(3):
    print("Iter:", it)
    result = model.generate(concatenated_warmup, language=language)
    print(result)
    print("---")

print("---")
print("Warmup elapsed:", time.time() - t0)

t0_start = time.time()
all_total_elapsed = 0
durations_total = 0

for batch in make_batches(input_features_batch, bs):
    # In our test script, we skip other batches becase it leads to torch recompiles
    if len(batch) != bs:
        continue
        
    t0 = time.time()

    # Gather data
    concatenated_batch = torch.cat([b['features'] for b in batch], dim=0)
    durations = sum([b['duration'] for b in batch])
    
    generated_ids = model.generate(concatenated_batch, language=language)

    rec_elapsed = time.time() - t0

    print("---")
    print("Recognition elapsed:", rec_elapsed)

    t0 = time.time()

    transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)

    decoding_elapsed = time.time() - t0
    
    print("---")
    print("Decoding elapsed:", decoding_elapsed)

    for transcription in transcriptions:
        print(transcription)

    total_elapsed = rec_elapsed + decoding_elapsed
    
    print("Duration:", durations)
    print("RTF:", round(total_elapsed / durations, 4))

    all_total_elapsed += total_elapsed
    durations_total += durations

    print("***")

print('----')

print("All Duration:", durations_total)
print("All RTF:", round(all_total_elapsed / durations_total, 4))

print('All elapsed:', time.time() - t0_start)
