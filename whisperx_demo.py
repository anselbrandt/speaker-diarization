import whisperx
import gc
import os
from dotenv import load_dotenv

load_dotenv()

token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")

device = "cuda"
audio_file = "544_short.mp3"
batch_size = 16
compute_type = "float16"

# 1. Transcribe with original whisper (batched)
model = whisperx.load_model("large-v2", device, compute_type=compute_type)

# save model to local path (optional)
# model_dir = "/path/"
# model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)
# before alignment
for line in result["segments"]:
    print(line)

# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model

# 2. Align whisper output
model_a, metadata = whisperx.load_align_model(
    language_code=result["language"], device=device
)
result = whisperx.align(
    result["segments"], model_a, metadata, audio, device, return_char_alignments=False
)

# after alignment
for line in result["segments"]:
    print(line)

# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

# 3. Assign speaker labels
diarize_model = whisperx.DiarizationPipeline(use_auth_token=token, device=device)

# add min/max number of speakers if known
diarize_segments = diarize_model(audio)
# diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

result = whisperx.assign_word_speakers(diarize_segments, result)
for line in diarize_segments:
    print(line)

# segments are now assigned speaker IDs
for line in result["segments"]:
    print(line)
