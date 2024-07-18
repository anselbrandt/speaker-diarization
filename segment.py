import os
from mutagen.mp3 import MP3

import time
from datetime import timedelta

from dotenv import load_dotenv
from pyannote.audio import Audio

from utils import split_waveform_by_timestamps

import whisperx

load_dotenv()

token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")

device = "cuda"
batch_size = 16
compute_type = "float16"
model_dir = "./data"


def processfile(file):
    startTime = time.time()
    (path, filename, dir) = file
    audio_file = path
    episode_dir = (
        dir.replace("files", "out")
        + "/"
        + filename.replace(" ", "_").replace(".mp3", "/")
    )

    model = whisperx.load_model(
        "large-v2",
        device,
        compute_type=compute_type,
        download_root=model_dir,
        language="en",
    )

    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)

    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    diarize_model = whisperx.DiarizationPipeline(use_auth_token=token, device=device)

    diarize_segments = diarize_model(audio, min_speakers=2, max_speakers=2)

    result = whisperx.assign_word_speakers(diarize_segments, result)

    timestamps = []
    for segment in result["segments"]:
        if "speaker" in segment.keys():
            speaker = segment["speaker"]
        else:
            speaker = "unknown"
        timestamp = (segment["start"], segment["end"], speaker)
        timestamps.append(timestamp)

    selected_timestamps = sorted(
        timestamps, key=lambda tup: tup[1] - tup[0], reverse=True
    )[:10]

    io = Audio(mono="downmix", sample_rate=16000)
    waveform, sample_rate = io(audio_file)

    split_waveform_by_timestamps(
        waveform, sample_rate, episode_dir, selected_timestamps
    )

    endTime = time.time()
    delta = endTime - startTime
    elapsed = str(timedelta(seconds=delta))

    print(filename, elapsed)


root = "files"

dirs = [
    os.path.join(root, dir)
    for dir in os.listdir(root)
    if os.path.isdir(os.path.join(root, dir))
]

for dir in dirs:
    outpath = dir.replace("files", "out")
    os.makedirs(outpath, exist_ok=True)
    files = []
    sum = 0
    for file in os.listdir(dir):
        if ".DS_Store" not in file:
            if os.path.isfile(os.path.join(dir, file)):
                filename = os.path.join(dir, file)
                files.append(filename)
                audio = MP3(filename)
                length = audio.info.length
                sum = sum + length
    print(dir.replace("files/", ""), len(files), "files,", timedelta(seconds=sum))

files = [
    (os.path.join(dir, file), file, dir)
    for dir in dirs
    for file in os.listdir(dir)
    if os.path.isfile(os.path.join(dir, file))
    if ".DS_Store" not in file
]

print(len(files), "files total")

for file in files:
    processfile(file)
