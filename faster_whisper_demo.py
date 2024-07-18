from faster_whisper import WhisperModel
import time
from datetime import timedelta
import torchaudio
from util import split_waveform_by_timestamps

startTime = time.time()

model_size = "large-v3"

input_file = "544_short.mp3"
output_file = "faster_whisper_out.txt"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, info = model.transcribe(input_file, beam_size=5)

print("Processing %s" % input_file)

outfile = open(output_file, "w")

print(
    "Detected language '%s' with probability %f"
    % (info.language, info.language_probability)
)

timestamps = []

for segment in segments:
    line = "[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text)
    outfile.write(line + "\n")
    timestamps.append((segment.start, segment.end, ""))

outfile.close()

endTime = time.time()
delta = endTime - startTime
elapsed = str(timedelta(seconds=delta))

print("Execution time: %s" % elapsed)

waveform, sample_rate = torchaudio.load(input_file)

split_waveform_by_timestamps(waveform, sample_rate, "faster_whisper_output", timestamps)
