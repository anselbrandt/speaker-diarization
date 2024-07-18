# Speaker Diarization

### [Faster Whisper](https://github.com/SYSTRAN/faster-whisper)

Make sure the following is in path in `.bashrc`:

```
export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64\${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}:${PYENV_ROOT}/versions/3.12.3/lib/python3.12/site-packages/nvidia/cublas/lib:${PYENV_ROOT}/versions/3.12.3/lib/python3.12/site-packages/nvidia/cudnn/lib
```

### [WhisperX](https://github.com/m-bain/whisperX/tree/main) (local)

#### [Missing argument for TranscriptionOptions.__new__()](https://github.com/m-bain/whisperX/issues/808)

Faster Whisper 1.0.3 requires `hotwords` argument.

Add `"hotwords": None`, in to `default_asr_options` in `whisperx/asr.py`

### Pyannote

Audio file should be imported like this, otherwise diarization will only use CPU:

```
from pyannote.audio import Audio

io = Audio(mono='downmix', sample_rate=16000)
waveform, sample_rate = io("audio.mp3")

diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})
```

#### Saving JSON

```
import json

with open("segments.json", "w") as outfile:
    json.dump(result["segments"], outfile)
```

####

```
from pyannote.audio import Audio
from pyannote.core import Segment

SAMPLE_CHUNK = Segment(15, 20)

audio_reader = Audio(sample_rate=model.hparams.sample_rate)
waveform, sample_rate = audio_reader.crop(wav_file, SAMPLE_CHUNK)
```