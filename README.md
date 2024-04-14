# Speaker Diarization

Audio file should be imported like this, otherwise `diarization` will only use CPU:

```
io = Audio(mono='downmix', sample_rate=16000)
waveform, sample_rate = io("audio.mp3")

diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})
```
https://github.com/pyannote/pyannote-audio/issues/1652#issuecomment-1952136528

### Saving JSON

```
import json

with open("segments.json", "w") as outfile:
    json.dump(result["segments"], outfile)
```

## Pyannote Crop Convenience Function

```
from pyannote.audio import Audio
from pyannote.core import Segment

SAMPLE_CHUNK = Segment(15, 20)

audio_reader = Audio(sample_rate=model.hparams.sample_rate)
waveform, sample_rate = audio_reader.crop(wav_file, SAMPLE_CHUNK)
```