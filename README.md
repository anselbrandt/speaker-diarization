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
