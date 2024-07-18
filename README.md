# Speaker Diarization

### [Faster Whisper](https://github.com/SYSTRAN/faster-whisper)

Make sure the following is in path in `.bashrc`:

```
export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64\${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}:${PYENV_ROOT}/versions/3.12.3/lib/python3.12/site-packages/nvidia/cublas/lib:${PYENV_ROOT}/versions/3.12.3/lib/python3.12/site-packages/nvidia/cudnn/lib
```

