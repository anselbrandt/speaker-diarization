import wave
import os

def split_wav_by_timestamps(input_file, output_dir, timestamps, sortby='speaker'):

    with wave.open(input_file, 'rb') as wf:
        params = wf.getparams()

        for i, (start_time, end_time, speaker) in enumerate(timestamps):
            start_frame = int(start_time * params[2])
            end_frame = int(end_time * params[2])

            wf.setpos(start_frame)

            chunk_data = wf.readframes(end_frame - start_frame)

            if sortby=='timestamp':
                output_file = os.path.join(output_dir, f'{speaker}_{start_time}_{end_time}.wav')
            else:
                output_file = os.path.join(output_dir, f'{start_time}_{end_time}_{speaker}.wav')
            with wave.open(output_file, 'wb') as chunk_wf:
                chunk_wf.setparams(params)
                chunk_wf.writeframes(chunk_data)

def aggregate_timestamps(timestamps):
    aggregated = []
    previous = timestamps[0]
    for timestamp in timestamps:
        start, end, speaker = timestamp
        prevstart, prevend, prevspeaker = previous
        if speaker == prevspeaker:
            previous = (prevstart, end, speaker)
        else:
            aggregated.append(previous)
            previous = timestamp
    return aggregated