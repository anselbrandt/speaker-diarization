import wave
import os

def split_wav_by_timestamps(input_file, output_dir, timestamps, sortby='speaker'):
    # Open the WAV file
    with wave.open(input_file, 'rb') as wf:
        # Get the parameters of the input WAV file
        params = wf.getparams()

        # Iterate through the timestamps and split the WAV file accordingly
        for i, (start_time, end_time, speaker) in enumerate(timestamps):
            # Calculate the start and end frame indices based on timestamps
            start_frame = int(start_time * params[2])
            end_frame = int(end_time * params[2])

            # Set the position to the start frame
            wf.setpos(start_frame)

            # Read the frames from start to end frame
            chunk_data = wf.readframes(end_frame - start_frame)

            # Create a new output WAV file for each chunk
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